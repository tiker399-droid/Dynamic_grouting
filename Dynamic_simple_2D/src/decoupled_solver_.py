"""
解耦求解器 - 顺序求解压力场和位移场（无量纲化版本）
压力方程：包含土骨架速度项（使用上一时间步的位移速度近似）
位移方程：线弹性（压力作为已知体力）
粘度随时间变化（浆液粘度）
所有变量均已无量纲化，求解完成后可通过特征尺度恢复物理值
"""

import numpy as np
from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py import PETSc
import ufl
import logging


class DecoupledSolver:
    def __init__(self, comm, materials, bc_manager, V_u, V_p, config):
        self.comm = comm
        self.mat = materials
        self.bc_manager = bc_manager
        self.V_u = V_u
        self.V_p = V_p
        self.config = config
        self.rank = comm.Get_rank()
        self.logger = logging.getLogger(f"DecoupledSolver_rank{self.rank}")

        # -------------------- 材料常数（物理值）--------------------
        self.g = self.mat.g                     # 重力向量 (m/s²)
        self.k0 = self.mat.k0                    # 初始渗透率 (m²)
        self.rho_g = self.mat.rho_g               # 浆液密度 (kg/m³)
        self.rho_s = self.mat.rho_s               # 土颗粒密度 (kg/m³)
        self.phi0 = self.mat.phi0                 # 初始孔隙度
        self.E = self.mat.E                       # 杨氏模量 (Pa)
        self.nu = self.mat.nu                     # 泊松比
        self.mu0 = self.mat.mu_g0                 # 浆液初始粘度 (Pa·s)
        self.rho_w = self.mat.rho_w                # 水密度 (kg/m³)
        self.g_mag = self.mat.g_magnitude          # 重力加速度大小 (m/s²)

        # -------------------- 几何特征尺度 --------------------
        self.L = float(config['geometry']['height'])      # 特征长度 (m)

        # -------------------- 压力特征尺度 --------------------
        # 采用最大注浆压力作为压力特征尺度
        self.P = float(config['materials']['grout']['pressure'])    # 特征压力 (Pa)

        # -------------------- 位移特征尺度 --------------------
        # 由弹性关系 U = P * L / E
        self.U = self.P * self.L / self.E          # 特征位移 (m)

        # -------------------- 时间特征尺度 --------------------
        # 由压力扩散项与骨架速度项平衡导出：T = L² * μ0 / (k0 * E)
        self.T = (self.L**2 * self.mu0) / (self.k0 * self.E)   # 特征时间 (s)
        self.T = float(config['simulation']['total_time'])

        # -------------------- 无量纲参数 --------------------
        # 重力数 Gr = ρ_g g L / P  （用于压力方程重力项）
        self.Gr = self.rho_g * self.g_mag * self.L / self.P
        # 体力数 Gr_b = ρ_b g L / P  （用于位移方程体力项）
        rho_bulk = (1 - self.phi0) * self.rho_s + self.phi0 * self.rho_w
        self.Gr_b = rho_bulk * self.g_mag * self.L / self.P

        if self.rank == 0:
            self.logger.info(f"特征长度 L = {self.L:.3e} m")
            self.logger.info(f"特征压力 P = {self.P:.3e} Pa")
            self.logger.info(f"特征位移 U = {self.U:.3e} m")
            self.logger.info(f"特征时间 T = {self.T:.3e} s")
            self.logger.info(f"重力数 Gr = {self.Gr:.3e}")
            self.logger.info(f"体力数 Gr_b = {self.Gr_b:.3e}")

        # -------------------- 转换为 UFL 常数 --------------------
        self.Gr_const = fem.Constant(self.V_u.mesh, self.Gr)
        self.Gr_b_const = fem.Constant(self.V_u.mesh, self.Gr_b)

        # 将一些常数包装为 UFL 常数（用于方程组装）
        self.k0_const = self.mat.k0_constant       # 渗透率 (假设为常数)
        self.rho_g_const = self.mat.rho_g_constant
        self.rho_s_const = self.mat.rho_s_constant
        self.rho_w_const = self.mat.rho_w_constant
        self.phi0_const = self.mat.phi0_constant
        self.g_const = self.mat.g

        # 粘度相关：mu_current_constant 是物理粘度 (随时间变化)，需要无量纲化
        # 注意：mu_current_constant 在每次时间步前会由 materials.update_time_dependent_properties 更新
        self.mu0_const = fem.Constant(self.V_u.mesh, self.mu0)   # 初始粘度常数

        # -------------------- 求解器选项 --------------------
        self.pressure_ksp_type = config.get('pressure_ksp_type', 'cg')
        self.pressure_pc_type = config.get('pressure_pc_type', 'hypre')
        self.displacement_ksp_type = config.get('displacement_ksp_type', 'gmres')
        self.displacement_pc_type = config.get('displacement_pc_type', 'lu')
        self.ksp_rtol = config.get('ksp_rtol', 1e-10)
        self.ksp_atol = config.get('ksp_atol', 1e-12)
        self.ksp_max_it = config.get('ksp_max_it', 1000)

        # -------------------- 位移历史（无量纲位移）-------------------
        self.u_prev = None          # u^* 在 t_n
        self.u_prev2 = None         # u^* 在 t_{n-1}

    def solve(self, dt, time, u, p, u_prev, p_prev):
        """
        求解一个时间步
        Parameters:
            dt: 物理时间步长 (s)
            time: 当前物理时间 (s)
            u: 位移函数 (更新为新的无量纲位移)
            p: 压力函数 (更新为新的无量纲压力)
            u_prev: 上一时间步的无量纲位移
            p_prev: 上一时间步的无量纲压力 (未使用，保留接口)
        """
        # 更新与时间相关的材料属性（如粘度）
        self.mat.update_time_dependent_properties(time)

        # 更新位移历史
        self.u_prev2 = self.u_prev
        self.u_prev = u_prev

        # 求解压力
        self._solve_pressure(p, time, dt)

        # 求解位移
        self._solve_displacement(u, p, time)

        return True, 0

    def _solve_pressure(self, p_func, time, dt):
        """组装并求解无量纲压力方程"""
        bcs_p = self.bc_manager.get_pressure_bcs()   # 边界条件应为无量纲值
        v_p = ufl.TestFunction(self.V_p)
        p_trial = ufl.TrialFunction(self.V_p)

        # 当前无量纲粘度 μ* = μ(t) / μ0
        mu_phys = self.mat.mu_current_constant       # 物理粘度 (更新后)
        mu_star = mu_phys / self.mu0_const
        # 为避免除以零，确保 mu0_const 不为零；这里假设 mu0 > 0

        # 左端：扩散项 (1/μ*) ∇p* · ∇v
        a = fem.form(ufl.inner((1.0 / mu_star) * ufl.grad(p_trial), ufl.grad(v_p)) * ufl.dx)

        # 右端：重力项 Gr * e_g · ∇v
        # 重力方向向下，假设坐标系竖直向上为正，则 e_g = (0, 0, -1) 或 (0, -1)
        mesh = self.V_p.mesh
        gdim = mesh.geometry.dim
        if gdim == 2:
            e_g = ufl.as_vector([0, -1])
        else:
            e_g = ufl.as_vector([0, 0, -1])
        L_grav = self.Gr_const * ufl.inner((1.0 / mu_star) * e_g, ufl.grad(v_p)) * ufl.dx

        # 右端：骨架速度项 ∫ v_s^* · ∇v dx  其中 v_s^* = (u_prev - u_prev2) / (dt/T)
        if self.u_prev is not None and self.u_prev2 is not None:
            # 无量纲时间步长
            dt_star = dt / self.T * 1e5
            v_s_star = (self.u_prev - self.u_prev2) / dt_star
            L_body = 0#ufl.inner(v_s_star, ufl.grad(v_p)) * ufl.dx
        else:
            L_body = 0

        L = fem.form(L_grav + L_body)

        # 组装矩阵
        A = petsc.assemble_matrix(a, bcs=bcs_p)
        A.assemble()

        # 组装右端项
        b = petsc.assemble_vector(L)
        petsc.apply_lifting(b, [a], [bcs_p])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs_p)

        # 创建求解器
        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")                     # 直接法
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        ksp.setTolerances(rtol=self.ksp_rtol, atol=self.ksp_atol, max_it=self.ksp_max_it)

        # 求解
        ksp.solve(b, p_func.x.petsc_vec)
        p_func.x.scatter_forward()

        # 简单检查解（无量纲压力应大致在 O(1) 附近）
        p_array = p_func.x.array
        if self.rank == 0:
            print(f"无量纲压力范围: min={p_array.min():.6e}, max={p_array.max():.6e}")
            reason = ksp.getConvergedReason()
            print(f"压力求解器收敛原因: {reason}")

        ksp.destroy()

    def _solve_displacement(self, u_func, p_func, time):
        """组装并求解无量纲位移方程"""
        bcs_u = self.bc_manager.get_displacement_bcs()   # 边界条件应为无量纲值
        v_u = ufl.TestFunction(self.V_u)
        u_trial = ufl.TrialFunction(self.V_u)

        # 无量纲弹性参数 (除以 E)
        lambda_phys, mu_phys = self.mat.get_lame_parameters()   # 物理值
        lambda_star = lambda_phys / self.E
        mu_star = mu_phys / self.E

        def sigma_star(u):
            eps = ufl.sym(ufl.grad(u))
            return lambda_star * ufl.tr(eps) * ufl.Identity(len(u)) + 2 * mu_star * eps

        # 左端：弹性项
        a = fem.form(ufl.inner(sigma_star(u_trial), ufl.grad(v_u)) * ufl.dx)

        # 右端：压力耦合项 (p* ∇·v)  注意原方程是 + ∇p 项，分部积分后得到 -p ∇·v，这里根据实际弱形式调整
        # 从原始物理方程 ∇·σ = -ρg - ∇p 出发，乘以虚位移 v 并分部积分，得到：
        # ∫ σ:∇v dx = ∫ ρg·v dx + ∫ p ∇·v dx （假设边界项已处理）
        # 因此右端应为 + p ∇·v
        L_pressure = ufl.inner(p_func, ufl.div(v_u)) * ufl.dx

        # 右端：体力项 Gr_b e_g · v
        mesh = self.V_u.mesh
        gdim = mesh.geometry.dim
        if gdim == 2:
            e_g = ufl.as_vector([0, -1])
        else:
            e_g = ufl.as_vector([0, 0, -1])
        L_body = self.Gr_b_const * ufl.inner(e_g, v_u) * ufl.dx

        L = fem.form(L_pressure + L_body)

        # 组装矩阵
        A = petsc.assemble_matrix(a, bcs=bcs_u)
        A.assemble()

        # 组装右端项
        b = petsc.assemble_vector(L)
        petsc.apply_lifting(b, [a], [bcs_u])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs_u)

        # 创建求解器
        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(A)
        ksp.setType(self.displacement_ksp_type)
        pc = ksp.getPC()
        pc.setType(self.displacement_pc_type)
        if self.displacement_pc_type == "lu":
            pc.setFactorSolverType("mumps")
        ksp.setTolerances(rtol=self.ksp_rtol, atol=self.ksp_atol, max_it=self.ksp_max_it)

        ksp.solve(b, u_func.x.petsc_vec)
        u_func.x.scatter_forward()

        # 检查解
        u_array = u_func.x.array
        if self.rank == 0:
            print(f"无量纲位移范围: min={u_array.min():.6e}, max={u_array.max():.6e}")
            reason = ksp.getConvergedReason()
            print(f"位移求解器收敛原因: {reason}")

        ksp.destroy()

    # ---------- 辅助方法：将无量纲解恢复为物理值 ----------
    def u_to_physical(self, u_star):
        """将无量纲位移转换为物理位移"""
        return u_star * self.U

    def p_to_physical(self, p_star):
        """将无量纲压力转换为物理压力"""
        return p_star * self.P

    def t_to_physical(self, t_star):
        """将无量纲时间转换为物理时间"""
        return t_star * self.T

    def u_star_from_physical(self, u_phys):
        """将物理位移转换为无量纲位移"""
        return u_phys / self.U

    def p_star_from_physical(self, p_phys):
        """将物理压力转换为无量纲压力"""
        return p_phys / self.P