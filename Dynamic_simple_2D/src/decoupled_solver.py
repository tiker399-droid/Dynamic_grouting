"""
解耦求解器 - 顺序求解压力场和位移场
压力方程：包含土骨架速度项（使用上一时间步的位移速度近似）
位移方程：线弹性（压力作为已知体力）
粘度随时间变化（浆液粘度）
独立函数空间版本，包含详细调试检查和完整无量纲化
"""

import numpy as np
from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py import PETSc
import ufl
import logging


class DecoupledSolver:
    """
    解耦求解器，每个时间步内依次求解压力场和位移场。
    压力方程：∇·(v_s) + ∇·(k/μ (∇p - ρg)) = 0
    位移方程：∇·σ' - α∇p - ρ_bulk g = 0
    其中 v_s 用上一时间步的位移速度近似。
    无量纲化版本。
    """

    def __init__(self, comm, materials, bc_manager, V_u, V_p, config):
        """
        初始化解耦求解器

        Args:
            comm: MPI 通信器
            materials: 材料属性管理器
            bc_manager: 边界条件管理器（提供 get_pressure_bcs 和 get_displacement_bcs 方法）
            V_u: 位移函数空间
            V_p: 压力函数空间
            config: 配置字典（用于求解器选项）
        """
        self.comm = comm
        self.mat = materials
        self.bc_manager = bc_manager
        self.V_u = V_u
        self.V_p = V_p
        self.config = config

        self.rank = comm.Get_rank()
        self.logger = logging.getLogger(f"DecoupledSolver_rank{self.rank}")

        # 材料常数
        self.g = self.mat.g                 # 重力向量 (有量纲)
        self.k0 = self.mat.k0                # 渗透率常数
        self.rho_g = self.mat.rho_g          # 浆液密度
        self.rho_s = self.mat.rho_s
        self.phi0 = self.mat.phi0
        self.E = self.mat.E
        self.nu = self.mat.nu

        # 随时间变化的粘度常数（由 materials 维护）
        self.mu = self.mat.mu_current_constant

        # --- 计算参考量和缩放因子 ---
        self.L0 = config['geometry']['height']
        self.rho_w = self.mat.rho_w
        self.g_mag = self.mat.g_magnitude
        self.P0 = self.rho_w * self.g_mag * self.L0
        self.U0 = self.P0 * self.L0 / self.E
        self.mu_w = self.mat.mu_w
        self.t0 = self.L0**2 * self.mu_w / (self.k0 * self.E)

        self.scale_p = self.L0**2 * self.mu_w / (self.k0 * self.P0)
        self.scale_u = 1.0 / (self.E * self.L0)
        self.scale_up = 1.0 / (self.P0 * self.L0)
        self.scale_vs = self.t0 / (self.U0 * self.L0)

        # 将缩放因子转换为 UFL 常数
        self.scale_p_const = fem.Constant(self.V_u.mesh, self.scale_p)
        self.scale_u_const = fem.Constant(self.V_u.mesh, self.scale_u)
        self.scale_up_const = fem.Constant(self.V_u.mesh, self.scale_up)
        self.scale_vs_const = fem.Constant(self.V_u.mesh, self.scale_vs)

        # 材料常数已经是 fem.Constant，可以直接使用
        self.k0_const = self.mat.k0_constant
        self.mu_const = self.mat.mu_current_constant  # 随时间变化
        self.rho_g_const = self.mat.rho_g_constant
        self.rho_s_const = self.mat.rho_s_constant
        self.phi0_const = self.mat.phi0_constant
        self.g_const = self.mat.g  # 已经是 fem.Constant

        # 求解器选项
        self.pressure_ksp_type = config.get('pressure_ksp_type', 'cg')
        self.pressure_pc_type = config.get('pressure_pc_type', 'hypre')
        self.displacement_ksp_type = config.get('displacement_ksp_type', 'gmres')
        self.displacement_pc_type = config.get('displacement_pc_type', 'lu')
        self.ksp_rtol = config.get('ksp_rtol', 1e-10)
        self.ksp_atol = config.get('ksp_atol', 1e-12)
        self.ksp_max_it = config.get('ksp_max_it', 1000)

        # 存储两个之前的位移场（用于计算土骨架速度）
        self.u_prev = None   # 上一时间步的位移
        self.u_prev2 = None  # 上上时间步的位移

    def solve(self, dt, time, u, p, u_prev, p_prev):
        """
        执行一个时间步的解耦求解

        Args:
            dt: 时间步长 (有量纲)
            time: 当前时间 (有量纲)
            u: 当前位移函数（将被更新）
            p: 当前压力函数（将被更新）
            u_prev: 上一步位移函数
            p_prev: 上一步压力函数

        Returns:
            (converged, iterations): 收敛标志和迭代次数（此处始终返回 True, 0）
        """
        # 更新时间相关的材料属性（例如粘度）
        self.mat.update_time_dependent_properties(time)

        # 更新位移历史
        self.u_prev2 = self.u_prev
        self.u_prev = u_prev

        # 打印位移历史统计（用于调试）
        if self.u_prev is not None:
            u_prev_array = self.u_prev.x.array
            print(f"u_prev: min={u_prev_array.min():.6e}, max={u_prev_array.max():.6e}, NaN={np.any(np.isnan(u_prev_array))}")
        if self.u_prev2 is not None:
            u_prev2_array = self.u_prev2.x.array
            print(f"u_prev2: min={u_prev2_array.min():.6e}, max={u_prev2_array.max():.6e}, NaN={np.any(np.isnan(u_prev2_array))}")
        '''
        print(f"L0 = {self.L0}")
        print(f"P0 = {self.P0:.3e}")
        print(f"U0 = {self.U0:.3e}")
        print(f"t0 = {self.t0:.3e}")
        print(f"scale_p = {self.scale_p:.3e}")
        print(f"scale_u = {self.scale_u:.3e}")
        print(f"scale_up = {self.scale_up:.3e}")
        print(f"scale_vs = {self.scale_vs:.3e}")
        '''
        # 1. 求解压力场（使用上一时间步的位移速度）
        self._solve_pressure(p, time, dt)

        # 2. 求解位移场（使用新的压力）
        self._solve_displacement(u, p, time)

        # 检查侧面压力边界值
        self.check_side_pressure(p, self.bc_manager, self.V_u.mesh, self.comm)

        return True, 0

    def _solve_pressure(self, p_func, time, dt):
        """
        求解压力场（稳态扩散方程 + 土骨架速度项），无量纲形式
        """
        print("===== 进入 _solve_pressure =====")
        # 获取压力边界条件
        bcs_p = self.bc_manager.get_pressure_bcs()
        print(f"压力边界条件数量: {len(bcs_p)}")

        # 定义试验函数和测试函数
        v_p = ufl.TestFunction(self.V_p)
        p_trial = ufl.TrialFunction(self.V_p)

        # 使用 UFL 常数
        k = self.k0_const
        mu = self.mu_const
        rho = self.rho_g_const
        g = self.g_const
        scale_p = self.scale_p_const
        scale_vs = self.scale_vs_const

        print(f"k0 = {k}")
        print(f"mu = {mu}")
        print(f"rho_g = {rho}")
        print(f"g = {g.value}")

        # 双线性形式（扩散项），已应用缩放因子
        a = fem.form(ufl.inner((k / mu) * ufl.grad(p_trial), ufl.grad(v_p)) * ufl.dx)

        # 构建线性形式表达式（基础重力项），同样应用缩放
        L_expr = ufl.inner((k / mu) * rho * g, ufl.grad(v_p)) * ufl.dx

        # 如果有上一时间步的位移，添加土骨架速度项（需应用 scale_vs）
        if self.u_prev is not None and self.u_prev2 is not None:
            '''
            # 计算位移历史的最大绝对值（无量纲）
            max_u = np.max(np.abs(self.u_prev.x.array - self.u_prev2.x.array))
            # 设定阈值（无量纲），例如 0.1（对应物理位移约 8 mm，可根据需要调整）
            threshold = 100000

            if max_u < threshold:
            '''    
            v_s = (self.u_prev - self.u_prev2) / dt
            L_expr -=  1e-6 * ufl.div(v_s) * v_p * ufl.dx
            '''
            else:
                print(f"警告: 位移过大 (max_u={max_u:.3e})，跳过土骨架速度项")
            '''
        L = fem.form(L_expr)

        # 组装矩阵和向量
        A = petsc.assemble_matrix(a, bcs=bcs_p)
        A.assemble()

        # 检查矩阵对角元
        diag = A.getDiagonal()
        diag_array = diag.getArray()
        print(f"压力矩阵对角元范围: min={diag_array.min():.6e}, max={diag_array.max():.6e}")
        if np.any(np.isinf(diag_array)):
            print("压力矩阵对角元包含 Inf！")
        if np.any(np.isnan(diag_array)):
            print("压力矩阵对角元包含 NaN！")
        zero_diag = np.where(np.abs(diag_array) < 1e-12)[0]
        if len(zero_diag) > 0:
            print(f"压力矩阵有 {len(zero_diag)} 个零对角元")

        b = petsc.assemble_vector(L)
        petsc.apply_lifting(b, [a], [bcs_p])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs_p)

        # 检查右端项
        b_array = b.getArray()
        print(f"压力右端项范围: min={b_array.min():.6e}, max={b_array.max():.6e}")
        if np.any(np.isnan(b_array)):
            print("压力右端项包含 NaN！")
            raise RuntimeError("压力右端项包含 NaN")
        if np.any(np.isinf(b_array)):
            print("压力右端项包含 Inf！")
            raise RuntimeError("压力右端项包含 Inf")

        # 创建求解器
        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")   # 直接求解
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")

        # 求解
        ksp.solve(b, p_func.x.petsc_vec)
        p_func.x.scatter_forward()

        # 检查解
        p_array = p_func.x.array
        print(f"压力解范围: min={p_array.min():.6e}, max={p_array.max():.6e}")
        if np.any(np.isinf(p_array)):
            print("压力解包含 Inf！")
            raise RuntimeError("压力解包含 Inf")
        if np.any(np.isnan(p_array)):
            print("压力解包含 NaN！")
            raise RuntimeError("压力解包含 NaN")

        if self.rank == 0:
            reason = ksp.getConvergedReason()
            self.logger.info(f"压力求解器收敛原因: {reason}")

        ksp.destroy()

    def _solve_displacement(self, u_func, p_func, time):
        """
        求解位移场（线弹性，压力作为体力），无量纲形式
        """
        print("===== 进入 _solve_displacement =====")
        # 获取位移边界条件
        bcs_u = self.bc_manager.get_displacement_bcs()
        print(f"位移边界条件数量: {len(bcs_u)}")

        # 定义试验函数和测试函数
        v_u = ufl.TestFunction(self.V_u)
        u_trial = ufl.TrialFunction(self.V_u)

        # 材料属性
        def sigma(u):
            epsilon = ufl.sym(ufl.grad(u))
            lambda_, mu = self.mat.get_lame_parameters()
            return lambda_ * ufl.tr(epsilon) * ufl.Identity(len(u)) + 2 * mu * epsilon

        # 整体密度（常数，使用浆液密度）
        rho_bulk = (1 - self.phi0_const) * self.rho_s_const + self.phi0_const * self.rho_g_const
        self.scale_u = self.scale_u_const
        self.scale_up = self.scale_up_const

        # 打印位移求解前的压力解
        p_array = p_func.x.array
        print(f"位移求解前压力解范围: min={p_array.min():.6e}, max={p_array.max():.6e}")

        # 变分形式
        a = fem.form(ufl.inner(sigma(u_trial), ufl.grad(v_u)) * ufl.dx)
        L = fem.form(self.mat.alpha * p_func * ufl.div(v_u) * ufl.dx
                     + ufl.dot(rho_bulk * self.g, v_u) * ufl.dx)

        # 组装矩阵和向量
        A = petsc.assemble_matrix(a, bcs=bcs_u)
        A.assemble()
        b = petsc.assemble_vector(L)
        petsc.apply_lifting(b, [a], [bcs_u])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs_u)

        # 检查右端项
        b_array = b.getArray()
        print(f"位移右端项范围: min={b_array.min():.6e}, max={b_array.max():.6e}")
        if np.any(np.isnan(b_array)):
            print("位移右端项包含 NaN！")
            raise RuntimeError("位移右端项包含 NaN")
        if np.any(np.isinf(b_array)):
            print("位移右端项包含 Inf！")
            nan_idx = np.where(np.isinf(b_array))[0]
            print(f"Inf 索引: {nan_idx[:10]}")
            raise RuntimeError("位移右端项包含 Inf")

        # 创建求解器
        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(A)
        ksp.setType(self.displacement_ksp_type)
        ksp.setTolerances(rtol=self.ksp_rtol, atol=self.ksp_atol, max_it=self.ksp_max_it)
        pc = ksp.getPC()
        pc.setType(self.displacement_pc_type)
        if self.displacement_pc_type == "lu":
            pc.setFactorSolverType("mumps")

        # 求解
        ksp.solve(b, u_func.x.petsc_vec)
        u_func.x.scatter_forward()

        # 检查解
        u_array = u_func.x.array
        '''
        # 求解完成后，强制将异常自由度的值置为 0
        bad_indices = np.where(np.abs(u_array) > 1000)[0]
        if len(bad_indices) > 0:
            print(f"发现{len(bad_indices)}个异常自由度，将它们归零")
            u_func.x.array[bad_indices] = 0.0
            u_func.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        '''
        print(f"位移解范围: min={u_array.min():.6e}, max={u_array.max():.6e}")

        if np.any(np.isinf(u_array)):
            print("位移解包含 Inf！")
            # 可保存位移解以供分析
            np.savetxt(f"u_inf_t{time}.txt", u_array)
            raise RuntimeError("位移解包含 Inf")
        if np.any(np.isnan(u_array)):
            print("位移解包含 NaN！")
            raise RuntimeError("位移解包含 NaN")

        if self.rank == 0:
            reason = ksp.getConvergedReason()
            self.logger.info(f"位移求解器收敛原因: {reason}")

        ksp.destroy()

    def check_side_pressure(self, p_func, bc_manager, mesh, comm):
        """
        计算侧面压力边界上的压力值统计并打印
        """
        from dolfinx import fem
        import numpy as np
        fdim = mesh.topology.dim - 1
        V_p = p_func.function_space
        side_markers = ['marker_103', 'marker_104']
        all_facets = []
        for marker in side_markers:
            if marker in bc_manager.boundary_geometries:
                facets = bc_manager.boundary_geometries[marker]['facets']
                all_facets.append(facets)
        if not all_facets:
            return
        all_facets = np.concatenate(all_facets).astype(np.int32)
        dofs = fem.locate_dofs_topological(V_p, fdim, all_facets)
        p_vals = p_func.x.array[dofs]
        if comm.rank == 0:
            print(f"侧面压力边界最终值: min={p_vals.min():.2f}, max={p_vals.max():.2f}")