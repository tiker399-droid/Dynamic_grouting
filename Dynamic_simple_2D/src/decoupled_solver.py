"""
解耦求解器 - 顺序求解压力场和位移场
压力方程：包含土骨架速度项（使用上一时间步的位移速度近似）
位移方程：线弹性（压力作为已知体力）
粘度随时间变化（浆液粘度）
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
    """

    def __init__(self, comm, materials, bc_manager, function_space, config):
        """
        初始化解耦求解器

        Args:
            comm: MPI 通信器
            materials: 材料属性管理器
            bc_manager: 边界条件管理器（需提供 get_pressure_bcs 和 get_displacement_bcs 方法）
            function_space: 混合函数空间 [位移, 压力]
            config: 配置字典（用于求解器选项）
        """
        self.comm = comm
        self.mat = materials
        self.bc_manager = bc_manager
        self.W = function_space
        self.config = config

        self.rank = comm.Get_rank()
        self.logger = logging.getLogger(f"DecoupledSolver_rank{self.rank}")

        # 提取子空间
        self.V_u = self.W.sub(0)
        self.V_p = self.W.sub(1)

        # 材料常数
        self.g = self.mat.g
        self.k0 = self.mat.k0                # 渗透率常数
        self.rho_g = self.mat.rho_g          # 浆液密度
        self.rho_s = self.mat.rho_s
        self.phi0 = self.mat.phi0
        self.E = self.mat.E
        self.nu = self.mat.nu

        # 随时间变化的粘度常数（由 materials 维护）
        self.mu = self.mat.mu_current_constant

        # 求解器选项
        self.pressure_ksp_type = config.get('pressure_ksp_type', 'cg')
        self.pressure_pc_type = config.get('pressure_pc_type', 'hypre')
        self.displacement_ksp_type = config.get('displacement_ksp_type', 'gmres')
        self.displacement_pc_type = config.get('displacement_pc_type', 'lu')
        self.ksp_rtol = config.get('ksp_rtol', 1e-10)
        self.ksp_atol = config.get('ksp_atol', 1e-12)
        self.ksp_max_it = config.get('ksp_max_it', 1000)

        # 存储两个之前的位移场（用于计算土骨架速度）
        self.u_prev = None   # 上一时间步的位移（混合函数中的位移部分）
        self.u_prev2 = None  # 上上时间步的位移

    def solve(self, dt, time, solution, solution_prev, **kwargs):
        """
        执行一个时间步的解耦求解

        Args:
            dt: 时间步长
            time: 当前时间
            solution: 当前步解（混合函数，会被更新）
            solution_prev: 上一步解（混合函数）
            **kwargs: 其他参数

        Returns:
            (converged, iterations): 收敛标志和迭代次数（此处始终返回 True, 0）
        """
        # 更新时间相关的材料属性（例如粘度）
        self.mat.update_time_dependent_properties(time)

        # 提取上一时间步的位移
        u_prev = solution_prev.sub(0)

        # 更新位移历史
        self.u_prev2 = self.u_prev
        self.u_prev = u_prev

        # 1. 求解压力场（使用上一时间步的位移速度）
        self._solve_pressure(solution.sub(1), time, dt)

        # 2. 求解位移场（使用新的压力）
        self._solve_displacement(solution.sub(0), solution.sub(1), time)

        return True, 0

    def _solve_pressure(self, p_func, time, dt):
        """
        求解压力场（稳态扩散方程 + 土骨架速度项）
        """
        # 获取所有边界条件
        all_bcs = self.bc_manager.get_boundary_conditions()
        # 筛选出压力边界
        bcs_p = [bc for bc in all_bcs if bc.function_space == self.V_p or 
                 (hasattr(bc.function_space, 'parent') and bc.function_space.parent == self.V_p)]

        # 定义试验函数和测试函数
        v_p = ufl.TestFunction(self.V_p)
        p_trial = ufl.TrialFunction(self.V_p)

        # 材料属性（使用当前时间步的值）
        k = self.k0
        mu = self.mu              # 随时间变化的粘度常数
        rho = self.rho_g           # 浆液密度
        g = self.g

        # 变分形式
        a = fem.form(ufl.inner((k / mu) * ufl.grad(p_trial), ufl.grad(v_p)) * ufl.dx)

        # 构建线性形式表达式（基础重力项）
        L_expr = ufl.inner((k / mu) * rho * g, ufl.grad(v_p)) * ufl.dx

        # 如果有上一时间步的位移，添加土骨架速度项
        if self.u_prev is not None and self.u_prev2 is not None:
            # 土骨架速度 v_s = (u_prev - u_prev2) / dt
            v_s = (self.u_prev - self.u_prev2) / dt
            # 在连续性方程中，div(v_s) 项应移到右端，即 - div(v_s) * v_p
            L_expr -= ufl.div(v_s) * v_p * ufl.dx
        else:
            # 初始步没有历史位移，忽略该项
            pass

        L = fem.form(L_expr)

        # 组装矩阵和向量
        A = petsc.assemble_matrix(a, bcs=bcs_p)
        A.assemble()
        b = petsc.assemble_vector(L)
        petsc.apply_lifting(b, [a], [bcs_p])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs_p)

        # 创建求解器
        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(A)
        ksp.setType(self.pressure_ksp_type)
        ksp.setTolerances(rtol=self.ksp_rtol, atol=self.ksp_atol, max_it=self.ksp_max_it)

        # 设置预条件（直接求解器更稳定）
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        ksp.setType("preonly")

        # 求解
        ksp.solve(b, p_func.x.petsc_vec)
        p_func.x.scatter_forward()

        if self.rank == 0:
            reason = ksp.getConvergedReason()
            self.logger.info(f"压力求解器收敛原因: {reason}")

        ksp.destroy()

    def _solve_displacement(self, u_func, p_func, time):
        """
        求解位移场（线弹性，压力作为体力）
        """
        # 获取所有边界条件
        all_bcs = self.bc_manager.get_boundary_conditions()
        bcs_u = [bc for bc in all_bcs if bc.function_space == self.V_u or 
                 (hasattr(bc.function_space, 'parent') and bc.function_space.parent == self.V_u)]

        # 定义试验函数和测试函数
        v_u = ufl.TestFunction(self.V_u)
        u_trial = ufl.TrialFunction(self.V_u)

        # 材料属性
        def sigma(u):
            epsilon = ufl.sym(ufl.grad(u))
            lambda_, mu = self.mat.get_lame_parameters()
            return lambda_ * ufl.tr(epsilon) * ufl.Identity(len(u)) + 2 * mu * epsilon

        # 整体密度（常数，使用浆液密度？此处沿用原公式，可调整）
        rho_bulk = (1 - self.phi0) * self.rho_s + self.phi0 * self.rho_g

        # 变分形式
        a = fem.form(ufl.inner(sigma(u_trial), ufl.grad(v_u)) * ufl.dx)
        L = fem.form(self.mat.alpha * p_func * ufl.div(v_u) * ufl.dx
                     + ufl.inner(rho_bulk * self.g, v_u) * ufl.dx)

        # 组装矩阵和向量
        A = petsc.assemble_matrix(a, bcs=bcs_u)
        A.assemble()
        b = petsc.assemble_vector(L)
        petsc.apply_lifting(b, [a], [bcs_u])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs_u)

        # 创建求解器
        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(A)
        ksp.setType(self.displacement_ksp_type)
        ksp.setTolerances(rtol=self.ksp_rtol, atol=self.ksp_atol, max_it=self.ksp_max_it)

        # 设置预条件
        pc = ksp.getPC()
        pc.setType(self.displacement_pc_type)
        if self.displacement_pc_type == "lu":
            pc.setFactorSolverType("mumps")

        # 求解
        ksp.solve(b, u_func.x.petsc_vec)
        u_func.x.scatter_forward()

        if self.rank == 0:
            reason = ksp.getConvergedReason()
            self.logger.info(f"位移求解器收敛原因: {reason}")

        ksp.destroy()