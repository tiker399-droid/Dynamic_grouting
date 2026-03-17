"""
解耦求解器 - 顺序求解压力场和位移场
压力方程：包含土骨架速度项（使用上一时间步的位移速度近似）
位移方程：线弹性（压力作为已知体力）
粘度随时间变化（浆液粘度）
独立函数空间版本，包含完整无量纲化和经验缩放
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

        # 材料常数
        self.g = self.mat.g
        self.k0 = self.mat.k0
        self.rho_g = self.mat.rho_g
        self.rho_s = self.mat.rho_s
        self.phi0 = self.mat.phi0
        self.E = self.mat.E
        self.nu = self.mat.nu
        self.mu = self.mat.mu_current_constant  # 随时间变化

        # 参考量和缩放因子
        self.L0 = config['geometry']['height']
        self.rho_w = self.mat.rho_w
        self.g_mag = self.mat.g_magnitude
        self.P0 = self.rho_w * self.g_mag * self.L0
        self.U0 = self.P0 * self.L0 / self.E
        self.mu0 = self.mat.mu_g0
        self.t0 = self.k0 * self.mu0 / (self.L0**2 * self.P0)
        self.t1 = self.L0**2 * self.mu0 / (self.k0 * self.P0)

        self.scale_p = 1.0#self.t1
        self.scale_u = 1.0#1.0 / self.E
        self.scale_up = 1.0#1.0 / self.E
        self.scale_vs = 1e-5

        # 转换为 UFL 常数
        self.scale_p_const = fem.Constant(self.V_u.mesh, self.scale_p)
        self.scale_u_const = fem.Constant(self.V_u.mesh, self.scale_u)
        self.scale_up_const = fem.Constant(self.V_u.mesh, self.scale_up)
        self.scale_vs_const = fem.Constant(self.V_u.mesh, self.scale_vs)

        # 材料常数（已为 fem.Constant）
        self.k0_const = self.mat.k0_constant
        self.mu_const = self.mat.mu_current_constant
        self.rho_g_const = self.mat.rho_g_constant
        self.rho_s_const = self.mat.rho_s_constant
        self.rho_w_const = self.mat.rho_w_constant
        self.phi0_const = self.mat.phi0_constant
        self.g_const = self.mat.g

        # 求解器选项
        self.pressure_ksp_type = config.get('pressure_ksp_type', 'cg')
        self.pressure_pc_type = config.get('pressure_pc_type', 'hypre')
        self.displacement_ksp_type = config.get('displacement_ksp_type', 'gmres')
        self.displacement_pc_type = config.get('displacement_pc_type', 'lu')
        self.ksp_rtol = config.get('ksp_rtol', 1e-10)
        self.ksp_atol = config.get('ksp_atol', 1e-12)
        self.ksp_max_it = config.get('ksp_max_it', 1000)

        # 位移历史
        self.u_prev = None
        self.u_prev2 = None

    def solve(self, dt, time, u, p, u_prev, p_prev):
        self.mat.update_time_dependent_properties(time)
        self.u_prev2 = self.u_prev
        self.u_prev = u_prev
        self._solve_pressure(p, time, dt)
        self._solve_displacement(u, p, time)
        return True, 0

    def _solve_pressure(self, p_func, time, dt):
        bcs_p = self.bc_manager.get_pressure_bcs()
        v_p = ufl.TestFunction(self.V_p)
        p_trial = ufl.TrialFunction(self.V_p)

        k = self.k0_const
        mu = self.mu_const
        rho = self.rho_g_const
        g = self.g_const
        scale_p = self.scale_p_const
        scale_vs = self.scale_vs_const

        a = fem.form(scale_p * ufl.inner((k / mu) * ufl.grad(p_trial), ufl.grad(v_p)) * ufl.dx)
        L_expr = scale_p * ufl.inner((k / mu) * rho * g, ufl.grad(v_p)) * ufl.dx

        if self.u_prev is not None and self.u_prev2 is not None:
            v_s = (self.u_prev - self.u_prev2) / dt
            L_expr -= scale_vs * ufl.div(v_s) * v_p * ufl.dx

        L = fem.form(L_expr)

        A = petsc.assemble_matrix(a, bcs=bcs_p)
        A.assemble()
        b = petsc.assemble_vector(L)
        petsc.apply_lifting(b, [a], [bcs_p])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs_p)

        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        ksp.solve(b, p_func.x.petsc_vec)
        p_func.x.scatter_forward()


        # 检查解
        p_array = p_func.x.array
        print(f"压力解范围: min={p_array.min():.6e}, max={p_array.max():.6e}")

        if self.rank == 0:
            reason = ksp.getConvergedReason()
            self.logger.info(f"压力求解器收敛原因: {reason}")

        ksp.destroy()

    def _solve_displacement(self, u_func, p_func, time):
        bcs_u = self.bc_manager.get_displacement_bcs()
        v_u = ufl.TestFunction(self.V_u)
        u_trial = ufl.TrialFunction(self.V_u)

        def sigma(u):
            epsilon = ufl.sym(ufl.grad(u))
            lambda_, mu = self.mat.get_lame_parameters()
            return lambda_ * ufl.tr(epsilon) * ufl.Identity(len(u)) + 2 * mu * epsilon

        rho_bulk = (1 - self.phi0_const) * self.rho_s_const + self.phi0_const * self.rho_w_const

        a = fem.form(self.scale_u_const * ufl.inner(sigma(u_trial), ufl.grad(v_u)) * ufl.dx)
        L = fem.form(self.scale_up_const * self.mat.alpha * p_func * ufl.div(v_u) * ufl.dx
                     + self.scale_u_const * ufl.dot(rho_bulk * self.g, v_u) * ufl.dx)

        A = petsc.assemble_matrix(a, bcs=bcs_u)
        A.assemble()
        b = petsc.assemble_vector(L)
        petsc.apply_lifting(b, [a], [bcs_u])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs_u)

        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(A)
        ksp.setType(self.displacement_ksp_type)
        ksp.setTolerances(rtol=self.ksp_rtol, atol=self.ksp_atol, max_it=self.ksp_max_it)
        pc = ksp.getPC()
        pc.setType(self.displacement_pc_type)
        if self.displacement_pc_type == "lu":
            pc.setFactorSolverType("mumps")
        ksp.solve(b, u_func.x.petsc_vec)
        u_func.x.scatter_forward()


        # 检查解
        u_array = u_func.x.array
        print(f"位移解范围: min={u_array.min():.6e}, max={u_array.max():.6e}")

        if self.rank == 0:
            reason = ksp.getConvergedReason()
            self.logger.info(f"位移求解器收敛原因: {reason}")

        ksp.destroy()