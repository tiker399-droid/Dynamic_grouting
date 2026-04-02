"""
解耦求解器 - 顺序求解压力场和位移场
压力方程：包含土骨架速度项（使用上一时间步的位移速度近似）
位移方程：线弹性（压力作为已知体力）
粘度随时间变化（浆液粘度）
独立函数空间版本

修改说明：
1. 在求解器内部自动计算初始平衡位移场 u_initial；
2. 每一步先求“总位移” u_total，再做 u = u_total - u_initial；
3. 因而对外部而言，u 始终表示“扣除初始场后的净位移”；
4. 压力方程中的位移历史也使用净位移历史，初始场不会进入后续增量演化。
"""

import numpy as np
from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py import PETSc
import ufl
import logging
from mpi4py import MPI


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
        self.mesh = V_u.mesh

        # 材料常数
        self.g = self.mat.g
        self.k0 = self.mat.k0
        self.rho_g = self.mat.rho_g
        self.rho_s = self.mat.rho_s
        self.phi0 = self.mat.phi0
        self.E = self.mat.E
        self.nu = self.mat.nu
        self.mu = self.mat.mu_current_constant

        # 直接使用材料常量
        self.k0_const = self.mat.k0_constant
        self.mu_const = self.mat.mu_current_constant
        self.rho_g_const = self.mat.rho_g_constant
        self.rho_s_const = self.mat.rho_s_constant
        self.rho_w_const = self.mat.rho_w_constant
        self.phi0_const = self.mat.phi0_constant
        self.g_const = self.mat.g
        self.alpha_const = self.mat.alpha_constant
        self.storage_const = self.mat.storage_coefficient_constant

        # 求解器选项
        self.coupling_max_iter = config.get('solver', {}).get('coupling_max_iterations', 20)
        self.coupling_tol = config.get('solver', {}).get('coupling_tolerance', 1e-6)

        self.pressure_ksp_type = config.get('pressure_ksp_type', 'preonly')
        self.pressure_pc_type = config.get('pressure_pc_type', 'lu')
        self.displacement_ksp_type = config.get('displacement_ksp_type', 'gmres')
        self.displacement_pc_type = config.get('displacement_pc_type', 'lu')
        self.ksp_rtol = config.get('ksp_rtol', 1e-10)
        self.ksp_atol = config.get('ksp_atol', 1e-12)
        self.ksp_max_it = config.get('ksp_max_it', 1000)

        # 净位移历史：u^n, u^(n-1)
        self.u_hist_1 = fem.Function(self.V_u, name="u_hist_1")
        self.u_hist_2 = fem.Function(self.V_u, name="u_hist_2")
        self._u_history_initialized = False

        # 总位移临时场（内部使用）
        self.u_total = fem.Function(self.V_u, name="u_total")

        # 初始平衡位移场
        self.u_initial = fem.Function(self.V_u, name="u_initial")
        self._initial_field_ready = False

        # 启用初始场消去
        self.subtract_initial_field = config.get("subtract_initial_field", True)

        # 初始化初始场
        self._compute_initial_displacement()

    def solve(self, dt, time, u, p, u_prev, p_prev):
        self.mat.update_time_dependent_properties(time)

        # 初始化净位移历史（第一次调用时）
        if not self._u_history_initialized:
            self.u_hist_1.x.array[:] = u_prev.x.array[:]
            self.u_hist_2.x.array[:] = u_prev.x.array[:]
            self.u_hist_1.x.scatter_forward()
            self.u_hist_2.x.scatter_forward()
            self._u_history_initialized = True

        # 初始化当前步的字段（用于迭代）
        u_k = u       # 直接使用传入的 u，避免额外分配
        p_k = p

        # 保存迭代前的值，用于收敛判断
        u_old = fem.Function(self.V_u)
        p_old = fem.Function(self.V_p)
        u_old.x.array[:] = u_k.x.array[:]
        p_old.x.array[:] = p_k.x.array[:]

        converged = False
        for iter_count in range(1, self.coupling_max_iter + 1):
            # 1. 求解压力方程，使用当前位移 u_k 计算 eps_v_rate
            self._solve_pressure(p_k, p_prev, dt, time, u_k)

            # 2. 求解位移方程，使用当前压力 p_k
            self._solve_displacement(u_k, p_k, time)

            # 3. 检查收敛性（比较前后两次迭代的变化）
            # 计算位移的相对变化（L2 范数）
            delta_u = u_k.x.array - u_old.x.array
            norm_delta_u = np.linalg.norm(delta_u)
            norm_u = np.linalg.norm(u_k.x.array)
            rel_change_u = norm_delta_u / (norm_u + 1e-12)

            # 计算压力的相对变化
            delta_p = p_k.x.array - p_old.x.array
            norm_delta_p = np.linalg.norm(delta_p)
            norm_p = np.linalg.norm(p_k.x.array)
            rel_change_p = norm_delta_p / (norm_p + 1e-12)

            # 如果变化小于容差，则收敛
            if rel_change_u < self.coupling_tol and rel_change_p < self.coupling_tol:
                converged = True
                if self.rank == 0:
                    self.logger.info(f"耦合迭代收敛，迭代次数: {iter_count}, "
                                    f"Δu={rel_change_u:.2e}, Δp={rel_change_p:.2e}")
                break

            # 否则，更新旧值，继续迭代
            u_old.x.array[:] = u_k.x.array[:]
            p_old.x.array[:] = p_k.x.array[:]

        if not converged and self.rank == 0:
            self.logger.warning(f"耦合迭代未收敛，达到最大迭代次数 {self.coupling_max_iter}")

        # 更新净位移历史（使用最终收敛的位移）
        self.u_hist_2.x.array[:] = self.u_hist_1.x.array[:]
        self.u_hist_1.x.array[:] = u_k.x.array[:]
        self.u_hist_2.x.scatter_forward()
        self.u_hist_1.x.scatter_forward()

        return converged, iter_count

    def _global_l2_norm(self, expr):
        local_val = fem.assemble_scalar(fem.form(ufl.inner(expr, expr) * ufl.dx))
        global_val = self.comm.allreduce(local_val, op=MPI.SUM)
        return np.sqrt(max(global_val, 0.0))

    def _sigma(self, u):
        epsilon = ufl.sym(ufl.grad(u))
        lambda_, mu = self.mat.get_lame_parameters()
        gdim = self.mesh.geometry.dim
        return lambda_ * ufl.tr(epsilon) * ufl.Identity(gdim) + 2.0 * mu * epsilon

    def _compute_initial_displacement(self):
        """
        在当前框架内直接计算初始平衡位移场：
        - 压力采用静水压力 p_init = rho_w * g * (H - y)
        - 位移边界条件沿用当前 bc_manager 的位移边界
        - 求得的 u_initial 在后续每一步中被直接扣除
        """
        if not self.subtract_initial_field:
            if self.rank == 0:
                self.logger.info("未启用初始场消去。")
            self._initial_field_ready = False
            return

        try:
            v_u = ufl.TestFunction(self.V_u)
            u_trial = ufl.TrialFunction(self.V_u)
            bcs_u = self.bc_manager.get_displacement_bcs()

            # 几何高度：优先从 config 读
            geom_cfg = self.config.get("geometry", {})
            foundation_height = float(geom_cfg.get("height", 13.0))

            x = ufl.SpatialCoordinate(self.mesh)
            vert_axis = self.mesh.geometry.dim - 1  # 2D -> y, 3D -> z
            g_mag = float(self.mat.g_magnitude)
            p_init_expr = self.rho_w_const * g_mag * (foundation_height - x[vert_axis])

            rho_bulk = (1.0 - self.phi0_const) * self.rho_s_const + self.phi0_const * self.rho_w_const

            a = fem.form(ufl.inner(self._sigma(u_trial), ufl.sym(ufl.grad(v_u))) * ufl.dx)
            L = fem.form(
                self.alpha_const * p_init_expr * ufl.div(v_u) * ufl.dx
                + ufl.dot(rho_bulk * self.g, v_u) * ufl.dx
            )

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

            ksp.solve(b, self.u_initial.x.petsc_vec)
            self.u_initial.x.scatter_forward()

            if self.rank == 0:
                arr = self.u_initial.x.array
                reason = ksp.getConvergedReason()
                print(
                    f"初始平衡位移计算完成: min={arr.min():.6e}, max={arr.max():.6e}, KSP={reason}"
                )

            ksp.destroy()
            A.destroy()
            b.destroy()

            self._initial_field_ready = True

        except Exception as e:
            self._initial_field_ready = False
            if self.rank == 0:
                self.logger.warning(f"初始平衡位移计算失败，将不做初始场消去: {e}")

    def _solve_pressure(self, p_func, p_prev_func, dt, time, u_cur):
        bcs_p = self.bc_manager.get_pressure_bcs()
        v_p = ufl.TestFunction(self.V_p)
        p_trial = ufl.TrialFunction(self.V_p)

        k = self.k0_const
        mu = self.mu_const
        alpha = self.alpha_const
        S = self.storage_const

        dt_const = fem.Constant(self.mesh, PETSc.ScalarType(dt))

        # 使用当前位移 u_cur 和上一时间步位移 u_hist_2 计算体积应变率
        eps_v_rate = (ufl.div(u_cur) - ufl.div(self.u_hist_2)) / dt_const

        a_expr = (
             #(S / dt_const) * p_trial * v_p * ufl.dx   # 如果需要存储项，可取消注释
            + ufl.inner((k / mu) * ufl.grad(p_trial), ufl.grad(v_p)) * ufl.dx
        )

        L_expr = (
             #(S / dt_const) * p_prev_func * v_p * ufl.dx
            + ufl.inner((k / mu) * self.rho_g_const * self.g_const, ufl.grad(v_p)) * ufl.dx
            - 1e-4 * alpha * eps_v_rate * v_p * ufl.dx
        )

        a = fem.form(a_expr)
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
        ksp.setTolerances(rtol=self.ksp_rtol, atol=self.ksp_atol, max_it=self.ksp_max_it)

        ksp.solve(b, p_func.x.petsc_vec)
        p_func.x.scatter_forward()

        if self.rank == 0:
            reason = ksp.getConvergedReason()
            self.logger.debug(f"[Pressure] t={time:.3f}, dt={dt:.3e}, "
                            f"min(p)={p_func.x.array.min():.6e}, "
                            f"max(p)={p_func.x.array.max():.6e}, KSP={reason}")

        ksp.destroy()
        A.destroy()
        b.destroy()

    def _solve_displacement(self, u_func, p_func, time):
        """
        先求总位移 u_total，再减去初始场 u_initial，得到净位移 u_func。
        对外部而言，u_func 始终表示“相对初始平衡态的附加位移”。
        """
        bcs_u = self.bc_manager.get_displacement_bcs()
        v_u = ufl.TestFunction(self.V_u)
        u_trial = ufl.TrialFunction(self.V_u)

        rho_bulk = (1.0 - self.phi0_const) * self.rho_s_const + self.phi0_const * self.rho_w_const

        a = fem.form(ufl.inner(self._sigma(u_trial), ufl.sym(ufl.grad(v_u))) * ufl.dx)
        L = fem.form(
            self.alpha_const * p_func * ufl.div(v_u) * ufl.dx
            + ufl.dot(rho_bulk * self.g, v_u) * ufl.dx
        )

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

        # 先解总位移
        ksp.solve(b, self.u_total.x.petsc_vec)
        self.u_total.x.scatter_forward()

        # 再扣除初始平衡场
        if self.subtract_initial_field and self._initial_field_ready:
            u_func.x.array[:] = self.u_total.x.array[:] - self.u_initial.x.array[:]
        else:
            u_func.x.array[:] = self.u_total.x.array[:]
        u_func.x.scatter_forward()

        if self.rank == 0:
            total_arr = self.u_total.x.array
            net_arr = u_func.x.array
            reason = ksp.getConvergedReason()
            print(f"总位移范围: min={total_arr.min():.6e}, max={total_arr.max():.6e}")
            print(f"净位移范围: min={net_arr.min():.6e}, max={net_arr.max():.6e}")
            self.logger.info(f"位移求解器收敛原因: {reason}")

        ksp.destroy()
        A.destroy()
        b.destroy()