"""
解耦求解器 - 固定应力分裂(FSS)稳定化版本（轴对称）

坐标约定：
    x[0] = r（径向），x[1] = z（竖向）
    积分微元：dV = 2π r dr dz，弱形式中略去常数 2π，所有积分乘以 r

轴对称与平面应变的弱形式区别：
    ① 散度多出环向分量：div_axi(u) = div(u) + u[0]/r
    ② 位移方程双线性形式多出环向应力贡献：σ_θθ * v_r / r
    ③ 所有体积分乘以 r（径向坐标）

FSS 方法原理（与坐标系无关）：
    在压力方程中引入稳定化项 β = α²/K_dr，使顺序解耦无条件稳定。
    K_dr = λ + 2μ/3 = E / (3(1-2ν))（排水体积模量，轴对称问题用三维值）
    收缩因子 ρ = S·K_dr / (S·K_dr + α²) ≈ 1.9e-3，2~3 次内迭代即收敛。

FSS 迭代格式（每个时间步 t^n → t^{n+1}）：
    初始化：u^{(0)} = u^n，p^{0} = p^n
    for k = 0, 1, ..., n_inner-1:
        压力步：(S+β)/dt·p^{k+1} - ∇·(k/μ·∇p^{k+1}) =
                S/dt·p^n + β/dt·p^k - α/dt·(div_axi(u^{(k)}) - div_axi(u^n))
        位移步：∇·σ'_axi(u^{k+1}) = α·∇p^{k+1} + ρ_b·g
    收敛后：p^{n+1} = p^{n+1,n_inner}，u^{n+1} = u^{(n_inner)}
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

        # 材料标量值
        self.k0  = self.mat.k0
        self.E   = self.mat.E
        self.nu  = self.mat.nu

        # FEniCS 常量对象
        self.k0_const      = self.mat.k0_constant
        self.mu_const      = self.mat.mu_current_constant
        self.rho_g_const   = self.mat.rho_g_constant
        self.rho_s_const   = self.mat.rho_s_constant
        self.rho_w_const   = self.mat.rho_w_constant
        self.phi0_const    = self.mat.phi0_constant
        self.alpha_const   = self.mat.alpha_constant
        self.storage_const = self.mat.storage_coefficient_constant

        # ------------------------------------------------------------------
        # 轴对称坐标：r = x[0]
        # ------------------------------------------------------------------
        self._x = ufl.SpatialCoordinate(self.mesh)
        self._r = self._x[0]   # 径向坐标，用于积分权重和环向项

        # ------------------------------------------------------------------
        # FSS 稳定化参数
        # 轴对称问题本质是三维旋转对称，K_dr 取三维排水体积模量：
        #   K_dr = λ + 2μ/3 = E / (3(1-2ν))
        # 原平面应变代码误用 λ+μ，此处修正。
        # ------------------------------------------------------------------
        lambda_val, mu_lame_val = self.mat.get_lame_parameters()
        K_dr_val = float(lambda_val) + 2.0 * float(mu_lame_val) / 3.0   # 三维排水体积模量
        alpha_scalar = float(self.mat.alpha)
        self.beta_fss = alpha_scalar ** 2 / K_dr_val

        if self.rank == 0:
            self.logger.info(
                f"FSS 稳定化（轴对称）: "
                f"K_dr = {K_dr_val/1e6:.3f} MPa, "
                f"β = α²/K_dr = {self.beta_fss:.3e} /Pa, "
                f"S_phys = {float(self.storage_const.value):.3e} /Pa, "
                f"收缩因子 ρ ≈ {float(self.storage_const.value)*K_dr_val / (float(self.storage_const.value)*K_dr_val + alpha_scalar**2):.3e}"
            )

        # 求解器选项
        solver_cfg = config.get('solver', {})
        self.n_fss_iter             = int(solver_cfg.get('fss_inner_iterations', 3))
        self.displacement_ksp_type  = config.get('displacement_ksp_type', 'preonly')
        self.displacement_pc_type   = config.get('displacement_pc_type', 'lu')
        self.ksp_rtol               = config.get('ksp_rtol', 1e-10)
        self.ksp_atol               = config.get('ksp_atol', 1e-12)
        self.ksp_max_it             = config.get('ksp_max_it', 1000)

        # FSS 内迭代缓冲区
        self.u_hist_1        = fem.Function(self.V_u, name="u_hist_1")
        self.u_iter          = fem.Function(self.V_u, name="u_iter")
        self.u_n_snapshot    = fem.Function(self.V_u, name="u_n_snapshot")
        self.p_iter          = fem.Function(self.V_p, name="p_iter")
        self.u_total         = fem.Function(self.V_u, name="u_total")
        self.u_initial       = fem.Function(self.V_u, name="u_initial")

        self._u_history_initialized = False
        self._initial_field_ready   = False
        self.subtract_initial_field = config.get("subtract_initial_field", True)

        # 计算初始平衡位移
        self._compute_initial_displacement()

        if self.rank == 0:
            self.logger.info(f"FSS 内迭代次数: {self.n_fss_iter} 次/时间步")

    # =========================================================================
    # 轴对称算子（核心改动）
    # =========================================================================
    def _div_axi(self, u):
        """
        轴对称散度（体积应变）：
            ε_v = ∂u_r/∂r + u_r/r + ∂u_z/∂z
                = div(u) + u[0]/r
        """
        return ufl.div(u) + u[0] / self._r

    def _eps_rz(self, u):
        """r-z 平面内对称应变张量（2×2），不含环向分量"""
        return ufl.sym(ufl.grad(u))

    def _sigma_rz(self, u):
        """
        r-z 平面内有效应力张量（2×2）。
        注意散度使用轴对称版本（含 u_r/r），以保证与环向应力的一致性。
        """
        lambda_, mu = self.mat.get_lame_parameters()
        eps_v = self._div_axi(u)
        return lambda_ * eps_v * ufl.Identity(2) + 2.0 * mu * self._eps_rz(u)

    def _sigma_tt(self, u):
        """
        环向有效应力标量：
            σ'_θθ = λ·ε_v + 2μ·(u_r/r)
        """
        lambda_, mu = self.mat.get_lame_parameters()
        eps_v = self._div_axi(u)
        return lambda_ * eps_v + 2.0 * mu * u[0] / self._r

    # =========================================================================
    # 主求解入口
    # =========================================================================
    def solve(self, dt, time, u, p, u_prev, p_prev):
        """
        执行一个时间步的 FSS 解耦求解（轴对称）。

        流程：
          1. 保存时间步起点 u^n
          2. 初始化 u^{(0)} = u^n，p^{0} = p^n
          3. for k in range(n_fss_iter):
               a. 压力步（轴对称弱形式，乘 r）
               b. 位移步（轴对称弱形式，乘 r，含环向项）
               c. 更新 u^{(k+1)}，p^{k}
          4. 更新历史
        """
        self.mat.update_time_dependent_properties(time)

        # 初始化历史（仅第一步）
        if not self._u_history_initialized:
            self.u_hist_1.x.array[:] = u_prev.x.array[:]
            self.u_hist_1.x.scatter_forward()
            self._u_history_initialized = True

        # 保存时间步起点 u^n
        self.u_n_snapshot.x.array[:] = self.u_hist_1.x.array[:]
        self.u_n_snapshot.x.scatter_forward()

        # 初始化迭代起点
        self.u_iter.x.array[:] = self.u_n_snapshot.x.array[:]
        self.u_iter.x.scatter_forward()
        self.p_iter.x.array[:] = p_prev.x.array[:]
        self.p_iter.x.scatter_forward()

        if self.rank == 0:
            print(f"时间步: t={time:.3f}s, dt={dt:.3e}s")

        # FSS 内迭代
        for k in range(self.n_fss_iter):
            self._solve_pressure(p, p_prev, self.p_iter, dt, time,
                                 self.u_iter, self.u_n_snapshot)
            self.p_iter.x.array[:] = p.x.array[:]
            self.p_iter.x.scatter_forward()

            self._solve_displacement(u, p, time)

            # 相对误差输出
            if self.rank == 0:
                diff = fem.Function(self.V_u)
                diff.x.array[:] = u.x.array[:] - self.u_iter.x.array[:]
                diff.x.scatter_forward()
                diff_norm = self._global_l2_norm(diff)
                u_norm    = self._global_l2_norm(u) + 1e-12
                print(f"  [FSS k={k+1}/{self.n_fss_iter}] 位移相对变化: {diff_norm/u_norm:.2e}")
                u_arr = u.x.array
                print(f"  净位移范围: [{u_arr.min():.4e}, {u_arr.max():.4e}] m")

            self.u_iter.x.array[:] = u.x.array[:]
            self.u_iter.x.scatter_forward()

        # 更新历史
        self.u_hist_1.x.array[:] = u.x.array[:]
        self.u_hist_1.x.scatter_forward()

        return True, 0

    # =========================================================================
    # 辅助方法
    # =========================================================================
    def _global_l2_norm(self, expr):
        """计算标量或向量函数的全局 L2 范数"""
        local_val  = fem.assemble_scalar(fem.form(ufl.inner(expr, expr) * ufl.dx))
        global_val = self.comm.allreduce(local_val, op=MPI.SUM)
        return np.sqrt(max(global_val, 0.0))

    def _compute_initial_displacement(self):
        """
        计算初始平衡位移（重力 + 静水压力），轴对称弱形式。
        用于从总位移中减去初始场，得到注浆引起的净位移。
        """
        if not self.subtract_initial_field:
            self._initial_field_ready = False
            return
        try:
            v_u    = ufl.TestFunction(self.V_u)
            u_trial = ufl.TrialFunction(self.V_u)
            bcs_u  = self.bc_manager.get_displacement_bcs()

            geom_cfg = self.config.get("geometry", {})
            H        = float(geom_cfg.get("height", 13.0))
            g_mag    = float(self.mat.g_magnitude)
            r        = self._r
            x        = self._x

            # 初始静水压力场
            p_init_expr = self.rho_w_const * g_mag * (H - x[1])

            # 初始整体密度（常数孔隙率）
            rho_bulk = (1.0 - self.phi0_const) * self.rho_s_const \
                     + self.phi0_const * self.rho_w_const

            # 双线性形式（轴对称）：含环向应力贡献，乘 r
            a = fem.form(
                (
                    ufl.inner(self._sigma_rz(u_trial), self._eps_rz(v_u))
                    + self._sigma_tt(u_trial) * v_u[0] / r
                ) * r * ufl.dx
            )

            # 线性形式（轴对称）：压力耦合用轴对称散度，乘 r
            L = fem.form(
                (
                    self.alpha_const * p_init_expr * self._div_axi(v_u)
                    + ufl.dot(rho_bulk * self.mat.g, v_u)
                ) * r * ufl.dx
            )

            A = petsc.assemble_matrix(a, bcs=bcs_u)
            A.assemble()
            b = petsc.assemble_vector(L)
            petsc.apply_lifting(b, [a], [bcs_u])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                          mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, bcs_u)

            ksp = PETSc.KSP().create(self.comm)
            ksp.setOperators(A)
            ksp.setType("preonly")
            pc = ksp.getPC()
            pc.setType("lu")
            pc.setFactorSolverType("mumps")
            ksp.setTolerances(rtol=self.ksp_rtol, atol=self.ksp_atol,
                              max_it=self.ksp_max_it)
            ksp.solve(b, self.u_initial.x.petsc_vec)
            self.u_initial.x.scatter_forward()

            if self.rank == 0:
                arr = self.u_initial.x.array
                print(f"初始平衡位移: min={arr.min():.4e}, "
                      f"max={arr.max():.4e} m, KSP={ksp.getConvergedReason()}")

            ksp.destroy(); A.destroy(); b.destroy()
            self._initial_field_ready = True

        except Exception as e:
            self._initial_field_ready = False
            if self.rank == 0:
                self.logger.warning(f"初始平衡位移计算失败，不做消去: {e}")

    # =========================================================================
    # 压力方程（FSS 稳定化，轴对称）
    # =========================================================================
    def _solve_pressure(self, p_func, p_prev_func, p_iter_func, dt, time,
                        u_current, u_prev_ts):
        """
        FSS 稳定化 Biot 压力方程（轴对称弱形式）。

        强形式（后向 Euler，FSS）：
            (S+β)/dt·p^{k+1} - ∇·(k/μ·∇p^{k+1})
                = S/dt·p^n + β/dt·p^k - α/dt·(ε_v^{(k)} - ε_v^n)

        其中 ε_v = div_axi(u) = div(u) + u_r/r。

        弱形式（乘 r 后积分）：
            a(p, q) = (S+β)/dt · ∫ p·q · r dx
                    + ∫ (k/μ) · ∇p·∇q · r dx
            L(q)    = S/dt · ∫ p^n·q · r dx
                    + β/dt · ∫ p^k·q · r dx
                    - α/dt · ∫ Δε_v · q · r dx
        """
        bcs_p  = self.bc_manager.get_pressure_bcs()
        v_p    = ufl.TestFunction(self.V_p)
        p_trial = ufl.TrialFunction(self.V_p)
        r      = self._r

        mu     = self.mu_const
        alpha  = self.alpha_const
        S      = self.storage_const

        S_eff_val = float(S.value) + self.beta_fss
        S_eff     = fem.Constant(self.mesh, PETSc.ScalarType(S_eff_val))
        dt_const  = fem.Constant(self.mesh, PETSc.ScalarType(dt))

        # ---- 孔隙率与渗透率（基于当前位移，轴对称散度）----
        phi0  = self.phi0_const
        eps_v = self._div_axi(u_current)           # ← 轴对称散度
        phi   = phi0 + (alpha - phi0) * eps_v

        phi_min = fem.Constant(self.mesh, PETSc.ScalarType(0.01))
        phi_max = fem.Constant(self.mesh, PETSc.ScalarType(0.99))
        phi = ufl.conditional(ufl.lt(phi, phi_min), phi_min,
              ufl.conditional(ufl.gt(phi, phi_max), phi_max, phi))

        k0_val       = float(self.k0_const.value)
        k_perm_expr  = k0_val * (phi**3 / ((1.0 - phi)**2 + 1e-10)) / (phi0**3 / ((1.0 - phi0)**2 + 1e-10))
        mobility     = k_perm_expr / mu

        # ---- FSS 耦合项（轴对称散度）----
        delta_eps_v = self._div_axi(u_current) - self._div_axi(u_prev_ts)  # ← 轴对称

        # ---- 双线性形式（乘 r）----
        a_expr = (
            S_eff / dt_const * p_trial * v_p * r * ufl.dx          # ← 乘 r
            + ufl.inner(mobility * ufl.grad(p_trial),
                        ufl.grad(v_p)) * r * ufl.dx                 # ← 乘 r
        )

        # ---- 线性形式（乘 r）----
        L_expr = (
            float(S.value) / dt_const * p_prev_func * v_p * r * ufl.dx    # ← 乘 r
            + self.beta_fss / dt_const * p_iter_func * v_p * r * ufl.dx   # ← 乘 r
            - alpha / dt_const * delta_eps_v * v_p * r * ufl.dx            # ← 乘 r
        )

        a = fem.form(a_expr)
        L = fem.form(L_expr)

        A = petsc.assemble_matrix(a, bcs=bcs_p)
        A.assemble()
        b = petsc.assemble_vector(L)
        petsc.apply_lifting(b, [a], [bcs_p])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs_p)

        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        ksp.setTolerances(rtol=self.ksp_rtol, atol=self.ksp_atol,
                          max_it=self.ksp_max_it)
        ksp.solve(b, p_func.x.petsc_vec)
        p_func.x.scatter_forward()

        if self.rank == 0:
            p_arr  = p_func.x.array
            mu_val = float(mu.value)
            self.logger.debug(
                f"[Pressure] t={time:.3f}s, dt={dt:.3e}s, μ={mu_val:.5f} Pa·s, "
                f"p∈[{p_arr.min():.4e}, {p_arr.max():.4e}] Pa"
            )
        ksp.destroy(); A.destroy(); b.destroy()

    # =========================================================================
    # 位移方程（轴对称弱形式）
    # =========================================================================
    def _solve_displacement(self, u_func, p_func, time):
        """
        准静态线弹性位移方程（轴对称弱形式）。

        强形式：∇·σ'_axi(u) - α∇p + ρ_b·g = 0

        弱形式（乘 r 后积分，含环向应力项）：
            a(u, v) = ∫ [σ'_rz : ε_rz(v) + σ'_θθ · v_r/r] · r dx
            L(v)    = ∫ [α · p · div_axi(v) + ρ_b · g · v] · r dx

        说明：
            σ'_rz : ε_rz(v) 覆盖 r-z 平面内的四个应力分量
            σ'_θθ · v_r/r   是环向应力对虚功的贡献（笛卡尔弱形式中不存在此项）
            div_axi(v) = div(v) + v[0]/r  保证与轴对称强形式的压力梯度项一致
        """
        bcs_u   = self.bc_manager.get_displacement_bcs()
        v_u     = ufl.TestFunction(self.V_u)
        u_trial = ufl.TrialFunction(self.V_u)
        r       = self._r

        # 当前孔隙率（用轴对称散度）
        phi0     = self.phi0_const
        phi      = phi0 + (self.alpha_const - phi0) * self._div_axi(u_func)  # ← 轴对称
        rho_bulk = (1.0 - phi) * self.rho_s_const + phi * self.rho_w_const

        # ---- 双线性形式（乘 r，含环向项）----
        a = fem.form(
            (
                ufl.inner(self._sigma_rz(u_trial), self._eps_rz(v_u))  # r-z 平面贡献
                + self._sigma_tt(u_trial) * v_u[0] / r                  # 环向应力贡献 ← 新增
            ) * r * ufl.dx                                               # ← 乘 r
        )

        # ---- 线性形式（乘 r，压力用轴对称散度）----
        L = fem.form(
            (
                self.alpha_const * p_func * self._div_axi(v_u)          # ← 轴对称散度
                + ufl.dot(rho_bulk * self.mat.g, v_u)
            ) * r * ufl.dx                                               # ← 乘 r
        )

        A = petsc.assemble_matrix(a, bcs=bcs_u)
        A.assemble()
        b = petsc.assemble_vector(L)
        petsc.apply_lifting(b, [a], [bcs_u])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs_u)

        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(A)
        ksp.setType(self.displacement_ksp_type)
        ksp.setTolerances(rtol=self.ksp_rtol, atol=self.ksp_atol,
                          max_it=self.ksp_max_it)
        pc = ksp.getPC()
        pc.setType(self.displacement_pc_type)
        if self.displacement_pc_type == "lu":
            pc.setFactorSolverType("mumps")
        ksp.solve(b, self.u_total.x.petsc_vec)
        self.u_total.x.scatter_forward()

        # 净位移 = 总位移 - 初始平衡位移
        if self.subtract_initial_field and self._initial_field_ready:
            u_func.x.array[:] = (self.u_total.x.array[:]
                                  - self.u_initial.x.array[:])
        else:
            u_func.x.array[:] = self.u_total.x.array[:]
        u_func.x.scatter_forward()

        if self.rank == 0:
            total_arr = self.u_total.x.array
            net_arr   = u_func.x.array
            reason    = ksp.getConvergedReason()
            self.logger.debug(
                f"[Displacement] 总位移: [{total_arr.min():.4e}, {total_arr.max():.4e}] m, "
                f"净位移: [{net_arr.min():.4e}, {net_arr.max():.4e}] m, KSP={reason}"
            )
        ksp.destroy(); A.destroy(); b.destroy()