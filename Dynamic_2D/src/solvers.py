"""
求解器管理器 - 非线性耦合求解器（适配 DOLFINx 0.9.0）
基于 PETSc 的牛顿法求解多物理场耦合系统
"""

from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
import numpy as np
from dolfinx.fem.petsc import assemble_vector, assemble_matrix
from typing import Optional, Dict, Any, Tuple, Callable
import logging


class SolverManager:
    """
    非线性求解器管理器（DOLFINx 0.9.0 API）
    使用 NewtonSolver 和 NonlinearProblem
    """

    def __init__(
        self,
        comm,
        config: Dict[str, Any],
        function_space: fem.FunctionSpace,
        weak_form_builder,
        bc_manager
    ):
        """
        初始化求解器管理器

        Args:
            comm: MPI 通信器
            config: 求解器配置字典（包含 'solver' 键）
            function_space: 混合函数空间
            weak_form_builder: 弱形式构建器实例
            bc_manager: 边界条件管理器实例
        """
        self.comm = comm
        self.config = config.get('solver', {})
        self.function_space = function_space
        self.weak_form_builder = weak_form_builder
        self.bc_manager = bc_manager

        # 日志
        self.rank = comm.Get_rank()
        self.logger = logging.getLogger(f"SolverManager_rank{self.rank}")

        # 从配置提取参数（确保类型转换）
        self.max_iterations = int(self.config.get('max_iterations', 20))
        self.relative_tolerance = float(self.config.get('relative_tolerance', 1e-6))
        self.absolute_tolerance = float(self.config.get('absolute_tolerance', 1e-8))
        self.linear_solver = self.config.get('linear_solver', 'mumps')
        self.preconditioner = self.config.get('preconditioner', 'ilu')
        self.report_convergence = self.config.get('report_convergence', False)

        # 当前问题状态
        self.problem = None
        self.solver = None

        # 统计信息
        self.iteration_counts = []      # 每个时间步的迭代次数

        if self.rank == 0:
            self.logger.info("求解器管理器初始化完成")
            self.logger.info(f"  最大迭代次数: {self.max_iterations}")
            self.logger.info(f"  相对容差: {self.relative_tolerance:.1e}")
            self.logger.info(f"  线性求解器: {self.linear_solver}")

    def solve(self, dt, time, solution, solution_prev, boundary_conditions=None, materials=None):
        import numpy as np
        arr = solution.x.array
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            self.logger.error("Initial solution contains NaN/Inf!")
            return False, 0
        else:
            self.logger.info(f"Initial solution OK: min={arr.min():.3e}, max={arr.max():.3e}")
        # 1. 强制清理旧的求解器和问题对象，帮助 Python GC 和 PETSc 释放资源
        self.solver = None
        self.problem = None

        # 2. 获取边界条件
        bcs = self.bc_manager.get_boundary_conditions()
        self.logger.info(f"获取到 {len(bcs)} 个边界条件")

        # 3. 构建弱形式（你的现有代码）
        print(f"Rank {self.rank}: Before building forms")
        import sys; sys.stdout.flush()
        F_ufl, J_ufl = self.weak_form_builder.build_form(
            dt, time, solution, solution_prev, bcs)
        print(f"Rank {self.rank}: After building forms")
        sys.stdout.flush()
        # 4. 设置初始猜测
        solution.x.array[:] = solution_prev.x.array[:]

        # 5. **创建全新的 NonlinearProblem 和 NewtonSolver**
        self.problem = NonlinearProblem(F_ufl, solution, bcs, J_ufl)
        print(f"Rank {self.rank}: After creating NonlinearProblem")
        sys.stdout.flush()
        self.solver = NewtonSolver(self.comm, self.problem)

        # 6. 设置求解器参数
        self.solver.max_it = self.max_iterations
        self.solver.rtol = self.relative_tolerance
        self.solver.atol = self.absolute_tolerance

        # 7. **重要：重新配置 KSP，避免继承旧选项**
        ksp = self.solver.krylov_solver
        opts = PETSc.Options()
        # 使用唯一的选项前缀，避免不同求解器实例间的选项冲突
        prefix = f"grouting_{time:.3f}_"  # 或用随机字符串
        ksp.setOptionsPrefix(prefix)
        opts.prefixPush(prefix)


        # 设置线性求解器选项（保持你的原有逻辑）
        if self.linear_solver == "mumps":
            opts["ksp_type"] = "preonly"
            opts["pc_type"] = "lu"
            opts["pc_factor_mat_solver_type"] = "mumps"
        elif self.linear_solver == "gmres":
            opts["ksp_type"] = "gmres"
            opts["ksp_gmres_restart"] = 30
            opts["pc_type"] = self.preconditioner
        elif self.linear_solver == "bcgs":
            opts["ksp_type"] = "bcgs"
            opts["pc_type"] = self.preconditioner
            #opts["sub_pc_type"] = "ilu"
            #opts["sub_pc_factor_levels"] = 0
        else:
            opts["ksp_type"] = "cg"
            opts["pc_type"] = self.preconditioner

        opts["ksp_monitor"] = None
        opts["ksp_converged_reason"] = None
        opts["ksp_rtol"] = 1e-6
        opts["ksp_atol"] = 1e-8
        opts["ksp_max_it"] = 10000
        opts["pc_factor_shift_type"] = "NONZERO"
        opts["ksp_monitor_singular_value"] = None
        opts.prefixPop()
        ksp.setFromOptions()
        print(f"Rank {self.rank}: About to call solver.solve at time {time}")
        import sys
        sys.stdout.flush()   # 确保立即输出

        # 手动组装残差向量
        b = assemble_vector(self.problem.L)   # L 是残差形式
        b.assemble()
        b_array = b.getArray()
        print(f"残差非零元素数量: {np.count_nonzero(b_array)}")
        print(f"残差最大值: {np.max(np.abs(b_array))}")
        if np.any(np.isnan(b_array)):
            nan_indices = np.where(np.isnan(b_array))[0]
            print(f"残差向量包含 NaN！索引：{nan_indices[:20]}")
            # 保存到文件供分析
            np.savetxt(f"residual_nan_rank{self.rank}.txt", b_array)
            return False, 0
        else:
            self.logger.info("残差向量正常（无 NaN/Inf）")

        # 手动组装雅可比矩阵
        A = assemble_matrix(self.problem.a, bcs=bcs)   # a 是雅可比形式
        A.assemble()
        # 检查矩阵是否有 NaN（需通过向量方式检查对角元）

        diag = A.getDiagonal()
        diag_array = diag.getArray()
        if np.any(np.isnan(diag_array)):
            print("雅可比矩阵对角元包含 NaN!")
        if np.any(np.isinf(diag_array)):
            print("雅可比矩阵对角元包含 Inf!")
        zero_diag = np.where(np.abs(diag_array) < 1e-12)[0]
        if len(zero_diag) > 0:
            self.logger.warning(f"雅可比矩阵有 {len(zero_diag)} 个零对角元,前10个:{zero_diag[:10]}")
        else:
            print("雅各比矩阵无零对角元")

        # 8. 求解
        try:
            n, converged = self.solver.solve(solution)
            self.iteration_counts.append(n)
            return converged, n
        except Exception as e:
            self.logger.error(f"求解器异常: {e}")
            print(f"Rank {self.rank}: Exception during solver.solve: {e}")
            import traceback
            traceback.print_exc()
            return False, 0

    def get_statistics(self) -> Dict[str, Any]:
        """
        返回求解器统计信息

        Returns:
            字典包含迭代次数历史、平均迭代次数等
        """
        if not self.iteration_counts:
            return {}

        stats = {
            "total_steps": len(self.iteration_counts),
            "total_iterations": sum(self.iteration_counts),
            "average_iterations": np.mean(self.iteration_counts),
            "max_iterations": np.max(self.iteration_counts),
            "min_iterations": np.min(self.iteration_counts),
        }
        return stats

    def reset_statistics(self):
        """重置统计信息"""
        self.iteration_counts.clear()
