"""
求解器管理器 - 非线性耦合求解器（适配 DOLFINx 0.9.0）
基于 PETSc 的牛顿法求解多物理场耦合系统
"""

from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
import numpy as np
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

    def solve(
        self,
        dt: float,
        time: float,
        solution: fem.Function,
        solution_prev: fem.Function,
        boundary_conditions=None,       # 保留参数，但实际从 bc_manager 获取
        materials=None                   # 保留参数，便于未来扩展
    ) -> Tuple[bool, int]:
        """
        求解当前时间步

        Args:
            dt: 时间步长
            time: 当前时间
            solution: 当前步解函数（将被更新）
            solution_prev: 上一步解函数
            boundary_conditions: 保留，兼容性
            materials: 保留，兼容性

        Returns:
            (converged, iterations): 是否收敛，非线性迭代次数
        """
        # 获取当前边界条件
        bcs = self.bc_manager.get_boundary_conditions()

        # 构建当前时间步的弱形式（残差和雅可比）
        try:
            F_ufl, J_ufl = self.weak_form_builder.build_form(
                dt, time, solution, solution_prev, boundary_conditions=bcs
            )
        except Exception as e:
            self.logger.error(f"弱形式构建失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False, 0

        # 设置初始猜测（从上一步解开始）
        solution.x.array[:] = solution_prev.x.array[:]

        # 创建非线性问题（使用 DOLFINx 0.9.0 内置的 NonlinearProblem）
        try:
            self.problem = NonlinearProblem(F_ufl, solution, bcs, J_ufl)
        except Exception as e:
            self.logger.error(f"问题创建失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False, 0

        # 创建求解器
        if self.solver is None:
            self.solver = NewtonSolver(self.comm, self.problem)

            # 设置求解器参数
            self.solver.max_it = self.max_iterations
            self.solver.rtol = self.relative_tolerance
            self.solver.atol = self.absolute_tolerance

            # 设置线性求解器
            opts = PETSc.Options()
            prefix = "grouting_"
            self.solver.krylov_solver.setOptionsPrefix(prefix)
            opts.prefixPush(prefix)

            if self.linear_solver == "mumps":
                opts["ksp_type"] = "preonly"
                opts["pc_type"] = "lu"
                opts["pc_factor_mat_solver_type"] = "mumps"
            elif self.linear_solver == "gmres":
                opts["ksp_type"] = "gmres"
                opts["pc_type"] = self.preconditioner
            else:
                opts["ksp_type"] = "cg"
                opts["pc_type"] = self.preconditioner

            opts["ksp_rtol"] = 1e-8
            opts["ksp_atol"] = 1e-12

            opts.prefixPop()
            self.solver.krylov_solver.setFromOptions()

        # 求解
        try:
            n, converged = self.solver.solve(solution)
            self.iteration_counts.append(n)

            if self.report_convergence and self.rank == 0:
                self.logger.debug(
                    f"时间步 {time:.2f}s, dt={dt:.3e}, "
                    f"迭代次数: {n}, 收敛: {converged}"
                )

            return converged, n

        except Exception as e:
            self.logger.error(f"求解器异常: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
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
