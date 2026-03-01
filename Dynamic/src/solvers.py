"""
求解器管理器 - 非线性耦合求解器
基于 PETSc 的牛顿法求解多物理场耦合系统
"""

from dolfinx import fem, nls, log
from petsc4py import PETSc
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging


class SolverManager:
    """
    非线性求解器管理器
    负责组装残差和雅可比矩阵，驱动非线性迭代，处理收敛性
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
        self.logger = logging.getLogger(f"SolverManager_rank{comm.Get_rank()}")

        # 从配置提取参数
        self.max_iterations = self.config.get('max_iterations', 20)
        self.relative_tolerance = self.config.get('relative_tolerance', 1e-6)
        self.absolute_tolerance = self.config.get('absolute_tolerance', 1e-8)
        self.linear_solver = self.config.get('linear_solver', 'mumps')
        self.preconditioner = self.config.get('preconditioner', 'ilu')
        self.report_convergence = self.config.get('report_convergence', False)

        # 求解器对象（延迟创建）
        self.solver = None
        self.problem = None

        # 统计信息
        self.iteration_counts = []      # 每个时间步的迭代次数
        self.linear_iterations = []     # 每个非线性步的线性迭代次数（可选）

        if self.comm.rank == 0:
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
        # 获取当前边界条件（从边界条件管理器）
        bcs = self.bc_manager.get_boundary_conditions()

        # 构建当前时间步的弱形式（残差和雅可比）
        try:
            F, J = self.weak_form_builder.build_form(
                dt, time, solution, solution_prev, bcs
            )
        except Exception as e:
            self.logger.error(f"弱形式构建失败: {e}")
            return False, 0

        # 创建或更新非线性问题
        # 注意：每次时间步可能因 dt 或材料属性变化导致 F/J 变化，
        # 因此每次都需要重新创建问题对象。
        self.problem = fem.petsc.NonlinearProblem(F, solution, bcs=bcs, J=J)

        # 创建或重置求解器
        if self.solver is None:
            self._create_solver()
        else:
            # 重置求解器状态（可选，但 NewtonSolver 内部会处理）
            self.solver.reset()

        # 设置初始猜测（从上一步解开始）
        solution.x.array[:] = solution_prev.x.array[:]

        # 求解
        try:
            n, converged = self.solver.solve(self.problem)
            self.iteration_counts.append(n)

            if self.report_convergence and self.comm.rank == 0:
                self.logger.debug(
                    f"时间步 {time:.2f}s, dt={dt:.3e}, "
                    f"迭代次数: {n}, 收敛: {converged}"
                )

            return converged, n

        except Exception as e:
            self.logger.error(f"求解器异常: {e}")
            return False, 0

    def _create_solver(self):
        """创建 PETSc 牛顿求解器并设置参数"""
        self.solver = nls.petsc.NewtonSolver(self.comm, self.problem)

        # 设置非线性求解器参数
        self.solver.max_it = self.max_iterations
        self.solver.rtol = self.relative_tolerance
        self.solver.atol = self.absolute_tolerance

        # 设置线性求解器（KSP）选项
        ksp = self.solver.krylov_solver
        opts = PETSc.Options()

        # 使用选项前缀隔离，避免影响其他求解器
        prefix = "grouting_solver_"
        ksp.setOptionsPrefix(prefix)
        opts.prefixPush(prefix)

        # 配置线性求解器类型
        if self.linear_solver == "mumps":
            # 直接求解器
            opts["ksp_type"] = "preonly"
            opts["pc_type"] = "lu"
            opts["pc_factor_mat_solver_type"] = "mumps"
        elif self.linear_solver == "gmres":
            # 迭代求解器
            opts["ksp_type"] = "gmres"
            opts["pc_type"] = self.preconditioner
            # 可设置更多选项，如重启次数等
        else:
            # 默认使用超收敛
            opts["ksp_type"] = "cg"
            opts["pc_type"] = self.preconditioner

        # 可选：设置线性容差
        opts["ksp_rtol"] = 1e-8
        opts["ksp_atol"] = 1e-12

        # 应用选项
        ksp.setFromOptions()
        opts.prefixPop()

        if self.comm.rank == 0:
            self.logger.debug(
                f"求解器创建完成，前缀 '{prefix}'，线性求解器类型 {self.linear_solver}"
            )

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
        self.linear_iterations.clear()