"""
多物理场耦合注浆模拟 - 主控制器（重写版）
基于混合物理论，模拟地基固结注浆过程
坐标系：竖直向上为正，重力向量 (0,0,-9.81)
混合函数空间顺序：[位移, 压力, 孔隙度, 浓度]
注：使用压力扩散方程，达西速度从压力场导出
"""

import logging
import os
from pathlib import Path
from mpi4py import MPI
import yaml
import ufl
from dolfinx import fem, io, mesh
import numpy as np
from typing import Optional, Dict, Any, Callable

# 导入自定义模块
from materials import MaterialProperties
from boundary_conditions import DynamicBoundaryConditionsManager
from weak_forms import WeakFormBuilder
from solvers import SolverManager
from time_stepping import TimeStepManager
from output_manager import OutputManager


class MultiphysicsGroutingSimulation:
    """
    多物理场耦合注浆模拟器主控制器
    """

    def __init__(self, config_file: str = "config/grouting_config.yaml",
                 mesh_file: str = "meshes/foundation_drilling_model.msh",
                 output_dir: str = "results"):
        """
        初始化模拟器

        Args:
            config_file: YAML配置文件路径
            mesh_file: Gmsh网格文件路径
            output_dir: 结果输出目录
        """

        # 获取当前脚本所在目录（即 src 文件夹）
        script_dir = Path(__file__).parent.resolve()
        # 项目根目录为 src 的父目录（即 Dynamic 文件夹）
        project_root = script_dir.parent

        # 将传入的路径（相对路径）转换为基于项目根目录的绝对路径
        self.config_file = project_root / config_file
        self.mesh_file = project_root / mesh_file
        self.output_dir = project_root / output_dir
        
        # MPI设置
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # 模拟状态
        self.time = 0.0
        self.time_step = 0
        self.converged = True

        # 模块初始化状态
        self._modules_initialized = False

        # 配置字典（将在 _load_configuration 中填充）
        self.config = None

        # 模块占位符
        self.mesh = None
        self.cell_tags = None
        self.facet_tags = None
        self.W = None                     # 混合函数空间
        self.solution = None               # 当前步解
        self.solution_prev = None           # 上一步解
        self.fields = {}                   # 场变量字典（用于调试/输出）

        self.materials = None
        self.bc_manager = None
        self.weak_form_builder = None
        self.solver_manager = None
        self.time_controller = None
        self.output_manager = None

        # 日志设置
        self._setup_logging()

        # 验证输入文件
        self._validate_input_files()

        # 加载配置
        self._load_configuration()

        if self.rank == 0:
            self.logger.info(f"多物理场注浆模拟器初始化完成")
            self.logger.info(f"进程数: {self.size}")
            self.logger.info(f"配置文件: {self.config_file}")
            self.logger.info(f"网格文件: {self.mesh_file}")
            self.logger.info(f"输出目录: {self.output_dir}")

    # ------------------------------------------------------------------
    # 初始化辅助方法
    # ------------------------------------------------------------------
    def _setup_logging(self):
        """设置分级日志系统"""
        logger_name = f"GroutingSim_rank{self.rank}"
        self.logger = logging.getLogger(logger_name)

        # 从环境变量或默认值设置日志级别
        log_level = logging.DEBUG if os.environ.get('DEBUG_MODE', 'false').lower() == 'true' else logging.INFO
        self.logger.setLevel(log_level)

        # 清除已有处理器
        if self.logger.handlers:
            self.logger.handlers.clear()

        formatter = logging.Formatter(
            '[%(asctime)s | Rank %(process)d | %(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )

        # 控制台处理器（主进程详细，其他进程简洁）
        console_level = logging.INFO if self.rank == 0 else logging.WARNING
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(console_level)
        self.logger.addHandler(console_handler)

        # 文件处理器（所有进程）
        if self.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.comm.Barrier()

        log_file = self.output_dir / f"simulation_rank{self.rank}.log"
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

        self.logger.debug("日志系统初始化完成")

    def _validate_input_files(self):
        """验证所有输入文件的存在性和可读性"""
        if self.rank == 0:
            missing = []
            for fpath, desc in [(self.config_file, "配置文件"), (self.mesh_file, "网格文件")]:
                if not fpath.exists():
                    missing.append(f"{desc}: {fpath}")
            if missing:
                raise FileNotFoundError("缺少必要的输入文件:\n" + "\n".join(missing))
            self.logger.info("输入文件验证通过")

    def _load_configuration(self):
        """加载并验证配置文件"""
        try:
            if self.rank == 0:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f) or {}
            else:
                config_data = None

            config_data = self.comm.bcast(config_data, root=0)
            self.config = self._apply_defaults_and_validate(config_data)

            if self.rank == 0:
                self.logger.info("配置文件加载成功")
        except Exception as e:
            self.logger.error(f"配置文件加载失败: {e}")
            raise RuntimeError(f"配置文件加载失败: {e}")

    def _apply_defaults_and_validate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用默认值并验证配置完整性"""
        # 默认配置（基于文献参数）
        defaults = {
            'simulation': {
                'total_time': 3600.0,
                'dt_initial': 1.0,
                'dt_min': 0.1,
                'dt_max': 60.0,
                'save_frequency': 10,
                'max_steps': 10000,
                'tolerance': 1e-6,
                'max_consecutive_failures': 5,
                'adaptive_strategy': 'basic',
            },
            'materials': {
                'soil': {
                    'E': 20e6,
                    'nu': 0.3,
                    'phi0': 0.45,
                    'k0': 1e-12,
                    'biot_coefficient': 1.0,
                    'rho_s': 2650.0,
                },
                'grout': {
                    'rho_g': 1800.0,
                    'mu_g0': 0.01,
                    'xi': 1.56,
                    'filtration_coeff': 0.75,
                    'pressure': 220e3,
                    'duration': 600.0,
                    'rise_time': 60.0,
                    'pressure_mode': 'linear_increase',
                },
                'water': {
                    'rho_w': 1000.0,
                    'mu_w': 0.001,
                }
            },
            'geometry': {
                'height': 13.0,
                'depth': 10.0,
                'drill_radius': 0.05,
            },
            'solver': {
                'type': 'newton',
                'max_iterations': 20,
                'relative_tolerance': 1e-6,
                'absolute_tolerance': 1e-8,
                'linear_solver': 'mumps',
                'preconditioner': 'ilu',
                'report_convergence': False,
            },
            'output': {
                'format': 'xdmf',
                'fields': ['displacement', 'pressure', 'porosity', 'concentration', 'darcy_velocity'],
                'write_frequency': 10,
                'monitor_frequency': 100,
            },
            'grouting_termination': {
                'holding_time': 1200.0,          # 20分钟
                'injection_rate_threshold': 2.0,  # L/min
                'holding_rate_threshold': 2.0,    # L/min
            }
        }

        # 递归合并
        def merge(base, update):
            for key, val in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(val, dict):
                    merge(base[key], val)
                else:
                    base[key] = val
            return base

        result = merge(defaults.copy(), config)

        # 可选：添加参数合理性检查
        # 这里仅记录警告，不中断
        if result['materials']['soil']['phi0'] <= 0 or result['materials']['soil']['phi0'] >= 1:
            self.logger.warning(f"孔隙度 phi0={result['materials']['soil']['phi0']} 不在(0,1)内")
        return result

    # ------------------------------------------------------------------
    # 模块初始化
    # ------------------------------------------------------------------
    def _initialize_modules(self):
        """按顺序初始化所有计算模块"""
        # 避免重复初始化
        if self._modules_initialized:
            if self.rank == 0:
                self.logger.info("模块已初始化，跳过重复初始化")
            return

        if self.rank == 0:
            self.logger.info("开始初始化各模块...")

        try:
            # 1. 网格处理器
            self._initialize_mesh_handler()

            # 2. 材料属性管理器
            self._initialize_material_properties()

            # 3. 函数空间
            self._initialize_function_spaces()

            # 4. 时间步进控制器（需要在边界条件之前初始化）
            self._initialize_time_stepping()

            # 5. 边界条件管理器
            self._initialize_boundary_conditions()

            # 6. 弱形式构建器
            self._initialize_weak_forms()

            # 7. 求解器管理器
            self._initialize_solver_manager()

            # 8. 输出管理器
            self._initialize_output_manager()

            # 标记为已初始化
            self._modules_initialized = True

            if self.rank == 0:
                self.logger.info("所有模块初始化完成")
        except Exception as e:
            self.logger.error(f"模块初始化失败: {e}", exc_info=True)
            raise

    def _initialize_mesh_handler(self):
        """从 Gmsh 文件读取网格"""
        from dolfinx.io import gmshio
        from dolfinx.mesh import create_cell_partitioner, GhostMode

        partitioner = create_cell_partitioner(GhostMode.shared_facet)
        self.mesh, self.cell_tags, self.facet_tags = gmshio.read_from_msh(
            filename=str(self.mesh_file),
            comm=self.comm,
            rank=0,
            gdim=3,
            partitioner=partitioner
        )

        if self.rank == 0:
            num_cells = self.mesh.topology.index_map(3).size_global
            num_vertices = self.mesh.geometry.x.shape[0]
            self.logger.info(f"网格信息: {num_cells} 个单元, {num_vertices} 个节点")

    def _initialize_material_properties(self):
        """初始化材料属性管理器"""
        self.materials = MaterialProperties(
            config=self.config,
            mesh=self.mesh,
            comm=self.comm
        )

    def _initialize_function_spaces(self):
        """创建混合函数空间 [u, p, phi, c]"""
        gdim = self.mesh.geometry.dim

        # 使用 basix 创建有限元（新版 DOLFINx API）
        import basix.ufl

        # 通过拓扑获取单元类型名称
        cell_type = self.mesh.topology.cell_name()

        # 标量元素（用于 p, phi, c）
        P1 = basix.ufl.element("Lagrange", cell_type, 1, shape=())

        # 向量元素（用于 u）
        P1_vec = basix.ufl.element("Lagrange", cell_type, 1, shape=(gdim,))

        # 创建混合元素
        from basix.ufl import mixed_element as create_mixed_element
        mixed_element = create_mixed_element([P1_vec, P1, P1, P1])

        # 新版 DOLFINx API 使用 functionspace
        self.W = fem.functionspace(self.mesh, mixed_element)

        # 创建解函数
        self.solution = fem.Function(self.W, name="Solution")
        self.solution_prev = fem.Function(self.W, name="Solution_previous")

        # 提取各场分量（用于方便访问）
        self.u, self.p, self.phi, self.c = ufl.split(self.solution)
        self.u_n, self.p_n, self.phi_n, self.c_n = ufl.split(self.solution_prev)

        self.fields = {
            'displacement': self.u,
            'pressure': self.p,
            'porosity': self.phi,
            'concentration': self.c
        }

        # 设置初始条件
        self._set_initial_conditions()

        if self.rank == 0:
            dof_count = self.W.dofmap.index_map.size_global * self.W.dofmap.index_map_bs
            self.logger.info(f"函数空间: {dof_count} 个自由度")

    def _set_initial_conditions(self):
        phi0 = self.config['materials']['soil']['phi0']

        # 直接对混合函数的子空间视图赋值
        self.solution.sub(2).x.array[:] = phi0
        self.solution_prev.sub(2).x.array[:] = phi0

        # 其他场设为零
        for idx in [0, 1, 3]:
            self.solution.sub(idx).x.array[:] = 0.0
            self.solution_prev.sub(idx).x.array[:] = 0.0

        if self.rank == 0:
            self.logger.info("初始条件设置完成")

    def _initialize_boundary_conditions(self):
        """初始化边界条件管理器"""
        self.bc_manager = DynamicBoundaryConditionsManager(
            mesh_obj=self.mesh,
            facet_tags=self.facet_tags,
            materials=self.materials,
            config=self.config,
            function_space=self.W,
            time=self.time,
            time_controller=self.time_controller
        )

    def _initialize_weak_forms(self):
        """初始化弱形式构建器"""
        self.weak_form_builder = WeakFormBuilder(
            function_space=self.W,
            materials=self.materials,
            config=self.config,
            fields=self.fields
        )

    def _initialize_solver_manager(self):
        """初始化求解器管理器"""
        self.solver_manager = SolverManager(
            comm=self.comm,
            config=self.config,
            function_space=self.W,
            weak_form_builder=self.weak_form_builder,
            bc_manager=self.bc_manager
        )

    def _initialize_time_stepping(self):
        """初始化时间步进控制器"""
        self.time_controller = TimeStepManager(
            config=self.config,
            comm=self.comm
        )
        # 注册注入率计算回调
        self.time_controller.set_injection_rate_calculator(self._compute_injection_rate)

    def _initialize_output_manager(self):
        """初始化输出管理器"""
        self.output_manager = OutputManager(
            comm=self.comm,
            config=self.config,
            mesh=self.mesh,
            output_dir=self.output_dir
        )

    # ------------------------------------------------------------------
    # 注入率计算（关键功能）
    # ------------------------------------------------------------------
    def _compute_injection_rate(self) -> float:
        """
        计算当前注入率 (L/min)
        通过对注浆孔边界上的浆液通量积分得到
        注：1 m³/s = 60000 L/min
        使用从压力导出的达西速度 q_darcy = -(k/μ)(∇p - ρg)
        """
        if self.solution is None or self.bc_manager is None:
            return 0.0

        # 获取浓度场
        c_func = self.solution.sub(3).collapse()
        p_func = self.solution.sub(1).collapse()
        phi_func = self.solution.sub(2).collapse()

        # 从边界几何中获取所有注浆孔的面
        inlet_facets = []
        for name, info in self.bc_manager.boundary_geometries.items():
            if name.startswith('grout_inlet_') or name == 'grout_inlet_fallback':
                inlet_facets.extend(info['facets'])

        if not inlet_facets:
            return 0.0

        inlet_facets = np.array(inlet_facets, dtype=np.int32)

        # 定义边界测量
        fdim = self.mesh.topology.dim - 1

        # 创建临时标记
        from dolfinx.mesh import meshtags
        inlet_marker = 1000
        inlet_meshtags = meshtags(self.mesh, fdim, inlet_facets, np.full(len(inlet_facets), inlet_marker, dtype=np.int32))

        # 法向向量
        n = ufl.FacetNormal(self.mesh)

        # 获取当前时间的材料属性
        k = self.materials.calculate_permeability(phi_func)
        mu = self.materials.calculate_viscosity(c_func, self.time)
        rho = self.materials.calculate_density(c_func)
        g = self.materials.g

        # 计算达西速度：q_darcy = -(k/μ)(∇p - ρg)
        q_darcy = -(k / mu) * (ufl.grad(p_func) - rho * g)

        # 定义积分测量
        ds_inlet = ufl.Measure("ds", domain=self.mesh, subdomain_data=inlet_meshtags)

        # 积分：c * (q_darcy·n) 在注浆孔边界上
        flux_form = fem.form(c_func * ufl.dot(q_darcy, n) * ds_inlet(inlet_marker))
        flux = fem.assemble_scalar(flux_form)

        # 转换为 L/min (假设单位：m³/s -> L/min)
        injection_rate = flux * 60000.0
        return injection_rate

    # ------------------------------------------------------------------
    # 运行主循环
    # ------------------------------------------------------------------
    def run(self, total_time: Optional[float] = None):
        """
        运行多物理场耦合模拟

        Args:
            total_time: 可选，覆盖配置文件中的总模拟时间
        """
        if self.rank == 0:
            self.logger.info("=" * 60)
            self.logger.info("开始多物理场注浆模拟")
            self.logger.info("=" * 60)

        try:
            # 1. 初始化所有模块
            self._initialize_modules()

            # 2. 设置模拟参数
            sim_config = self.config['simulation']
            total_time = total_time or sim_config['total_time']
            max_steps = sim_config.get('max_steps', 10000)
            save_freq = self.config['output'].get('write_frequency', 10)

            if self.rank == 0:
                self.logger.info(f"总模拟时间: {total_time:.1f}秒")
                self.logger.info(f"最大时间步数: {max_steps}")
                self.logger.info(f"保存频率: 每 {save_freq} 步")

            # 3. 初始化材料时间相关属性
            self.materials.update_time_dependent_properties(self.time)

            # 4. 时间循环
            self.time_step = 0
            self.converged = True

            # 进度报告间隔
            report_interval = max(10, min(100, max_steps // 20))

            while self.time < total_time - 1e-10 and self.time_step < max_steps:
                self.time_step += 1

                # 获取时间步长
                dt, should_continue = self.time_controller.advance(converged=self.converged)
                if not should_continue:
                    self.logger.info("时间步进控制器指示终止模拟")
                    break

                self.time = self.time_controller.time

                # 更新时间相关材料属性
                self.materials.update_time_dependent_properties(self.time)

                # 更新边界条件
                self.bc_manager.update(self.time)

                # 进度报告
                if self.rank == 0 and (self.time_step % report_interval == 0 or self.time_step == 1):
                    progress = min(self.time / total_time * 100, 100.0)
                    self.logger.info(
                        f"时间步 {self.time_step}: t={self.time:.1f}s, dt={dt:.3f}s, 进度={progress:.1f}%"
                    )

                # 求解当前时间步
                if self.rank == 0 and self.time_step % report_interval == 0:
                    self.logger.debug(f"开始求解时间步 {self.time_step}...")

                success, iterations = self.solver_manager.solve(
                    dt=dt,
                    time=self.time,
                    solution=self.solution,
                    solution_prev=self.solution_prev,
                    materials=self.materials
                )

                if not success:
                    self.logger.warning(f"时间步 {self.time_step} 求解失败")
                    self.converged = False

                    # 减小时间步长重试
                    reduced_dt = self.time_controller.reduce_time_step()
                    if reduced_dt > self.time_controller.min_dt:
                        self.logger.info(f"减小时间步长为 {reduced_dt:.3f}s 并重试")
                        self.time = self.time_controller.time  # 已回退
                        self.time_step -= 1  # 步数不增加
                        continue
                    else:
                        self.logger.error("时间步长已达最小值，模拟终止")
                        break

                self.converged = True

                # 更新注浆状态（压力和注入率）
                current_pressure = self.bc_manager.get_current_pressure_value()
                # 注入率已在 update_grouting_status 内部通过回调自动计算，但也可以显式传入
                self.time_controller.update_grouting_status(current_pressure)

                # 计算衍生场（如果需要）
                derived_fields = self.materials.calculate_all_derived(self.fields, self.time)

                # 保存结果
                self.output_manager.write_timestep(
                    time=self.time,
                    time_step=self.time_step,
                    solution=self.solution,
                    derived_fields=derived_fields
                )

                # 更新上一时间步的解（交换指针以提高效率）
                self.solution_prev, self.solution = self.solution, self.solution_prev

            # 5. 模拟结束处理
            if self.converged:
                if self.rank == 0:
                    self.logger.info("=" * 60)
                    self.logger.info("模拟成功完成")
                    self.logger.info(f"总时间步数: {self.time_step}")
                    self.logger.info(f"最终时间: {self.time:.1f}s")
                    self.logger.info("=" * 60)
            else:
                if self.rank == 0:
                    self.logger.warning("=" * 60)
                    self.logger.warning("模拟提前终止")
                    self.logger.info(f"已计算时间步数: {self.time_step}")
                    self.logger.info(f"最终时间: {self.time:.1f}s")
                    self.logger.info("=" * 60)

            # 6. 保存最终结果（如果最后一步未保存）
            if self.time_step % self.config['output'].get('write_frequency', 10) != 0:
                derived_fields = self.materials.calculate_all_derived(self.fields, self.time)
                self.output_manager.write_timestep(
                    time=self.time,
                    time_step=self.time_step,
                    solution=self.solution,
                    derived_fields=derived_fields
                )

        except KeyboardInterrupt:
            if self.rank == 0:
                self.logger.warning("模拟被用户中断")
            self._cleanup()
            raise
        except Exception as e:
            if self.rank == 0:
                self.logger.error(f"模拟运行失败: {e}", exc_info=True)
            self._cleanup()
            raise
        else:
            self._cleanup()

    # ------------------------------------------------------------------
    # 清理与结果摘要
    # ------------------------------------------------------------------
    def _cleanup(self):
        """清理资源（关闭文件等）"""
        try:
            if hasattr(self, 'output_manager') and self.output_manager:
                self.output_manager.close()
            if self.rank == 0:
                self.logger.info("模拟资源已清理")
        except Exception as e:
            self.logger.warning(f"清理资源时出错: {e}")

    def get_results_summary(self) -> Dict[str, Any]:
        """获取模拟结果摘要"""
        summary = {
            'time': self.time,
            'time_steps': self.time_step,
            'converged': self.converged,
            'fields': list(self.fields.keys()) if self.fields else [],
            'config_file': str(self.config_file),
            'output_dir': str(self.output_dir)
        }
        if hasattr(self, 'mesh') and self.mesh:
            summary['mesh_cells'] = self.mesh.topology.index_map(3).size_global
        return summary


# ------------------------------------------------------------------
# 主程序入口
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import os
    import pathlib

    # 禁用 HDF5 文件锁定（避免在 MPI 环境中的文件锁问题）
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

    # 获取脚本所在目录，作为相对路径的基准
    script_dir = pathlib.Path(__file__).resolve().parent.parent
    default_config = str(script_dir / "config" / "grouting_config.yaml")
    default_mesh = str(script_dir / "meshes" / "foundation_drilling_model.msh")
    default_output = str(script_dir / "results")

    parser = argparse.ArgumentParser(description="多物理场注浆模拟")
    parser.add_argument("--config", type=str, default=default_config,
                        help="配置文件路径")
    parser.add_argument("--mesh", type=str, default=default_mesh,
                        help="网格文件路径")
    parser.add_argument("--output", type=str, default=default_output,
                        help="输出目录路径")
    parser.add_argument("--time", type=float, default=None,
                        help="总模拟时间（秒）")
    parser.add_argument("--debug", action="store_true",
                        help="启用调试模式（详细日志）")

    args = parser.parse_args()

    if args.debug:
        os.environ['DEBUG_MODE'] = 'true'

    sim = MultiphysicsGroutingSimulation(
        config_file=args.config,
        mesh_file=args.mesh,
        output_dir=args.output
    )

    try:
        sim.run(total_time=args.time)
        summary = sim.get_results_summary()
        if sim.rank == 0:
            print("\n" + "=" * 60)
            print("模拟摘要:")
            print(f"  状态: {'成功' if summary['converged'] else '失败'}")
            print(f"  最终时间: {summary['time']:.1f}秒")
            print(f"  模拟场: {', '.join(summary['fields'])}")
            print(f"  输出目录: {summary['output_dir']}")
            print("=" * 60)
    except Exception as e:
        if sim.rank == 0:
            print(f"模拟错误: {e}")
        exit(1)