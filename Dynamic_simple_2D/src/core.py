"""
多物理场耦合注浆模拟 - 主控制器（简化两场系统，独立空间）
基于混合物理论，模拟地基固结注浆过程
坐标系：竖直向上为正，二维中 y 向上，重力向量 (0,-9.81)
场顺序：位移(u), 压力(p) 分别独立
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
from decoupled_solver import DecoupledSolver
from time_stepping import TimeStepManager
from output_manager import OutputManager


class MultiphysicsGroutingSimulation:
    """
    多物理场注浆模拟器主控制器
    简化两场系统 [位移, 压力]，独立函数空间
    """

    def __init__(self, config_file: str = "config/grouting_config.yaml",
                 mesh_file: str = "meshes/foundation_axisymmetric.msh",
                 output_dir: str = "results"):
        """
        初始化模拟器

        Args:
            config_file: YAML配置文件路径
            mesh_file: Gmsh网格文件路径（二维网格）
            output_dir: 结果输出目录
        """
        # 获取当前脚本所在目录（即 src 文件夹）
        script_dir = Path(__file__).parent.resolve()
        project_root = script_dir.parent

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

        # 配置字典
        self.config = None

        # 模块占位符
        self.mesh = None
        self.cell_tags = None
        self.facet_tags = None
        self.V_u = None          # 位移函数空间
        self.V_p = None          # 压力函数空间
        self.u = None            # 当前位移
        self.p = None            # 当前压力
        self.u_prev = None       # 上一步位移
        self.p_prev = None       # 上一步压力
        self.fields = {}         # 场变量字典（用于调试/输出）

        self.materials = None
        self.bc_manager = None
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
            self.logger.info(f"多物理场注浆模拟器（简化两场系统，独立空间）初始化完成")
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

        log_level = logging.DEBUG if os.environ.get('DEBUG_MODE', 'false').lower() == 'true' else logging.INFO
        self.logger.setLevel(log_level)

        if self.logger.handlers:
            self.logger.handlers.clear()

        formatter = logging.Formatter(
            '[%(asctime)s | Rank %(process)d | %(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )

        console_level = logging.INFO if self.rank == 0 else logging.WARNING
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(console_level)
        self.logger.addHandler(console_handler)

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
                'dimension': 2,
            },
            'solver': {
                'type': 'newton',
                'max_iterations': 20,
                'relative_tolerance': 1e-6,
                'absolute_tolerance': 1e-8,
                'linear_solver': 'mumps',
                'preconditioner': 'lu',
                'report_convergence': False,
            },
            'output': {
                'format': 'xdmf',
                'fields': ['displacement', 'pressure'],
                'write_frequency': 10,
                'monitor_frequency': 100,
            },
            'grouting_termination': {
                'holding_time': 1200.0,
                'injection_rate_threshold': 2.0,
                'holding_rate_threshold': 2.0,
            }
        }

        def merge(base, update):
            for key, val in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(val, dict):
                    merge(base[key], val)
                else:
                    base[key] = val
            return base

        result = merge(defaults.copy(), config)

        if result['materials']['soil']['phi0'] <= 0 or result['materials']['soil']['phi0'] >= 1:
            self.logger.warning(f"孔隙度 phi0={result['materials']['soil']['phi0']} 不在(0,1)内")
        return result

    # ------------------------------------------------------------------
    # 模块初始化
    # ------------------------------------------------------------------
    def _initialize_modules(self):
        """按顺序初始化所有计算模块"""
        if self._modules_initialized:
            if self.rank == 0:
                self.logger.info("模块已初始化，跳过重复初始化")
            return

        if self.rank == 0:
            self.logger.info("开始初始化各模块...")

        try:
            self._initialize_mesh_handler()
            self._initialize_material_properties()
            self._initialize_function_spaces()
            self._initialize_time_stepping()
            self._initialize_boundary_conditions()
            self._initialize_solver_manager()
            self._initialize_output_manager()

            self._modules_initialized = True

            if self.rank == 0:
                self.logger.info("所有模块初始化完成")
        except Exception as e:
            self.logger.error(f"模块初始化失败: {e}", exc_info=True)
            raise

    def _initialize_mesh_handler(self):
        """从 Gmsh 文件读取二维网格"""
        from dolfinx.io import gmshio
        from dolfinx.mesh import create_cell_partitioner, GhostMode

        partitioner = create_cell_partitioner(GhostMode.shared_facet)
        self.mesh, self.cell_tags, self.facet_tags = gmshio.read_from_msh(
            filename=str(self.mesh_file),
            comm=self.comm,
            rank=0,
            gdim=2,
            partitioner=partitioner
        )

        if self.rank == 0:
            num_cells = self.mesh.topology.index_map(2).size_global
            num_vertices = self.mesh.geometry.x.shape[0]
            self.logger.info(f"网格信息: {num_cells} 个单元, {num_vertices} 个节点")

    def _initialize_material_properties(self):
        """初始化材料属性管理器（自动适应维度）"""
        self.materials = MaterialProperties(
            config=self.config,
            mesh=self.mesh,
            comm=self.comm
        )

    def _initialize_function_spaces(self):
        """创建独立的函数空间：位移(向量P2) 和 压力(标量P1)"""
        gdim = self.mesh.geometry.dim
        cell_type = self.mesh.topology.cell_name()
        import basix.ufl

        # 位移空间 (向量，二阶)
        P2_vec = basix.ufl.element("Lagrange", cell_type, 1, shape=(gdim,))
        self.V_u = fem.functionspace(self.mesh, P2_vec)

        # 压力空间 (标量，一阶)
        P1 = basix.ufl.element("Lagrange", cell_type, 1, shape=())
        self.V_p = fem.functionspace(self.mesh, P1)

        self.u = fem.Function(self.V_u, name="Displacement")
        self.p = fem.Function(self.V_p, name="Pressure")
        self.u_prev = fem.Function(self.V_u, name="Displacement_previous")
        self.p_prev = fem.Function(self.V_p, name="Pressure_previous")

        self.fields = {
            'displacement': self.u,
            'pressure': self.p
        }

        self._set_initial_conditions()

        if self.rank == 0:
            dof_u = self.V_u.dofmap.index_map.size_global * self.V_u.dofmap.index_map_bs
            dof_p = self.V_p.dofmap.index_map.size_global
            self.logger.info(f"函数空间: 位移自由度 {dof_u}, 压力自由度 {dof_p}")

    def _set_initial_conditions(self):
        """设置初始条件"""
        # 位移初始为0
        self.u.x.array[:] = 0.0
        self.u_prev.x.array[:] = 0.0

        # 压力初始化为静水压力（有量纲）
        H = self.config['geometry']['height']
        rho_w = self.materials.rho_w
        g = self.materials.g_magnitude

        def hydrostatic_pressure(x):
            return rho_w * g * (H - x[1])

        self.p.interpolate(hydrostatic_pressure)
        self.p_prev.interpolate(hydrostatic_pressure)

        p_array = self.p.x.array
        if np.any(np.isnan(p_array)):
            print("压力插值产生了 NaN!")

        if self.rank == 0:
            self.logger.info("初始条件设置完成：压力初始化为静水压力，位移为零")

    def _initialize_boundary_conditions(self):
        """初始化边界条件管理器，传入独立的函数空间"""
        self.bc_manager = DynamicBoundaryConditionsManager(
            mesh_obj=self.mesh,
            facet_tags=self.facet_tags,
            materials=self.materials,
            config=self.config,
            V_u=self.V_u,
            V_p=self.V_p,
            time=self.time,
            time_controller=self.time_controller
        )

    def _initialize_solver_manager(self):
        """初始化解耦求解器"""
        self.solver_manager = DecoupledSolver(
            comm=self.comm,
            materials=self.materials,
            bc_manager=self.bc_manager,
            V_u=self.V_u,
            V_p=self.V_p,
            config=self.config
        )

    def _initialize_time_stepping(self):
        """初始化时间步进控制器"""
        self.time_controller = TimeStepManager(
            config=self.config,
            comm=self.comm
        )

    def _initialize_output_manager(self):
        """初始化输出管理器"""
        self.output_manager = OutputManager(
            comm=self.comm,
            config=self.config,
            mesh=self.mesh,
            output_dir=self.output_dir
        )

    # ------------------------------------------------------------------
    # 运行主循环
    # ------------------------------------------------------------------
    def run(self, total_time: Optional[float] = None):
        """运行多物理场耦合模拟"""
        if self.rank == 0:
            self.logger.info("=" * 60)
            self.logger.info("开始多物理场注浆模拟（简化两场系统，独立空间）")
            self.logger.info("=" * 60)

        try:
            self._initialize_modules()

            sim_config = self.config['simulation']
            total_time = total_time or sim_config['total_time']
            max_steps = sim_config.get('max_steps', 10000)
            save_freq = self.config['output'].get('write_frequency', 10)

            if self.rank == 0:
                self.logger.info(f"总模拟时间: {total_time:.1f}秒")
                self.logger.info(f"最大时间步数: {max_steps}")
                self.logger.info(f"保存频率: 每 {save_freq} 步")

            self.materials.update_time_dependent_properties(self.time)

            self.time_step = 0
            self.converged = True

            report_interval = max(10, min(100, max_steps // 20))

            while self.time < total_time - 1e-10 and self.time_step < max_steps:
                self.time_step += 1

                dt, should_continue = self.time_controller.advance(converged=self.converged)
                if not should_continue:
                    self.logger.info("时间步进控制器指示终止模拟")
                    break

                self.time = self.time_controller.time

                self.materials.update_time_dependent_properties(self.time)
                self.bc_manager.update(self.time)

                if self.rank == 0 and (self.time_step % report_interval == 0 or self.time_step == 1):
                    progress = min(self.time / total_time * 100, 100.0)
                    self.logger.info(
                        f"时间步 {self.time_step}: t={self.time:.1f}s, dt={dt:.3f}s, 进度={progress:.1f}%"
                    )

                success, iterations = self.solver_manager.solve(
                    dt=dt,
                    time=self.time,
                    u=self.u,
                    p=self.p,
                    u_prev=self.u_prev,
                    p_prev=self.p_prev
                )

                if not success:
                    self.logger.warning(f"时间步 {self.time_step} 求解失败")
                    self.converged = False

                    reduced_dt = self.time_controller.reduce_time_step()
                    if reduced_dt > self.time_controller.min_dt:
                        self.logger.info(f"减小时间步长为 {reduced_dt:.3f}s 并重试")
                        self.time = self.time_controller.time
                        self.time_step -= 1
                        continue
                    else:
                        self.logger.error("时间步长已达最小值，模拟终止")
                        break

                self.converged = True

                derived_fields = self.materials.calculate_all_derived(self.fields, self.time)

                self.output_manager.write_timestep(
                    time=self.time,
                    time_step=self.time_step,
                    u=self.u,
                    p=self.p,
                    derived_fields=derived_fields
                )

                # 交换当前步与上一步
                self.u_prev, self.u = self.u, self.u_prev
                self.p_prev, self.p = self.p, self.p_prev

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

            # 如果最后一步未保存，再保存一次
            if self.time_step % self.config['output'].get('write_frequency', 10) != 0:
                derived_fields = self.materials.calculate_all_derived(self.fields, self.time)
                self.output_manager.write_timestep(
                    time=self.time,
                    time_step=self.time_step,
                    u=self.u,
                    p=self.p,
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
        """清理资源"""
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
            summary['mesh_cells'] = self.mesh.topology.index_map(2).size_global
        return summary


# ------------------------------------------------------------------
# 主程序入口
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import os
    import pathlib

    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

    script_dir = pathlib.Path(__file__).resolve().parent.parent
    default_config = str(script_dir / "config" / "grouting_config.yaml")
    default_mesh = str(script_dir / "meshes" / "foundation_axisymmetric.msh")
    default_output = str(script_dir / "results")

    parser = argparse.ArgumentParser(description="多物理场注浆模拟（简化两场系统，独立空间）")
    parser.add_argument("--config", type=str, default=default_config,
                        help="配置文件路径")
    parser.add_argument("--mesh", type=str, default=default_mesh,
                        help="二维网格文件路径")
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