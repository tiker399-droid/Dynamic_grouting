"""
多物理场耦合注浆模拟 - 主控制器
"""

import logging
from pathlib import Path
from mpi4py import MPI
import yaml
import ufl
from dolfinx import fem, io
import numpy as np
from typing import Optional, Dict, Any

class MultiphysicsGroutingSimulation:
    """
    多物理场耦合注浆模拟器 - 基于混合物理论
    模拟TBM掘进期间砂土中的注浆渗透与过滤
    """
    
    def __init__(self, config_file: str = "Dynamic/config/grouting_config.yaml", 
                 mesh_file: str = "Dynamic/meshes/foundation_drilling_model.msh", 
                 output_dir: str = "Dynamic/results"):
        """
        初始化模拟器
        
        Args:
            config_file: YAML配置文件路径
            mesh_file: Gmsh网格文件路径  
            output_dir: 结果输出目录
        """
        # MPI设置
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # 文件路径
        self.config_file = Path(config_file)
        self.mesh_file = Path(mesh_file)
        self.output_dir = Path(output_dir)
        
        # 模拟状态
        self.time = 0.0
        self.time_step = 0
        self.converged = False
        
        # 关键变量字典（用于存储各物理场）
        self.fields = {}
        
        # 初始化步骤
        self._setup_logging()
        self._validate_input_files()
        self._load_configuration()
        
        # 日志记录初始化完成
        if self.rank == 0:
            self.logger.info(f"多物理场注浆模拟器初始化完成")
            self.logger.info(f"进程数: {self.size}")
            self.logger.info(f"配置文件: {self.config_file}")
            self.logger.info(f"网格文件: {self.mesh_file}")
            self.logger.info(f"输出目录: {self.output_dir}")
    
    def _setup_logging(self):
        """设置分级日志系统"""
        logger_name = f"GroutingSim_rank{self.rank}"
        self.logger = logging.getLogger(logger_name)
        
        # 设置日志级别
        log_level = logging.DEBUG if self.config.get('debug', False) else logging.INFO
        self.logger.setLevel(log_level)
        
        # 清除已有处理器
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 日志格式
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
            missing_files = []
            
            for file_path, file_type in [
                (self.config_file, "配置文件"),
                (self.mesh_file, "网格文件")
            ]:
                if not file_path.exists():
                    missing_files.append(f"{file_type}: {file_path}")
            
            if missing_files:
                error_msg = "缺少必要的输入文件:\n" + "\n".join(missing_files)
                raise FileNotFoundError(error_msg)
            
            self.logger.info("输入文件验证通过")
    
    def _load_configuration(self):
        """加载并验证配置文件"""
        try:
            if self.rank == 0:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f) or {}
            else:
                config_data = None
            
            # 广播配置到所有进程
            config_data = self.comm.bcast(config_data, root=0)
            
            # 应用默认配置并验证
            self.config = self._apply_defaults_and_validate(config_data)
            
            if self.rank == 0:
                self.logger.info("配置文件加载成功")
                
        except Exception as e:
            error_msg = f"配置文件加载失败: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _apply_defaults_and_validate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用默认值并验证配置完整性"""
        # 默认配置（基于文献参数）
        defaults = {
            'simulation': {
                'total_time': 3600.0,      # 总模拟时间 (秒)
                'dt_initial': 1.0,         # 初始时间步长
                'dt_min': 0.1,             # 最小时间步长
                'dt_max': 60.0,            # 最大时间步长
                'save_frequency': 10,      # 保存频率
                'max_steps': 10000,        # 最大步数
                'tolerance': 1e-6,         # 收敛容差
            },
            'materials': {
                'soil': {
                    'E': 20e6,             # 杨氏模量 (Pa)
                    'nu': 0.3,             # 泊松比
                    'phi0': 0.45,          # 初始孔隙度
                    'K0': 0.5,             # 静止土压力系数
                    'friction_angle': 30.0, # 摩擦角 (°)
                    'cohesion': 10e3,      # 黏聚力 (Pa)
                    'dilatancy_angle': 5.0 # 剪胀角 (°)
                },
                'grout': {
                    'rho_g': 1800,         # 密度 (kg/m³)
                    'mu_g0': 0.01,         # 初始粘度 (Pa·s)
                    'xi': 1.56,            # 粘度增长常数
                    'filtration_coeff': 0.75, # 过滤系数
                    'pressure': 220e3,     # 注浆压力 (Pa)
                    'duration': 600.0      # 注浆持续时间 (s)
                },
                'water': {
                    'rho_w': 1000,         # 密度 (kg/m³)
                    'mu_w': 0.001,         # 粘度 (Pa·s)
                    'K_w': 1e-4            # 水力传导率 (m/s)
                }
            },
            'solver': {
                'type': 'newton',          # 求解器类型
                'max_iterations': 20,      # 最大迭代次数
                'relative_tolerance': 1e-6,
                'absolute_tolerance': 1e-8,
                'linear_solver': 'mumps',  # 线性求解器
                'preconditioner': 'ilu'    # 预条件子
            },
            'output': {
                'format': 'xdmf',          # 输出格式
                'fields': ['displacement', 'pressure', 'porosity', 'concentration', 'darcy_velocity'],
                'write_frequency': 10
            }
        }
        
        # 递归合并配置
        def merge_dict(base, update):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
            return base
        
        result = merge_dict(defaults.copy(), config)
        
        # 验证关键参数
        required_params = [
            ('simulation.total_time', float),
            ('materials.soil.phi0', float),
            ('materials.grout.pressure', float),
            ('materials.water.K_w', float)
        ]
        
        for param_path, param_type in required_params:
            try:
                keys = param_path.split('.')
                value = result
                for key in keys:
                    value = value[key]
                if not isinstance(value, param_type):
                    self.logger.warning(f"参数 {param_path} 类型不正确，期望 {param_type}")
            except KeyError:
                self.logger.warning(f"缺少关键参数: {param_path}")
        
        return result
    
    def _initialize_modules(self):
        """初始化所有计算模块"""
        modules_initialized = []
        
        try:
            # 1. 网格处理器
            self.logger.info("初始化网格处理器...")
            self._initialize_mesh_handler()
            modules_initialized.append('mesh')
            
            # 2. 材料属性管理器
            self.logger.info("初始化材料属性管理器...")
            self._initialize_material_properties()
            modules_initialized.append('materials')
            
            # 3. 函数空间
            self.logger.info("初始化函数空间...")
            self._initialize_function_spaces()
            modules_initialized.append('function_spaces')
            
            # 4. 边界条件管理器
            self.logger.info("初始化边界条件管理器...")
            self._initialize_boundary_conditions()
            modules_initialized.append('boundary_conditions')
            
            # 5. 弱形式构建器
            self.logger.info("初始化弱形式构建器...")
            self._initialize_weak_forms()
            modules_initialized.append('weak_forms')
            
            # 6. 求解器管理器
            self.logger.info("初始化求解器管理器...")
            self._initialize_solver_manager()
            modules_initialized.append('solver')
            
            # 7. 时间步进控制器
            self.logger.info("初始化时间步进控制器...")
            self._initialize_time_stepping()
            modules_initialized.append('time_stepping')
            
            # 8. 输出管理器
            self.logger.info("初始化输出管理器...")
            self._initialize_output_manager()
            modules_initialized.append('output')
            
            if self.rank == 0:
                self.logger.info(f"所有模块初始化完成: {', '.join(modules_initialized)}")
                
        except Exception as e:
            error_msg = f"模块初始化失败: {e}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)
    
    def _initialize_mesh_handler(self):
        """初始化网格处理器"""
        try:
            from dolfinx.io import gmshio
            from dolfinx.mesh import create_cell_partitioner, GhostMode
            
            # 创建网格分区器
            partitioner = create_cell_partitioner(GhostMode.shared_facet)
            
            # 读取Gmsh网格
            self.mesh, self.cell_tags, self.facet_tags = gmshio.read_from_msh(
                filename=str(self.mesh_file),
                comm=self.comm,
                rank=0,
                gdim=3,
                partitioner=partitioner
            )
            
            # 记录网格信息
            if self.rank == 0:
                num_cells = self.mesh.topology.index_map(3).size_global
                num_vertices = self.mesh.geometry.x.shape[0]
                self.logger.info(f"网格信息: {num_cells} 个单元, {num_vertices} 个节点")
                
        except ImportError as e:
            raise ImportError(f"无法导入网格处理模块: {e}")
        except Exception as e:
            raise RuntimeError(f"网格加载失败: {e}")
    
    def _initialize_material_properties(self):
        """初始化材料属性管理器"""
        try:
            # 假设存在 materials.py 模块
            from materials import MaterialProperties
            
            self.materials = MaterialProperties(
                config=self.config['materials'],
                mesh=self.mesh,
                comm=self.comm
            )
            
        except ImportError:
            # 备用：创建简单的材料管理器
            self.logger.warning("使用内置简单材料管理器")
            self._create_simple_material_manager()
        except Exception as e:
            raise RuntimeError(f"材料属性管理器初始化失败: {e}")
    
    def _create_simple_material_manager(self):
        """创建简单材料管理器（备用）"""
        class SimpleMaterialManager:
            def __init__(self, config, mesh, comm):
                self.config = config
                self.mesh = mesh
                self.comm = comm
                
                # 土体参数
                self.soil = config['soil']
                # 浆液参数
                self.grout = config['grout']
                # 水参数
                self.water = config['water']
                
        self.materials = SimpleMaterialManager(
            config=self.config['materials'],
            mesh=self.mesh,
            comm=self.comm
        )
    
    def _initialize_function_spaces(self):
        """初始化函数空间（基于文献方程）"""
        try:
            # 几何维度
            gdim = self.mesh.geometry.dim
            
            # 定义有限元类型（文献中为混合单元）
            # 位移: 矢量P1元
            # 压力: 标量P1元  
            # 孔隙度: 标量P1元
            # 浓度: 标量P1元
            # 达西流速: 矢量P1元
            
            P1 = ufl.FiniteElement("CG", self.mesh.ufl_cell(), 1)
            P1_vec = ufl.VectorElement("CG", self.mesh.ufl_cell(), 1, dim=gdim)
            
            # 创建混合函数空间 [位移, 压力, 孔隙度, 浓度, 达西流速]
            mixed_element = ufl.MixedElement([P1_vec, P1, P1, P1, P1_vec])
            self.W = fem.FunctionSpace(self.mesh, mixed_element)
            
            # 创建当前和上一时间步的函数
            self.solution = fem.Function(self.W, name="Solution")
            self.solution_prev = fem.Function(self.W, name="Solution_previous")
            
            # 提取各物理场分量
            self._extract_field_components()
            
            # 设置初始条件
            self._set_initial_conditions()
            
            if self.rank == 0:
                self.logger.info(f"函数空间: {self.W.dofmap.index_map.size_global} 个自由度")
                
        except Exception as e:
            raise RuntimeError(f"函数空间初始化失败: {e}")
    
    def _extract_field_components(self):
        """从混合函数中提取各物理场分量"""
        # 使用ufl.split提取当前时间步分量
        self.u, self.p, self.phi, self.c, self.q_w = ufl.split(self.solution)
        
        # 上一时间步分量
        self.u_n, self.p_n, self.phi_n, self.c_n, self.q_w_n = ufl.split(self.solution_prev)
        
        # 存储为字段字典
        self.fields = {
            'displacement': self.u,
            'pressure': self.p,
            'porosity': self.phi,
            'concentration': self.c,
            'darcy_velocity': self.q_w
        }
    
    def _set_initial_conditions(self):
        """根据文献设置初始条件"""
        try:
            # 获取初始孔隙度
            phi0 = self.materials.soil.get('phi0', 0.45)
            
            # 1. 初始孔隙度（均匀分布）
            phi_func = self.solution.sub(2).collapse()
            phi_func.x.array[:] = phi0
            self.solution_prev.sub(2).collapse().x.array[:] = phi0
            
            # 2. 初始浓度（无浆液）
            c_func = self.solution.sub(3).collapse()
            c_func.x.array[:] = 0.0
            self.solution_prev.sub(3).collapse().x.array[:] = 0.0
            
            # 3. 初始位移（零位移）
            u_func = self.solution.sub(0).collapse()
            u_func.x.array[:] = 0.0
            self.solution_prev.sub(0).collapse().x.array[:] = 0.0
            
            # 4. 初始达西流速（零流速）
            q_func = self.solution.sub(4).collapse()
            q_func.x.array[:] = 0.0
            self.solution_prev.sub(4).collapse().x.array[:] = 0.0
            
            # 5. 初始压力（静水压力）
            self._set_initial_hydrostatic_pressure()
            
            if self.rank == 0:
                self.logger.info("初始条件设置完成")
                
        except Exception as e:
            raise RuntimeError(f"初始条件设置失败: {e}")
    
    def _set_initial_hydrostatic_pressure(self):
        """设置初始静水压力分布"""
        # 获取几何坐标
        mesh = self.mesh
        coords = mesh.geometry.x
        
        # 假设水面在地表
        water_table = np.max(coords[:, 2])
        
        # 获取压力场的自由度坐标
        V_p = self.solution.sub(1).function_space
        dof_coords = V_p.tabulate_dof_coordinates()
        
        # 计算静水压力：p = ρ_w * g * h
        rho_w = self.materials.water.get('rho_w', 1000.0)
        g = 9.81
        
        pressure_values = np.zeros_like(self.solution.sub(1).collapse().x.array)
        for i, coord in enumerate(dof_coords):
            z = coord[2]
            depth = water_table - z
            pressure_values[i] = rho_w * g * max(0.0, depth)
        
        # 设置压力场
        self.solution.sub(1).collapse().x.array[:] = pressure_values
        self.solution_prev.sub(1).collapse().x.array[:] = pressure_values
    
    def _initialize_boundary_conditions(self):
        """初始化边界条件管理器"""
        try:
            # 假设存在 boundary_conditions.py 模块
            from boundary_conditions import BoundaryConditionsManager
            
            self.bc_manager = BoundaryConditionsManager(
                mesh=self.mesh,
                facet_tags=self.facet_tags,
                materials=self.materials,
                config=self.config,
                function_space=self.W,
                time=self.time
            )
            
        except Exception as e:
            raise RuntimeError(f"边界条件管理器初始化失败: {e}")
    
    def _initialize_weak_forms(self):
        """初始化弱形式构建器"""
        try:
            # 假设存在 weak_forms.py 模块
            from weak_forms import WeakFormBuilder
            
            self.weak_form_builder = WeakFormBuilder(
                function_space=self.W,
                materials=self.materials,
                config=self.config,
                fields=self.fields
            )
            
        except Exception as e:
            raise RuntimeError(f"弱形式构建器初始化失败: {e}")
    
    def _initialize_solver_manager(self):
        """初始化求解器管理器"""
        try:
            # 假设存在 solvers.py 模块
            from solvers import SolverManager
            
            self.solver_manager = SolverManager(
                comm=self.comm,
                config=self.config['solver'],
                function_space=self.W,
                weak_form_builder=self.weak_form_builder,
                bc_manager=self.bc_manager
            )
            
        except Exception as e:
            raise RuntimeError(f"求解器管理器初始化失败: {e}")
    
    def _initialize_time_stepping(self):
        """初始化时间步进控制器"""
        try:
            # 假设存在 time_stepping.py 模块
            from time_stepping import TimeSteppingController
            
            self.time_controller = TimeSteppingController(
                config=self.config['simulation'],
                initial_time=self.time
            )
            
        except Exception as e:
            raise RuntimeError(f"时间步进控制器初始化失败: {e}")
    
    
    def _initialize_output_manager(self):
        """初始化输出管理器"""
        try:
            # 创建输出目录
            if self.rank == 0:
                (self.output_dir / "results").mkdir(exist_ok=True)
            self.comm.Barrier()
            
            # 创建XDMF文件用于输出
            output_file = self.output_dir / "results" / "simulation_results.xdmf"
            self.xdmf_writer = io.XDMFFile(self.comm, str(output_file), "w")
            self.xdmf_writer.write_mesh(self.mesh)
            
            self.save_counter = 0
            self.save_frequency = self.config['output'].get('write_frequency', 10)
            
        except Exception as e:
            self.logger.warning(f"输出管理器初始化失败: {e}")
            self.xdmf_writer = None
    
    def _save_results(self):
        """保存当前时间步的结果"""
        if self.xdmf_writer is None or self.save_counter % self.save_frequency != 0:
            return
        
        try:
            # 提取各物理场
            fields_to_save = self.config['output'].get('fields', [])
            
            for field_name in fields_to_save:
                if field_name == 'displacement':
                    u_func = self.solution.sub(0).collapse()
                    u_func.name = "Displacement"
                    self.xdmf_writer.write_function(u_func, self.time)
                elif field_name == 'pressure':
                    p_func = self.solution.sub(1).collapse()
                    p_func.name = "Pressure"
                    self.xdmf_writer.write_function(p_func, self.time)
                elif field_name == 'porosity':
                    phi_func = self.solution.sub(2).collapse()
                    phi_func.name = "Porosity"
                    self.xdmf_writer.write_function(phi_func, self.time)
                elif field_name == 'concentration':
                    c_func = self.solution.sub(3).collapse()
                    c_func.name = "Concentration"
                    self.xdmf_writer.write_function(c_func, self.time)
                # 注意：达西流速是矢量场
            
            self.save_counter += 1
            
            if self.rank == 0 and self.save_counter % (self.save_frequency * 10) == 0:
                self.logger.info(f"结果已保存，时间: {self.time:.1f}s")
                
        except Exception as e:
            self.logger.warning(f"保存结果失败: {e}")
    
    def run(self, total_time: Optional[float] = None):
        """运行多物理场耦合模拟"""
        
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
            
            if self.rank == 0:
                self.logger.info(f"总模拟时间: {total_time}秒")
                self.logger.info(f"最大时间步数: {max_steps}")
            
            # 3. 时间步进循环
            step_count = 0
            self.converged = True
            
            while self.time_controller.should_continue() and step_count < max_steps:
                step_count += 1
                
                # 更新当前时间步
                dt = self.time_controller.advance()
                self.time = self.time_controller.time
                
                if self.rank == 0 and step_count % 100 == 0:
                    progress = self.time_controller.get_progress() * 100
                    self.logger.info(f"时间步 {step_count}: 时间={self.time:.1f}s, dt={dt:.3f}s, 进度={progress:.1f}%")
                
                # 更新边界条件（随时间变化）
                self.bc_manager.update(self.time)
                
                # 获取当前边界条件
                bcs = self.bc_manager.get_boundary_conditions()
                
                # 求解当前时间步
                success = self.solver_manager.solve(
                    dt=dt,
                    time=self.time,
                    solution=self.solution,
                    solution_prev=self.solution_prev,
                    boundary_conditions=bcs
                )
                
                if not success:
                    self.logger.warning(f"时间步 {step_count} 求解失败")
                    self.converged = False
                    break
                
                # 保存结果
                self._save_results()
                
                # 更新上一时间步的解
                self.solution_prev.x.array[:] = self.solution.x.array[:]
            
            # 4. 模拟完成
            if self.converged:
                if self.rank == 0:
                    self.logger.info("=" * 60)
                    self.logger.info("模拟成功完成")
                    self.logger.info(f"总时间步数: {step_count}")
                    self.logger.info(f"最终时间: {self.time:.1f}s")
                    self.logger.info("=" * 60)
            else:
                if self.rank == 0:
                    self.logger.warning("模拟提前终止")
            
            # 5. 清理资源
            self._cleanup()
            
        except Exception as e:
            if self.rank == 0:
                self.logger.error(f"模拟运行失败: {e}", exc_info=True)
            raise RuntimeError(f"模拟失败: {e}")
    
    def _cleanup(self):
        """清理模拟资源"""
        try:
            if hasattr(self, 'xdmf_writer') and self.xdmf_writer:
                self.xdmf_writer.close()
                
            if self.rank == 0:
                self.logger.info("模拟资源已清理")
                
        except Exception as e:
            self.logger.warning(f"清理资源时出错: {e}")
    
    def get_results_summary(self):
        """获取模拟结果摘要"""
        summary = {
            'time': self.time,
            'time_steps': self.time_step,
            'converged': self.converged,
            'fields': list(self.fields.keys()),
            'config_file': str(self.config_file),
            'output_dir': str(self.output_dir)
        }
        
        if hasattr(self, 'mesh'):
            summary['mesh_cells'] = self.mesh.topology.index_map(3).size_global
        
        return summary


# 主程序入口
if __name__ == "__main__":
    # 解析命令行参数
    import argparse
    
    parser = argparse.ArgumentParser(description="多物理场注浆模拟")
    parser.add_argument("--config", type=str, default="Dynamic/config/grouting_config.yaml",
                       help="配置文件路径")
    parser.add_argument("--mesh", type=str, default="Dynamic/meshes/foundation_drilling_model.msh",
                       help="网格文件路径")
    parser.add_argument("--output", type=str, default="Dynamic/results",
                       help="输出目录路径")
    parser.add_argument("--time", type=float, default=None,
                       help="总模拟时间（秒）")
    
    args = parser.parse_args()
    
    # 创建并运行模拟器
    simulator = MultiphysicsGroutingSimulation(
        config_file=args.config,
        mesh_file=args.mesh,
        output_dir=args.output
    )
    
    try:
        simulator.run(total_time=args.time)
        
        # 输出摘要
        summary = simulator.get_results_summary()
        if simulator.rank == 0:
            print("\n" + "="*60)
            print("模拟摘要:")
            print(f"  状态: {'成功' if summary['converged'] else '失败'}")
            print(f"  最终时间: {summary['time']:.1f}秒")
            print(f"  模拟场: {', '.join(summary['fields'])}")
            print(f"  输出目录: {summary['output_dir']}")
            print("="*60)
            
    except Exception as e:
        if simulator.rank == 0:
            print(f"模拟错误: {e}")
        exit(1)