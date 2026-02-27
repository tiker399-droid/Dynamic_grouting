"""
动态边界条件管理器 - 地基固结注浆模拟
基于几何位置动态识别边界，处理随时间变化的边界条件
"""

import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
import logging
from typing import List, Dict, Any, Tuple, Optional, Callable
from enum import Enum

class BoundaryType(Enum):
    """边界类型枚举"""
    DISPLACEMENT = "displacement"
    PRESSURE = "pressure"
    CONCENTRATION = "concentration"
    POROSITY = "porosity"

class PressureEvolutionMode(Enum):
    """压力演化模式"""
    LINEAR_INCREASE = "linear_increase"  # 线性增加然后稳定
    PULSE = "pulse"  # 脉冲注浆

class GroutingStage(Enum):
    """注浆阶段枚举"""
    BEFORE_GROUTING = "before_grouting"  # 注浆前
    GROUTING_RISING = "grouting_rising"  # 注浆压力上升期
    GROUTING_STEADY = "grouting_steady"  # 注浆压力稳定期
    AFTER_GROUTING = "after_grouting"  # 注浆后

class DynamicBoundaryConditionsManager:
    """动态边界条件管理器 - 基于几何位置识别边界"""
    
    def __init__(self, mesh_obj, facet_tags, materials, config, function_space, time=0.0):
        """
        初始化动态边界条件管理器
        
        Args:
            mesh_obj: 计算网格
            facet_tags: 面标记
            materials: 材料管理器
            config: 配置字典
            function_space: 混合函数空间
            time: 初始时间
        """
        # 基础设置
        self.mesh = mesh_obj
        self.facet_tags = facet_tags
        self.materials = materials
        self.config = config
        self.W = function_space  # 混合函数空间
        self.time = time
        self.comm = mesh_obj.comm
        self.rank = self.comm.Get_rank()
        self.dim = mesh_obj.geometry.dim
        
        # 设置日志
        self.logger = logging.getLogger(f"DynBCManager_rank{self.rank}")
        
        # 存储边界条件的列表
        self.bcs = []
        
        # 注浆阶段和时间
        self.grouting_stage = GroutingStage.BEFORE_GROUTING
        
        # 从配置获取参数
        self._extract_parameters()
        
        # 存储子空间信息
        self._initialize_subspaces()
        
        # 存储边界几何信息
        self.boundary_geometries = {}
        self._identify_boundary_geometries()
        
        # 存储当前边界值
        self.current_values = {
            'grouting_pressure': 0.0,
            'grouting_concentration': 0.0,
            'is_grouting_active': False
        }
        
        # 存储压力演化函数
        self.pressure_evolution_func = self._create_pressure_evolution_function()
        
        # 创建初始边界条件
        self._create_initial_boundary_conditions()
        
        if self.rank == 0:
            self.logger.info("动态边界条件管理器初始化完成")
            self.logger.info(f"几何边界识别结果: {len(self.boundary_geometries)} 个边界类型")
            self.logger.info(f"注浆参数: 最大压力={self.pressure_max/1000:.1f}kPa, "
                           f"持续时间={self.grouting_duration}s, "
                           f"上升时间={self.rise_time}s")
    
    def _extract_parameters(self):
        """从配置中提取参数"""
        grout_config = self.config.get('materials', {}).get('grout', {})
        geometry_config = self.config.get('geometry', {})
        
        # 注浆压力参数
        self.pressure_max = grout_config.get('pressure', 220e3)  # 最大注浆压力 (Pa)
        self.grouting_duration = grout_config.get('duration', 600.0)  # 注浆总持续时间 (s)
        self.rise_time = grout_config.get('rise_time', 60.0)  # 压力上升时间 (s)
        
        # 几何参数
        self.foundation_height = geometry_config.get('height', 13.0)  # 地基高度 (m)
        self.drill_depth = geometry_config.get('depth', 10.0)  # 钻孔深度 (m)
        self.drill_radius = geometry_config.get('drill_radius', 0.3)  # 钻孔半径 (m)
        
        # 材料参数
        water_config = self.config.get('materials', {}).get('water', {})
        self.water_density = water_config.get('rho_w', 1000.0)
        self.gravity = 9.81
        
        # 压力演化模式
        pressure_mode = grout_config.get('pressure_mode', 'linear_increase')
        try:
            self.pressure_mode = PressureEvolutionMode(pressure_mode)
        except ValueError:
            self.pressure_mode = PressureEvolutionMode.LINEAR_INCREASE
    
    def _initialize_subspaces(self):
        """初始化各物理场的子空间"""
        # 从混合空间提取各子空间
        self.subspaces = {
            'displacement': self.W.sub(0),
            'pressure': self.W.sub(1),
            'porosity': self.W.sub(2),
            'concentration': self.W.sub(3)
        }
    
    def _identify_boundary_geometries(self):
        """基于几何位置识别边界"""
        self.boundary_geometries.clear()
        
        # 获取几何信息
        facet_geom = self.mesh.geometry.x
        fdim = self.mesh.topology.dim - 1
        facet_to_vertex = self.mesh.topology.connectivity(fdim, 0)
        
        # 存储所有已识别的面
        identified_facets = set()
        
        # 1. 识别标记的边界 (101-107)
        for marker in range(101, 108):
            facets = self.facet_tags.find(marker)
            if facets.shape[0] > 0:
                self.boundary_geometries[f'marker_{marker}'] = {
                    'facets': facets,
                    'type': 'marked',
                    'marker': marker
                }
                identified_facets.update(facets)
        
        # 2. 识别顶部边界 (自由排水边界) - 通过高度判断
        top_facets = []
        for facet in np.where(self.facet_tags.values >= 0)[0]:
            if facet in identified_facets:
                continue
            vertices = facet_to_vertex.links(facet)
            avg_z = np.mean(facet_geom[vertices, 2])
            if abs(avg_z - self.foundation_height) < 0.1:  # 允许微小误差
                top_facets.append(facet)
        
        if top_facets:
            top_facets = np.array(top_facets, dtype=np.int32)
            self.boundary_geometries['top'] = {
                'facets': top_facets,
                'type': 'top',
                'height': self.foundation_height
            }
            identified_facets.update(top_facets)
        
        # 3. 识别钻孔壁面上的注浆孔位置
        if 'marker_101' in self.boundary_geometries:
            drill_facets = self.boundary_geometries['marker_101']['facets']
            self._identify_grout_inlets(drill_facets, facet_geom, facet_to_vertex, identified_facets)
        
        # 记录识别结果
        if self.rank == 0:
            for name, info in self.boundary_geometries.items():
                self.logger.debug(f"边界 '{name}': {info['facets'].shape[0]} 个面")
    
    def _identify_grout_inlets(self, drill_facets, facet_geom, facet_to_vertex, identified_facets):
        """识别钻孔壁面上的注浆孔位置"""
        # 注浆孔深度位置 (从顶部向下)
        grout_depths = [
            self.foundation_height - self.drill_depth + 1.6,
            self.foundation_height - self.drill_depth + 1.2,
            self.foundation_height - self.drill_depth + 0.8,
            self.foundation_height - self.drill_depth + 0.4,
            self.foundation_height - self.drill_depth
        ]
        
        for i, target_z in enumerate(grout_depths):
            inlet_facets = []
            
            for facet in drill_facets:
                if facet in identified_facets:
                    continue
                vertices = facet_to_vertex.links(facet)
                center = np.mean(facet_geom[vertices], axis=0)
                z = center[2]
                if abs(z - target_z) < 0.02:  # 微小误差
                    inlet_facets.append(facet)
            
            if inlet_facets:
                inlet_facets = np.array(inlet_facets, dtype=np.int32)
                self.boundary_geometries[f'grout_inlet_{i+1}'] = {
                    'facets': inlet_facets,
                    'type': 'grout_inlet',
                    'height': target_z,
                    'depth_from_top': self.foundation_height - target_z
                }
                identified_facets.update(inlet_facets)
                
                if self.rank == 0:
                    self.logger.debug(f"识别注浆孔 {i+1}: 高度={target_z:.2f}m, {len(inlet_facets)} 个面")
    
    def _create_pressure_evolution_function(self) -> Callable[[float], float]:
        """
        创建压力演化函数
        
        Returns:
            返回一个函数，输入时间，输出注浆压力
        """
        if self.pressure_mode == PressureEvolutionMode.LINEAR_INCREASE:
            def pressure_func(t):
                if t < self.rise_time:
                    # 线性上升期
                    return self.pressure_max * (t / self.rise_time)
                elif t < self.grouting_duration:
                    # 稳定期
                    return self.pressure_max
                else:
                    # 注浆结束后
                    return 0.0
        
        elif self.pressure_mode == PressureEvolutionMode.PULSE:
            # 脉冲注浆：周期性的压力变化
            pulse_period = self.config.get('materials', {}).get('grout', {}).get('pulse_period', 60.0)
            pulse_duty_cycle = self.config.get('materials', {}).get('grout', {}).get('pulse_duty_cycle', 0.5)
            
            def pressure_func(t):
                if t < self.grouting_duration:
                    # 在脉冲周期内
                    phase = (t % pulse_period) / pulse_period
                    if phase < pulse_duty_cycle:
                        # 脉冲开启
                        return self.pressure_max
                    else:
                        # 脉冲关闭
                        return 0.0
                else:
                    return 0.0
        
        else:
            # 默认线性增加
            def pressure_func(t):
                if t < self.rise_time:
                    return self.pressure_max * (t / self.rise_time)
                elif t < self.grouting_duration:
                    return self.pressure_max
                else:
                    return 0.0
        
        return pressure_func
    
    def _determine_grouting_stage(self, time: float) -> GroutingStage:
        """确定当前注浆阶段"""
        if time < 0:
            return GroutingStage.BEFORE_GROUTING
        elif time < self.rise_time:
            return GroutingStage.GROUTING_RISING
        elif time < self.grouting_duration:
            return GroutingStage.GROUTING_STEADY
        else:
            return GroutingStage.AFTER_GROUTING
    
    def _create_initial_boundary_conditions(self):
        """创建初始边界条件"""
        self.bcs.clear()
        
        # 1. 位移边界条件（不随时间变化）
        self._create_displacement_bcs()
        
        # 2. 压力边界条件
        self._create_pressure_bcs()
        
        # 3. 初始时不设置浓度边界条件（注浆前）
        
        if self.rank == 0:
            self.logger.info(f"创建了 {len(self.bcs)} 个初始边界条件")
    
    def _create_displacement_bcs(self):
        """创建位移边界条件"""
        try:
            fdim = self.mesh.topology.dim - 1
            V_u = self.subspaces['displacement']
            
            # 底面固定边界 (marker 107)
            if 'marker_107' in self.boundary_geometries:
                facets_bottom = self.boundary_geometries['marker_107']['facets']
                zero_vector = np.zeros(self.dim, dtype=np.float64)
                bc_u_bottom = fem.dirichletbc(
                    zero_vector,
                    fem.locate_dofs_topological(V_u, fdim, facets_bottom),
                    V_u
                )
                self.bcs.append(bc_u_bottom)
                
                if self.rank == 0:
                    self.logger.debug(f"创建底面固定位移边界条件: {facets_bottom.shape[0]} 个面")
            
            # 侧面边界约束
            side_markers = ['marker_103', 'marker_104', 'marker_105', 'marker_106']
            side_constraints = [(0, 0), (1, 0), (2, 1), (3, 1)]  # (索引, 分量)
            
            for (idx, component) in side_constraints:
                marker_name = side_markers[idx]
                if marker_name in self.boundary_geometries:
                    facets = self.boundary_geometries[marker_name]['facets']
                    if facets.shape[0] > 0:
                        dofs = fem.locate_dofs_topological(
                            V_u.sub(component), fdim, facets
                        )
                        bc_u_side = fem.dirichletbc(
                            np.float64(0.0), dofs, V_u.sub(component)
                        )
                        self.bcs.append(bc_u_side)
            
        except Exception as e:
            self.logger.warning(f"创建位移边界条件失败: {e}")
    
    def _create_pressure_bcs(self):
        """创建压力边界条件"""
        try:
            fdim = self.mesh.topology.dim - 1
            V_p = self.subspaces['pressure']
            
            # 1. 顶部自由排水边界 (零压力)
            if 'top' in self.boundary_geometries:
                facets_top = self.boundary_geometries['top']['facets']
                bc_p_top = fem.dirichletbc(
                    np.float64(0.0),
                    fem.locate_dofs_topological(V_p, fdim, facets_top),
                    V_p
                )
                self.bcs.append(bc_p_top)
                
                if self.rank == 0:
                    self.logger.debug(f"创建顶部零压力边界条件: {facets_top.shape[0]} 个面")
            
            # 2. 注浆孔压力边界条件
            self._update_grouting_pressure_bcs(V_p, fdim)
            
            # 3. 侧面水压力边界条件
            self._create_water_pressure_bcs(V_p, fdim)
            
        except Exception as e:
            self.logger.warning(f"创建压力边界条件失败: {e}")
    
    def _create_water_pressure_bcs(self, V_p, fdim):
        """创建水压力边界条件（静态水压力分布）"""
        try:
            # 侧面边界 (marker 103-106)
            side_markers = ['marker_103', 'marker_104', 'marker_105', 'marker_106']
            
            # 创建水压力表达式：p = ρg * (高度 - z)
            x = ufl.SpatialCoordinate(self.mesh)
            water_pressure_expr = self.water_density * self.gravity * (self.foundation_height - x[2])
            
            # 将表达式转换为Function
            water_pressure_func = fem.Function(V_p)
            water_pressure_func.interpolate(
                fem.Expression(water_pressure_expr, V_p.element.interpolation_points())
            )
            
            # 应用到侧面边界
            for marker_name in side_markers:
                if marker_name in self.boundary_geometries:
                    facets = self.boundary_geometries[marker_name]['facets']
                    if facets.shape[0] > 0:
                        dofs = fem.locate_dofs_topological(V_p, fdim, facets)
                        bc_p_side = fem.dirichletbc(water_pressure_func, dofs)
                        self.bcs.append(bc_p_side)
            
            if self.rank == 0:
                self.logger.debug("创建侧面水压力边界条件")
                
        except Exception as e:
            self.logger.warning(f"创建水压力边界条件失败: {e}")
    
    def _update_grouting_pressure_bcs(self, V_p, fdim):
        """更新注浆孔压力边界条件"""
        try:
            # 计算当前注浆压力
            current_pressure = self.pressure_evolution_func(self.time)
            self.current_values['grouting_pressure'] = current_pressure
            self.current_values['is_grouting_active'] = current_pressure > 0
            
            # 查找所有注浆孔边界
            grout_inlets = [name for name in self.boundary_geometries.keys() 
                           if name.startswith('grout_inlet_')]
            
            # 为每个注浆孔创建压力边界条件
            for inlet_name in grout_inlets:
                facets = self.boundary_geometries[inlet_name]['facets']
                
                # 创建压力函数
                pressure_func = fem.Function(V_p)
                pressure_func.x.array[:] = current_pressure
                
                # 创建边界条件
                dofs = fem.locate_dofs_topological(V_p, fdim, facets)
                bc_p_inlet = fem.dirichletbc(pressure_func, dofs)
                self.bcs.append(bc_p_inlet)
                
                if self.rank == 0 and current_pressure > 0:
                    self.logger.debug(f"创建注浆孔压力边界条件: {inlet_name}, "
                                    f"压力={current_pressure/1000:.1f}kPa")
            
        except Exception as e:
            self.logger.warning(f"更新注浆压力边界条件失败: {e}")
    
    def _update_concentration_bcs(self):
        """更新浓度边界条件（只在注浆期间设置）"""
        try:
            fdim = self.mesh.topology.dim - 1
            V_c = self.subspaces['concentration']
            
            # 只在注浆期间设置浓度边界条件
            if self.current_values['is_grouting_active']:
                current_concentration = 1.0  # 注浆期间浓度为1
                self.current_values['grouting_concentration'] = current_concentration
                
                # 查找所有注浆孔边界
                grout_inlets = [name for name in self.boundary_geometries.keys() 
                               if name.startswith('grout_inlet_')]
                
                # 为每个注浆孔创建浓度边界条件
                for inlet_name in grout_inlets:
                    facets = self.boundary_geometries[inlet_name]['facets']
                    
                    # 创建浓度函数
                    conc_func = fem.Function(V_c)
                    conc_func.x.array[:] = current_concentration
                    
                    # 创建边界条件
                    dofs = fem.locate_dofs_topological(V_c, fdim, facets)
                    bc_c_inlet = fem.dirichletbc(conc_func, dofs)
                    self.bcs.append(bc_c_inlet)
                
                if self.rank == 0:
                    self.logger.debug(f"创建浓度边界条件: 浓度=1.0 (注浆中)")
            else:
                # 不注浆时，不设置浓度边界条件（自然边界条件）
                self.current_values['grouting_concentration'] = 0.0
                if self.rank == 0 and self.time > 0:
                    self.logger.debug("移除浓度边界条件 (注浆结束)")
                    
        except Exception as e:
            self.logger.warning(f"更新浓度边界条件失败: {e}")
    
    def update(self, time: float, material_props: Optional[Dict] = None):
        """
        更新随时间变化的边界条件
        
        Args:
            time: 当前时间
            material_props: 材料属性更新字典（可选）
        """
        old_time = self.time
        self.time = time
        
        # 更新注浆阶段
        new_stage = self._determine_grouting_stage(time)
        if new_stage != self.grouting_stage and self.rank == 0:
            self.logger.info(f"注浆阶段变化: {self.grouting_stage.value} -> {new_stage.value}")
        self.grouting_stage = new_stage
        
        # 清空边界条件列表
        self.bcs.clear()
        
        # 重新创建所有边界条件
        self._create_displacement_bcs()  # 位移边界条件不变
        self._create_pressure_bcs()      # 压力边界条件更新
        self._update_concentration_bcs() # 浓度边界条件（只在注浆期间设置）
        
        # 记录更新信息
        if self.rank == 0 and abs(time - old_time) > 1e-6:
            pressure_kpa = self.current_values['grouting_pressure'] / 1000
            concentration = self.current_values['grouting_concentration']
            stage = self.grouting_stage.value
            
            self.logger.debug(f"边界条件更新: t={time:.1f}s, 阶段={stage}, "
                            f"压力={pressure_kpa:.1f}kPa, "
                            f"浓度={concentration}, "
                            f"注浆中={self.current_values['is_grouting_active']}")
    
    def get_boundary_conditions(self) -> List:
        """
        获取当前边界条件列表
        
        Returns:
            边界条件列表
        """
        return self.bcs
    
    def get_boundary_info(self) -> Dict:
        """
        获取边界信息摘要
        
        Returns:
            边界信息字典
        """
        info = {
            'num_bcs': len(self.bcs),
            'current_time': self.time,
            'grouting_stage': self.grouting_stage.value,
            'grouting_pressure': self.current_values['grouting_pressure'],
            'grouting_concentration': self.current_values['grouting_concentration'],
            'is_grouting_active': self.current_values['is_grouting_active'],
            'boundary_types': list(self.boundary_geometries.keys())
        }
        
        # 统计各类型边界条件数量
        bc_types = {}
        for bc in self.bcs:
            # 简化统计：根据函数空间判断类型
            if hasattr(bc, 'function_space'):
                space = bc.function_space
                for name, subspace in self.subspaces.items():
                    if space is subspace or (hasattr(space, '_parent') and space._parent is subspace):
                        bc_types[name] = bc_types.get(name, 0) + 1
                        break
        
        info['bc_types'] = bc_types
        
        return info
    
    def get_current_pressure_value(self) -> float:
        """获取当前注浆压力值"""
        return self.current_values['grouting_pressure']
    
    def get_current_concentration_value(self) -> float:
        """获取当前注浆浓度值"""
        return self.current_values['grouting_concentration']
    
    def is_grouting_active(self) -> bool:
        """检查是否正在注浆"""
        return self.current_values['is_grouting_active']
    
    def save_boundary_state(self, filename: str):
        """
        保存边界状态到文件（用于调试）
        
        Args:
            filename: 文件名
        """
        try:
            import pickle
            state = {
                'time': self.time,
                'grouting_stage': self.grouting_stage.value,
                'current_values': self.current_values,
                'boundary_types': list(self.boundary_geometries.keys()),
                'num_bcs': len(self.bcs)
            }
            
            if self.rank == 0:
                with open(filename, 'wb') as f:
                    pickle.dump(state, f)
                self.logger.debug(f"边界状态保存到 {filename}")
                
        except Exception as e:
            self.logger.warning(f"保存边界状态失败: {e}")