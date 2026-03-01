"""
动态边界条件管理器 - 基于几何位置识别边界，处理随时间变化的边界条件
支持位移约束、静水压力、注浆压力演化（线性/脉冲）、浓度边界自动切换
坐标系约定：竖直向上为正
"""

import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
import logging
from typing import List, Dict, Any, Tuple, Optional, Callable
from enum import Enum
from pathlib import Path


class BoundaryType(Enum):
    """边界类型枚举"""
    DISPLACEMENT = "displacement"
    PRESSURE = "pressure"
    CONCENTRATION = "concentration"
    POROSITY = "porosity"


class PressureEvolutionMode(Enum):
    """压力演化模式"""
    LINEAR_INCREASE = "linear_increase"   # 线性上升→稳定→归零
    PULSE = "pulse"                         # 脉冲注浆（周期性）
    CONSTANT = "constant"                    # 恒定压力（用于测试）


class GroutingStage(Enum):
    """注浆阶段枚举（与时间步进控制器同步）"""
    BEFORE_GROUTING = "before_grouting"
    PRESSURE_RISING = "pressure_rising"
    PRESSURE_STEADY = "pressure_steady"
    AFTER_GROUTING = "after_grouting"


class DynamicBoundaryConditionsManager:
    """
    动态边界条件管理器
    根据时间更新注浆压力、浓度边界，并管理位移约束
    """

    def __init__(
        self,
        mesh_obj: mesh.Mesh,
        facet_tags: mesh.MeshTags,
        materials,
        config: Dict[str, Any],
        function_space: fem.FunctionSpace,
        time: float = 0.0,
        time_controller=None
    ):
        """
        初始化边界条件管理器

        Args:
            mesh_obj: 计算网格
            facet_tags: 面标记（来自 gmsh 读取）
            materials: 材料属性管理器
            config: 完整配置字典
            function_space: 混合函数空间 [u, p, φ, c, q]
            time: 初始时间
        """
        self.mesh = mesh_obj
        self.facet_tags = facet_tags
        self.materials = materials
        self.config = config
        self.W = function_space
        self.time = time
        self.comm = mesh_obj.comm
        self.rank = self.comm.Get_rank()
        self.time_controller = time_controller

        # 日志
        self.logger = logging.getLogger(f"BCManager_rank{self.rank}")

        # 边界条件存储
        self.bcs = []          # 当前有效的狄利克雷边界条件列表

        # 几何维度
        self.gdim = mesh_obj.geometry.dim

        # 提取几何与注浆参数
        self._extract_parameters()

        # 初始化子空间引用
        self._init_subspaces()

        # 识别几何边界（基于标记和几何位置）
        self.boundary_geometries = {}
        self._identify_boundaries()

        # 创建压力演化函数
        self.pressure_func = self._create_pressure_evolution_function()

        # 当前边界值状态
        self.current_values = {
            'grouting_pressure': 0.0,
            'grouting_concentration': 0.0,
            'is_grouting_active': False,
            'grouting_stage': GroutingStage.BEFORE_GROUTING.value
        }

        # 初始化边界条件（创建所有不随时间变化的位移边界，以及初始的压力/浓度边界）
        self._create_initial_bcs()

        if self.rank == 0:
            self.logger.info("边界条件管理器初始化完成")
            self.logger.info(f"  识别到的边界类型: {list(self.boundary_geometries.keys())}")
            self.logger.info(f"  注浆压力模式: {self.pressure_mode.value}")
            self.logger.info(f"  最大压力: {self.pressure_max/1000:.1f} kPa")
            self.logger.info(f"  注浆持续时间: {self.grouting_duration} s")

    # ------------------------------------------------------------------
    # 参数提取
    # ------------------------------------------------------------------
    def _extract_parameters(self):
        """从配置中提取几何、注浆相关参数"""
        grout_config = self.config.get('materials', {}).get('grout', {})
        geom_config = self.config.get('geometry', {})
        water_config = self.config.get('materials', {}).get('water', {})

        # 注浆压力参数（确保类型转换）
        self.pressure_max = float(grout_config.get('pressure', 220e3))           # Pa
        self.grouting_duration = float(grout_config.get('duration', 600.0))      # s
        self.rise_time = float(grout_config.get('rise_time', 60.0))              # s
        pressure_mode_str = grout_config.get('pressure_mode', 'linear_increase')
        try:
            self.pressure_mode = PressureEvolutionMode(pressure_mode_str)
        except ValueError:
            self.pressure_mode = PressureEvolutionMode.LINEAR_INCREASE

        # 脉冲参数（如果使用脉冲模式）
        self.pulse_period = grout_config.get('pulse_period', 60.0)
        self.pulse_duty_cycle = grout_config.get('pulse_duty_cycle', 0.5)

        # 几何参数
        self.foundation_height = geom_config.get('height', 13.0)          # m
        self.drill_depth = geom_config.get('depth', 10.0)                 # m
        self.drill_radius = geom_config.get('drill_radius', 0.05)          # m

        # 水参数（用于静水压力）
        self.water_density = water_config.get('rho_w', 1000.0)             # kg/m³
        self.gravity = 9.81                                                # m/s²

    # ------------------------------------------------------------------
    # 子空间初始化
    # ------------------------------------------------------------------
    def _init_subspaces(self):
        """获取各物理场的子空间引用"""
        self.subspaces = {
            'displacement': self.W.sub(0),
            'pressure': self.W.sub(1),
            'porosity': self.W.sub(2),
            'concentration': self.W.sub(3),
            'darcy_velocity': self.W.sub(4)
        }

    # ------------------------------------------------------------------
    # 几何边界识别
    # ------------------------------------------------------------------
    def _identify_boundaries(self):
        """
        基于面标记和几何位置识别物理边界
        填充 self.boundary_geometries 字典，每个条目包含 'facets' (np.ndarray) 和 'type'
        """
        # 获取拓扑信息
        tdim = self.mesh.topology.dim
        fdim = tdim - 1
        facet_to_cell = self.mesh.topology.connectivity(fdim, tdim)
        facet_geom = self.mesh.geometry.x
        facet_to_vertex = self.mesh.topology.connectivity(fdim, 0)

        identified = set()   # 已标记的面，避免重复

        # 1. 读取 gmsh 标记的边界（假设标记 101~107）
        for marker in range(101, 108):
            facets = self.facet_tags.find(marker)
            if facets.shape[0] > 0:
                self.boundary_geometries[f'marker_{marker}'] = {
                    'facets': facets,
                    'type': 'marked',
                    'marker': marker
                }
                identified.update(facets)

        # 2. 识别顶部自由排水边界（通过高度）
        top_facets = []
        for facet in range(self.facet_tags.indices.shape[0]):   # 遍历所有面
            if facet in identified:
                continue
            vertices = facet_to_vertex.links(facet)
            avg_z = np.mean(facet_geom[vertices, 2])
            if abs(avg_z - self.foundation_height) < 0.05:      # 高度容差 5cm
                top_facets.append(facet)

        if top_facets:
            top_facets = np.array(top_facets, dtype=np.int32)
            self.boundary_geometries['top'] = {
                'facets': top_facets,
                'type': 'top',
                'height': self.foundation_height
            }
            identified.update(top_facets)

        # 3. 识别注浆孔位置（在钻孔壁面上）
        if 'marker_101' in self.boundary_geometries:
            drill_facets = self.boundary_geometries['marker_101']['facets']
            self._identify_grout_inlets(drill_facets, facet_geom, facet_to_vertex, identified)

        # 记录识别结果
        if self.rank == 0:
            for name, info in self.boundary_geometries.items():
                self.logger.debug(f"  边界 '{name}': {info['facets'].shape[0]} 个面")

    def _identify_grout_inlets(self, drill_facets, facet_geom, facet_to_vertex, identified):
        """
        从钻孔壁面中识别注浆孔位置（根据深度分层）
        """
        # 注浆孔深度位置（从顶部向下）
        # 假设五个注浆孔均匀分布在钻孔底部 2m 范围内
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
                if facet in identified:
                    continue
                vertices = facet_to_vertex.links(facet)
                center = np.mean(facet_geom[vertices], axis=0)
                z = center[2]
                if abs(z - target_z) < 0.02:          # 2cm 容差
                    inlet_facets.append(facet)

            if inlet_facets:
                inlet_facets = np.array(inlet_facets, dtype=np.int32)
                self.boundary_geometries[f'grout_inlet_{i+1}'] = {
                    'facets': inlet_facets,
                    'type': 'grout_inlet',
                    'height': target_z,
                    'depth_from_top': self.foundation_height - target_z
                }
                identified.update(inlet_facets)

        # 如果未识别到任何注浆孔，使用整个钻孔壁面作为备用
        if not any(name.startswith('grout_inlet') for name in self.boundary_geometries):
            if self.rank == 0:
                self.logger.warning("未识别到分层注浆孔，将使用整个钻孔壁面作为注浆边界")
            # 过滤掉已识别的面
            remaining = [f for f in drill_facets if f not in identified]
            if remaining:
                self.boundary_geometries['grout_inlet_fallback'] = {
                    'facets': np.array(remaining, dtype=np.int32),
                    'type': 'grout_inlet',
                    'height': None
                }

    # ------------------------------------------------------------------
    # 压力演化函数
    # ------------------------------------------------------------------
    def _create_pressure_evolution_function(self) -> Callable[[float], float]:
        """
        根据配置创建压力演化函数 p(t) (Pa)
        """
        if self.pressure_mode == PressureEvolutionMode.LINEAR_INCREASE:
            def func(t):
                if t < self.rise_time:
                    return self.pressure_max * (t / self.rise_time)
                elif t < self.grouting_duration:
                    return self.pressure_max
                else:
                    return 0.0

        elif self.pressure_mode == PressureEvolutionMode.PULSE:
            def func(t):
                if t < self.grouting_duration:
                    # 周期脉冲
                    phase = (t % self.pulse_period) / self.pulse_period
                    return self.pressure_max if phase < self.pulse_duty_cycle else 0.0
                else:
                    return 0.0

        elif self.pressure_mode == PressureEvolutionMode.CONSTANT:
            def func(t):
                return self.pressure_max if t < self.grouting_duration else 0.0

        else:
            # 默认线性
            def func(t):
                return self.pressure_max if t < self.grouting_duration else 0.0

        return func

    # ------------------------------------------------------------------
    # 边界条件创建（内部方法）
    # ------------------------------------------------------------------
    def _create_initial_bcs(self):
        """创建初始边界条件（位移固定边界 + 初始压力/浓度）"""
        self.bcs.clear()

        # 1. 位移边界条件（不随时间变化）
        self._create_displacement_bcs()

        # 2. 压力边界条件（包括静水压力和注浆压力）
        self._create_pressure_bcs()

        # 3. 浓度边界条件（注浆前无浓度边界）
        #   （在 update 中会根据注浆状态添加）

        if self.rank == 0:
            self.logger.info(f"初始边界条件创建完成，共 {len(self.bcs)} 个")

    def _create_displacement_bcs(self):
        """创建位移固定边界（底面固定，侧面法向约束）"""
        fdim = self.mesh.topology.dim - 1
        V_u = self.subspaces['displacement']

        # 底面固定 (marker 107)
        if 'marker_107' in self.boundary_geometries:
            facets = self.boundary_geometries['marker_107']['facets']
            # 使用 Function 作为边界值，需要 collapse 子空间
            V_u_collapsed, u_to_parent = V_u.collapse()
            zero_vec_func = fem.Function(V_u_collapsed)
            zero_vec_func.x.array[:] = 0.0
            dofs = fem.locate_dofs_topological(V_u, fdim, facets)
            bc = fem.dirichletbc(zero_vec_func, dofs)
            self.bcs.append(bc)

        # 侧面法向约束 (markers 103,104,105,106 对应四个侧面)
        # 假设 x=0 面 marker 103，x=Lx 面 marker 104，y=0 面 marker 105，y=Ly 面 marker 106
        side_markers = [('marker_103', 0),   # x=0, 约束 x 方向
                        ('marker_104', 0),   # x=Lx, 约束 x 方向
                        ('marker_105', 1),   # y=0, 约束 y 方向
                        ('marker_106', 1)]   # y=Ly, 约束 y 方向

        for marker_name, comp in side_markers:
            if marker_name in self.boundary_geometries:
                facets = self.boundary_geometries[marker_name]['facets']
                if facets.shape[0] > 0:
                    # 子空间的边界条件需要用 collapse
                    V_comp = V_u.sub(comp)
                    V_comp_collapsed, comp_to_parent = V_comp.collapse()
                    zero_func = fem.Function(V_comp_collapsed)
                    zero_func.x.array[:] = 0.0
                    dofs = fem.locate_dofs_topological(V_comp, fdim, facets)
                    bc = fem.dirichletbc(zero_func, dofs)
                    self.bcs.append(bc)

    def _create_pressure_bcs(self):
        """
        创建压力边界条件：
        - 顶部自由排水边界 (p=0)
        - 侧面静水压力分布 p = ρ_w g (H - z)
        - 注浆孔压力（当前时间值）
        """
        fdim = self.mesh.topology.dim - 1
        V_p = self.subspaces['pressure']

        # 1. 顶部零压力
        if 'top' in self.boundary_geometries:
            facets = self.boundary_geometries['top']['facets']
            V_p_collapsed, p_to_parent = V_p.collapse()
            zero_func = fem.Function(V_p_collapsed)
            zero_func.x.array[:] = 0.0
            dofs = fem.locate_dofs_topological(V_p, fdim, facets)
            bc = fem.dirichletbc(zero_func, dofs)
            self.bcs.append(bc)

        # 2. 侧面静水压力
        # 创建静水压力表达式（作为函数）
        V_p_collapsed, p_to_parent = V_p.collapse()
        water_pressure_func = fem.Function(V_p_collapsed)
        water_pressure_func.interpolate(lambda x: self.water_density * self.gravity * (self.foundation_height - x[2]))

        side_markers = ['marker_103', 'marker_104', 'marker_105', 'marker_106']
        for marker in side_markers:
            if marker in self.boundary_geometries:
                facets = self.boundary_geometries[marker]['facets']
                if facets.shape[0] > 0:
                    dofs = fem.locate_dofs_topological(V_p, fdim, facets)
                    bc = fem.dirichletbc(water_pressure_func, dofs)
                    self.bcs.append(bc)

        # 3. 注浆孔压力（基于当前时间）
        self._update_grouting_pressure_bcs(V_p, fdim)

    def _update_grouting_pressure_bcs(self, V_p, fdim):
        """更新注浆孔压力边界条件（根据 self.time）"""
        if self.time_controller is not None:
            stage = self.time_controller.grouting_stage
            if stage == GroutingStage.AFTER_GROUTING:
                current_pressure = 0.0
            else:
                current_pressure = self.pressure_func(self.time)
        else:
            current_pressure = self.pressure_func(self.time)

        self.current_values['grouting_pressure'] = current_pressure

        self.current_values['is_grouting_active'] = (current_pressure > 1.0)  # 忽略微小值

        # 查找所有注浆孔边界
        grout_inlets = [name for name in self.boundary_geometries.keys()
                        if name.startswith('grout_inlet_')]

        # 如果没有注浆孔，使用钻孔壁面作为后备
        if not grout_inlets and 'grout_inlet_fallback' in self.boundary_geometries:
            grout_inlets = ['grout_inlet_fallback']

        # 创建压力函数（为每个注浆孔单独创建，但值相同）
        V_p_collapsed, p_to_parent = V_p.collapse()
        pressure_func = fem.Function(V_p_collapsed)
        pressure_func.x.array[:] = current_pressure

        for inlet_name in grout_inlets:
            facets = self.boundary_geometries[inlet_name]['facets']
            if facets.shape[0] > 0:
                dofs = fem.locate_dofs_topological(V_p, fdim, facets)
                bc = fem.dirichletbc(pressure_func, dofs)
                self.bcs.append(bc)

        if self.rank == 0 and current_pressure > 0:
            self.logger.debug(f"注浆孔压力更新: {current_pressure/1000:.2f} kPa")

    def _update_concentration_bcs(self):
        """更新浓度边界条件（注浆期间设 c=1，否则自然边界）"""
        fdim = self.mesh.topology.dim - 1
        V_c = self.subspaces['concentration']

        # 移除之前的浓度边界条件（如果有）
        # 由于每次 update 都会重新创建所有边界，因此这里不需要单独移除

        if self.current_values['is_grouting_active']:
            # 注浆期间，在注浆孔施加 c=1
            V_c_collapsed, c_to_parent = V_c.collapse()
            conc_func = fem.Function(V_c_collapsed)
            conc_func.x.array[:] = 1.0

            grout_inlets = [name for name in self.boundary_geometries.keys()
                            if name.startswith('grout_inlet_')]
            if not grout_inlets and 'grout_inlet_fallback' in self.boundary_geometries:
                grout_inlets = ['grout_inlet_fallback']

            for inlet_name in grout_inlets:
                facets = self.boundary_geometries[inlet_name]['facets']
                if facets.shape[0] > 0:
                    dofs = fem.locate_dofs_topological(V_c, fdim, facets)
                    bc = fem.dirichletbc(conc_func, dofs)
                    self.bcs.append(bc)

            self.current_values['grouting_concentration'] = 1.0
            if self.rank == 0:
                self.logger.debug("浓度边界已激活 (c=1)")
        else:
            self.current_values['grouting_concentration'] = 0.0
            # 不添加任何浓度边界，自然边界条件（通量为零）自动生效

    # ------------------------------------------------------------------
    # 公共方法
    # ------------------------------------------------------------------
    def update(self, time: float, material_props: Optional[Dict] = None):
        """
        更新边界条件到给定时间

        Args:
            time: 当前时间
            material_props: 可选的材料属性更新（当前未使用）
        """
        old_time = self.time
        self.time = time

        # 确定注浆阶段（可选，用于日志）
        if time < 0:
            stage = GroutingStage.BEFORE_GROUTING
        elif time < self.rise_time:
            stage = GroutingStage.PRESSURE_RISING
        elif time < self.grouting_duration:
            stage = GroutingStage.PRESSURE_STEADY
        else:
            stage = GroutingStage.AFTER_GROUTING
        self.current_values['grouting_stage'] = stage.value

        # 重新创建所有边界条件（因为压力、浓度已变化）
        self.bcs.clear()
        self._create_displacement_bcs()   # 位移不变
        self._create_pressure_bcs()       # 内部调用 _update_grouting_pressure_bcs
        self._update_concentration_bcs()   # 浓度边界条件

        # 日志（仅在时间变化显著时）
        if self.rank == 0 and abs(time - old_time) > 1e-6:
            self.logger.debug(
                f"边界条件更新: t={time:.2f}s, 阶段={stage.value}, "
                f"压力={self.current_values['grouting_pressure']/1000:.2f}kPa, "
                f"浓度激活={self.current_values['is_grouting_active']}"
            )

    def get_boundary_conditions(self) -> List:
        """返回当前有效的狄利克雷边界条件列表"""
        return self.bcs

    def get_boundary_info(self) -> Dict[str, Any]:
        """返回边界信息摘要"""
        info = {
            'num_bcs': len(self.bcs),
            'current_time': self.time,
            'grouting_stage': self.current_values['grouting_stage'],
            'grouting_pressure': self.current_values['grouting_pressure'],
            'is_grouting_active': self.current_values['is_grouting_active'],
            'boundary_types': list(self.boundary_geometries.keys())
        }
        return info

    def is_grouting_active(self) -> bool:
        """返回当前是否正在注浆"""
        return self.current_values['is_grouting_active']

    def save_boundary_state(self, filename: str):
        """
        保存边界状态到文件（用于调试）
        """
        if self.rank != 0:
            return
        try:
            import pickle
            state = {
                'time': self.time,
                'grouting_stage': self.current_values['grouting_stage'],
                'grouting_pressure': self.current_values['grouting_pressure'],
                'grouting_concentration': self.current_values['grouting_concentration'],
                'boundary_geometries': {k: v['facets'].shape[0] for k, v in self.boundary_geometries.items()}
            }
            with open(filename, 'wb') as f:
                pickle.dump(state, f)
            self.logger.debug(f"边界状态已保存到 {filename}")
        except Exception as e:
            self.logger.warning(f"保存边界状态失败: {e}")