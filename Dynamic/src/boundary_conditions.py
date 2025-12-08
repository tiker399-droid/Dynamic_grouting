"""
边界条件管理
处理位移、压力、浓度等边界条件
"""

import numpy as np
from dolfinx import fem
from dolfinx.mesh import locate_entities
from mpi4py import MPI
import ufl
from petsc4py import PETSc
import logging

logger = logging.getLogger(__name__)


class BoundaryConditions:
    """
    边界条件管理器
    
    处理：
    1. 位移边界条件
    2. 压力边界条件（注浆压力、排水边界）
    3. 浓度边界条件（注浆孔）
    4. 孔隙度边界条件
    """
    
    def __init__(self, mesh, facet_tags, materials, config):
        """
        初始化边界条件管理器
        
        Args:
            mesh: 网格
            facet_tags: 面标记
            materials: 材料属性管理器
            config: 配置字典
        """
        self.mesh = mesh
        self.facet_tags = facet_tags
        self.materials = materials
        self.config = config
        self.comm = mesh.comm
        
        # 存储边界条件
        self.bcs_displacement = []
        self.bcs_pressure = []
        self.bcs_concentration = []
        self.bcs_porosity = []
        
        # 存储随时间变化的边界条件函数
        self.time_dependent_bcs = {}
        
        logger.info("BoundaryConditions initialized")
    
    def create_all_boundary_conditions(self, W):
        """
        创建所有边界条件
        
        Args:
            W: 混合函数空间
        
        Returns:
            边界条件列表
        """
        all_bcs = []
        
        # 创建位移边界条件
        self._create_displacement_bcs(W)
        all_bcs.extend(self.bcs_displacement)
        
        # 创建压力边界条件
        self._create_pressure_bcs(W)
        all_bcs.extend(self.bcs_pressure)
        
        # 创建浓度边界条件
        self._create_concentration_bcs(W)
        all_bcs.extend(self.bcs_concentration)
        
        # 创建孔隙度边界条件（通常为自然边界条件）
        
        logger.info(f"Created {len(all_bcs)} boundary conditions")
        return all_bcs
    
    def _create_displacement_bcs(self, W):
        """创建位移边界条件"""
        mesh = self.mesh
        fdim = mesh.topology.dim - 1
        
        # 获取位移子空间
        V_u = W.sub(0)
        
        # 1. 底部固定边界 (标记108)
        if 108 in self.facet_tags.values:
            facets_bottom = self.facet_tags.find(108)
            dofs_bottom = fem.locate_dofs_topological(V_u, fdim, facets_bottom)
            
            # 固定所有方向
            bc_bottom = fem.dirichletbc(
                PETSc.ScalarType((0.0, 0.0, 0.0)), 
                dofs_bottom, 
                V_u
            )
            self.bcs_displacement.append(bc_bottom)
            logger.info("  - Bottom fixed (all directions)")
        
        # 2. 侧面法向约束
        # 标记103: x=0面, 约束x方向
        if 103 in self.facet_tags.values:
            facets_x0 = self.facet_tags.find(103)
            dofs_x0 = fem.locate_dofs_topological(V_u.sub(0), fdim, facets_x0)
            bc_x0 = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_x0, V_u.sub(0))
            self.bcs_displacement.append(bc_x0)
            logger.info("  - x=0 face (x-direction fixed)")
        
        # 标记104: x=max面, 约束x方向
        if 104 in self.facet_tags.values:
            facets_xmax = self.facet_tags.find(104)
            dofs_xmax = fem.locate_dofs_topological(V_u.sub(0), fdim, facets_xmax)
            bc_xmax = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_xmax, V_u.sub(0))
            self.bcs_displacement.append(bc_xmax)
            logger.info("  - x=max face (x-direction fixed)")
        
        # 标记105: y=0面, 约束y方向
        if 105 in self.facet_tags.values:
            facets_y0 = self.facet_tags.find(105)
            dofs_y0 = fem.locate_dofs_topological(V_u.sub(1), fdim, facets_y0)
            bc_y0 = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_y0, V_u.sub(1))
            self.bcs_displacement.append(bc_y0)
            logger.info("  - y=0 face (y-direction fixed)")
        
        # 标记106: y=max面, 约束y方向
        if 106 in self.facet_tags.values:
            facets_ymax = self.facet_tags.find(106)
            dofs_ymax = fem.locate_dofs_topological(V_u.sub(1), fdim, facets_ymax)
            bc_ymax = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_ymax, V_u.sub(1))
            self.bcs_displacement.append(bc_ymax)
            logger.info("  - y=max face (y-direction fixed)")
    
    def _create_pressure_bcs(self, W):
        """创建压力边界条件"""
        mesh = self.mesh
        fdim = mesh.topology.dim - 1
        
        # 获取压力子空间
        V_p = W.sub(1)
        
        # 1. 顶部排水边界 (标记107) - 零压力
        if 107 in self.facet_tags.values:
            facets_top = self.facet_tags.find(107)
            dofs_top = fem.locate_dofs_topological(V_p, fdim, facets_top)
            
            # 地表压力为大气压（相对压力为0）
            p_top = fem.Constant(mesh, PETSc.ScalarType(0.0))
            bc_top = fem.dirichletbc(p_top, dofs_top, V_p)
            self.bcs_pressure.append(bc_top)
            logger.info("  - Top drainage (p=0)")
        
        # 2. 注浆孔压力边界 (标记101) - 随时间变化
        if 101 in self.facet_tags.values:
            facets_grout = self.facet_tags.find(101)
            dofs_grout = fem.locate_dofs_topological(V_p, fdim, facets_grout)
            
            # 创建随时间变化的压力函数
            p_grout_func = self._create_grout_pressure_function(mesh)
            
            # 存储用于时间更新
            self.time_dependent_bcs['grout_pressure'] = {
                'dofs': dofs_grout,
                'function': p_grout_func,
                'subspace': V_p
            }
            
            # 创建边界条件
            bc_grout = fem.dirichletbc(p_grout_func, dofs_grout, V_p)
            self.bcs_pressure.append(bc_grout)
            logger.info("  - Grout hole (time-dependent pressure)")
    
    def _create_grout_pressure_function(self, mesh):
        """
        创建随时间变化的注浆压力函数
        
        压力施加过程（文献中）：
        1. 线性上升阶段 (ramp_time)
        2. 保持阶段 (hold_time)
        3. 移除阶段
        """
        from .time_functions import GroutPressureFunction
        
        # 从配置获取参数
        grout_config = self.config['boundary_conditions']['grout_pressure']
        p_max = grout_config['p_max']  # 最大注浆压力
        ramp_time = grout_config['ramp_time']  # 上升时间
        hold_time = grout_config['hold_time']  # 保持时间
        
        # 创建压力函数
        p_func = GroutPressureFunction(mesh, p_max, ramp_time, hold_time)
        
        return p_func
    
    def _create_concentration_bcs(self, W):
        """创建浓度边界条件"""
        mesh = self.mesh
        fdim = mesh.topology.dim - 1
        
        # 获取浓度子空间
        V_c = W.sub(3)
        
        # 注浆孔浓度边界 (标记101) - c=1 (纯浆液)
        if 101 in self.facet_tags.values:
            facets_grout = self.facet_tags.find(101)
            dofs_grout = fem.locate_dofs_topological(V_c, fdim, facets_grout)
            
            # 纯浆液浓度
            c_grout = fem.Constant(mesh, PETSc.ScalarType(1.0))
            bc_grout = fem.dirichletbc(c_grout, dofs_grout, V_c)
            self.bcs_concentration.append(bc_grout)
            logger.info("  - Grout hole (c=1)")
    
    def update_time_dependent_bcs(self, t: float):
        """
        更新随时间变化的边界条件
        
        Args:
            t: 当前时间
        """
        for name, bc_info in self.time_dependent_bcs.items():
            if name == 'grout_pressure':
                # 更新注浆压力
                bc_func = bc_info['function']
                if hasattr(bc_func, 'update'):
                    bc_func.update(t)
                
                # 如果是Function类型，需要插值更新
                if isinstance(bc_func, fem.Function):
                    # 重新计算压力值
                    new_pressure = self._compute_grout_pressure(t)
                    bc_func.x.array[:] = new_pressure
                    bc_func.x.scatter_forward()
    
    def _compute_grout_pressure(self, t: float) -> np.ndarray:
        """
        计算注浆孔压力值
        
        Args:
            t: 当前时间
        
        Returns:
            压力值数组
        """
        grout_config = self.config['boundary_conditions']['grout_pressure']
        p_max = grout_config['p_max']
        ramp_time = grout_config['ramp_time']
        hold_time = grout_config['hold_time']
        
        # 计算压力值
        if t <= ramp_time:
            # 线性上升阶段
            p = (t / ramp_time) * p_max
        elif t <= ramp_time + hold_time:
            # 保持阶段
            p = p_max
        else:
            # 压力移除后（模拟结束后）
            p = 0.0
        
        return np.array([p], dtype=PETSc.ScalarType)