"""
材料属性管理器 - 简化核心版本
包含主程序直接需要的功能
"""

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI
import logging

class MaterialProperties:
    """材料属性管理器 - 核心功能版本"""
    
    def __init__(self, config, mesh, comm):
        """
        初始化材料属性管理器
        
        Args:
            config: 配置文件字典
            mesh: 计算网格
            comm: MPI通信器
        """
        # 基础设置
        self.comm = comm
        self.rank = comm.Get_rank()
        self.mesh = mesh
        
        # 设置日志
        self.logger = logging.getLogger(f"Materials_rank{self.rank}")
        
        # 提取材料参数（从主程序传入的config结构）
        self._extract_parameters(config)
        
        # 初始化常量和基础设置
        self._initialize_constants()
        
        # 输出初始化信息
        if self.rank == 0:
            self.logger.info("材料属性管理器初始化完成")
            self.logger.info(f"初始孔隙度: {self.phi0:.3f}")
            self.logger.info(f"浆液初始粘度: {self.mu_g0:.4f} Pa·s")
            self.logger.info(f"过滤系数: {self.lambda_f:.3f}")
    
    def _extract_parameters(self, config):
        """
        从配置字典中提取材料参数
        
        注意：假设config结构为：
        config = {
            'materials': {
                'soil': {...},
                'grout': {...},
                'water': {...}
            }
        }
        """
        # 获取材料配置部分
        materials_config = config.get('materials', {})
        
        # 提取土壤参数
        soil_config = materials_config.get('soil', {})
        self.phi0 = soil_config.get('phi0', 0.40)  # 初始孔隙度
        self.k0 = soil_config.get('k0', 1e-12)     # 初始渗透率 (m²)
        
        # 提取浆液参数
        grout_config = materials_config.get('grout', {})
        self.rho_g = grout_config.get('rho_g', 1800.0)     # 浆液密度 (kg/m³)
        self.mu_g0 = grout_config.get('mu_g0', 0.01)       # 浆液初始粘度 (Pa·s)
        self.xi = grout_config.get('xi', 1.56)            # 粘度增长常数 (1/s)
        self.lambda_f = grout_config.get('filtration_coeff', 0.75)  # 过滤系数
        
        # 提取水参数
        water_config = materials_config.get('water', {})
        self.rho_w = water_config.get('rho_w', 1000.0)     # 水密度 (kg/m³)
        self.mu_w = water_config.get('mu_w', 0.001)        # 水粘度 (Pa·s)
        
        # 存储完整配置以供参考
        self.soil_config = soil_config
        self.grout_config = grout_config
        self.water_config = water_config
    
    def _initialize_constants(self):
        """初始化常量和基础设置"""
        # 重力加速度向量 (m/s²)，假设z方向向下
        self.g_vector = np.array([0.0, 0.0, -9.81])
        
        # 创建FEniCS常数对象
        self.g = fem.Constant(self.mesh, self.g_vector)  # 重力向量
        
        # 创建材料常数（用于弱形式）
        self.k0_constant = fem.Constant(self.mesh, self.k0)
        self.mu_g0_constant = fem.Constant(self.mesh, self.mu_g0)
        self.lambda_f_constant = fem.Constant(self.mesh, self.lambda_f)
        
        # 当前时间和粘度缓存（用于性能优化）
        self._current_time = 0.0
        self._current_mu_g = self.mu_g0
    
    def update_time_dependent_properties(self, time):
        """
        更新时间相关的材料属性
        
        Args:
            time: 当前时间 (秒)
            
        Returns:
            dict: 更新后的时间相关属性
        """
        self._current_time = time
        
        # 更新浆液粘度（指数增长）
        # μ_g(t) = μ_g0 * exp(ξ * t)
        self._current_mu_g = self.mu_g0 * np.exp(self.xi * time)
        
        # 返回更新后的属性字典（用于调试和输出）
        updated_props = {
            'time': time,
            'grout_viscosity': self._current_mu_g,
            'grout_viscosity_constant': fem.Constant(self.mesh, self._current_mu_g)
        }
        
        if self.rank == 0 and abs(time) > 1e-10:  # 非初始时间
            self.logger.debug(f"更新时间相关属性: t={time:.1f}s, μ_g={self._current_mu_g:.4e} Pa·s")
        
        return updated_props
    
    def calculate_permeability(self, phi):
        """
        Kozeny-Carman关系：计算渗透率
        
        Args:
            phi: 孔隙度场（UFL表达式或标量）
            
        Returns:
            渗透率表达式 k = k0 * (φ^3) / (1-φ)^2
        """
        # 添加小量避免除零
        denominator = (1 - phi)**2 + 1e-10
        return self.k0 * (phi**3) / denominator
    
    def calculate_viscosity(self, c, time=None):
        """
        计算混合粘度
        
        Args:
            c: 浆液浓度场（UFL表达式，0-1之间）
            time: 当前时间，如果为None则使用缓存的时间
            
        Returns:
            混合粘度表达式 μ = c*μ_g(t) + (1-c)*μ_w
        """
        # 使用指定的时间或缓存的时间
        current_time = time if time is not None else self._current_time
        
        # 计算当前浆液粘度
        mu_g = self.mu_g0 * ufl.exp(self.xi * current_time)
        
        # 混合粘度
        return c * mu_g + (1 - c) * self.mu_w
    
    def calculate_density(self, c):
        """
        计算混合物密度
        
        Args:
            c: 浆液浓度场（UFL表达式，0-1之间）
            
        Returns:
            混合物密度表达式 ρ = c*ρ_g + (1-c)*ρ_w
        """
        return c * self.rho_g + (1 - c) * self.rho_w
    
    def calculate_filtration_rate(self, c, q_w):
        """
        过滤定律：计算浆液捕获速率
        
        Args:
            c: 浆液浓度场
            q_w: 达西流速场（向量场）
            
        Returns:
            过滤速率表达式 ȯn = λ_f * c * |q_w|
        """
        # 计算速度幅值（添加小量避免零除）
        qw_magnitude = ufl.sqrt(ufl.dot(q_w, q_w) + 1e-10)
        
        return self.lambda_f * c * qw_magnitude
    
    def darcy_velocity(self, p, phi, c, time=None):
        """
        达西定律：计算渗流速度
        
        Args:
            p: 压力场
            phi: 孔隙度场
            c: 浓度场
            time: 当前时间
            
        Returns:
            达西流速表达式 q_w = - (k/μ) * (∇p - ρg)
        """
        # 计算渗透率和粘度
        k = self.calculate_permeability(phi)
        mu = self.calculate_viscosity(c, time)
        
        # 计算混合物密度
        rho = self.calculate_density(c)
        
        # 达西定律
        return - (k / mu) * (ufl.grad(p) - rho * self.g)
    
    def calculate_all_derived(self, solution_fields, time):
        """
        计算所有衍生物理场（表达式形式）
        
        Args:
            solution_fields: 字典，包含各物理场
                {'displacement': u, 'pressure': p, 'porosity': phi, 
                 'concentration': c, 'darcy_velocity': q_w}
            time: 当前时间
            
        Returns:
            dict: 衍生场表达式字典
        """
        derived_fields = {}
        
        try:
            # 从solution_fields中提取基本场
            phi = solution_fields.get('porosity')
            c = solution_fields.get('concentration')
            q_w = solution_fields.get('darcy_velocity')
            
            # 计算各衍生场（如果基本场存在）
            if phi is not None:
                derived_fields['permeability'] = self.calculate_permeability(phi)
            
            if c is not None:
                derived_fields['viscosity'] = self.calculate_viscosity(c, time)
                derived_fields['density'] = self.calculate_density(c)
            
            if c is not None and q_w is not None:
                derived_fields['filtration_rate'] = self.calculate_filtration_rate(c, q_w)
            
            # 还可以计算其他衍生场
            if 'pressure' in solution_fields and phi is not None and c is not None:
                p = solution_fields['pressure']
                derived_fields['darcy_velocity_calc'] = self.darcy_velocity(p, phi, c, time)
            
            return derived_fields
            
        except Exception as e:
            self.logger.warning(f"计算衍生场时出错: {e}")
            return {}
    
    def get_material_parameters(self):
        """
        获取所有材料参数（用于调试和输出）
        
        Returns:
            dict: 材料参数字典
        """
        return {
            'soil': {
                'phi0': self.phi0,
                'k0': self.k0
            },
            'grout': {
                'rho_g': self.rho_g,
                'mu_g0': self.mu_g0,
                'xi': self.xi,
                'lambda_f': self.lambda_f
            },
            'water': {
                'rho_w': self.rho_w,
                'mu_w': self.mu_w
            },
            'gravity': self.g_vector.tolist()
        }
    
    def validate_parameters(self):
        """
        验证材料参数的合理性
        
        Returns:
            tuple: (是否有效, 错误信息列表)
        """
        errors = []
        
        # 检查孔隙度范围
        if self.phi0 <= 0 or self.phi0 >= 1:
            errors.append(f"初始孔隙度 phi0={self.phi0} 不在(0,1)范围内")
        
        # 检查粘度非负
        if self.mu_g0 <= 0:
            errors.append(f"浆液初始粘度 mu_g0={self.mu_g0} 必须大于0")
        
        if self.mu_w <= 0:
            errors.append(f"水粘度 mu_w={self.mu_w} 必须大于0")
        
        # 检查密度正定
        if self.rho_g <= 0:
            errors.append(f"浆液密度 rho_g={self.rho_g} 必须大于0")
        
        if self.rho_w <= 0:
            errors.append(f"水密度 rho_w={self.rho_w} 必须大于0")
        
        # 检查过滤系数非负
        if self.lambda_f < 0:
            errors.append(f"过滤系数 lambda_f={self.lambda_f} 不能为负")
        
        valid = len(errors) == 0
        
        if not valid and self.rank == 0:
            for error in errors:
                self.logger.warning(f"材料参数错误: {error}")
        
        return valid, errors
    
    def create_initial_fields(self):
        """
        创建初始材料场（可选功能）
        
        Returns:
            dict: 初始场字典
        """
        # 创建初始孔隙度场（均匀分布）
        P1 = ufl.FiniteElement("CG", self.mesh.ufl_cell(), 1)
        phi_space = fem.FunctionSpace(self.mesh, P1)
        
        phi_init = fem.Function(phi_space, name="initial_porosity")
        phi_init.x.array[:] = self.phi0
        
        return {
            'porosity': phi_init,
            'permeability': self.calculate_permeability(phi_init)
        }