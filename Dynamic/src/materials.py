"""
材料属性管理器 - 增强版
包含弹性参数、Biot系数等，支持完整的控制方程
"""

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI
import logging

class MaterialProperties:
    """材料属性管理器 - 增强版"""
    
    def __init__(self, config, mesh, comm):
        """
        初始化材料属性管理器
        
        Args:
            config: 配置文件字典
            mesh: 计算网格
            comm: MPI通信器
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.mesh = mesh
        self.logger = logging.getLogger(f"Materials_rank{self.rank}")
        
        # 提取所有材料参数
        self._extract_parameters(config)
        
        # 初始化常量和基础设置
        self._initialize_constants()
        
        # 验证参数
        valid, errors = self.validate_parameters()
        if not valid and self.rank == 0:
            for err in errors:
                self.logger.warning(f"材料参数警告: {err}")
        
        if self.rank == 0:
            self.logger.info("材料属性管理器初始化完成")
            self.logger.info(f"初始孔隙度: {self.phi0:.3f}")
            self.logger.info(f"杨氏模量: {self.E/1e6:.1f} MPa, 泊松比: {self.nu:.2f}")
            self.logger.info(f"浆液初始粘度: {self.mu_g0:.4f} Pa·s")
    
    def _extract_parameters(self, config):
        """从配置字典中提取所有材料参数"""
        materials_config = config.get('materials', {})

        # 土壤参数 - 确保类型转换
        soil = materials_config.get('soil', {})
        self.phi0 = float(soil.get('phi0', 0.40))              # 初始孔隙度
        self.k0 = float(soil.get('k0', 1e-12))                 # 初始渗透率 (m²)
        self.E = float(soil.get('E', 20e6))                    # 杨氏模量 (Pa)
        self.nu = float(soil.get('nu', 0.3))                  # 泊松比
        self.alpha = float(soil.get('biot_coefficient', 1.0)) # Biot系数
        # 可选：土颗粒密度（如果需要）
        self.rho_s = float(soil.get('rho_s', 2020.0))         # 土颗粒密度 (kg/m³)

        # 浆液参数 - 确保类型转换
        grout = materials_config.get('grout', {})
        self.rho_g = float(grout.get('rho_g', 1800.0))        # 浆液密度 (kg/m³)
        self.mu_g0 = float(grout.get('mu_g0', 0.01))          # 浆液初始粘度 (Pa·s)
        self.xi = float(grout.get('xi', 1.56))                 # 粘度增长常数 (1/s)
        self.lambda_f = float(grout.get('filtration_coeff', 0.75)) # 过滤系数

        # 水参数 - 确保类型转换
        water = materials_config.get('water', {})
        self.rho_w = float(water.get('rho_w', 1000.0))        # 水密度 (kg/m³)
        self.mu_w = float(water.get('mu_w', 0.001))           # 水粘度 (Pa·s)
        
        # 重力加速度（z轴向上为正）
        self.g_magnitude = -9.81                               # m/s²
        
        # 存储原始配置（供调试）
        self.soil = soil
        self.grout = grout
        self.water = water
    
    def _initialize_constants(self):
        """初始化FEniCS常数对象"""
        # 重力向量 (z向上为正)
        self.g_vector = np.array([0.0, 0.0, self.g_magnitude], dtype=np.float64)
        self.g = fem.Constant(self.mesh, self.g_vector)

        # 材料常数（用于弱形式）- 使用 PETSc.ScalarType 确保兼容性
        from petsc4py import PETSc
        scalar_dtype = PETSc.ScalarType
        
        self.k0_constant = fem.Constant(self.mesh, scalar_dtype(self.k0))
        self.mu_g0_constant = fem.Constant(self.mesh, scalar_dtype(self.mu_g0))
        self.lambda_f_constant = fem.Constant(self.mesh, scalar_dtype(self.lambda_f))
        
        # 弹性常数（可转换为UFL常数，但直接用标量在表达式中也可）
        # 这里不创建Constant，因为E和nu在表达式中直接使用数字
        
        # 当前时间和粘度缓存
        self._current_time = 0.0
        self._current_mu_g = self.mu_g0
    
    # ---------- 本构关系 ----------
    
    def calculate_permeability(self, phi):
        """
        Kozeny-Carman关系：计算渗透率
        k = k0 * (φ^3) / (1-φ)^2
        添加小量避免除零
        """
        denominator = (1 - phi)**2 + 1e-10
        return self.k0 * (phi**3) / denominator
    
    def calculate_viscosity(self, c, time=None):
        """
        计算混合粘度
        μ = c * μ_g(t) + (1-c) * μ_w
        """
        current_time = time if time is not None else self._current_time
        mu_g = self.mu_g0 * ufl.exp(self.xi * current_time)
        return c * mu_g + (1 - c) * self.mu_w
    
    def calculate_density(self, c):
        """计算混合物密度 ρ = c ρ_g + (1-c) ρ_w"""
        return c * self.rho_g + (1 - c) * self.rho_w
    
    def calculate_bulk_density(self, phi, c):
        """
        计算饱和多孔介质的整体密度
        ρ_bulk = (1 - φ) * ρ_s + φ * [c ρ_g + (1-c) ρ_w]
        
        Args:
            phi: 孔隙度场
            c: 浓度场
        
        Returns:
            整体密度表达式
        """
        # 流体密度
        rho_f = c * self.rho_g + (1 - c) * self.rho_w
        # 整体密度
        return (1 - phi) * self.rho_s + phi * rho_f

    # 注：4场系统中不再使用过滤速率，因为达西速度q不再是独立未知量
    # def calculate_filtration_rate(self, c, q):
    #     """过滤定律：ȯn = λ_f * c * |q|"""
    #     q_mag = ufl.sqrt(ufl.dot(q, q) + 1e-10)
    #     return self.lambda_f * c * q_mag

    def calculate_darcy_velocity(self, p, phi, c, time=None):
        """
        达西定律表达式：q = - (k/μ) * (∇p - ρ g)
        注意：g 方向向下为正，∇p - ρg 在重力向下时正确
        这个函数用于从压力场导出达西速度（用于后处理或输运方程）
        """
        k = self.calculate_permeability(phi)
        mu = self.calculate_viscosity(c, time)
        rho = self.calculate_density(c)
        return - (k / mu) * (ufl.grad(p) - rho * self.g)
    
    # ---------- 弹性参数 ----------
    
    def get_lame_parameters(self):
        """
        计算Lame常数 λ 和 μ
        返回 (lambda, mu) 两个标量（可转换为UFL常数）
        """
        lambda_ = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        mu = self.E / (2 * (1 + self.nu))
        return lambda_, mu
    
    def effective_stress(self, u):
        """
        计算有效应力张量 σ' (线弹性)
        输入位移场 u (UFL表达式)，返回二阶张量
        """
        epsilon = ufl.sym(ufl.grad(u))
        lambda_, mu = self.get_lame_parameters()
        return lambda_ * ufl.tr(epsilon) * ufl.Identity(len(u)) + 2 * mu * epsilon
    
    def biot_coefficient(self):
        """返回Biot系数 α"""
        return self.alpha

    # ---------- 解耦求解器专用方法 ----------
    
    def get_default_porosity(self):
        """返回默认孔隙率（用于简化模型）"""
        return self.phi0

    def calculate_permeability_scalar(self, phi):
        """
        计算标量渗透率（用于解耦求解器）
        k = k0 * (φ^3) / (1-φ)^2
        """
        denominator = (1 - phi)**2 + 1e-10
        return self.k0 * (phi**3) / denominator

    def calculate_viscosity_scalar(self, c, time=None):
        """
        计算标量粘度（用于解耦求解器）
        μ = c * μ_g(t) + (1-c) * μ_w
        """
        current_time = time if time is not None else self._current_time
        mu_g = self.mu_g0 * np.exp(self.xi * current_time)
        return c * mu_g + (1 - c) * self.mu_w

    def stress_tensor_linear(self, u):
        """
        计算线性应力张量（小变形）
        σ = λ tr(ε) I + 2 μ ε
        """
        epsilon = ufl.sym(ufl.grad(u))
        lambda_, mu = self.get_lame_parameters()
        return lambda_ * ufl.tr(epsilon) * ufl.Identity(len(u)) + 2 * mu * epsilon
    
    # ---------- 时间相关属性 ----------
    
    def update_time_dependent_properties(self, time):
        """
        更新时间相关的材料属性（主要是浆液粘度）
        """
        self._current_time = time
        self._current_mu_g = self.mu_g0 * np.exp(self.xi * time/60)
        
        updated_props = {
            'time': time,
            'grout_viscosity': self._current_mu_g,
            'grout_viscosity_constant': fem.Constant(self.mesh, self._current_mu_g)
        }
        
        if self.rank == 0 and time > 1e-10:
            self.logger.debug(f"更新粘度: μ_g = {self._current_mu_g:.4e} Pa·s")
        
        return updated_props
    
    # ---------- 衍生场计算 ----------

    def calculate_all_derived(self, solution_fields, time):
        """计算所有衍生物理场（表达式形式）"""
        derived = {}
        try:
            phi = solution_fields.get('porosity')
            c = solution_fields.get('concentration')
            p = solution_fields.get('pressure')

            if phi is not None:
                derived['permeability'] = self.calculate_permeability(phi)
            if c is not None:
                derived['viscosity'] = self.calculate_viscosity(c, time)
                derived['density'] = self.calculate_density(c)
            # 4场系统中：从压力场导出达西速度
            if p is not None and phi is not None and c is not None:
                derived['darcy_velocity'] = self.calculate_darcy_velocity(p, phi, c, time)
        except Exception as e:
            self.logger.warning(f"计算衍生场时出错: {e}")
        return derived
    
    # ---------- 参数验证 ----------
    
    def validate_parameters(self):
        """验证材料参数的物理合理性"""
        errors = []
        if self.phi0 <= 0 or self.phi0 >= 1:
            errors.append(f"孔隙度 phi0={self.phi0} 不在(0,1)内")
        if self.E <= 0:
            errors.append(f"杨氏模量 E={self.E} 必须大于0")
        if self.nu <= -1 or self.nu >= 0.5:
            errors.append(f"泊松比 nu={self.nu} 应在(-1,0.5)内")
        if self.k0 <= 0:
            errors.append(f"初始渗透率 k0={self.k0} 必须大于0")
        if self.mu_g0 <= 0:
            errors.append(f"浆液初始粘度 mu_g0={self.mu_g0} 必须大于0")
        if self.mu_w <= 0:
            errors.append(f"水粘度 mu_w={self.mu_w} 必须大于0")
        if self.rho_g <= 0:
            errors.append(f"浆液密度 rho_g={self.rho_g} 必须大于0")
        if self.rho_w <= 0:
            errors.append(f"水密度 rho_w={self.rho_w} 必须大于0")
        if self.lambda_f < 0:
            errors.append(f"过滤系数 lambda_f={self.lambda_f} 不能为负")
        return len(errors) == 0, errors
    
    # ---------- 辅助方法 ----------
    
    def get_material_parameters(self):
        """返回所有材料参数字典（用于输出/调试）"""
        return {
            'soil': {
                'phi0': self.phi0,
                'k0': self.k0,
                'E': self.E,
                'nu': self.nu,
                'alpha': self.alpha,
                'rho_s': self.rho_s
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
            'gravity': self.g_magnitude
        }