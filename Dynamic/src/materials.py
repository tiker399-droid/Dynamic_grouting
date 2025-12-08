"""
材料属性管理
处理随时间变化的材料参数和本构关系
"""

import numpy as np
from typing import Callable
import ufl
from dolfinx import fem
from petsc4py import PETSc


class MaterialProperties:
    """
    管理所有材料属性和本构关系
    
    基于文献中的方程：
    - 粘度硬化: μ_g(t) = μ_g0 * exp(ξt) [Eq. 8]
    - 混合物粘度: μ = cμ_g + (1-c)μ_w [Eq. 7]
    - 混合物密度: ρ̄ = cρ_g + (1-c)ρ_w [Eq. 6]
    - 渗透率: k = k0 * φ³/(1-φ)² [Eq. 5]
    - 过滤速率: n̂ = λ_f * c * |q_w| [Eq. 9]
    """
    
    def __init__(self, params: Dict):
        """
        初始化材料参数
        
        Args:
            params: 材料参数字典
        """
        # 从配置加载参数
        self._load_parameters(params)
        
        # 计算拉梅常数
        self._compute_lame_constants()
        
        # 预计算常数
        self.gravity = np.array([0.0, 0.0, -9.81])
        
        logger.info("MaterialProperties initialized")
    
    def _load_parameters(self, params: Dict):
        """加载并验证参数"""
        # 土壤参数
        self.E = params['soil']['E']  # 杨氏模量 (Pa)
        self.nu = params['soil']['nu']  # 泊松比
        self.phi0 = params['soil']['phi0']  # 初始孔隙度
        self.k0 = params['soil']['k0']  # 参考渗透率 (m²)
        self.lambda_f = params['soil']['lambda_f']  # 过滤系数
        
        # 浆液参数
        self.rho_g = params['grout']['rho_g']  # 浆液密度 (kg/m³)
        self.mu_g0 = params['grout']['mu_g0']  # 初始粘度 (Pa·s)
        self.xi = params['grout']['xi']  # 硬化速率常数
        
        # 水参数
        self.rho_w = params['water']['rho_w']  # 水密度 (kg/m³)
        self.mu_w = params['water']['mu_w']  # 水粘度 (Pa·s)
        
        # 验证参数
        self._validate_parameters()
    
    def _validate_parameters(self):
        """验证参数合理性"""
        assert self.E > 0, "杨氏模量必须为正"
        assert 0 <= self.nu < 0.5, f"泊松比必须在[0, 0.5)之间: {self.nu}"
        assert 0 < self.phi0 < 1, f"初始孔隙度必须在(0, 1)之间: {self.phi0}"
        assert self.k0 > 0, "渗透率必须为正"
        assert self.lambda_f >= 0, "过滤系数必须非负"
        assert self.mu_g0 > 0, "浆液粘度必须为正"
        assert self.xi >= 0, "硬化速率常数必须非负"
    
    def _compute_lame_constants(self):
        """计算拉梅常数"""
        self.lambda_lame = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu_lame = self.E / (2 * (1 + self.nu))
        
        logger.debug(f"Lame constants: λ={self.lambda_lame:.2e}, μ={self.mu_lame:.2e}")
    
    def grout_viscosity(self, t: float) -> float:
        """
        浆液粘度随时间硬化 [Eq. 8]
        
        Args:
            t: 时间 (s)
        
        Returns:
            浆液粘度 (Pa·s)
        """
        return self.mu_g0 * np.exp(self.xi * t)
    
    def mixture_viscosity(self, c: ufl.Scalar, t: float, mesh) -> ufl.Scalar:
        """
        混合物粘度 [Eq. 7]
        
        Args:
            c: 浆液浓度场
            t: 时间
            mesh: 网格
        
        Returns:
            混合物粘度表达式
        """
        mu_g = fem.Constant(mesh, self.grout_viscosity(t))
        mu_w = fem.Constant(mesh, self.mu_w)
        
        return c * mu_g + (1 - c) * mu_w
    
    def mixture_density(self, c: ufl.Scalar, mesh) -> ufl.Scalar:
        """
        混合物密度 [Eq. 6]
        
        Args:
            c: 浆液浓度场
            mesh: 网格
        
        Returns:
            混合物密度表达式
        """
        rho_g = fem.Constant(mesh, self.rho_g)
        rho_w = fem.Constant(mesh, self.rho_w)
        
        return c * rho_g + (1 - c) * rho_w
    
    def permeability(self, phi: ufl.Scalar) -> ufl.Scalar:
        """
        Kozeny-Carman渗透率模型 [Eq. 5]
        
        Args:
            phi: 孔隙度场
        
        Returns:
            渗透率表达式
        """
        # 添加小量避免除零
        epsilon = 1e-10
        return self.k0 * phi**3 / ((1 - phi)**2 + epsilon)
    
    def filtration_rate(self, c: ufl.Scalar, q_w: ufl.Vector) -> ufl.Scalar:
        """
        过滤速率 [Eq. 9]
        
        Args:
            c: 浆液浓度场
            q_w: 达西流速场
        
        Returns:
            过滤速率表达式
        """
        # 计算流速模量（添加小量避免除零）
        q_norm = ufl.sqrt(ufl.dot(q_w, q_w) + 1e-10)
        return self.lambda_f * c * q_norm
    
    def create_gravity_vector(self, mesh) -> fem.Constant:
        """
        创建重力向量常数
        
        Args:
            mesh: 网格
        
        Returns:
            重力向量
        """
        return fem.Constant(mesh, PETSc.ScalarType(self.gravity))
    
    def create_body_force(self, mesh, include_buoyancy: bool = True) -> fem.Constant:
        """
        创建体力向量（考虑浮力）
        
        Args:
            mesh: 网格
            include_buoyancy: 是否包括浮力
        
        Returns:
            体力向量
        """
        if include_buoyancy:
            # 有效重度 = 饱和重度 - 水重度
            gamma_sat = 20000  # 饱和土重度 N/m³ (20 kN/m³)
            gamma_w = 9800  # 水重度 N/m³
            gamma_eff = gamma_sat - gamma_w
            
            body_force = np.array([0.0, 0.0, -gamma_eff])
        else:
            body_force = np.array([0.0, 0.0, 0.0])
        
        return fem.Constant(mesh, PETSc.ScalarType(body_force))