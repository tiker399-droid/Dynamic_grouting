"""
弱形式构建器 - 多物理场耦合注浆模拟 (简化两场系统)
基于混合物理论的控制方程，使用混合有限元方法
场顺序： [位移(u), 压力(p)]
时间离散：向后欧拉（完全隐式）
注：使用压力扩散方程，不再独立求解达西速度q
自动适应二维/三维网格
"""

import ufl
from dolfinx import fem
from mpi4py import MPI
import logging
import numpy as np

class WeakFormBuilder:
    """
    多物理场耦合弱形式构建器
    组装非线性残差 F 和雅可比矩阵 J
    """
    
    def __init__(self, function_space, materials, config, fields, cell_tags=None):
        """
        初始化弱形式构建器
        
        Args:
            function_space: 混合函数空间 (dolfinx.fem.FunctionSpace)
            materials: 材料属性管理器 (MaterialProperties 实例)
            config: 配置字典 (包含求解器参数等)
            fields: 场变量名字典 (用于调试，可选)
            cell_tags: 细胞标记 (用于区域积分，可选)
        """
        self.W = function_space
        self.mat = materials
        self.config = config
        self.fields = fields or {}
        self.cell_tags = cell_tags
        self.mesh = function_space.mesh  # 从函数空间获取网格
        self.rank = 0
        self.cell_tags = cell_tags
        if self.rank == 0 and cell_tags is not None:
            print(f"Init: 接收到的标记数量 = {len(cell_tags.values)}")
        
        # 日志
        self.logger = logging.getLogger(f"WeakFormBuilder_rank{MPI.COMM_WORLD.Get_rank()}")
        
        # 检查函数空间维度
        num_subspaces = self.W.num_sub_spaces
        if num_subspaces != 2:
            self.logger.warning(f"函数空间应有2个子空间，当前有{num_subspaces}个")

    def build_form(self, dt, time, solution, solution_prev, boundary_conditions):
        """
        构建当前时间步的残差形式和雅可比矩阵
        
        Args:
            dt: 时间步长 (float)
            time: 当前时间 (float)
            solution: 当前步解 (dolfinx.fem.Function)
            solution_prev: 上一步解 (dolfinx.fem.Function)
            boundary_conditions: 边界条件列表 (由边界条件管理器提供)
            
        Returns:
            F: 残差变分形式 (UFL 形式)
            J: 雅可比矩阵 (UFL 形式)
        """
        # --- 测试函数 ---
        v_u, v_p = ufl.TestFunctions(self.W)

        # --- 当前步未知函数 ---
        u, p = ufl.split(solution)

        # --- 上一步解 ---
        u_n, p_n = ufl.split(solution_prev)
        
        # --- 时间离散 ---
        # 土骨架速度 (向后欧拉)
        v_s = (u - u_n) / dt

        # 重力向量 (来自材料类，已自动适应维度)
        g = self.mat.g

        # --- 材料属性 (使用随时间变化的浆液粘度，浆液密度常数) ---
        # 初始孔隙度（常数）
        phi0 = self.mat.phi0_constant
        # 渗透率（基于常数孔隙度）
        k = self.mat.calculate_permeability(phi0)
        # 粘度：随时间变化的浆液粘度常数
        mu = self.mat.mu_current_constant
        # 流体密度：浆液密度常数
        rho = self.mat.rho_g_constant
        # 整体密度（常数）
        rho_bulk = self.mat.calculate_bulk_density(phi0, None)  # 忽略浓度

        # --- 定义积分度量 ---
        if self.cell_tags is not None:
            markers = self.cell_tags.values
            if 1 in markers:
                dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.cell_tags,
                                 metadata={"quadrature_degree": 2})
                dx_domain = dx(1)
            else:
                self.logger.warning("细胞标记中未找到标记1，将使用全区域积分")
                dx_domain = ufl.dx
        else:
            dx_domain = ufl.dx

        # --- 定义参考量（用于无量纲化）---
        L0 = self.config['geometry']['height']           # 13.0 m
        E0 = self.mat.E                                   # 20e6 Pa
        rho_w = self.mat.rho_w
        g_mag = self.mat.g_magnitude
        P0 = rho_w * g_mag * L0                           # 参考压力（基于水密度）
        U0 = P0 * L0 / E0                                  # 参考位移
        k0 = self.mat.k0
        mu0 = self.mat.mu_g0                              # 参考粘度（初始粘度）
        t0 = L0**2 * mu0 / (k0 * E0)                      # 参考时间
        scale_continuity = t0 / (U0 * L0)

        # 动量方程缩放因子：使弹性项无量纲
        scale_u = 1.0 / (E0 * L0)
        scale_p_force = 1.0 / (P0 * L0)
        scale_body = scale_u

        # --- 动量平衡方程 (F_u) ---
        sigma_eff = self.mat.effective_stress(u)
        alpha = self.mat.biot_coefficient()

        F_u = (scale_u * ufl.inner(sigma_eff, ufl.grad(v_u)) * dx_domain
               - scale_p_force * alpha * p * ufl.div(v_u) * dx_domain
               - scale_body * ufl.inner(rho_bulk * g, v_u) * dx_domain)
        
        # --- 连续性方程 (F_p) ---
        F_p = (ufl.div(v_s) * v_p * dx_domain +
               scale_continuity * ufl.inner((k / mu) * (ufl.grad(p) - rho * g), ufl.grad(v_p)) * dx_domain)

        # --- 总残差 ---
        F = F_u + F_p

        # --- 雅可比矩阵 (自动微分) ---
        du = ufl.TrialFunction(self.W)
        J = ufl.derivative(F, solution, du)

        # 记录调试信息
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"弱形式构建完成，时间={time:.2f}, dt={dt:.3e}")

        return F, J
    
    def build_mass_matrix(self):
        """构建质量矩阵（可选）"""
        pass
    
    def build_stiffness_matrix(self):
        """构建刚度矩阵（可选）"""
        pass