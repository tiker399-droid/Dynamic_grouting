"""
弱形式构建器 - 多物理场耦合注浆模拟
基于混合物理论的控制方程，使用混合有限元方法
场顺序： [位移(u), 压力(p), 孔隙度(φ), 浓度(c)]
时间离散：向后欧拉（完全隐式）
注：使用压力扩散方程，不再独立求解达西速度q
自动适应二维/三维网格
"""

import ufl
from dolfinx import fem
from mpi4py import MPI
import logging

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
        
        # 日志
        self.logger = logging.getLogger(f"WeakFormBuilder_rank{MPI.COMM_WORLD.Get_rank()}")
        
        # 检查函数空间维度
        num_subspaces = self.W.num_sub_spaces
        if num_subspaces != 4:
            self.logger.warning(f"函数空间应有4个子空间，当前有{num_subspaces}个")
    
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
        v_u, v_p, v_phi, v_c = ufl.TestFunctions(self.W)

        # --- 当前步未知函数 ---
        u, p, phi, c = ufl.split(solution)

        # --- 上一步解 ---
        u_n, p_n, phi_n, c_n = ufl.split(solution_prev)
        
        # --- 时间离散 ---
        # 土骨架速度 (向后欧拉)
        v_s = (u - u_n) / dt

        # 重力向量 (来自材料类，已自动适应维度)
        g = self.mat.g

        # --- 材料属性 (使用当前步值) ---
        # 渗透率 k(φ)
        k = self.mat.calculate_permeability(phi)
        # 混合粘度 μ(c, t)
        mu = self.mat.calculate_viscosity(c, time)
        # 混合物密度 ρ(c)
        rho = self.mat.calculate_density(c)
        # 整体密度
        rho_bulk = self.mat.calculate_bulk_density(phi, c)

        # --- 根据达西定律计算的等效流速场 (用于输运方程) ---
        # q = -(k/μ)(∇p - ρg)
        q_darcy = -(k / mu) * (ufl.grad(p) - rho * g)

        abs_q = ufl.sqrt(ufl.inner(q_darcy, q_darcy) + 1e-12)   # 加小量避免导数奇异
        n_hat = self.mat.lambda_f_constant * c * abs_q          # λ_f * c * |q|

        # --- 定义积分度量 ---
        # 如果提供了细胞标记且包含标记1，则仅在地基区域（标记1）积分
        if self.cell_tags is not None:
            # 获取所有细胞标记值
            markers = self.cell_tags.values
            if 1 in markers:
                # 创建带子域的积分度量
                dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.cell_tags,
                                 metadata={"quadrature_degree": 2})
                dx_domain = dx(1)  # 仅标记为1的区域（地基）
            else:
                # 标记1不存在，回退到全区域
                self.logger.warning("细胞标记中未找到标记1，将使用全区域积分")
                dx_domain = ufl.dx
        else:
            dx_domain = ufl.dx  # 默认全区域积分

        # --- 动量平衡方程 (F_u) ---
        # 有效应力 σ'(u)
        sigma_eff = self.mat.effective_stress(u)
        # Biot系数 α
        alpha = self.mat.biot_coefficient()

        F_u = ufl.inner(sigma_eff, ufl.grad(v_u)) * dx_domain \
              - alpha * p * ufl.div(v_u) * dx_domain \
              - ufl.inner(rho_bulk * g, v_u) * dx_domain
        
        # --- 连续性方程 (F_p) ---
        F_p = (ufl.div(v_s)) * v_p * dx_domain \
              + ufl.inner((k / mu) * (ufl.grad(p) - rho * g), ufl.grad(v_p)) * dx_domain
        
        # --- 孔隙率演化方程 (F_phi) ---
        epsilon_diff_phi = 1e-6   # 原为 1e-8，增大至 1e-6
        F_phi = (- (phi - phi_n) / dt * v_phi) * dx_domain \
                + ufl.div(v_s) * v_phi * dx_domain \
                + ufl.inner(phi * v_s, ufl.grad(v_phi)) * dx_domain \
                + epsilon_diff_phi * ufl.dot(ufl.grad(phi), ufl.grad(v_phi)) * dx_domain \
                - n_hat * v_phi * dx_domain      # 添加过滤源项（注意符号：方程中为 -n_hat）

        # --- 浓度输运方程 (F_c) ---
        epsilon_diff = 1e-6        # 原为 1e-6，可保持不变或也增大（此处保持 1e-6）
        F_c = ((c * phi - c_n * phi_n) / dt * v_c) * dx_domain \
            - ufl.inner(c * q_darcy, ufl.grad(v_c)) * dx_domain \
            - ufl.inner(c * phi * v_s, ufl.grad(v_c)) * dx_domain \
            + epsilon_diff * ufl.dot(ufl.grad(c), ufl.grad(v_c)) * dx_domain \
            + n_hat * v_c * dx_domain           # 添加过滤源项（符号：方程中为 +n_hat）
        
        # --- 总残差 ---
        F = F_u + F_p + F_phi + F_c

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