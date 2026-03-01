"""
弱形式构建器 - 多物理场耦合注浆模拟
基于混合物理论的控制方程，使用混合有限元方法
场顺序： [位移(u), 压力(p), 孔隙度(φ), 浓度(c), 达西速度(q)]
时间离散：向后欧拉（完全隐式）
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
        if num_subspaces != 5:
            self.logger.warning(f"函数空间应有5个子空间，当前有{num_subspaces}个")
    
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
        v_u, v_p, v_phi, v_c, v_q = ufl.TestFunctions(self.W)
        
        # --- 当前步未知函数 ---
        u, p, phi, c, q = ufl.split(solution)
        
        # --- 上一步解 ---
        u_n, p_n, phi_n, c_n, q_n = ufl.split(solution_prev)
        
        # --- 时间离散 ---
        # 土骨架速度 (向后欧拉)
        v_s = (u - u_n) / dt
        
        # --- 材料属性 (使用当前步值) ---
        # 渗透率 k(φ)
        k = self.mat.calculate_permeability(phi)
        # 混合粘度 μ(c, t)
        mu = self.mat.calculate_viscosity(c, time)
        # 混合物密度 ρ(c)
        rho = self.mat.calculate_density(c)
        # 整体密度
        rho_bulk = self.mat.calculate_bulk_density(phi, c)   
        # 过滤速率 ^n = λ_f c |q|
        n_hat = self.mat.calculate_filtration_rate(c, q)
        
        # 重力向量 (来自材料类，已定义为 fem.Constant)
        g = self.mat.g
        
        # --- 动量平衡方程 (F_u) ---
        # 有效应力 σ'(u)
        sigma_eff = self.mat.effective_stress(u)
        # Biot系数 α
        alpha = self.mat.biot_coefficient()
        
        # 定义体积积分度量：如果提供了细胞标记，则只在地基区域（标记1）积分
        if self.cell_tags is not None:
            # 创建带子域的积分度量
            dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.cell_tags)
            dx_domain = dx(1)  # 仅标记为1的区域（地基）
        else:
            dx_domain = ufl.dx  # 默认全区域积分
        
        # 弱形式：∫ σ' : ∇v_u dx - ∫ α p (∇·v_u) dx - ∫ ρ g · v_u dx = 0
        F_u = ufl.inner(sigma_eff, ufl.grad(v_u)) * dx_domain \
              - alpha * p * ufl.div(v_u) * dx_domain \
              - ufl.inner(rho_bulk * g, v_u) * dx_domain
        
        # --- 连续性方程 (F_p) ---
        # ∇·q + ∇·v_s = 0
        # 乘以测试函数 v_p，积分
        F_p = (ufl.div(q) + ufl.div(v_s)) * v_p * dx_domain
        
        # --- 孔隙率演化方程 (F_phi) ---
        # -∂φ/∂t + ∇·v_s - ∇·(φ v_s) = ^n
        # 时间离散：- (φ - φ_n)/dt + ∇·v_s - ∇·(φ v_s) = ^n
        # 对 ∇·(φ v_s) 项分部积分：∫ -∇·(φ v_s) v_phi dx = ∫ φ v_s · ∇v_phi dx - ∫ φ v_s·n v_phi ds
        # 忽略边界项（由BC处理），得到：
        F_phi = (- (phi - phi_n) / dt * v_phi) * dx_domain \
                + ufl.div(v_s) * v_phi * dx_domain \
                + ufl.inner(phi * v_s, ufl.grad(v_phi)) * dx_domain \
                - n_hat * v_phi * dx_domain
        
        # --- 浓度输运方程 (F_c) ---
        # ∂(cφ)/∂t + ∇·(c q) + ∇·(c φ v_s) = -^n
        # 时间离散： (cφ - c_n φ_n)/dt + ∇·(c q) + ∇·(c φ v_s) = -^n
        # 对 ∇·(c q) 和 ∇·(c φ v_s) 分部积分：
        # ∫ ∇·(c q) v_c dx = -∫ c q · ∇v_c dx + ∫ c q·n v_c ds
        # ∫ ∇·(c φ v_s) v_c dx = -∫ c φ v_s · ∇v_c dx + ∫ c φ v_s·n v_c ds
        # 忽略边界项，得到：
        F_c = ((c * phi - c_n * phi_n) / dt * v_c) * dx_domain \
              - ufl.inner(c * q, ufl.grad(v_c)) * dx_domain \
              - ufl.inner(c * phi * v_s, ufl.grad(v_c)) * dx_domain \
              + n_hat * v_c * dx_domain
        
        # --- 达西定律 (F_q) ---
        # q + (k/μ)(∇p - ρ g) = 0
        # 乘以向量测试函数 v_q
        F_q = ufl.inner(q + (k / mu) * (ufl.grad(p) - rho * g), v_q) * dx_domain
        
        # --- 总残差 ---
        F = F_u + F_p + F_phi + F_c + F_q
        
        # --- 雅可比矩阵 (自动微分) ---
        du = ufl.TrialFunction(self.W)
        J = ufl.derivative(F, solution, du)
        
        # 记录调试信息（可选）
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"弱形式构建完成，时间={time:.2f}, dt={dt:.3e}")
        
        return F, J
    
    def build_mass_matrix(self):
        """
        构建质量矩阵（用于瞬态项，可选）
        返回 (M, None) 或直接返回 M
        这里仅作为示例，实际求解中不需要单独的质量矩阵
        """
        # 如果需要，可以在这里实现
        pass
    
    def build_stiffness_matrix(self):
        """
        构建刚度矩阵（可选）
        """
        pass