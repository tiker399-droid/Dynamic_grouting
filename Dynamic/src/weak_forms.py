"""
弱形式构建器 - 多物理场耦合注浆模拟
基于混合物理论的控制方程，使用混合有限元方法
场顺序： [位移(u), 压力(p), 孔隙度(φ), 浓度(c)]
时间离散：向后欧拉（完全隐式）
注：使用压力扩散方程，不再独立求解达西速度q
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

        # 重力向量 (来自材料类，已定义为 fem.Constant)
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
        # 注意：这是从压力场导出的，不是独立未知量
        q_darcy = -(k / mu) * (ufl.grad(p) - rho * g)
        
        # --- 动量平衡方程 (F_u) ---
        # 有效应力 σ'(u)
        sigma_eff = self.mat.effective_stress(u)
        # Biot系数 α
        alpha = self.mat.biot_coefficient()

        # 定义体积积分度量：使用全区域积分，避免零对角元问题
        # 如果需要只在地基区域积分，应该在材料属性中控制，而不是在积分度量中
        dx_domain = ufl.dx  # 默认全区域积分

        # 弱形式：∫ σ' : ∇v_u dx - ∫ α p (∇·v_u) dx - ∫ ρ g · v_u dx = 0
        F_u = ufl.inner(sigma_eff, ufl.grad(v_u)) * dx_domain \
              - alpha * p * ufl.div(v_u) * dx_domain \
              - ufl.inner(rho_bulk * g, v_u) * dx_domain
        
        # --- 连续性方程 (F_p) ---
        # 使用原连续性方程: ∇·q + ∇·v_s = 0
        # 但用达西定律替换q: q = -(k/μ)(∇p - ρg)
        # 分部积分：-∫∇·[(k/μ)(∇p - ρg)] v_p dx = ∫ (k/μ)(∇p - ρg)·∇v_p dx - 边界项
        # 边界项忽略（由BC处理），得到：
        F_p = (ufl.div(v_s)) * v_p * dx_domain \
              + ufl.inner((k / mu) * (ufl.grad(p) - rho * g), ufl.grad(v_p)) * dx_domain
        
        # --- 孔隙率演化方程 (F_phi) ---
        # -∂φ/∂t + ∇·v_s - ∇·(φ v_s) = 0
        # 时间离散：- (φ - φ_n)/dt + ∇·v_s - ∇·(φ v_s) = 0
        # 对 ∇·(φ v_s) 项分部积分：∫ -∇·(φ v_s) v_phi dx = ∫ φ v_s · ∇v_phi dx - ∫ φ v_s·n v_phi ds
        # 忽略边界项（由BC处理），得到：
        F_phi = (- (phi - phi_n) / dt * v_phi) * dx_domain \
                + ufl.div(v_s) * v_phi * dx_domain \
                + ufl.inner(phi * v_s, ufl.grad(v_phi)) * dx_domain

        # --- 浓度输运方程 (F_c) ---
        # ∂(cφ)/∂t + ∇·(c q_darcy) + ∇·(c φ v_s) = 0
        # 使用从压力导出的达西速度 q_darcy = -(k/μ)(∇p - ρg)
        # 时间离散： (cφ - c_n φ_n)/dt + ∇·(c q_darcy) + ∇·(c φ v_s) = 0
        # 对 ∇·(c q_darcy) 和 ∇·(c φ v_s) 分部积分：
        # 忽略边界项，得到：
        # 添加人工扩散项 ε∇c·∇v_c 以避免雅可比矩阵零对角元
        epsilon_diff = 1e-6  # 人工扩散系数
        F_c = ((c * phi - c_n * phi_n) / dt * v_c) * dx_domain \
              - ufl.inner(c * q_darcy, ufl.grad(v_c)) * dx_domain \
              - ufl.inner(c * phi * v_s, ufl.grad(v_c)) * dx_domain \
              + epsilon_diff * ufl.dot(ufl.grad(c), ufl.grad(v_c)) * dx_domain

        # --- 总残差 ---
        F = F_u + F_p + F_phi + F_c

        # --- 雅可比矩阵 (自动微分) ---
        du = ufl.TrialFunction(self.W)
        J = ufl.derivative(F, solution, du)

        # 注意：DOLFINx 0.9.0 的 NonlinearProblem 会自动编译 UFL 形式
        # 这里直接返回 UFL 形式，不预先编译
        # 编译工作由 NonlinearProblem 在初始化时自动完成
        
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