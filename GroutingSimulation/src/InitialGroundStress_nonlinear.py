# 该模块定义了InitialGroundStress类，用于计算和存储初始地应力状态
# 暂时未考虑土层内部材料参数的不均匀性
# 平衡方程：
# 本构方程：Mohr-Coulomb本构模型

from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl
import math


class MCDPConstitutive:
    """MC-DP本构模型实现类"""
    
    def __init__(self, lambda_, mu_, c, phi, psi):
        self.lambda_ = lambda_
        self.mu_ = mu_
        self.c = c
        self.phi = phi
        self.psi = psi
        # 弹性刚度矩阵
        # self.D_elastic = self.elastic_stiffness_tensor()

    def elastic_stiffness_tensor(self):
        """计算弹性刚度张量-需修改"""
        I = ufl.Identity(3)
        return self.lambda_ * ufl.outer(I, I) + 2 * self.mu_ * ufl.outer(I, I)

    def elastic_stress(self, u):
        """计算弹性应力"""
        return self.lambda_ * ufl.nabla_div(u) * ufl.Identity(3) + 2 * self.mu_ * epsilon(u)

    def elastic_tangent_operator(self, u, v):
        """计算弹性切线算子-什么作用-如何推导求出的"""
        return (self.lambda_ * ufl.nabla_div(u) * ufl.nabla_div(v) + 
                2 * self.mu_ * ufl.inner(epsilon(u), epsilon(v))) * ufl.dx

    def stress_invariant(self, sigma):
        """计算应力不变量"""
        I1 = ufl.tr(sigma)
        s = sigma - (I1/3) * ufl.Identity(3)  # 偏应力
        
        J2 = 0.5 * ufl.inner(s, s)
        J3 = ufl.det(s)
        
        # 计算应力洛德角
        q = ufl.sqrt(3 * J2)
        sin_3theta = - (3 * ufl.sqrt(3) / 2) * J3 / (q**3 + 1e-16)
        sin_3theta = ufl.Max(ufl.Min(sin_3theta, 1.0), -1.0)  # 限制在[-1,1]范围内
        theta = ufl.asin(sin_3theta) / 3.0
        
        return I1, J2, theta

    def yield_function(self, sigma, c, phi):
        """计算屈服函数"""
        I1, J2, theta = self.stress_invariant(sigma)
        sqrt_J2 = ufl.sqrt(J2 + 1e-16)
        
        term1 = sqrt_J2 * (1/ufl.sqrt(3) * ufl.sin(theta) * ufl.sin(phi) + ufl.cos(theta))
        term2 = I1/3 * ufl.sin(phi)
        term3 = c * ufl.cos(phi)
        
        f = term1 - term2 - term3
        return f

    def plastic_potential(self, sigma, psi):
        """计算塑性势函数"""
        I1, J2, theta = self.stress_invariant(sigma)
        alpha = 2 * ufl.sin(psi) / (ufl.sqrt(3) * (3 - ufl.sin(psi)))
        g = ufl.sqrt(J2 + 1e-16) - alpha * I1
        return g

    def compute_consistent_tangent(self, sigma, plastic_multiplier):
        """计算一致切线模量张量"""
        f = self.yield_function(sigma, self.c, self.phi)
        if f <= 0:
            return self.D_elastic
        else:
            reduction = 0.1
            return reduction * self.D_elastic


def epsilon(u):
    """计算应变张量"""
    return ufl.sym(ufl.grad(u))


def nonlinear_solver(domain, V_u, f, p, bcs, mc_dp_model):
    """非线性求解器"""
    
    # 初始化位移场
    u_sol = fem.Function(V_u)
    u_old = fem.Function(V_u)

    # NR迭代参数
    max_global_iter = 20
    tolerance = 1e-8
    residual_norm = 1.0
    convergence_history = []

    if MPI.COMM_WORLD.rank == 0:
        print("开始非线性迭代求解初始地应力场...")
        print('=' * 50)

    for global_iter in range(max_global_iter):
        if MPI.COMM_WORLD.rank == 0:
            print(f"全局迭代第 {global_iter + 1} 步")
        
        def residual_form(v):
            """定义残差形式-如何定义的"""
            current_stress = mc_dp_model.elastic_stress(u_sol)
            return (-ufl.inner(current_stress, epsilon(v)) * ufl.dx + 
                    ufl.dot(f, v) * ufl.dx + 
                    p * ufl.div(v) * ufl.dx)
        
        def tangent_form(u, v):
            """定义切线形式"""
            return mc_dp_model.elastic_tangent_operator(u, v)
        
        # 组装矩阵和向量
        a_tangent = fem.form(tangent_form(ufl.TrialFunction(V_u), ufl.TestFunction(V_u)))
        L_residual = fem.form(residual_form(ufl.TestFunction(V_u)))
        
        A = petsc.assemble_matrix(a_tangent, bcs=bcs)
        A.assemble()

        b_residual = fem.petsc.assemble_vector(L_residual)
        fem.petsc.apply_lifting(b_residual, [a_tangent], [bcs])
        b_residual.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # 应用边界条件
        for bc in bcs:
            bc.set(b_residual, u_sol.x.petsc_vec)
        
        # 计算残差范数
        residual_norm = b_residual.norm()
        convergence_history.append(residual_norm)

        if MPI.COMM_WORLD.rank == 0:
            print(f"  残差范数: {residual_norm:.6e}")

        # 收敛检查
        if residual_norm < tolerance:
            if MPI.COMM_WORLD.rank == 0:
                print(f'在{global_iter + 1}次迭代后收敛')
                print('=' * 50)
            break

        # 求解位移增量
        du = fem.Function(V_u)
        ksp = PETSc.KSP().create(domain.comm)
        ksp.setOperators(A)
        ksp.setType("cg")
        ksp.getPC().setType("hypre")
        ksp.setTolerances(rtol=1e-10)

        ksp.solve(b_residual, du.x.petsc_vec)
        
        # 更新位移
        u_sol.x.array[:] += du.x.array[:]
        u_old.x.array[:] = u_sol.x.array[:]

        ksp.destroy()
        A.destroy()
        b_residual.destroy()
    else:
        if MPI.COMM_WORLD.rank == 0:
            print("达到最大迭代次数，未收敛")
            print(f"最终残差范数: {residual_norm:.6e}")
            print('=' * 50)
    
    return u_sol, convergence_history


def main():
    """主函数"""
    # 网格参数
    Lx, Ly, Lz = 10, 10, 10
    nx, ny, nz = 10, 10, 10
    cell_type = mesh.CellType.hexahedron

    domain = mesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [Lx, Ly, Lz]], 
                            [nx, ny, nz], cell_type=cell_type)

    tols = 1e-8

    def bottom_facets(x):
        return np.isclose(x[2], 0.0, atol=tols)

    def top_facets(x):
        return np.isclose(x[2], Lz, atol=tols)

    def around_facets(x):
        return (np.isclose(x[0], 0.0, atol=tols) | np.isclose(x[0], Lx, atol=tols) |
                np.isclose(x[1], 0.0, atol=tols) | np.isclose(x[1], Ly, atol=tols))

    # 获取边界facet
    tdim = domain.topology.dim
    fdim = tdim - 1
    facet_bottom = mesh.locate_entities_boundary(domain, fdim, bottom_facets)
    facet_top = mesh.locate_entities_boundary(domain, fdim, top_facets)
    facet_around = mesh.locate_entities_boundary(domain, fdim, around_facets)

    from dolfinx.mesh import meshtags
    facet_indices = np.hstack([facet_bottom, facet_top, facet_around])
    facet_markers = np.hstack([np.full(len(facet_bottom), 1, dtype=np.int32),
                               np.full(len(facet_top), 2, dtype=np.int32),
                               np.full(len(facet_around), 3, dtype=np.int32)])

    mt = meshtags(domain, fdim, facet_indices, facet_markers)
    
    if MPI.COMM_WORLD.rank == 0:
        print(f"Bottom facets: {len(facet_bottom)}")
        print(f"Top facets: {len(facet_top)}")
        print(f"Around facets: {len(facet_around)}")
        print(f"Total facets tagged: {len(facet_indices)}")
        print("Unique markers in mt:", np.unique(mt.values))

    # 材料参数
    E = 20e6  # Young's modulus
    nu = 0.3   # Poisson's ratio
    rhos_s = 2000  # Soil density
    rho_w = 1000  # Water density
    g = 9.81  # Gravitational acceleration
    gamma_sat = rhos_s * g  # Saturated unit weight

    lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu_ = E / (2 * (1 + nu))

    # 设置体力
    f = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, -gamma_sat)))

    # 设置孔隙水压力
    x = ufl.SpatialCoordinate(domain)
    p_expression = rho_w * g * (Lz - x[2])
    V = fem.functionspace(domain, ("CG", 1))
    p = fem.Function(V)
    p.interpolate(fem.Expression(p_expression, V.element.interpolation_points()))

    # 位移函数空间
    V_u = fem.functionspace(domain, ('CG', 1, (tdim,)))

    # 设置边界条件
    # 底部边界条件：固定
    bottoms_dofs = fem.locate_dofs_topological(V_u, fdim, facet_bottom)
    bc_bottom = fem.dirichletbc(PETSc.ScalarType((0, 0, 0)), bottoms_dofs, V_u)
    
    # 周围边界条件：法向约束
    def x0_boundary(x):
        return np.isclose(x[0], 0.0, atol=tols)

    def xLx_boundary(x):
        return np.isclose(x[0], Lx, atol=tols)

    def y0_boundary(x):
        return np.isclose(x[1], 0.0, atol=tols)

    def yLy_boundary(x):
        return np.isclose(x[1], Ly, atol=tols)

    facets_x0 = mesh.locate_entities_boundary(domain, fdim, x0_boundary)
    facets_xLx = mesh.locate_entities_boundary(domain, fdim, xLx_boundary)
    facets_y0 = mesh.locate_entities_boundary(domain, fdim, y0_boundary)
    facets_yLy = mesh.locate_entities_boundary(domain, fdim, yLy_boundary)

    dofs_x0 = fem.locate_dofs_topological(V_u.sub(0), fdim, facets_x0)
    bc_x0 = fem.dirichletbc(PETSc.ScalarType(0), dofs_x0, V_u.sub(0))
    dofs_xLx = fem.locate_dofs_topological(V_u.sub(0), fdim, facets_xLx)
    bc_xLx = fem.dirichletbc(PETSc.ScalarType(0), dofs_xLx, V_u.sub(0))

    dofs_y0 = fem.locate_dofs_topological(V_u.sub(1), fdim, facets_y0)
    bc_y0 = fem.dirichletbc(PETSc.ScalarType(0), dofs_y0, V_u.sub(1))
    dofs_yLy = fem.locate_dofs_topological(V_u.sub(1), fdim, facets_yLy)
    bc_yLy = fem.dirichletbc(PETSc.ScalarType(0), dofs_yLy, V_u.sub(1))

    bcs = [bc_bottom, bc_x0, bc_xLx, bc_y0, bc_yLy]

    # MC-DP模型参数
    phi_rad = np.radians(30.0)  # 内摩擦角，单位：弧度
    psi_rad = np.radians(5.0)   # 剪胀角，单位：弧度
    c_val = 10e3                # 黏聚力，单位：Pa

    # 创建MC-DP本构模型
    mc_dp_model = MCDPConstitutive(lambda_, mu_, c_val, phi_rad, psi_rad)

    # 运行非线性求解
    u_final, convergence_history = nonlinear_solver(domain, V_u, f, p, bcs, mc_dp_model)
    
    return u_final, convergence_history


if __name__ == "__main__":
    u_final, convergence_history = main()