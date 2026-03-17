"""
初始状态求解器：仅求解位移方程，压力设为静水压力，得到重力作用下的初始位移。
用于与注浆后的位移对比，得到注浆引起的净位移。
"""

import numpy as np
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem import petsc, dirichletbc
from dolfinx.mesh import locate_entities_boundary
from mpi4py import MPI
from petsc4py import PETSc
import basix.ufl
import yaml
import logging
import sys
import os
from pathlib import Path

# 导入自定义材料模块
sys.path.append(os.path.dirname(__file__))
from materials import MaterialProperties

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("InitialState")


def main(config_file: str, mesh_file: str, output_dir: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # 读取配置
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # 读取网格
    from dolfinx.io import gmshio
    mesh, cell_tags, facet_tags = gmshio.read_from_msh(mesh_file, comm, rank=0, gdim=2)

    # 初始化材料属性
    materials = MaterialProperties(config, mesh, comm)

    # 几何参数
    H = config['geometry']['height']
    g = materials.g_magnitude
    rho_w = materials.rho_w
    rho_g = materials.rho_g

    # 创建函数空间
    gdim = mesh.geometry.dim
    cell_type = mesh.topology.cell_name()
    P2_vec = basix.ufl.element("Lagrange", cell_type, 2, shape=(gdim,))
    V_u = fem.functionspace(mesh, P2_vec)   # 位移空间
    P1 = basix.ufl.element("Lagrange", cell_type, 1, shape=())
    V_p = fem.functionspace(mesh, P1)       # 压力空间（仅用于插值）

    # 创建静水压力函数（有量纲）
    p_init = fem.Function(V_p, name="Initial_pressure")
    x = ufl.SpatialCoordinate(mesh)
    p_expression = rho_w * g * (H - x[1])
    p_init.interpolate(fem.Expression(p_expression, V_p.element.interpolation_points()))  # 二维 y 向上

    # 位移函数
    u = fem.Function(V_u, name="Initial_displacement")

    # --- 使用边界标记定义位移边界条件（与 MeshCreate.py 标记一致）---
    fdim = mesh.topology.dim - 1          # 边界维度（1D 边）
    bcs_u = []

    # 边界标记值
    MARKER_LEFT   = 103   # 地基左侧（约束 x 方向）
    MARKER_RIGHT  = 104   # 地基右侧（约束 x 方向）
    MARKER_BOTTOM = 107   # 地基底面（全固定）
    MARKER_HOLE1 = 101
    MARKER_HOLE2 = 102

    # 获取所有边界标记的唯一值
    unique_markers = np.unique(facet_tags.values)

    # 1. 底面全固定（位移为零，约束两个方向）
    if MARKER_HOLE2 in unique_markers:
        facets_bottom = facet_tags.find(MARKER_HOLE2)
        dofs_bottom = fem.locate_dofs_topological(V_u, fdim, facets_bottom)
        # 创建零向量函数
        zero_vector = fem.Function(V_u)
        zero_vector.x.array[:] = 0.0
        bc_bottom = dirichletbc(zero_vector, dofs_bottom)
        bcs_u.append(bc_bottom)
        if rank == 0:
            print(f"钻孔底面边界 (102)：找到 {len(facets_bottom)} 条边，施加全固定。") 

    if MARKER_BOTTOM in unique_markers:
        facets_bottom = facet_tags.find(MARKER_BOTTOM)
        dofs_bottom = fem.locate_dofs_topological(V_u, fdim, facets_bottom)
        # 创建零向量函数
        zero_vector = fem.Function(V_u)
        zero_vector.x.array[:] = 0.0
        bc_bottom = dirichletbc(zero_vector, dofs_bottom)
        bcs_u.append(bc_bottom)
        if rank == 0:
            print(f"底面边界 (107)：找到 {len(facets_bottom)} 条边，施加全固定。")

    # 2. 左侧法向约束（约束 x 方向位移）
    if MARKER_LEFT in unique_markers:
        facets_left = facet_tags.find(MARKER_LEFT)
        # 子空间：x 方向 (下标 0)
        dofs_left_x = fem.locate_dofs_topological(V_u.sub(0), fdim, facets_left)
        bc_left_x = dirichletbc(PETSc.ScalarType(0), dofs_left_x, V_u.sub(0))
        bcs_u.append(bc_left_x)
        if rank == 0:
            print(f"左侧边界 (103)：找到 {len(facets_left)} 条边，约束 x 方向。")

    # 3. 右侧法向约束（约束 x 方向位移）
    if MARKER_RIGHT in unique_markers:
        facets_right = facet_tags.find(MARKER_RIGHT)
        dofs_right_x = fem.locate_dofs_topological(V_u.sub(0), fdim, facets_right)
        bc_right_x = dirichletbc(PETSc.ScalarType(0), dofs_right_x, V_u.sub(0))
        bcs_u.append(bc_right_x)
        if rank == 0:
            print(f"右侧边界 (104)：找到 {len(facets_right)} 条边，约束 x 方向。")

    if MARKER_HOLE1 in unique_markers:
        facets_hole = facet_tags.find(MARKER_HOLE1)
        dofs_hole_x = fem.locate_dofs_topological(V_u.sub(0), fdim, facets_hole)
        bc_hole_x = dirichletbc(PETSc.ScalarType(0), dofs_hole_x, V_u.sub(0))
        bcs_u.append(bc_hole_x)
        if rank == 0:
            print(f"钻孔边界 (101)：找到 {len(facets_right)} 条边，约束 x 方向。")

    # 钻孔边界（106）自由，无需施加约束
    if rank == 0:
        print(f"总边界条件数：{len(bcs_u)}")
    
    # --- 定义弱形式 ---
    v_u = ufl.TestFunction(V_u)
    u_trial = ufl.TrialFunction(V_u)

    # 材料参数
    E = materials.E
    nu = materials.nu
    alpha = materials.alpha
    rho_s = materials.rho_s
    phi0 = materials.phi0

    # 弹性张量
    def sigma(u):
        epsilon = ufl.sym(ufl.grad(u))
        lambda_, mu = materials.get_lame_parameters()
        return lambda_ * ufl.tr(epsilon) * ufl.Identity(gdim) + 2 * mu * epsilon

    # 整体密度
    rho_bulk = (1 - phi0) * rho_s + phi0 * rho_w   # 饱和土体密度

    # 体力项
    body_force = rho_bulk * materials.g

    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags,
                        metadata={"quadrature_degree": 2})
    
    # 双线性形式（弹性）
    a = ufl.inner(sigma(u_trial), ufl.sym(ufl.grad(v_u))) * dx
    # 线性形式：压力项 + 体力项
    L = alpha * p_init * ufl.div(v_u) * dx + ufl.dot(body_force, v_u) * dx

    # 组装矩阵和向量
    A = petsc.assemble_matrix(fem.form(a), bcs=bcs_u)
    A.assemble()
    b = petsc.assemble_vector(fem.form(L))
    petsc.apply_lifting(b, [fem.form(a)], [bcs_u])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, bcs_u)

    # 创建求解器
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("gmres")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1e-10, atol=1e-10, max_it=1000)

    ksp.solve(b, u.x.petsc_vec)
    u.x.scatter_forward()

    # 检查解
    u_array = u.x.array
    if rank == 0:
        print(f"初始位移范围: min={u_array.min():.6e}, max={u_array.max():.6e}")
        if np.any(np.isnan(u_array)) or np.any(np.isinf(u_array)):
            print("错误：位移包含 NaN 或 Inf！")
        else:
            print("初始位移求解成功。")

    # 输出结果
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 创建 P1 向量空间用于输出
    P1_vec = basix.ufl.element("Lagrange", mesh.topology.cell_name(), 1, shape=(gdim,))
    V_u_P1 = fem.functionspace(mesh, P1_vec)
    u_P1 = fem.Function(V_u_P1)
    u_P1.interpolate(u)
    u_P1.name = "Displacement_initial"
    with io.XDMFFile(comm, output_dir / "initial_displacement.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u_P1)

    if rank == 0:
        print(f"初始位移已保存至 {output_dir}/initial_displacement.xdmf")

    ksp.destroy()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="计算初始状态位移")
    parser.add_argument("--config", type=str, default="config/grouting_config.yaml", help="配置文件路径")
    parser.add_argument("--mesh", type=str, default="meshes/foundation_drilling_2d.msh", help="网格文件路径")
    parser.add_argument("--output", type=str, default="results", help="输出目录")
    args = parser.parse_args()

    # 获取项目根目录
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    config_file = project_root / args.config
    mesh_file = project_root / args.mesh
    output_dir = project_root / args.output

    main(str(config_file), str(mesh_file), str(output_dir))