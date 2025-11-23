import gmsh
gmsh.initialize()# 初始化gmsh
membrane = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)# 创建一个圆盘
gmsh.model.occ.synchronize()# 同步几何模型,建立几何体和网格生成器之间的连接
gdim = 2
gmsh.model.addPhysicalGroup(gdim, [membrane], 1)
# 创建一个物理组,在有限元计算中,物理组用于标识网格的不同部分,如边界或子区域
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.05)# 设置网格的最小尺寸
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.05)# 设置网格的最大尺寸
gmsh.model.mesh.generate(gdim)# 生成网格

from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
domain, cell_tags, facet_tags = gmshio.model_to_mesh(
    gmsh.model, mesh_comm, gmsh_model_rank, gdim = gdim)
# 将gmsh模型转换为dolfinx网格

from dolfinx import fem
V = fem.functionspace(domain, ("Lagrange", 1))
# 创建一个一阶拉格朗日有限元函数空间

import ufl
from dolfinx import default_scalar_type
x = ufl.SpatialCoordinate(domain)# 定义空间坐标变量x
beta = fem.Constant(domain, default_scalar_type(12))# 定义常数beta=12
R0 = fem.Constant(domain, default_scalar_type(0.3))# 定义常数R0=0.3
p = 4 * ufl.exp(-beta **2 * (x[0]**2 + (x[1] - R0)**2))# 定义压力分布函数p

import numpy as np
def boundary(x):
     r = np.sqrt(x[0]**2 + x[1]**2)
     return np.isclose(r, 1)
boundary_dofs = fem.locate_dofs_geometrical(V, boundary)
bc = fem.dirichletbc(default_scalar_type(0), boundary_dofs, V)
# 定义边界条件,首先需要找到边界上的自由度,然后将这些自由度的值设定为0

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = p * v * ufl.dx
# 定义变分形式a(u,v)和线性形式L(v)
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
# 定义线性问题,指定边界条件和求解器选项
uh = problem.solve()# 求解线性问题,得到数值解uh

Q = fem.functionspace(domain, ("Lagrange", 5))
expr = fem.Expression(p, Q.element.interpolation_points())
pressure = fem.Function(Q)
pressure.interpolate(expr)# 将压力函数p插值到一个更高阶的函数空间Q中,以便于可视化
# 可视化数值解和压力分布

from dolfinx.plot import vtk_mesh
import pyvista
topology, cell_types, x = vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, x)

# Set deflection values and add it to plotter
grid.point_data["u"] = uh.x.array
warped = grid.warp_by_scalar("u", factor=25)

plotter = pyvista.Plotter()
plotter.add_mesh(warped, show_edges=True, show_scalar_bar=True, scalars="u")
plotter.scalar_bar.SetOrientationToVertical() # 垂直方向
plotter.scalar_bar.SetPosition(0.02, 0.25)  # 左侧位置
plotter.scalar_bar.SetPosition2(0.08, 0.5)  # 窄而高的颜色条
plotter.add_axes(line_width=3,x_color='red', y_color='green', z_color='blue')
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    plotter.screenshot("deflection.png")
# 可视化数值解


load_plotter = pyvista.Plotter()
p_grid = pyvista.UnstructuredGrid(*vtk_mesh(Q))# 获取更高阶函数空间Q的VTK表示
p_grid.point_data["p"] = pressure.x.array.real
warped_p = p_grid.warp_by_scalar("p", factor=0.5)
load_plotter.add_mesh(warped_p, show_scalar_bar=True, scalars = 'p')
load_plotter.scalar_bar.SetOrientationToVertical() # 垂直方向
load_plotter.scalar_bar.SetPosition(0.02, 0.25)  # 左侧位置
load_plotter.scalar_bar.SetPosition2(0.08, 0.5)  # 窄而高的颜色条
load_plotter.add_axes(line_width=3,x_color='red', y_color='green', z_color='blue')
load_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    load_plotter.show()
else:
    load_plotter.screenshot("load.png")
# 可视化压力分布