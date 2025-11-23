from mpi4py import MPI
# 并行计算通信标准
from dolfinx import mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD,8,8,mesh.CellType.quadrilateral)
# 创建一个单元为四边形的单位正方形网格，划分为8x8个单元，参数分别为(通信器, x方向单元数, y方向单元数, 单元类型)

from dolfinx.fem import functionspace
V = functionspace(domain,("Lagrange",1))
# 创建一个一阶拉格朗日有限元函数空间，自由度为确定有限元函数所需要的独立参数，在这里为顶点的值

from dolfinx import fem
uD = fem.Function(V)
uD.interpolate(lambda x: 1 +x[0]**2 + 2*x[1]**2)
# 定义一个函数uD，并将其插值为边界条件函数uD=1+x^2+2y^2,“边界”体现在下一步

import numpy
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology) # 获取边界面的索引
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets) # 获取边界面的自由度索引
bc = fem.dirichletbc(uD, boundary_dofs)
# 定义边界条件，首先需要找到边界上的自由度，然后将这些自由度的值设定为uD

import ufl
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
# 定义试探函数和检验函数

from dolfinx import default_scalar_type
f = fem.Constant(domain, default_scalar_type(-6))
# 定义右端项函数f=-6

a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx
# 定义变分形式a(u,v)和线性形式L(v)

from dolfinx.fem.petsc import LinearProblem
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}) # 定义线性问题，指定边界条件和求解器选项
uh = problem.solve()
# 求解线性问题，得到数值解uh

V2 = fem.functionspace(domain, ("Lagrange", 2))
uex = fem.Function(V2)
uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
# 定义解析解函数uex，并将其插值为解析解uex=1+x^2+2y^2

L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx) # 定义L2误差的变分形式，参数为数值解与解析解之差的平方*
error_local = fem.assemble_scalar(L2_error) # 计算局部误差
error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
# 计算L2误差，首先计算局部误差，然后通过MPI通信将所有进程的误差汇总，最后取平方根得到L2范数

error_max = numpy.max(numpy.abs(uD.x.array-uh.x.array))
# Only print the error on one process
if domain.comm.rank == 0:
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")
# 输出L2误差和最大误差

import pyvista
print(pyvista.global_theme.jupyter_backend)# 检查PyVista的Jupyter后端设置
from dolfinx import plot
domain.topology.create_connectivity(tdim, tdim)# 创建单元之间的连接性
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)# 获取网格的VTK表示，包括拓扑结构、单元类型和几何信息
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# 创建用于可视化的网格数据

plotter = pyvista.Plotter()# 创建一个PyVista绘图对象
plotter.add_mesh(grid, show_edges=True)# 将网格添加到绘图对象中，并显示边缘
plotter.view_xy()# 设置视图为XY平面
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("fundamentals_mesh.png")# 如果不是离屏渲染，则显示绘图，否则保存为图片
# 可视化网格

u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)# 获取函数空间V的VTK表示，包括拓扑结构、单元类型和几何信息
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)# 创建用于可视化的函数空间数据
u_grid.point_data["u"] = uh.x.array.real# 将数值解uh的值赋给网格的点数据，键名为"u"
u_grid.set_active_scalars("u")# 设置活动标量为"u"
u_plotter = pyvista.Plotter()# 创建另一个PyVista绘图对象
u_plotter.add_mesh(u_grid, show_edges=True)# 将函数空间网格添加到绘图对象中，并显示边缘
u_plotter.view_xy()# 设置视图为XY平面
if not pyvista.OFF_SCREEN:
    u_plotter.show()
# 可视化数值解

warped = u_grid.warp_by_scalar()# 根据标量值对网格进行变形，以更直观地显示数值解的变化
plotter2 = pyvista.Plotter()# 创建第三个PyVista绘图对象
plotter2.add_mesh(warped, show_edges=True, show_scalar_bar=True)# 将变形后的网格添加到绘图对象中，并显示边缘和标量条
if not pyvista.OFF_SCREEN:
    plotter2.show()
# 可视化变形后的数值解