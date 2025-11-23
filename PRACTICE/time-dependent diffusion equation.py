import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

t = 0 # 初始时间
T = 1.0 # 结束时间
num_steps = 50 # 时间步数
dt = T / num_steps # 时间步长

nx, ny = 50, 50
domain = mesh.create_rectangle(MPI.COMM_WORLD,
                               [np.array([-2,-2]), np.array([2,2])],
                               [nx, ny],
                               mesh.CellType.triangle)
V = fem.functionspace(domain,("Lagrange",1))

def initial_condition(x, a=5):
    return np.exp(-a*(x[0]**2 + x[1]**2))

u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

def boundary(x):
    return np.isclose(x[0], -2) | np.isclose(x[0], 2) | np.isclose(x[1], -2) | np.isclose(x[1], 2)
dofs = fem.locate_dofs_geometrical(V, boundary)
bc = fem.dirichletbc(PETSc.ScalarType(0), dofs, V)

uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = fem.Constant(domain, PETSc.ScalarType(0))
a = u * v *ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (u_n + dt * f) * v * ufl.dx
# Assemble forms?
bilinear_form = fem.form(a)# 编译双线性形式
linear_form = fem.form(L)# 编译线性形式

A = assemble_matrix(bilinear_form, bcs=[bc])# 组装刚度矩阵
A.assemble()# 完成矩阵组装，保证矩阵同步
b = create_vector(linear_form)# 创建右端向量

solver = PETSc.KSP().create(domain.comm)# 创建线性求解器
solver.setOperators(A)# 设置线性系统矩阵
solver.setType(PETSc.KSP.Type.PREONLY)# 设置求解器类型为直接求解（精度高，计算量大）
solver.getPC().setType(PETSc.PC.Type.LU)# 设置预处理器类型为LU分解

grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))  # 创建 PyVista 网格用于可视化

plotter = pyvista.Plotter()# 创建 PyVista 绘图对象
plotter.open_gif('u_time.gif', fps = 10)# 打开 GIF 文件以保存动画，设置输出为gif动画文件

grid.point_data["uh"] = uh.x.array.real # 将初始解数据添加到网格点数据中
warped = grid.warp_by_scalar("uh", factor=1.0) # 根据标量场变形网格以增强可视化效果

viridis = mpl.colormaps.get_cmap("viridis").resampled(25)# 从 matplotlib 获取 "viridis" 颜色映射，重新采样为 25 个离散颜色级别
sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2f",
             color="black", position_x=0.1, position_y=0.8,
             width=0.8, height=0.1) # 精细控制颜色条（标量条）的显示
renderer  = plotter.add_mesh(warped, show_edges=True, lighting=False,cmap=viridis, 
                             scalar_bar_args=sargs, clim=[0, max(uh.x.array.real)])
# 将变形网格添加到绘图对象中，设置显示参数，包括颜色映射和标量条

for i in range(num_steps):
    t += dt

    with b.localForm() as loc_b:
        loc_b.set(0)
        assemble_vector(b, linear_form)# 组装右端向量

    apply_lifting(b, [bilinear_form], bcs=[[bc]])# 应用提升以考虑边界条件对内部节点的影响，修正内部节点方程，使之符合物理意义
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)# 更新幽灵单元数据以确保并行一致性
    set_bc(b, [bc])# 应用边界条件到右端向量

    solver.solve(b, uh.x.petsc_vec)# 求解线性系统，得到当前时间步的解，uh的初始值会被覆盖
    uh.x.scatter_forward()# 确保解在所有处理器间同步

    u_n.x.array[:] = uh.x.array# 将当前解更新为下一时间步的初始条件

    new_warped = grid.warp_by_scalar("uh", factor=1.0)#创建变形网格
    warped.points[:, :] = new_warped.points# 更新变形网格的点坐标
    warped.point_data["uh"][:] = uh.x.array#.real # 更新变形网格的标量数据
    plotter.write_frame()# 写入当前帧到 GIF 文件
plotter.close()# 关闭绘图对象，完成 GIF 文件的写入

A.destroy()
b.destroy()
solver.destroy()
# 释放 PETSc 资源