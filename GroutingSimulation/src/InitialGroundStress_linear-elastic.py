# 该脚本用于计算初始地应力场，假设土体为线弹性材料
# 暂时未考虑土层内部材料参数的不均匀性
# 平衡方程：div(sigma) + f = 0

from mpi4py import MPI
import numpy as np
from petsc4py import PETSc

from dolfinx import mesh, fem
import ufl
# mesh parameters
Lx, Ly, Lz = 3, 3, 3
nx, ny, nz = 60, 60, 30
cell_type = mesh.CellType.hexahedron

domain = mesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0],
                            [Lx, Ly, Lz]], [nx, ny, nz],
                         cell_type=cell_type)


E = 20e6  # Young's modulus
nu = 0.3   # Poisson's ratio
rhos_sat = 2700  # Soil density
rho_w = 1000  # Water density
g = 9.81  # Gravitational acceleration
gamma_sat = rhos_sat * g # Saturated unit weight

lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
mu_ = E / (2 * (1 + nu))
# set body force
f = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, -gamma_sat)))
# set water pressure
x = ufl.SpatialCoordinate(domain)
p_expression = rho_w * g * (Lz - x[2])
V = fem.functionspace(domain, ("CG", 1))
p = fem.Function(V)
p.interpolate(fem.Expression(p_expression, V.element.interpolation_points()))
# 既可以插值函数也可以插值表达式

import pyvista
from dolfinx import plot
p_grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))
p_grid.point_data["p[kPa]"] = p.x.array.real/1000  # kPa
plotter = pyvista.Plotter()
plotter.add_mesh(p_grid, show_edges=True, scalars="p[kPa]")
plotter.show()

tdim = domain.topology.dim

V_u = fem.functionspace(domain, ('CG', 1, (tdim,)))
# print(V_u.dofmap.index_map.size_local * V_u.dofmap.index_map_bs)

u = ufl.TrialFunction(V_u)
v = ufl.TestFunction(V_u)
u_sol = fem.Function(V_u)

def epsilon(u):
    return ufl.sym(ufl.grad(u))# ufl.sym表示取张量的对称部分，根据小应变假设

def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(tdim) + 2 * mu_ * epsilon(u)
# ufl.nabla_div表示散度运算,计算得到体积应变，正负号存疑
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx - p * ufl.div(v) * ufl.dx

tols = 1e-8
def bottom_facets(x):
    return np.isclose(x[2], 0.0, atol=tols)

fdim = domain.topology.dim - 1
facet_bottom = mesh.locate_entities_boundary(domain, fdim, bottom_facets)
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

bcs = [bc_bottom, bc_x0,  bc_xLx, bc_y0, bc_yLy]

# 组装刚度矩阵和载荷向量
from dolfinx.fem import petsc
A = petsc.assemble_matrix(fem.form(a), bcs=bcs)
A.assemble()

b = fem.petsc.assemble_vector(fem.form(L))
fem.petsc.apply_lifting(b, [fem.form(a)], [bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

# 应用边界条件到载荷向量
for bc in bcs:
    bc.set(b, u_sol.x.petsc_vec)

# 创建KSP求解器
ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A)

# 设置求解器参数
ksp.setType("cg")  # 使用共轭梯度法
ksp.getPC().setType("hypre")  # 使用hypre预条件子
ksp.setTolerances(rtol=1e-10, atol=1e-10, max_it=1000)

# 设置求解器监视器（可选）
ksp.setMonitor(lambda _, its, rnorm: print(f"迭代 {its}: 残差 = {rnorm:.2e}")) if MPI.COMM_WORLD.rank == 0 else None

# 求解系统
if MPI.COMM_WORLD.rank == 0:
    print("开始求解线性系统...")

ksp.solve(b, u_sol.x.petsc_vec)
u_sol.x.scatter_forward()

# 检查求解结果
converged_reason = ksp.getConvergedReason()
if converged_reason > 0:
    if MPI.COMM_WORLD.rank == 0:
        print(f"求解成功! 迭代次数: {ksp.getIterationNumber()}, 最终残差: {ksp.getResidualNorm():.2e}")
else:
    if MPI.COMM_WORLD.rank == 0:
        print(f"求解失败! 收敛原因代码: {converged_reason}")
        # 尝试使用直接求解器作为备选
        print("尝试使用直接求解器...")
        
        ksp_direct = PETSc.KSP().create(domain.comm)
        ksp_direct.setOperators(A)
        ksp_direct.setType("preonly")
        ksp_direct.getPC().setType("lu")
        ksp_direct.getPC().setFactorSolverType("mumps")
        
        ksp_direct.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        ksp_direct.destroy()

# 释放求解器资源
ksp.destroy()

if MPI.COMM_WORLD.rank == 0:
    print("位移场求解完成")

# 可视化位移场 - 按照 seepage.py 的绘图风格
try:
    import pyvista
    from dolfinx import plot
    
    # 设置 PyVista 参数
    pyvista.set_jupyter_backend("static")
    
    # 创建位移场的网格
    topology, cell_types, geometry = plot.vtk_mesh(V_u)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    
    # 获取位移数据
    displacement = u_sol.x.array.reshape(geometry.shape[0], 3)
    
    # 将位移数据添加到网格
    grid.point_data["Displacement"] = displacement
    grid.point_data["Displacement_X"] = displacement[:, 0]
    grid.point_data["Displacement_Y"] = displacement[:, 1] 
    grid.point_data["Displacement_Z"] = displacement[:, 2]
    
    # 单独显示 Z 方向位移（对于地应力分析最重要）
    plotter_z = pyvista.Plotter(window_size=[1200, 900])
    
    plotter_z.add_mesh(grid,
                      show_edges=True,
                      show_scalar_bar=True,
                      scalars="Displacement_Z",
                      cmap="coolwarm")
    
    # 设置颜色条样式
    plotter_z.scalar_bar.SetOrientationToVertical()
    plotter_z.scalar_bar.SetPosition(0.02, 0.25)
    plotter_z.scalar_bar.SetPosition2(0.08, 0.5)
    
    plotter_z.add_title("Z-Displacement [m]")
    plotter_z.add_axes()
    
    if not pyvista.OFF_SCREEN:
        plotter_z.show()
    else:
        plotter_z.screenshot("z_displacement.png") 
    
    # 打印位移统计信息
    if MPI.COMM_WORLD.rank == 0:
        print("\n位移统计信息:")
        print(f"X方向最大位移: {np.max(np.abs(displacement[:, 0])):.6e} m")
        print(f"Y方向最大位移: {np.max(np.abs(displacement[:, 1])):.6e} m") 
        print(f"Z方向最大位移: {np.max(displacement[:, 2]):.6e} m")
        print(f"Z方向最小位移: {np.min(displacement[:, 2]):.6e} m")
        
        # 计算体积变化（近似）
        original_volume = Lx * Ly * Lz
        # 注意：这只是近似计算，精确计算需要积分
        avg_vertical_strain = np.mean(displacement[:, 2]) / Lz
        volume_change = original_volume * avg_vertical_strain
        print(f"近似体积变化: {volume_change:.6e} m³")

except ImportError:
    if MPI.COMM_WORLD.rank == 0:
        print("PyVista 未安装，无法进行可视化")
        print("请安装: pip install pyvista")

# 绘制z方向位移变化曲线
try:
    import matplotlib.pyplot as plt
    import matplotlib
    
    # 设置英文字体
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['mathtext.fontset'] = 'dejavusans'
    
    if MPI.COMM_WORLD.rank == 0:
        # 获取所有节点的坐标和z方向位移
        topology, cell_types, geometry = plot.vtk_mesh(V_u)
        displacement = u_sol.x.array.reshape(geometry.shape[0], 3)
        z_displacement = displacement[:, 2]
        
        # 提取模型中心线 (x=Lx/2, y=Ly/2) 附近的节点
        center_x = Lx / 2
        center_y = Ly / 2
        tolerance = 0.15 * min(Lx/nx, Ly/ny)  # 基于网格尺寸的容差
        
        # 筛选靠近中心线的节点
        center_indices = []
        center_z_coords = []
        center_z_displacements = []
        
        for i in range(geometry.shape[0]):
            x, y, z = geometry[i]
            if (abs(x - center_x) < tolerance and 
                abs(y - center_y) < tolerance):
                center_indices.append(i)
                center_z_coords.append(z)
                center_z_displacements.append(z_displacement[i])
        
        # 按z坐标排序
        if center_z_coords:  # 确保列表不为空
            sorted_indices = np.argsort(center_z_coords)
            sorted_z_coords = np.array(center_z_coords)[sorted_indices]
            sorted_z_displacements = np.array(center_z_displacements)[sorted_indices]
        
            # 创建位移-深度曲线图
            plt.figure(figsize=(10, 8))
            
            # 主图：位移随深度变化
            plt.plot(sorted_z_displacements * 1000, sorted_z_coords, 'b-', linewidth=2, label='Z-displacement')
            plt.xlabel('Z-displacement (mm)')
            plt.ylabel('Depth (m)')
            plt.title('Z-displacement vs Depth')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 深度向下为正
            plt.gca().invert_yaxis()
            
            plt.tight_layout()

            plt.show()
            # 打印中心线位移统计
            print("\nCenterline Z-displacement Statistics:")
            print(f"Top displacement: {sorted_z_displacements[-1]*1000:.3f} mm")
            print(f"Bottom displacement: {sorted_z_displacements[0]*1000:.3f} mm")
            print(f"Max displacement: {np.max(sorted_z_displacements)*1000:.3f} mm")
            print(f"Min displacement: {np.min(sorted_z_displacements)*1000:.3f} mm")
            print(f"Displacement range: {(np.max(sorted_z_displacements)-np.min(sorted_z_displacements))*1000:.3f} mm")
            
            plt.savefig("z_displacement_vs_depth.png", dpi=300)
        
except ImportError:
    if MPI.COMM_WORLD.rank == 0:
        print("Matplotlib not installed, cannot plot displacement curves")
        print("Please install: pip install matplotlib")

"""
from dolfinx import io
import os

# 创建输出目录
output_dir = 'GroutingSimulation/results'
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# 保存位移场和网格
with io.XDMFFile(domain.comm, f"{output_dir}/displacement_field.xdmf", "w") as xdmf:
    # 写入网格信息
    xdmf.write_mesh(domain)
    # 写入位移场
    xdmf.write_function(u_sol)

if MPI.COMM_WORLD.rank == 0:
    print(f"位移场已保存到 {output_dir}/displacement_field.xdmf")
"""

