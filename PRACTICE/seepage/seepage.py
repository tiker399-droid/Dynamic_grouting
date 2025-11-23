# 地基灌浆模拟：纯渗流分析
# 使用 FEniCSx (v0.7.2+) 
# 控制方程: ∇·(-(K/μ)∇P) = Q

from mpi4py import MPI
import numpy as np
from dolfinx import fem, plot
from dolfinx.fem import functionspace, Function, Constant, dirichletbc, Expression
from dolfinx.mesh import create_rectangle, CellType
from ufl import TrialFunction, TestFunction, inner, grad, dx

# 参数设置
Lx, Ly = 10.0, 10.0         # 模型尺寸 (m)
Nx, Ny = 10, 10             # 网格密度
injection_rate = 1e-3       # 注浆速率 (m³/s)
permeability = 1e-6         # 渗透率 (m²)
viscosity = 0.1             # 浆液粘度 (Pa·s)
injection_radius = 0.6      # 注浆区域半径 (m)

def main():
    comm = MPI.COMM_WORLD
    domain = create_rectangle(comm, 
                             [np.array([0.0, 0.0]), np.array([Lx, Ly])], 
                             [Nx, Ny], 
                             CellType.triangle)
    
    V = functionspace(domain, ("Lagrange", 1))
    
    def left_boundary(x):
        return np.isclose(x[0], 0.0)
    
    def right_boundary(x):
        return np.isclose(x[0], Lx)
    
    def top_boundary(x):
        return np.isclose(x[1], Ly)
    
    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)
    
    dofs_left = fem.locate_dofs_geometrical(V, left_boundary)
    dofs_right = fem.locate_dofs_geometrical(V, right_boundary)
    dofs_top = fem.locate_dofs_geometrical(V, top_boundary)
    dofs_bottom = fem.locate_dofs_geometrical(V, bottom_boundary)

    from dolfinx import default_scalar_type
    bc_left = dirichletbc(default_scalar_type(0), dofs_left, V)
    bc_right = dirichletbc(default_scalar_type(0), dofs_right, V)
    bc_top = dirichletbc(default_scalar_type(0), dofs_top, V)
    bc_bottom = dirichletbc(default_scalar_type(0), dofs_bottom, V)
    # bcs = [bc_left, bc_right] # 左右边界自由排水
    bcs = [bc_top]    
    # bcs = [bc_left, bc_right, bc_top, bc_bottom] # 上边界、左右边界自由排水，下边界静水压强，有问题
    # bcs = [bc_bottom, bc_top] # 下边界静水压强, 上边界自由排水，有问题
    Q = Function(V)

    def source_function(x):
        values = np.zeros(x.shape[1])
        center = np.array([Lx/2, Ly/2, 0.0])
        for i in range(x.shape[1]):
            dist = np.linalg.norm(x[0:2, i] - center[0:2])
            if dist < injection_radius:
                area = np.pi * injection_radius**2
                values[i] = injection_rate / area
        return values

    Q.interpolate(source_function)
    
    P = TrialFunction(V)  # 试验函数 (压力)
    v = TestFunction(V)   # 测试函数
    
    K = Constant(domain, permeability)    # 渗透率
    mu = Constant(domain, viscosity)      # 粘度
    
    a = (K / mu) * inner(grad(P), grad(v)) * dx
    L = Q * v * dx
    
    from dolfinx.fem.petsc import LinearProblem
    problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    pressure = problem.solve()
    print("渗流问题求解完成")
    
    P_array = pressure.x.array
    if comm.rank == 0:
        all_P_arrays = comm.gather(P_array, root=0)
        P_array = np.concatenate(all_P_arrays)
        print(f"压力场统计:")
        print(f"  最大值: {np.max(P_array):.2e} Pa")
        print(f"  最小值: {np.min(P_array):.2e} Pa") 
        print(f"  平均值: {np.mean(P_array):.2e} Pa")
        dof_coords = V.tabulate_dof_coordinates()
    
    source_center_array = np.array([Lx/2, Ly/2])
    distances = np.linalg.norm(dof_coords[:, :2] - source_center_array, axis=1)
    closest_dof_idx = np.argmin(distances)# 找到距离注浆点最近的自由度索引
    
    if distances[closest_dof_idx] < 1e-10:  # 精确匹配
        injection_pressure = P_array[closest_dof_idx]
        print(f"  注浆点压力: {injection_pressure:.2e} Pa")
    else:
        print(f"  注浆点不是自由度节点，最近点距离: {distances[closest_dof_idx]:.2e} m")
        print(f"  最近点压力: {P_array[closest_dof_idx]:.2e} Pa")
    
    # 可视化
    try:
        import pyvista
        pyvista.set_jupyter_backend("static")
        
        # 创建绘图器
        topology, cell_types, geometry = plot.vtk_mesh(domain, domain.topology.dim)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        
        # 将压力场添加到网格
        grid.point_data["Pressure"] = pressure.x.array.real
        grid.set_active_scalars("Pressure")
        # warped = grid.warp_by_scalar("Pressure", factor=1.0)
        
        # 创建绘图窗口
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True, scalars="Pressure")
        plotter.scalar_bar.SetOrientationToVertical() # 垂直方向
        plotter.scalar_bar.SetPosition(0.02, 0.25)  # 左侧位置
        plotter.scalar_bar.SetPosition2(0.08, 0.5)  # 窄而高的颜色条
        plotter.add_title("Pressure Distribution[Pa]")
        plotter.add_axes()
        plotter.view_xy()
        if not pyvista.OFF_SCREEN:
            plotter.show()
        else:
            plotter.screenshot("pressure_distribution.png") 
    except ImportError:
        if comm.rank == 0:
            print("PyVista 未安装，跳过可视化")
    return pressure

if __name__ == "__main__":
    pressure = main()

# 下一阶段，加入时间项，模拟非稳态渗流过程
# 速度场