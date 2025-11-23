# 地基灌浆模拟：纯渗流分析（带圆孔网格）- 修正版
# 使用 FEniCSx (v0.7.2+)
# 控制方程: ∇·(-(K/μ)∇P) = 0，在圆孔边界上施加压力边界条件
from mpi4py import MPI
import numpy as np
from dolfinx import fem, plot
import dolfinx.geometry
from dolfinx.fem import functionspace, Function, Constant, dirichletbc, petsc
from ufl import TrialFunction, TestFunction, inner, grad, dx
import gmsh
from dolfinx.io import gmshio

# 参数设置
Lx, Ly = 10.0, 10.0         # 模型尺寸 (m)
injection_pressure = 3e5    # 注浆压力 (Pa)
permeability = 1e-11        # 渗透率 (m²)
viscosity = 0.01            # 浆液粘度 (Pa·s)
density = 1800.0            # 浆液密度 (kg/m³)
a_g = 9.81                  # 重力加速度 (m/s²)
injection_radius = 0.3      # 注浆孔半径 (m)
mesh_resolution = 1         # 网格分辨率
refinement_radius = 3.0     # 细化区域半径
refinement_factor = 6.0     # 细化因子

def create_domain_with_hole(comm):
    """创建带圆孔的正方形网格，并在圆孔附近细化网格"""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    
    # 创建正方形
    rect = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
    
    # 创建圆孔
    center_x, center_y = Lx/2, Ly/2
    circle = gmsh.model.occ.addDisk(center_x, center_y, 0, injection_radius, injection_radius)
    
    # 从正方形中剪掉圆孔
    gmsh.model.occ.cut([(2, rect)], [(2, circle)])
    
    gmsh.model.occ.synchronize()
    
    # 标记边界
    boundaries = gmsh.model.getBoundary([(2, rect)], oriented=False)
    
    outer_boundaries = []
    inner_boundaries = []
    
    for dimTag in boundaries:
        dim, tag = dimTag
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        if np.isclose(com[0], Lx/2, atol=0.1) and np.isclose(com[1], Ly/2, atol=0.1):
            inner_boundaries.append(tag)
        else:
            outer_boundaries.append(tag)
    
    gmsh.model.addPhysicalGroup(1, outer_boundaries, 1)  # 外边界
    gmsh.model.addPhysicalGroup(1, inner_boundaries, 2)  # 内边界（圆孔）
    gmsh.model.addPhysicalGroup(2, [rect], 3)  # 面
    
    # 设置基础网格尺寸
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_resolution)
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_resolution/10)
    
    # 在圆孔附近创建细化区域
    # 方法1: 使用距离场进行渐变细化
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", inner_boundaries)
    
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", mesh_resolution/refinement_factor)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", mesh_resolution)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", injection_radius)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", refinement_radius)

    # 设置背景网格场
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
    
    # 生成网格
    gmsh.model.mesh.generate(2)
    
    # 导入到dolfinx
    domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    
    gmsh.finalize()
    
    return domain, facet_markers

def main():
    comm = MPI.COMM_WORLD
    
    # 创建带圆孔的网格
    if comm.rank == 0:
        print("创建带圆孔的网格...")
    domain, facet_markers = create_domain_with_hole(comm)
    
    V = functionspace(domain, ("Lagrange", 1))
    
    # 定义边界条件
    bcs = []
    def boundary(x):
        return np.isclose(x[1], Ly) | np.isclose(x[1], 0) | np.isclose(x[0], 0) | np.isclose(x[0], Lx)
    dofs_outer = fem.locate_dofs_geometrical(V, boundary)
    from dolfinx import default_scalar_type
    bc = dirichletbc(default_scalar_type(0), dofs_outer, V)
    bcs.append(bc)

    # 内边界（圆孔，标记为2）设置为注浆压力
    facets_inner = facet_markers.find(2)
    dofs_inner = fem.locate_dofs_topological(V, domain.topology.dim-1, facets_inner)# 圆孔边界自由度
    if len(dofs_inner) > 0:
        bc_inner = dirichletbc(default_scalar_type(injection_pressure), dofs_inner, V)
        bcs.append(bc_inner)
    

    if comm.rank == 0:
        print(f"边界条件数量: {len(bcs)}")
        print(f"外边界自由度: {len(dofs_outer)}")
        print(f"内边界自由度: {len(dofs_inner)}")
    
    # 变分形式
    P = TrialFunction(V)  # 试验函数 (压力)
    v = TestFunction(V)   # 测试函数
    
    K = Constant(domain, permeability)    # 渗透率
    mu = Constant(domain, viscosity)      # 粘度
    rho = Constant(domain, density)        # 流体密度 [kg/m³]
    Q = Constant(domain, 0.0)           # 内源项（无内源）
    
    # 控制方程: ∇·(-(K/μ)∇P) =  0 (无内源)
    a = (K/mu) * inner(grad(P), grad(v)) * dx
    L = Q * v * dx

    from dolfinx.fem.petsc import LinearProblem
    problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    pressure = problem.solve()
    

    if comm.rank == 0:
        print("渗流问题求解完成")
        
        P_array = pressure.x.array
        print(f"压力场统计:")
        print(f"  最大值: {np.max(P_array):.2e} Pa")
        print(f"  最小值: {np.min(P_array):.2e} Pa") 
        print(f"  平均值: {np.mean(P_array):.2e} Pa")
        
        # 计算圆孔边界上的平均压力
        dof_coords = V.tabulate_dof_coordinates()
        center = np.array([Lx/2, Ly/2])
        distances = np.linalg.norm(dof_coords[:, :2] - center, axis=1)
        hole_dofs = distances < injection_radius * 1.1  # 圆孔附近的自由度
        
        if np.any(hole_dofs):
            hole_pressures = P_array[hole_dofs]
            avg_hole_pressure = np.mean(hole_pressures)
            max_hole_pressure = np.max(hole_pressures)
            min_hole_pressure = np.min(hole_pressures)
            print(f"  圆孔边界压力统计:")
            print(f"    最大值: {max_hole_pressure:.2e} Pa")
            print(f"    最小值: {min_hole_pressure:.2e} Pa")
            print(f"    平均值: {avg_hole_pressure:.2e} Pa")

    V_vec = functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))# 向量函数空间
    velocity_expr = - (K / mu) * grad(pressure)# 达西速度表达式
    velocity = fem.Expression(velocity_expr, V_vec.element.interpolation_points())# 插值速度场
    velocity_function = Function(V_vec)# 速度函数
    velocity_function.interpolate(velocity)# 插值速度场到函数
    if comm.rank == 0:
        print("速度场计算完成")
    
        # 计算速度场统计
        velocity_array = velocity_function.x.array.reshape(-1, domain.geometry.dim)
        velocity_magnitude = np.linalg.norm(velocity_array, axis=1)
    
        print(f"速度场统计:")
        print(f"  最大速度: {np.max(velocity_magnitude):.2e} m/s")
        print(f"  最小速度: {np.min(velocity_magnitude):.2e} m/s")
        print(f"  平均速度: {np.mean(velocity_magnitude):.2e} m/s")
    
        # 计算圆孔边界上的平均速度
        if np.any(hole_dofs):
            hole_velocities = velocity_magnitude[hole_dofs]
            avg_hole_velocity = np.mean(hole_velocities)
            max_hole_velocity = np.max(hole_velocities)
            min_hole_velocity = np.min(hole_velocities)
            print(f"  圆孔边界速度统计:")
            print(f"    最大值: {max_hole_velocity:.2e} m/s")
            print(f"    最小值: {min_hole_velocity:.2e} m/s")
            print(f"    平均值: {avg_hole_velocity:.2e} m/s")
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
        
        # 创建绘图窗口
        plotter1 = pyvista.Plotter()
        plotter1.add_mesh(grid, show_edges=True, show_scalar_bar=True, scalars="Pressure")
        plotter1.add_title(f"Pressure Distribution[Pa] (P0 = {injection_pressure:.1e} Pa)")
        plotter1.add_axes()
        plotter1.view_xy()
        
        topology_vec, cell_types_vec, geometry_vec = plot.vtk_mesh(V_vec)
        grid_vec = pyvista.UnstructuredGrid(topology_vec, cell_types_vec, geometry_vec)
        velocity_data = velocity_function.x.array.reshape(len(grid_vec.points), 2)
        velocity_data_3d = np.zeros((len(grid_vec.points), 3))
        velocity_data_3d[:, :2] = velocity_data
        grid_vec["Velocity"] = velocity_data_3d * 1e2
        grid_vec["Velocity_Magnitude"] = np.linalg.norm(velocity_data_3d, axis=1)
        plotter2 = pyvista.Plotter()
        plotter2.add_mesh(grid, show_edges=True, show_scalar_bar=True, scalars="Pressure")
        print(grid_vec.points.shape, grid_vec["Velocity"].shape)
        plotter2.add_arrows(grid_vec.points, grid_vec["Velocity"], mag=10.0, color="red")
        plotter2.add_title("Velocity Field with Pressure Background")
        plotter2.add_axes()
        plotter2.view_xy()

        if not pyvista.OFF_SCREEN:
            plotter1.show()
            plotter2.show()
        else:
            plotter1.screenshot("pressure_distribution_with_hole.png")
            plotter2.screenshot("velocity_field_with_pressure.png")
            
    except ImportError:
        if comm.rank == 0:
            print("PyVista 未安装，跳过可视化")
    try:
        import matplotlib.pyplot as plt
        
        # 获取注浆中心坐标
        center_x, center_y = Lx/2, Ly/2
        
        # 方法1: 使用网格点数据直接提取
        # 获取网格点坐标和对应的压力值
        points = domain.geometry.x
        pressures = pressure.x.array
        
        # 提取x方向数据 (y = center_y 附近)
        y_tolerance = 0.1  # 容忍度
        x_mask = np.abs(points[:, 1] - center_y) < y_tolerance
        x_points = points[x_mask, 0]
        x_pressures = pressures[x_mask]
        
        # 提取y方向数据 (x = center_x 附近)
        x_tolerance = 0.1  # 容忍度
        y_mask = np.abs(points[:, 0] - center_x) < x_tolerance
        y_points = points[y_mask, 1]
        y_pressures = pressures[y_mask]
        
        # 对数据进行排序
        x_indices = np.argsort(x_points)
        x_points_sorted = x_points[x_indices]
        x_pressures_sorted = x_pressures[x_indices]
        
        y_indices = np.argsort(y_points)
        y_points_sorted = y_points[y_indices]
        y_pressures_sorted = y_pressures[y_indices]
        
        # 计算相对于中心的距离
        x_distances = x_points_sorted - center_x
        y_distances = y_points_sorted - center_y
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 绘制x方向压力分布
        ax1.plot(x_distances, x_pressures_sorted, 'b-', linewidth=2, marker='o', markersize=3)
        ax1.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='grouting center')
        ax1.axvline(x=-injection_radius, color='g', linestyle=':', alpha=0.7, label='grouting hole boundary')
        ax1.axvline(x=injection_radius, color='g', linestyle=':', alpha=0.7)
        ax1.set_xlabel('distance to the center (m)')
        ax1.set_ylabel('pressure (Pa)')
        ax1.set_title('X pressure Distribution')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 绘制y方向压力分布
        ax2.plot(y_distances, y_pressures_sorted, 'r-', linewidth=2, marker='s', markersize=3)
        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='grouting center')
        ax2.axvline(x=-injection_radius, color='g', linestyle=':', alpha=0.7, label='grouting hole boundary')
        ax2.axvline(x=injection_radius, color='g', linestyle=':', alpha=0.7)
        ax2.set_xlabel('distance to the center (m)')
        ax2.set_ylabel('pressure (Pa)')
        ax2.set_title('Y pressure Distribution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # 保存图像
        plt.savefig('pressure_profiles.png', dpi=300, bbox_inches='tight')
        
        if comm.rank == 0:
            print("压力分布曲线已保存为 'pressure_profiles.png'")
            
            # 显示图像（如果不在无头模式下）
            import os
            if 'DISPLAY' in os.environ:
                plt.show()
            else:
                print("无显示设备，图像已保存但未显示")
                
    except ImportError:
        if comm.rank == 0:
            print("Matplotlib 未安装，跳过压力分布曲线绘制")
    except Exception as e:
        if comm.rank == 0:
            print(f"绘制压力分布曲线时出错: {e}")
    return pressure, velocity_function, domain

if __name__ == "__main__":
    pressure, velocity_function, domain = main()