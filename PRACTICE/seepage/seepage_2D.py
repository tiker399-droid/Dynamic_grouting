# 地基灌浆模拟：纯渗流分析（带垂直钻孔网格）- 修正版
# 使用 FEniCSx (v0.7.2+)
# 控制方程: ∇·(-(k/μ)∇P+ρg∇z) = 0，在钻孔底部边界上施加压力边界条件
from mpi4py import MPI
import numpy as np
from dolfinx import fem, plot
from dolfinx.fem import functionspace, Function, Constant, dirichletbc, petsc
from ufl import TrialFunction, TestFunction, inner, grad, dx
import gmsh
from dolfinx.io import gmshio

# 参数设置
Lx, Ly = 20.0, 10.0         # 模型尺寸 (m)
injection_pressure = 3e5    # 注浆压力 (Pa)
permeability = 1e-11        # 渗透率 (m²)
viscosity = 0.01            # 浆液粘度 (Pa·s)
density = 1800.0            # 浆液密度 (kg/m³)
a_g = 9.81                  # 重力加速度 (m/s²)
borehole_width = 0.3        # 钻孔宽度 (m)
borehole_depth = 2.0        # 钻孔深度 (从顶部到底部的深度)
mesh_resolution = 1         # 网格分辨率
refinement_factor = 6.0     # 细化因子

def create_domain_with_vertical_borehole(comm):
    """创建带垂直钻孔的网格，钻孔从地面延伸到钻孔深度"""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    
    # 创建主区域
    rect = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
    
    # 创建垂直钻孔 (从顶部延伸到中心)
    center_x = Lx/2
    borehole_top = Ly  # 钻孔顶部在地面
    borehole_bottom = Ly - borehole_depth  # 钻孔底部在模型中心
    borehole_height = borehole_top - borehole_bottom  # 钻孔高度
    
    borehole_x = center_x - borehole_width/2
    borehole = gmsh.model.occ.addRectangle(borehole_x, borehole_bottom, 0, borehole_width, borehole_height)
    
    # 从主区域中剪掉钻孔
    gmsh.model.occ.cut([(2, rect)], [(2, borehole)])
    
    gmsh.model.occ.synchronize()
    
    # 标记边界
    boundaries = gmsh.model.getBoundary([(2, rect)], oriented=False)
    
    outer_boundaries = []
    borehole_side_walls = []  # 钻孔侧壁
    borehole_bottom_boundary = []  # 钻孔底部
    
    for dimTag in boundaries:
        dim, tag = dimTag
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        
        # 检查是否在钻孔区域内
        if (borehole_x <= com[0] <= borehole_x + borehole_width and 
            borehole_bottom <= com[1] <= borehole_top):
            
            # 检查是否是钻孔底部
            if np.isclose(com[1], borehole_bottom, atol=0.1):
                borehole_bottom_boundary.append(tag)
            # 检查是否是钻孔侧壁
            elif np.isclose(com[0], borehole_x, atol=0.1) or np.isclose(com[0], borehole_x + borehole_width, atol=0.1):
                borehole_side_walls.append(tag)
            # 钻孔顶部已经被挖空，不需要标记
        else:
            outer_boundaries.append(tag)
    
    gmsh.model.addPhysicalGroup(1, outer_boundaries, 1)  # 外边界
    gmsh.model.addPhysicalGroup(1, borehole_side_walls, 2)  # 钻孔侧壁（不透水边界）
    gmsh.model.addPhysicalGroup(1, borehole_bottom_boundary, 3)  # 钻孔底部（注浆压力边界）
    gmsh.model.addPhysicalGroup(2, [rect], 4)  # 面
    
    # 设置基础网格尺寸
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_resolution)
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_resolution/10)
    
    # 在钻孔附近创建细化区域
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", borehole_side_walls + borehole_bottom_boundary)
    
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", mesh_resolution/refinement_factor)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", mesh_resolution)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", borehole_width/2)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", borehole_width*2)

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
    
    # 创建带垂直钻孔的网格
    if comm.rank == 0:
        print("创建带垂直钻孔的网格...")
        print(f"钻孔尺寸: {borehole_width} m x {borehole_depth} m")
        print(f"钻孔从地面(y={Ly})延伸到中心(y={Ly - borehole_depth})")
    domain, facet_markers = create_domain_with_vertical_borehole(comm)
    
    V = functionspace(domain, ("Lagrange", 1))
    
    # 定义边界条件
    bcs = []
    
    # 地面边界（不包括钻孔顶部，因为已经被挖空）设置为0压力（大气压力）
    def ground_boundary(x):
        # x 是一个形状为 (3, n) 的数组，其中每一列是一个点的坐标
        # 排除钻孔区域
        center_x = Lx/2
        # 检查点是否在地面上且不在钻孔区域内
        on_ground = np.isclose(x[1], Ly)
        in_borehole_x = (center_x - borehole_width/2 <= x[0]) & (x[0] <= center_x + borehole_width/2)
        in_borehole = in_borehole_x & on_ground
        return on_ground & ~in_borehole
    
    dofs_ground = fem.locate_dofs_geometrical(V, ground_boundary)
    from dolfinx import default_scalar_type
    if len(dofs_ground) > 0:
        bc_ground = dirichletbc(default_scalar_type(0), dofs_ground, V)
        bcs.append(bc_ground)
    
    # 钻孔底部（标记为3）设置为注浆压力
    facets_borehole_bottom = facet_markers.find(3)
    dofs_borehole_bottom = fem.locate_dofs_topological(V, domain.topology.dim-1, facets_borehole_bottom)
    if len(dofs_borehole_bottom) > 0:
        bc_borehole = dirichletbc(default_scalar_type(injection_pressure), dofs_borehole_bottom, V)
        bcs.append(bc_borehole)
    
    # 钻孔侧壁（标记为2）为不透水边界，不需要设置狄利克雷边界条件
    # 在弱形式中，自然边界条件会自动处理
    
    if comm.rank == 0:
        print(f"边界条件数量: {len(bcs)}")
        print(f"地面边界自由度: {len(dofs_ground)}")
        print(f"钻孔底部边界自由度: {len(dofs_borehole_bottom)}")
    
    # 变分形式
    P = TrialFunction(V)  # 试验函数 (压力)
    v = TestFunction(V)   # 测试函数
    
    K = Constant(domain, permeability)    # 渗透率
    mu = Constant(domain, viscosity)      # 粘度
    rho = Constant(domain, density)        # 流体密度 [kg/m³]
    g = Constant(domain, a_g)            # 重力加速度 [m/s²]
    grad_z = Constant(domain, (0.0, 1.0))  # z方向梯度（二维中为y方向）
    
    # 控制方程: ∇·(-(K/μ)∇P) =  0 (无内源)
    # 钻孔侧壁为不透水边界，自然边界条件会自动满足
    a = (K/mu) * inner(grad(P), grad(v)) * dx
    L = - (K/mu) * rho * g * inner(grad_z, grad(v)) * dx

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
        
        # 计算钻孔底部边界上的平均压力
        borehole_bottom_x = Lx/2
        borehole_bottom_y = Ly - borehole_depth
        dof_coords = V.tabulate_dof_coordinates()
        
        # 找到钻孔底部附近的自由度
        bottom_tolerance = 0.1
        bottom_mask = (
            (np.abs(dof_coords[:, 0] - borehole_bottom_x) < borehole_width/2 + bottom_tolerance) & 
            (np.abs(dof_coords[:, 1] - borehole_bottom_y) < bottom_tolerance)
        )
        
        if np.any(bottom_mask):
            bottom_pressures = P_array[bottom_mask]
            avg_bottom_pressure = np.mean(bottom_pressures)
            max_bottom_pressure = np.max(bottom_pressures)
            min_bottom_pressure = np.min(bottom_pressures)
            print(f"  钻孔底部压力统计:")
            print(f"    最大值: {max_bottom_pressure:.2e} Pa")
            print(f"    最小值: {min_bottom_pressure:.2e} Pa")
            print(f"    平均值: {avg_bottom_pressure:.2e} Pa")

    V_vec = functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))# 向量函数空间
    velocity_expr = - (K / mu) * (grad(pressure) + rho * g * grad_z)# 达西速度表达式
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
    
        # 计算钻孔底部边界上的平均速度
        if np.any(bottom_mask):
            bottom_velocities = velocity_magnitude[bottom_mask]
            avg_bottom_velocity = np.mean(bottom_velocities)
            max_bottom_velocity = np.max(bottom_velocities)
            min_bottom_velocity = np.min(bottom_velocities)
            print(f"  钻孔底部速度统计:")
            print(f"    最大值: {max_bottom_velocity:.2e} m/s")
            print(f"    最小值: {min_bottom_velocity:.2e} m/s")
            print(f"    平均值: {avg_bottom_velocity:.2e} m/s")
    
    # 绘制压力分布曲线
    try:
        import matplotlib.pyplot as plt
        
        # 获取注浆中心坐标 (钻孔底部中心)
        center_x, center_y = Lx/2, Ly - borehole_depth
        
        # 方法: 使用网格点数据直接提取
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
        x_pressures_sorted = x_pressures[x_indices] / 1000  # 转换为kPa
        
        y_indices = np.argsort(y_points)
        y_points_sorted = y_points[y_indices]
        y_pressures_sorted = y_pressures[y_indices] / 1000  # 转换为kPa
        
        # 计算相对于中心的距离
        x_distances = x_points_sorted - center_x
        y_distances = y_points_sorted - center_y
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 绘制x方向压力分布
        ax1.plot(x_distances, x_pressures_sorted, 'b-', linewidth=2, marker='o', markersize=3)
        ax1.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='grouting center')
        ax1.axvline(x=-borehole_width/2, color='g', linestyle=':', alpha=0.7, label='borehole boundary')
        ax1.axvline(x=borehole_width/2, color='g', linestyle=':', alpha=0.7)
        ax1.set_xlabel('the distance to the center(m)')
        ax1.set_ylabel('pressure (kPa)')
        ax1.set_title('pressure Distribution in X direction')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 绘制y方向压力分布
        ax2.plot(y_distances, y_pressures_sorted, 'r-', linewidth=2, marker='s', markersize=3)
        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='grouting center')
        # ax2.axvline(x=-borehole_depth/2, color='g', linestyle=':', alpha=0.7, label='borehole bottom')
        # ax2.axvline(x=borehole_depth/2, color='g', linestyle=':', alpha=0.7, label='borehole top')
        ax2.set_xlabel('the distance to the center (m)')
        ax2.set_ylabel('pressure (kPa)')
        ax2.set_title('pressure Distribution in Y direction')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # 保存图像
        plt.savefig('pressure_profiles_vertical_borehole.png', dpi=300, bbox_inches='tight')
        
        if comm.rank == 0:
            print("压力分布曲线已保存为 'pressure_profiles_vertical_borehole.png'")
            
    except ImportError:
        if comm.rank == 0:
            print("Matplotlib 未安装，跳过压力分布曲线绘制")
    except Exception as e:
        if comm.rank == 0:
            print(f"绘制压力分布曲线时出错: {e}")
    
    # 可视化
    try:
        import pyvista
        pyvista.set_jupyter_backend("static")
        
        # 创建绘图器
        topology, cell_types, geometry = plot.vtk_mesh(domain, domain.topology.dim)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, show_edges=True)
        plotter.view_xy()
        plotter.show()
        
        # 将压力场添加到网格
        grid.point_data["Pressure"] = pressure.x.array.real / 1000  # 转换为kPa
        grid.set_active_scalars("Pressure")

        # 创建绘图窗口
        plotter1 = pyvista.Plotter()
        plotter1.add_mesh(grid, show_edges=True, show_scalar_bar=True, scalars="Pressure")
        plotter1.add_title(f"Pressure Distribution[kPa] (P0 = {injection_pressure / 1000:.1f} kPa)")
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
            plotter1.screenshot("pressure_distribution_vertical_borehole.png")
            plotter2.screenshot("velocity_field_vertical_borehole.png")
            
    except ImportError:
        if comm.rank == 0:
            print("PyVista 未安装，跳过可视化")
    
    return pressure, velocity_function, domain

if __name__ == "__main__":
    pressure, velocity_function, domain = main()