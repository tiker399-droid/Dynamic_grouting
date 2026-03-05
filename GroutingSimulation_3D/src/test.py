import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io
from dolfinx.fem import functionspace, Function, Constant, dirichletbc, petsc
from dolfinx.io import gmshio
import ufl
import pyvista
from dolfinx import plot
import os

# 读取Gmsh网格
comm = MPI.COMM_WORLD
msh, cell_markers, facet_markers = gmshio.read_from_msh("GroutingSimulation/results/MeshCreate/foundation_drilling_model.msh", comm, rank=0, gdim=3)

print(f"Mesh imported: {msh.topology.index_map(3).size_local} cells")
print(f"Cell markers: {np.unique(cell_markers.values)}")

# 材料参数
E = 2e6  # Young's modulus
nu = 0.3   # Poisson's ratio
rhos_sat = 2700  # Soil density
rho_w = 1000  # Water density
g = 9.81  # Gravitational acceleration
gamma_sat = rhos_sat * g  # Saturated unit weight

lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
mu_ = E / (2 * (1 + nu))

# 几何参数
foundation_size = 14.0
hole_depth = 12.0

# 创建钻孔区域识别器
class DrillholeIdentifier:
    def __init__(self, msh, cell_markers, hole_depth, foundation_size):
        self.msh = msh
        self.cell_markers = cell_markers
        self.hole_depth = hole_depth
        self.foundation_size = foundation_size
        
        # 预计算钻孔单元
        self.drillhole_cells = self._identify_drillhole_cells()
        
    def _identify_drillhole_cells(self):
        """识别钻孔区域的所有单元"""
        cells = np.arange(self.msh.topology.index_map(3).size_local)
        drillhole_cells = []

        x = self.msh.geometry.x
        cell_to_dofs = self.msh.geometry.dofmap

        foundation_length = 4.0
        foundation_width = 4.0
        hole_radius = 0.04
        center_x = foundation_length / 2
        center_y = foundation_width / 2
        hole_top_z = self.foundation_size      # 14.0
        hole_bottom_z = hole_top_z - self.hole_depth  # 2.0

        for cell in range(len(cells)):
            dofs = cell_to_dofs[cell]
            coords = x[dofs]
            cell_center = np.mean(coords, axis=0)
            xc, yc, zc = cell_center

            # 判断是否在钻孔区域内
            radial_dist = np.sqrt((xc - center_x)**2 + (yc - center_y)**2)
            if radial_dist <= hole_radius and zc >= hole_bottom_z and zc <= hole_top_z:
                drillhole_cells.append(cell)
        
        print(f"识别到 {len(drillhole_cells)} 个钻孔区域单元")
        print(f"钻孔区域Z坐标范围: [{hole_bottom_z:.2f}, {hole_top_z:.2f}]")
        
        return np.array(drillhole_cells, dtype=np.int32)
    
    def get_all_cells_meshtag(self):
        """创建包含所有单元的MeshTag"""
        all_cells = np.arange(self.msh.topology.index_map(3).size_local)
        all_values = np.ones(len(all_cells), dtype=np.int32)
        return mesh.meshtags(self.msh, self.msh.topology.dim, all_cells, all_values)
    
    def get_no_drillhole_meshtag(self):
        """创建去除钻孔区域后的MeshTag"""
        # 所有非钻孔单元
        all_cells = np.arange(self.msh.topology.index_map(3).size_local)
        non_drillhole_mask = ~np.isin(all_cells, self.drillhole_cells)
        non_drillhole_cells = all_cells[non_drillhole_mask]
        non_drillhole_values = np.ones(len(non_drillhole_cells), dtype=np.int32)
        
        print(f"去除钻孔区域后剩余 {len(non_drillhole_cells)} 个单元")
        return mesh.meshtags(self.msh, self.msh.topology.dim, non_drillhole_cells, non_drillhole_values)

# 初始化钻孔识别器
drillhole_identifier = DrillholeIdentifier(msh, cell_markers, hole_depth, foundation_size)

# 创建函数空间
V_u = functionspace(msh, ('CG', 1, (msh.topology.dim,)))  # 位移函数空间
V_p = functionspace(msh, ("CG", 1))  # 压力函数空间

# 定义应变和应力函数
def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(msh.topology.dim) + 2 * mu_ * epsilon(u)

# 设置体力
f = Constant(msh, PETSc.ScalarType((0.0, 0.0, -gamma_sat)))

# 设置静水压力
x = ufl.SpatialCoordinate(msh)
p_expression = rho_w * g * (foundation_size - x[2])
p_water = Function(V_p)
p_water.interpolate(fem.Expression(p_expression, V_p.element.interpolation_points()))

# 使用边界标记设置边界条件
print("Setting boundary conditions using facet markers...")
fdim = msh.topology.dim - 1

# 检查可用的边界标记
bc_list = []
if facet_markers is not None:
    unique_markers = np.unique(facet_markers.values)
    print(f"Available boundary markers: {unique_markers}")
    
    # 底部边界条件：固定（标记107）
    if 107 in unique_markers:
        facets_bottom = facet_markers.find(107)  # FoundationBottom
        bottoms_dofs = fem.locate_dofs_topological(V_u, fdim, facets_bottom)
        bc_bottom = dirichletbc(PETSc.ScalarType((0, 0, 0)), bottoms_dofs, V_u)
        bc_list.append(bc_bottom)
        print(f"Bottom boundary: {len(facets_bottom)} facets")
    else:
        print("Warning: Bottom boundary marker 107 not found!")
    
    # 四周边界条件：法向约束
    # x=0 边界（标记103）
    if 103 in unique_markers:
        facets_x0 = facet_markers.find(103)  # FoundationXmin
        dofs_x0 = fem.locate_dofs_topological(V_u.sub(0), fdim, facets_x0)
        bc_x0 = dirichletbc(PETSc.ScalarType(0), dofs_x0, V_u.sub(0))
        bc_list.append(bc_x0)
        print(f"X-min boundary: {len(facets_x0)} facets")
    
    # x=4 边界（标记104）
    if 104 in unique_markers:
        facets_xLx = facet_markers.find(104)  # FoundationXmax
        dofs_xLx = fem.locate_dofs_topological(V_u.sub(0), fdim, facets_xLx)
        bc_xLx = dirichletbc(PETSc.ScalarType(0), dofs_xLx, V_u.sub(0))
        bc_list.append(bc_xLx)
        print(f"X-max boundary: {len(facets_xLx)} facets")
    
    # y=0 边界（标记105）
    if 105 in unique_markers:
        facets_y0 = facet_markers.find(105)  # FoundationYmin
        dofs_y0 = fem.locate_dofs_topological(V_u.sub(1), fdim, facets_y0)
        bc_y0 = dirichletbc(PETSc.ScalarType(0), dofs_y0, V_u.sub(1))
        bc_list.append(bc_y0)
        print(f"Y-min boundary: {len(facets_y0)} facets")
    
    # y=4 边界（标记106）
    if 106 in unique_markers:
        facets_yLy = facet_markers.find(106)  # FoundationYmax
        dofs_yLy = fem.locate_dofs_topological(V_u.sub(1), fdim, facets_yLy)
        bc_yLy = dirichletbc(PETSc.ScalarType(0), dofs_yLy, V_u.sub(1))
        bc_list.append(bc_yLy)
        print(f"Y-max boundary: {len(facets_yLy)} facets")

bcs = bc_list

# 创建输出目录
output_dir = 'GroutingSimulation/results/drilling_simulation'
if comm.rank == 0:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# 求解函数
def solve_elastic_problem(meshtag, u_initial=None):
    """求解弹性问题"""
    # 定义积分测度
    dx_active = ufl.Measure("dx", domain=msh, subdomain_data=meshtag)
    
    # 变分形式
    u = ufl.TrialFunction(V_u)
    v = ufl.TestFunction(V_u)
    
    a = ufl.inner(sigma(u), epsilon(v)) * dx_active(1)
    L = ufl.dot(f, v) * dx_active(1) + p_water * ufl.div(v) * dx_active(1)
    
    # 创建解函数
    u_solution = Function(V_u)
    if u_initial is not None:
        u_solution.x.array[:] = u_initial.x.array
    else:
        u_solution.x.array[:] = 0.0
    
    # 组装系统
    A = petsc.assemble_matrix(fem.form(a), bcs=bcs)
    A.assemble()
    
    b = fem.petsc.assemble_vector(fem.form(L))
    fem.petsc.apply_lifting(b, [fem.form(a)], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    
    # 应用边界条件
    for bc in bcs:
        bc.set(b, u_solution.x.petsc_vec)
    
    # 创建求解器
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1e-10, atol=1e-10, max_it=1000)
    
    # 求解
    ksp.solve(b, u_solution.x.petsc_vec)
    u_solution.x.scatter_forward()
    
    # 检查求解结果
    converged_reason = ksp.getConvergedReason()
    if converged_reason > 0:
        if comm.rank == 0:
            print(f"求解成功! 迭代次数: {ksp.getIterationNumber()}, 最终残差: {ksp.getResidualNorm():.2e}")
    else:
        if comm.rank == 0:
            print(f"求解失败! 收敛原因代码: {converged_reason}")
    
    ksp.destroy()
    A.destroy()
    b.destroy()
    
    return u_solution

# 计算位移统计信息
def compute_displacement_stats(u):
    """计算位移统计信息"""
    displacement = u.x.array.reshape(-1, 3)
    stats = {
        'max_disp_x': np.max(displacement[:, 0]),
        'min_disp_x': np.min(displacement[:, 0]),
        'max_disp_y': np.max(displacement[:, 1]),
        'min_disp_y': np.min(displacement[:, 1]),
        'max_disp_z': np.max(displacement[:, 2]),
        'min_disp_z': np.min(displacement[:, 2]),
        'max_disp_magnitude': np.max(np.linalg.norm(displacement, axis=1)),
        'mean_disp_magnitude': np.mean(np.linalg.norm(displacement, axis=1))
    }
    return stats

# ==================== 状态1: 未挖除状态 ====================
if comm.rank == 0:
    print("\n" + "="*50)
    print("状态1: 计算未挖除状态")
    print("="*50)

# 获取包含所有单元的MeshTag
all_cells_meshtag = drillhole_identifier.get_all_cells_meshtag()

# 求解未挖除状态
u_initial = solve_elastic_problem(all_cells_meshtag)

# 计算统计信息
initial_stats = compute_displacement_stats(u_initial)

if comm.rank == 0:
    print(f"未挖除状态位移统计:")
    print(f"  最大位移幅值: {initial_stats['max_disp_magnitude']:.6e} m")
    print(f"  Z方向位移范围: [{initial_stats['min_disp_z']:.6e}, {initial_stats['max_disp_z']:.6e}] m")

# ==================== 状态2: 全部挖除状态 ====================
if comm.rank == 0:
    print("\n" + "="*50)
    print("状态2: 计算全部挖除状态")
    print("="*50)

# 获取去除钻孔区域后的MeshTag
no_drillhole_meshtag = drillhole_identifier.get_no_drillhole_meshtag()

# 求解全部挖除状态（使用初始状态作为初始猜测）
u_final = solve_elastic_problem(no_drillhole_meshtag, u_initial)

# 计算统计信息
final_stats = compute_displacement_stats(u_final)

if comm.rank == 0:
    print(f"全部挖除状态位移统计:")
    print(f"  最大位移幅值: {final_stats['max_disp_magnitude']:.6e} m")
    print(f"  Z方向位移范围: [{final_stats['min_disp_z']:.6e}, {final_stats['max_disp_z']:.6e}] m")

# ==================== 计算位移变化 ====================
if comm.rank == 0:
    print("\n" + "="*50)
    print("计算位移变化")
    print("="*50)

# 计算位移变化
u_change = Function(V_u)
u_change.x.array[:] = u_final.x.array[:] - u_initial.x.array[:]

# 将钻孔区域的位移变化设置为0
drillhole_nodes = set()
geometry_dofs = msh.geometry.dofmap
for cell in drillhole_identifier.drillhole_cells:
    nodes = geometry_dofs[cell]
    drillhole_nodes.update(nodes)

drillhole_nodes = list(drillhole_nodes)
print(f"[Rank {comm.rank}] 设置 {len(drillhole_nodes)} 个钻孔区域节点的位移变化为0")

for node in drillhole_nodes:
    dof_x = node * 3
    dof_y = node * 3 + 1
    dof_z = node * 3 + 2
    
    if dof_x < len(u_change.x.array):
        u_change.x.array[dof_x] = 0.0
        u_change.x.array[dof_y] = 0.0
        u_change.x.array[dof_z] = 0.0

u_change.x.scatter_forward()

# 计算位移变化统计
change_stats = compute_displacement_stats(u_change)

if comm.rank == 0:
    print(f"位移变化统计:")
    print(f"  最大位移变化幅值: {change_stats['max_disp_magnitude']:.6e} m")
    print(f"  Z方向位移变化范围: [{change_stats['min_disp_z']:.6e}, {change_stats['max_disp_z']:.6e}] m")

# ==================== 保存结果 ====================
if comm.rank == 0:
    print("\n保存位移场结果...")

# 保存位移场
with io.XDMFFile(comm, f"{output_dir}/initial_displacement.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(u_initial)

with io.XDMFFile(comm, f"{output_dir}/final_displacement.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(u_final)

with io.XDMFFile(comm, f"{output_dir}/displacement_change.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(u_change)

# ==================== 创建可视化 ====================
if comm.rank == 0:
    try:
        print("\n创建切片可视化...")
        
        # 创建PyVista网格
        topology, cell_types, geometry = plot.vtk_mesh(V_u)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        
        # 初始位移场切片
        displacement_initial = u_initial.x.array.reshape(geometry.shape[0], 3)
        grid.point_data["Initial_Displacement_Z"] = displacement_initial[:, 2]
        
        slice_initial = grid.slice(normal='x', origin=[2.0, 0, 0])
        
        plotter_initial = pyvista.Plotter(off_screen=True)
        plotter_initial.window_size = [1200, 900]
        plotter_initial.add_mesh(slice_initial, scalars="Initial_Displacement_Z", cmap="coolwarm", 
                                show_scalar_bar=True, clim=[np.min(displacement_initial[:, 2]), np.max(displacement_initial[:, 2])])
        plotter_initial.add_title("Initial Displacement Field - Slice at x=2.0")
        plotter_initial.add_axes()
        plotter_initial.view_vector((1, 0, 0))
        plotter_initial.camera.zoom(1.5)
        plotter_initial.screenshot(f"{output_dir}/initial_displacement_x2.png")
        plotter_initial.close()
        
        # 最终位移场切片
        displacement_final = u_final.x.array.reshape(geometry.shape[0], 3)
        grid.point_data["Final_Displacement_Z"] = displacement_final[:, 2]
        
        slice_final = grid.slice(normal='x', origin=[2.0, 0, 0])
        
        plotter_final = pyvista.Plotter(off_screen=True)
        plotter_final.window_size = [1200, 900]
        plotter_final.add_mesh(slice_final, scalars="Final_Displacement_Z", cmap="coolwarm", 
                              show_scalar_bar=True, clim=[np.min(displacement_final[:, 2]), np.max(displacement_final[:, 2])])
        plotter_final.add_title("Final Displacement Field - Slice at x=2.0")
        plotter_final.add_axes()
        plotter_final.view_vector((1, 0, 0))
        plotter_final.camera.zoom(1.5)
        plotter_final.screenshot(f"{output_dir}/final_displacement_x2.png")
        plotter_final.close()
        
        # 位移变化切片
        displacement_change = u_change.x.array.reshape(geometry.shape[0], 3)
        grid.point_data["Displacement_Change_Z"] = displacement_change[:, 2]
        
        slice_change = grid.slice(normal='x', origin=[2.0, 0, 0])
        
        plotter_change = pyvista.Plotter(off_screen=True)
        plotter_change.window_size = [1200, 900]
        plotter_change.add_mesh(slice_change, scalars="Displacement_Change_Z", cmap="coolwarm", 
                               show_scalar_bar=True, clim=[np.min(displacement_change[:, 2]), np.max(displacement_change[:, 2])])
        plotter_change.add_title("Displacement Change Field - Slice at x=2.0")
        plotter_change.add_axes()
        plotter_change.view_vector((1, 0, 0))
        plotter_change.camera.zoom(1.5)
        plotter_change.screenshot(f"{output_dir}/displacement_change_x2.png")
        plotter_change.close()
        
        print("切片可视化已保存")
        
    except Exception as e:
        print(f"切片可视化失败: {e}")

# ==================== 生成详细报告 ====================
if comm.rank == 0:
    print("\n" + "="*60)
    print("钻孔模拟位移统计报告")
    print("="*60)
    
    print(f"\n未挖除状态:")
    print(f"  最大位移幅值: {initial_stats['max_disp_magnitude']:.6e} m")
    print(f"  Z方向位移范围: [{initial_stats['min_disp_z']:.6e}, {initial_stats['max_disp_z']:.6e}] m")
    print(f"  平均位移幅值: {initial_stats['mean_disp_magnitude']:.6e} m")
    
    print(f"\n全部挖除状态:")
    print(f"  最大位移幅值: {final_stats['max_disp_magnitude']:.6e} m")
    print(f"  Z方向位移范围: [{final_stats['min_disp_z']:.6e}, {final_stats['max_disp_z']:.6e}] m")
    print(f"  平均位移幅值: {final_stats['mean_disp_magnitude']:.6e} m")
    
    print(f"\n位移变化:")
    print(f"  最大位移变化幅值: {change_stats['max_disp_magnitude']:.6e} m")
    print(f"  Z方向位移变化范围: [{change_stats['min_disp_z']:.6e}, {change_stats['max_disp_z']:.6e}] m")
    print(f"  平均位移变化幅值: {change_stats['mean_disp_magnitude']:.6e} m")
    
    # 计算关键指标
    max_settlement_change = final_stats['max_disp_z'] - initial_stats['max_disp_z']
    print(f"\n关键指标:")
    print(f"  最大沉降变化: {max_settlement_change:.6e} m")
    print(f"  位移增幅: {(final_stats['max_disp_magnitude'] - initial_stats['max_disp_magnitude'])/initial_stats['max_disp_magnitude']*100:.2f}%")

# 最终结果
if comm.rank == 0:
    print("\n=== 钻孔模拟完成 ===")
    print(f"结果保存在: {output_dir}")
    print(f"位移场文件: initial_displacement.xdmf, final_displacement.xdmf, displacement_change.xdmf")
    print(f"切片可视化: initial_displacement_x2.png, final_displacement_x2.png, displacement_change_x2.png")