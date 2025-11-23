import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io
from dolfinx.fem import functionspace, Function, Constant, dirichletbc, petsc
from dolfinx.io import gmshio
import ufl
import pyvista
from dolfinx import plot

# 读取Gmsh网格
comm = MPI.COMM_WORLD
msh, cell_markers, facet_markers = gmshio.read_from_msh("GroutingSimulation/results/MeshCreate/foundation_drilling_model.msh", comm, rank=0, gdim=3)

print(f"Mesh imported: {msh.topology.index_map(3).size_local} cells")
print(f"Cell markers: {np.unique(cell_markers.values)}")

# 材料参数
E = 20e6  # Young's modulus
nu = 0.3   # Poisson's ratio
rhos_sat = 2700  # Soil density
rho_w = 1000  # Water density
g = 9.81  # Gravitational acceleration
gamma_sat = rhos_sat * g  # Saturated unit weight

lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
mu_ = E / (2 * (1 + nu))

# 几何参数
foundation_size = 4.0
hole_depth = 2.0
num_layers = 10
layer_height = hole_depth / num_layers

# 创建分层管理器
class LayerRemovalManager:
    def __init__(self, msh, cell_markers, num_layers, hole_depth, foundation_size):
        self.msh = msh
        self.cell_markers = cell_markers
        self.num_layers = num_layers
        self.layer_height = hole_depth / num_layers
        self.foundation_size = foundation_size
        
        # 初始所有层都激活
        self.active_layers = set(range(num_layers))
        
        # 预计算每个单元所属的层
        self.cell_layers = self._compute_cell_layers()
        
    def _compute_cell_layers(self):
        """计算每个圆柱体单元所属的层"""
        cells = np.arange(self.msh.topology.index_map(3).size_local)
        drillhole_mask = self.cell_markers.values == 2  # 钻孔区域标记为2
        
        # 获取单元坐标
        cell_coordinates = self.msh.geometry.x[self.msh.geometry.dofmap]
        
        cell_layers = np.full(len(cells), -1)  # -1表示非圆柱体单元
        
        for cell in np.where(drillhole_mask)[0]:
            # 获取单元所有节点的z坐标
            cell_z_coords = cell_coordinates[cell, :, 2]
            cell_center_z = np.mean(cell_z_coords)
            
            # 计算从顶部开始的深度
            depth_from_top = self.foundation_size - cell_center_z
            
            # 计算所属层 (0到num_layers-1)
            layer = int(depth_from_top / self.layer_height)
            layer = min(layer, self.num_layers - 1)  # 确保在范围内
            cell_layers[cell] = layer
        
        return cell_layers
    
    def remove_layer(self, layer):
        """移除指定层"""
        if layer in self.active_layers:
            self.active_layers.remove(layer)
            print(f"Removed layer {layer}")
    
    def get_active_cells_mask(self):
        """获取当前激活单元的掩码"""
        active_mask = np.ones(len(self.cell_layers), dtype=bool)
        
        for cell, layer in enumerate(self.cell_layers):
            if layer != -1 and layer not in self.active_layers:
                active_mask[cell] = False  # 停用该单元
        
        return active_mask
    
    def create_active_meshtag(self):
        """创建激活单元的MeshTag"""
        active_cells = np.where(self.get_active_cells_mask())[0]
        active_values = np.ones(len(active_cells), dtype=np.int32)
        
        return mesh.meshtags(self.msh, self.msh.topology.dim, active_cells, active_values)

# 初始化分层管理器
layer_manager = LayerRemovalManager(msh, cell_markers, num_layers, hole_depth, foundation_size)

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
if facet_markers is not None:
    unique_markers = np.unique(facet_markers.values)
    print(f"Available boundary markers: {unique_markers}")
    
    # 底部边界条件：固定（标记107）
    if 107 in unique_markers:
        facets_bottom = facet_markers.find(107)  # FoundationBottom
        bottoms_dofs = fem.locate_dofs_topological(V_u, fdim, facets_bottom)
        bc_bottom = dirichletbc(PETSc.ScalarType((0, 0, 0)), bottoms_dofs, V_u)
        print(f"Bottom boundary: {len(facets_bottom)} facets")
    else:
        bc_bottom = None
        print("Warning: Bottom boundary marker 107 not found!")
    
    # 四周边界条件：法向约束
    bc_list = []
    if bc_bottom is not None:
        bc_list.append(bc_bottom)
    
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
import os
output_dir = 'GroutingSimulation/results/drilling_simulation'
if comm.rank == 0:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# 初始位移场（零位移）
u_current = Function(V_u)
u_prev = Function(V_u)  # 用于存储前一步的位移
u_0 = Function(V_u)# 存储初始位移场

# 存储每一步的位移统计信息
displacement_stats = []

# 分层挖除模拟
for step in range(num_layers + 1):  # 包括初始状态（step=0）
    if comm.rank == 0:
        print(f"\n=== 钻孔步骤 {step}/{num_layers} ===")
        print(f"当前挖除层数: {step}")
    
    # 更新激活单元（移除当前层）
    if step > 0:
        layer_manager.remove_layer(step - 1)  # 移除上一层
    
    # 创建激活单元的MeshTag
    active_meshtag = layer_manager.create_active_meshtag()
    
    # 定义在激活单元上的积分测度
    dx_active = ufl.Measure("dx", domain=msh, subdomain_data=active_meshtag)
    
    # 变分形式（只在激活单元上积分）
    u = ufl.TrialFunction(V_u)
    v = ufl.TestFunction(V_u)
    
    # 线性弹性变分形式
    a = ufl.inner(sigma(u), epsilon(v)) * dx_active(1)
    L = ufl.dot(f, v) * dx_active(1) - p_water * ufl.div(v) * dx_active(1)
    
    # 如果是第一步，使用零初始条件；否则使用前一步的解作为初始猜测
    if step == 0:
        u_current.x.array[:] = 0.0
    else:
        # 使用前一步的位移作为初始猜测
        u_current.x.array[:] = u_prev.x.array
    
    # 组装系统
    A = petsc.assemble_matrix(fem.form(a), bcs=bcs)
    A.assemble()
    
    b = fem.petsc.assemble_vector(fem.form(L))
    fem.petsc.apply_lifting(b, [fem.form(a)], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    
    # 应用边界条件
    for bc in bcs:
        bc.set(b, u_current.x.petsc_vec)
    
    # 创建求解器
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.setTolerances(rtol=1e-10, atol=1e-10, max_it=1000)
    
    # 求解
    if comm.rank == 0:
        print("开始求解线性系统...")
    
    ksp.solve(b, u_current.x.petsc_vec)
    u_current.x.scatter_forward()
    
    # 检查求解结果
    converged_reason = ksp.getConvergedReason()
    if converged_reason > 0:
        if comm.rank == 0:
            print(f"求解成功! 迭代次数: {ksp.getIterationNumber()}, 最终残差: {ksp.getResidualNorm():.2e}")
    else:
        if comm.rank == 0:
            print(f"求解失败! 收敛原因代码: {converged_reason}")
    
    ksp.destroy()
    
    if step == 0:
        u_0.x.array[:] = u_current.x.array

    # 计算位移统计信息
    displacement = u_current.x.array.reshape(-1, 3)
    active_mask = layer_manager.get_active_cells_mask()
    
    # 收集统计信息
    stats = {
        'max_disp_z': np.max(displacement[:, 2]),
        'max_disp_magnitude': np.max(np.linalg.norm(displacement, axis=1)),
        'active_cells': np.sum(active_mask),
        'total_cells': len(active_mask)
    }
    displacement_stats.append(stats)
    
    # 保存当前位移场用于下一步
    u_prev.x.array[:] = u_current.x.array
    
    # 打印当前步统计信息
    if comm.rank == 0:
        print(f"最大位移幅值: {stats['max_disp_magnitude']:.6e} m")
        print(f"激活单元数量: {stats['active_cells']} / {stats['total_cells']}")

# 只保存最后一步的位移场
if comm.rank == 0:
    print("\n保存最后一步位移场...")
with io.XDMFFile(comm, f"{output_dir}/final_displacement.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(u_current)

# 创建切片可视化
if comm.rank == 0:
    try:
        print("\n创建切片可视化...")
        # 创建PyVista网格
        topology, cell_types, geometry = plot.vtk_mesh(V_u)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        
        displacement = u_0.x.array.reshape(geometry.shape[0], 3)
        grid.point_data["Displacement"] = displacement
        grid.point_data["Displacement_X"] = displacement[:, 0]
        grid.point_data["Displacement_Y"] = displacement[:, 1]
        grid.point_data["Displacement_Z"] = displacement[:, 2]
        
        # 在x=2处创建切片
        slice_1 = grid.slice(normal='x', origin=[2.0, 0, 0])
        
        # 创建绘图器
        plotter = pyvista.Plotter(off_screen=True)
        plotter.window_size = [1200, 900]
        
        # 添加切片
        plotter.add_mesh(slice_1, scalars="Displacement_Z", cmap="coolwarm", 
                        show_scalar_bar=True, clim=[np.min(displacement[:, 2]), np.max(displacement[:, 2])])
        
        # 添加标题和坐标轴
        plotter.add_title(f"initial Displacement Field - Slice at x=2.0")
        plotter.add_axes()
        
        # 设置视角 - 从x轴正方向观察
        plotter.view_vector((1, 0, 0))
        plotter.camera.zoom(1.5)

        # 保存图像
        plotter.screenshot(f"{output_dir}/initial_displacement_x2.png")
        plotter.close()
        
        print(f"切片可视化已保存: {output_dir}/initial_displacement_x2.png")
        
        # 创建第二个切片在x=2处

        slice_y = grid.slice(normal='x', origin=[0, 2.0, 0])
        
        plotter2 = pyvista.Plotter(off_screen=True)
        plotter2.window_size = [1200, 900]
        
        plotter2.add_mesh(slice_y, scalars="Displacement_Z", cmap="coolwarm",
                         show_scalar_bar=True, clim=[np.min(displacement[:, 2]), np.max(displacement[:, 2])])
        
        plotter2.add_title(f"Final Displacement Field - Slice at y=2.0")
        plotter2.add_axes()
        
        # 设置视角 - 从x轴正方向观察
        plotter2.view_vector((0, 1, 0))
        plotter2.camera.zoom(1.5)

        plotter2.screenshot(f"{output_dir}/displacement_slice_y2.png")
        plotter2.close()
        
        print(f"切片可视化已保存: {output_dir}/displacement_slice_y2.png")
        
    except Exception as e:
        print(f"切片可视化失败: {e}")

# 生成详细的位移统计报告
if comm.rank == 0:
    print("\n" + "="*60)
    print("钻孔模拟位移统计报告")
    print("="*60)
    
    # 初始状态统计
    initial_stats = displacement_stats[0]
    final_stats = displacement_stats[-1]
    
    print(f"\n初始状态 (步骤 0):")
    print(f"  最大位移幅值: {initial_stats['max_disp_magnitude']:.6e} m")
    print(f"  激活单元数量: {initial_stats['active_cells']} / {initial_stats['total_cells']}")
    
    print(f"\n最终状态 (步骤 {num_layers}):")
    print(f"  最大位移幅值: {final_stats['max_disp_magnitude']:.6e} m")
    print(f"  激活单元数量: {final_stats['active_cells']} / {final_stats['total_cells']}")
    
    # 计算变化量
    z_disp_change = final_stats['max_disp_z'] - initial_stats['max_disp_z']
    magnitude_change = final_stats['max_disp_magnitude'] - initial_stats['max_disp_magnitude']
    
    print(f"\n变化量:")
    print(f"  Z方向最大位移变化: {z_disp_change:.6e} m")
    print(f"  最大位移幅值变化: {magnitude_change:.6e} m")

# 最终结果
if comm.rank == 0:
    print("\n=== 钻孔模拟完成 ===")
    print(f"结果保存在: {output_dir}")
    print(f"位移场文件: final_displacement.xdmf")
    print(f"切片可视化: displacement_slice_x2.png, displacement_slice_y2.png")
