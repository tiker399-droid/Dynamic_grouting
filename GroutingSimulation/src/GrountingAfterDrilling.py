import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, io, fem, plot
from dolfinx.fem import functionspace, Function, Constant, dirichletbc, petsc
from dolfinx.io import gmshio
import ufl
import h5py
import os

class GroutingSimulation:
    def __init__(self, msh, cell_markers, facet_markers, initial_displacement=None, grout_density=1800):
        self.msh = msh
        self.cell_markers = cell_markers
        self.facet_markers = facet_markers
        self.grout_density = grout_density
        self.rho_w = 1000  # 水密度
        self.g = 9.81      # 重力加速度
        self.p_z = 1e6

        # 材料参数（土体）
        self.E = 2e6      # 杨氏模量
        self.nu = 0.3      # 泊松比
        self.k = 1e-12     # 渗透系数 (m/s)
        self.mu = 1e-3     # 水的动力粘度 (Pa·s)
        
        # 计算拉梅常数
        self.lambda_ = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu_ = self.E / (2 * (1 + self.nu))
        
        # 创建函数空间 - 使用完整网格
        self.V_u = functionspace(msh, ('CG', 1, (msh.topology.dim,)))  # 位移空间
        self.V_p = functionspace(msh, ("CG", 1))  # 压力空间
        
        # 初始化位移场
        self.u = Function(self.V_u)
        self.u_initial = Function(self.V_u)
        if initial_displacement is not None:
            self.u.x.array[:] = initial_displacement.flatten()
            self.u_initial.x.array[:] = initial_displacement.flatten()
        else:
            self.u.x.array[:] = 0.0  # 如果没有初始位移，设为0
        self.u_increment = Function(self.V_u)
        # 初始化压力场
        self.p = Function(self.V_p)
        
        # 创建地基单元标记
        self.create_foundation_domain()
        
        # 设置边界条件
        self.set_boundary_conditions()
        
    def create_foundation_domain(self):
        """创建地基计算域（排除钻孔区域）"""
        print("创建地基计算域...")
        
        # 获取地基单元（标记为1）
        foundation_cells = np.where(self.cell_markers.values == 1)[0]
        
        if len(foundation_cells) == 0:
            raise ValueError("未找到标记为1的地基单元!")
        
        print(f"地基单元数量: {len(foundation_cells)}")
        
        # 直接使用现有的cell_markers，不创建新的MeshTags
        self.foundation_domain = self.cell_markers
        
        # 创建只在计算域上积分的测量
        self.dx_foundation = ufl.Measure("dx", domain=self.msh, subdomain_data=self.foundation_domain)
    
    def set_boundary_conditions(self):
        """设置边界条件 - 顶部压力为0,地基四周和钻孔施加静水压力分布"""
        fdim = self.msh.topology.dim - 1
        
        # 检查可用的边界标记
        if self.facet_markers is not None:
            unique_markers = np.unique(self.facet_markers.values)
            print(f"可用的边界标记: {unique_markers}")
            
            # 位移边界条件
            self.bcs_u = []
            
            # 底部边界条件：固定（标记107）
            if 107 in unique_markers:
                facets_bottom = self.facet_markers.find(107)  # FoundationBottom
                bottoms_dofs = fem.locate_dofs_topological(self.V_u, fdim, facets_bottom)
                bc_bottom = dirichletbc(PETSc.ScalarType((0, 0, 0)), bottoms_dofs, self.V_u)
                self.bcs_u.append(bc_bottom)
                print(f"底部位移边界: {len(facets_bottom)} 个面")
            
            # 四周边界条件：法向约束
            boundary_pairs = [
                (103, 0),  # FoundationXmin, x方向固定
                (104, 0),  # FoundationXmax, x方向固定
                (105, 1),  # FoundationYmin, y方向固定
                (106, 1)   # FoundationYmax, y方向固定
            ]
            
            for marker, component in boundary_pairs:
                if marker in unique_markers:
                    facets = self.facet_markers.find(marker)
                    dofs = fem.locate_dofs_topological(self.V_u.sub(component), fdim, facets)
                    bc = dirichletbc(PETSc.ScalarType(0), dofs, self.V_u.sub(component))
                    self.bcs_u.append(bc)
                    print(f"边界 {marker} 位移边界: {len(facets)} 个面")
            
            # 压力边界条件
            self.bcs_p = []
            
            foundation_size = 4.0  # 地基高度
            x = ufl.SpatialCoordinate(self.msh)
            
            # 通过几何位置识别顶部边界并施加0压力
            print("通过几何位置识别顶部边界...")
            facets_top = self.find_top_boundary_by_geometry()
            if len(facets_top) > 0:
                dofs_top = fem.locate_dofs_topological(self.V_p, fdim, facets_top)
                
                # 创建0压力函数
                p_top = Function(self.V_p)
                p_top.x.array[:] = 0.0  # 设置为0压力
                
                bc_top = dirichletbc(p_top, dofs_top)
                self.bcs_p.append(bc_top)
                print(f"顶部边界压力边界: {len(facets_top)} 个面")
            else:
                print("警告: 未找到顶部边界")
            
            # 四周边界施加静水压力分布（水密度）
            # 注意：静水压力从顶部开始计算，所以公式为 P = ρg*(顶部高度 - z)
            water_hydrostatic_pressure = self.rho_w * self.g * (foundation_size - x[2])
            
            # 钻孔边界施加浆液静止压力分布（浆液密度）
            grout_hydrostatic_pressure = self.p_z + self.grout_density * self.g * (foundation_size - x[2])
            
            # 创建压力表达式
            water_pressure_expr = fem.Expression(water_hydrostatic_pressure, self.V_p.element.interpolation_points())
            grout_pressure_expr = fem.Expression(grout_hydrostatic_pressure, self.V_p.element.interpolation_points())
            
            # 四周边界（标记103,104,105,106）- 施加水静水压力
            side_boundaries = [103, 104, 105, 106]  # FoundationXmin, Xmax, Ymin, Ymax
            
            for marker in side_boundaries:
                if marker in unique_markers:
                    facets_side = self.facet_markers.find(marker)
                    dofs_side = fem.locate_dofs_topological(self.V_p, fdim, facets_side)
                    
                    # 创建压力函数并插值
                    p_side = Function(self.V_p)
                    p_side.interpolate(water_pressure_expr)
                    
                    bc_side = dirichletbc(p_side, dofs_side)
                    self.bcs_p.append(bc_side)
                    print(f"边界 {marker} 水压力边界: {len(facets_side)} 个面")
            
            # 钻孔边界（标记101,102）- 施加浆液静止压力
            drill_boundaries = [101, 102]  # CylinderWall, CylinderBottom
            
            for marker in drill_boundaries:
                    facets_drill = self.facet_markers.find(marker)
                    dofs_drill = fem.locate_dofs_topological(self.V_p, fdim, facets_drill)
                    
                    # 创建压力函数并插值
                    p_drill = Function(self.V_p)
                    p_drill.interpolate(grout_pressure_expr)
                    
                    bc_drill = dirichletbc(p_drill, dofs_drill)
                    self.bcs_p.append(bc_drill)
                    print(f"钻孔边界 {marker} 压力边界: {len(facets_drill)} 个面")
            
            print(f"总共设置了 {len(self.bcs_u)} 个位移边界条件和 {len(self.bcs_p)} 个压力边界条件")
            
            # 注意：地基底部（标记107）没有设置压力边界条件，这意味着它是自然边界条件（不透水）
            print("地基底部设置为不透水边界（自然边界条件）")

    def find_top_boundary_by_geometry(self):
        """通过几何位置识别顶部边界"""
        top_facets = []
        foundation_size = 4.0
        tol = 1e-6
        
        # 获取所有边界面的中心坐标
        fdim = self.msh.topology.dim - 1
        all_facets = np.where(self.facet_markers.values >= 0)[0]
        
        # 获取边界面的几何信息
        facet_geom = self.msh.geometry.x
        facet_to_vertex = self.msh.topology.connectivity(fdim, 0)
        
        for facet in all_facets:
            # 获取边界面的顶点
            vertices = facet_to_vertex.links(facet)
            
            # 计算边界面的平均高度
            avg_z = np.mean(facet_geom[vertices, 2])
            
            # 如果平均高度接近地基顶部，则认为是顶部边界
            if abs(avg_z - foundation_size) < tol:
                top_facets.append(facet)
        
        print(f"通过几何位置找到 {len(top_facets)} 个顶部边界面")
        return np.array(top_facets, dtype=np.int32)
    
    def darcy_law(self):
        """求解达西定律得到压力场 - 使用钻孔边界条件"""
        print("求解达西定律...")
        
        # 达西定律: q = - (k/μ)(∇P - ρg)
        # 连续性方程: ∇·q = 0
        # 组合: ∇·(-(k/μ)(∇P - ρg)) = 0
        
        # 测试函数和试验函数
        p_trial = ufl.TrialFunction(self.V_p)
        p_test = ufl.TestFunction(self.V_p)
        
        # 水力传导系数
        K = self.k / self.mu
        
        # 达西定律变分形式
        a_darcy = fem.form(K * ufl.dot(ufl.grad(p_trial), ufl.grad(p_test)) * self.dx_foundation(1))
        
        # 重力项 - 注意重力方向向下，与z轴正方向相反
        gravity_vector = ufl.as_vector([0, 0, -1])
        L_darcy = fem.form(K * self.rho_w * self.g * ufl.dot(gravity_vector, ufl.grad(p_test)) * self.dx_foundation(1))
        
        # 组装和求解
        A_darcy = fem.petsc.assemble_matrix(a_darcy, bcs=self.bcs_p)
        A_darcy.assemble()
        
        b_darcy = fem.petsc.assemble_vector(L_darcy)
        fem.petsc.apply_lifting(b_darcy, [a_darcy], [self.bcs_p])
        b_darcy.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        # 应用边界条件
        fem.petsc.set_bc(b_darcy, self.bcs_p)
        
        # 创建求解器
        ksp_darcy = PETSc.KSP().create(self.msh.comm)
        ksp_darcy.setOperators(A_darcy)
        ksp_darcy.setType("cg")
        ksp_darcy.getPC().setType("hypre")
        ksp_darcy.setTolerances(rtol=1e-10, atol=1e-10, max_it=1000)
        
        # 求解压力场
        ksp_darcy.solve(b_darcy, self.p.x.petsc_vec)
        self.p.x.scatter_forward()
        
        ksp_darcy.destroy()
        print("达西定律求解完成")
    
    def equilibrium_equation(self):
        """求解平衡方程得到位移场"""
        print("求解平衡方程...")
        
        # 定义应变和应力
        def epsilon(u):
            return ufl.sym(ufl.grad(u))
        
        def sigma(u):
            return self.lambda_ * ufl.tr(epsilon(u)) * ufl.Identity(self.msh.topology.dim) + 2 * self.mu_ * epsilon(u)
        
        # 测试函数和试验函数
        u_trial = ufl.TrialFunction(self.V_u)
        u_test = ufl.TestFunction(self.V_u)
        
        # 体力 - 使用土体密度
        soil_density = 2700  # 土体密度 kg/m³
        f = Constant(self.msh, PETSc.ScalarType((0.0, 0.0, -soil_density * self.g)))
        
        # 平衡方程变分形式
        # 总应力 = 有效应力 + 孔隙压力
        a_elastic = fem.form(ufl.inner(sigma(u_trial), epsilon(u_test)) * self.dx_foundation(1))
        L_elastic = fem.form(ufl.dot(f, u_test) * self.dx_foundation(1) + self.p * ufl.div(u_test) * self.dx_foundation(1))
        
        # 组装和求解
        A_elastic = petsc.assemble_matrix(a_elastic, bcs=self.bcs_u)
        A_elastic.assemble()
        
        b_elastic = petsc.assemble_vector(L_elastic)
        petsc.apply_lifting(b_elastic, [a_elastic], [self.bcs_u])
        b_elastic.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        # 应用边界条件
        petsc.set_bc(b_elastic, self.bcs_u)
        
        # 创建求解器
        ksp_elastic = PETSc.KSP().create(self.msh.comm)
        ksp_elastic.setOperators(A_elastic)
        ksp_elastic.setType("cg")
        ksp_elastic.getPC().setType("hypre")
        ksp_elastic.setTolerances(rtol=1e-8, atol=1e-8, max_it=1000)
        
        # 求解位移场
        ksp_elastic.solve(b_elastic, self.u.x.petsc_vec)
        self.u.x.scatter_forward()
        
        ksp_elastic.destroy()
        print("平衡方程求解完成")
    
    def run_grouting_simulation(self):
        """运行灌浆模拟 - 单次计算"""
        print("开始灌浆模拟...")
        
        # 创建输出目录
        output_dir = "GroutingSimulation/results/grouting_simulation"
        if self.msh.comm.rank == 0 and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存初始状态
        self.save_results("initial", output_dir)
        self.print_statistics("initial")
        
        print("\n=== 灌浆过程 ===")
        
        # 求解达西定律得到压力场
        self.darcy_law()
        
        # 求解平衡方程得到位移场
        self.equilibrium_equation()
        
        # 保存结果
        self.save_results("final", output_dir)
        
        # 打印统计信息
        self.print_statistics("final")
        
        print(f"\n灌浆模拟完成! 结果保存在 {output_dir}")
    
    def save_results(self, step, output_dir):
        """保存结果"""
        # 保存位移场
        with io.XDMFFile(self.msh.comm, f"{output_dir}/displacement_{step}.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.msh)
            xdmf.write_function(self.u)
        
        # 保存压力场
        with io.XDMFFile(self.msh.comm, f"{output_dir}/pressure_{step}.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.msh)
            xdmf.write_function(self.p)
    
    def print_statistics(self, step):
        """打印统计信息"""
        displacement = self.u.x.array.reshape(-1, 3)
        pressure = self.p.x.array
        
        if self.msh.comm.rank == 0:
            print(f"{step} 状态统计:")
            print(f"  位移范围: [{np.min(displacement[:, 2]):.6e}, {np.max(displacement[:, 2]):.6e}] m")
            print(f"  压力范围: [{np.min(pressure):.2f}, {np.max(pressure):.2f}] Pa")
            print(f"  最大沉降: {-np.min(displacement[:, 2]):.6e} m")

def read_initial_displacement(h5_filename, msh):
    """从HDF5文件读取初始位移场"""
    print("读取初始位移场...")
    
    with h5py.File(h5_filename, 'r') as h5f:
        # 读取位移数据
        displacement_data = h5f['/Function/f/0'][:]
    
    print(f"位移数据形状: {displacement_data.shape}")
    
    # 检查数据是否与网格匹配
    num_nodes = msh.geometry.x.shape[0]
    if displacement_data.shape[0] != num_nodes:
        print(f"警告: 位移数据节点数 ({displacement_data.shape[0]}) 与网格节点数 ({num_nodes}) 不匹配!")
        return None
    
    return displacement_data

# 主程序
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    
    # 1. 读取网格
    print("读取网格...")
    msh, cell_markers, facet_markers = gmshio.read_from_msh(
        "GroutingSimulation/results/MeshCreate/foundation_drilling_model.msh", 
        comm, rank=0, gdim=3
    )
    
    print(f"网格导入: {msh.topology.index_map(3).size_local} 个单元")
    print(f"单元标记: {np.unique(cell_markers.values)}")
    
    # 2. 读取初始位移场（钻孔模拟结果）
    h5_filename = "GroutingSimulation/results/drilling_simulation/final_displacement.h5"
    initial_displacement = None
    
    if os.path.exists(h5_filename):
        initial_displacement = read_initial_displacement(h5_filename, msh)
    else:
        print(f"警告: 初始位移场文件 {h5_filename} 不存在，将使用零位移初始条件")
    
    # 3. 初始化灌浆模拟
    grouting_sim = GroutingSimulation(
        msh, 
        cell_markers, 
        facet_markers, 
        initial_displacement=initial_displacement,
        grout_density=1800
    )
    
    # 4. 运行灌浆模拟
    grouting_sim.run_grouting_simulation()
    grouting_sim.u_increment.x.array[:] = grouting_sim.u.x.array[:] - grouting_sim.u_initial.x.array[:]

import pyvista
    # 创建切片可视化
if comm.rank == 0:
    try:
        print("\n创建切片可视化...")
        # 创建PyVista网格
        topology, cell_types, geometry = plot.vtk_mesh(grouting_sim.V_u)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        
        displacement = grouting_sim.u_increment.x.array.reshape(geometry.shape[0], 3)
        grid.point_data["Displacement"] = displacement
        grid.point_data["Displacement_X"] = displacement[:, 0]
        grid.point_data["Displacement_Y"] = displacement[:, 1]
        grid.point_data["Displacement_Z"] = displacement[:, 2]
        
        # 在x=2处创建切片
        slice_x = grid.slice(normal='x', origin=[2.0, 0, 0])
        
        # 创建绘图器
        plotter = pyvista.Plotter(off_screen=True)
        plotter.window_size = [1200, 900]
        
        # 添加切片
        plotter.add_mesh(slice_x, scalars="Displacement_Z", cmap="coolwarm", 
                        show_scalar_bar=True, clim=[np.min(displacement[:, 2]), np.max(displacement[:, 2])])
        
        # 添加标题和坐标轴
        plotter.add_title(f"Final Displacement Field - Slice at x=2.0")
        plotter.add_axes()
        
        # 设置视角 - 从x轴正方向观察
        plotter.view_vector((1, 0, 0))
        plotter.camera.zoom(1.5)

        # 保存图像
        plotter.screenshot(f"GroutingSimulation/results/grouting_simulation/displacement_slice_x2.png")
        plotter.close()
        
        print(f"切片可视化已保存: GroutingSimulation/results/grouting_simulation/displacement_slice_x2.png")
        
        # 创建第二个切片在y=2处
        slice_y = grid.slice(normal='y', origin=[0, 2.0, 0])
        
        plotter2 = pyvista.Plotter(off_screen=True)
        plotter2.window_size = [1200, 900]
        
        plotter2.add_mesh(slice_y, scalars="Displacement_Z", cmap="coolwarm",
                         show_scalar_bar=True, clim=[np.min(displacement[:, 2]), np.max(displacement[:, 2])])
        
        plotter2.add_title(f"Final Displacement Field - Slice at y=2.0")
        plotter2.add_axes()
        
        # 设置视角 - 从x轴正方向观察
        plotter2.view_vector((0, 1, 0))
        plotter2.camera.zoom(1.5)

        plotter2.screenshot(f"GroutingSimulation/results/grouting_simulation/displacement_slice_y2.png")
        plotter2.close()
        
        print(f"切片可视化已保存: GroutingSimulation/results/grouting_simulation/displacement_slice_y2.png")
        
    except Exception as e:
        print(f"切片可视化失败: {e}")

if comm.rank == 0:
    try:
        print("\n创建切片可视化...")
        # 创建PyVista网格
        topology, cell_types, geometry = plot.vtk_mesh(grouting_sim.V_p)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        
        # 获取压力数据
        pressure_data = grouting_sim.p.x.array[:]
        
        # 将压力数据添加到网格
        grid.point_data["Pressure"] = pressure_data
        
        # 在x=2处创建切片
        slice_x = grid.slice(normal='x', origin=[2.0, 0, 0])
        
        # 创建绘图器
        plotter3 = pyvista.Plotter(off_screen=True)
        plotter3.window_size = [1200, 900]
        
        # 添加切片
        plotter3.add_mesh(slice_x, scalars="Pressure", cmap="coolwarm", 
                        show_scalar_bar=True, clim=[np.min(pressure_data), np.max(pressure_data)])
        
        # 添加标题和坐标轴
        plotter3.add_title(f"Final Pressure Field - Slice at x=2.0")
        plotter3.add_axes()
        
        # 设置视角 - 从x轴正方向观察
        plotter3.view_vector((1, 0, 0))
        plotter3.camera.zoom(1.5)

        # 保存图像
        plotter3.screenshot(f"GroutingSimulation/results/grouting_simulation/Pressure.png")
        plotter3.close()
        
        print(f"切片可视化已保存: GroutingSimulation/results/grouting_simulation/Pressure.png")
    except Exception as e:
        print(f"可视化创建失败: {e}")