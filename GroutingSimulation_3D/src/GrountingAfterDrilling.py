import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, io, fem, plot
from dolfinx.fem import functionspace, Function, Constant, dirichletbc, petsc
from dolfinx.io import gmshio
import ufl
import h5py
import os
import matplotlib.pyplot as plt
import pyvista


class GroutingSimulation:
    def __init__(self, msh, cell_markers, facet_markers, initial_displacement=None, grout_density=1800):
        self.msh = msh
        self.cell_markers = cell_markers
        self.facet_markers = facet_markers
        self.grout_density = grout_density
        
        # 物理常数
        self.rho_w = 1000    # 水密度 (kg/m³)
        self.g = 9.81        # 重力加速度 (m/s²)
        self.p_z = 4.2e5     # 注浆压力 (Pa)

        # 材料参数
        self.E = 20e6        # 杨氏模量 (Pa)
        self.nu = 0.3        # 泊松比
        self.k = 1e-12       # 渗透系数 (m/s)
        self.mu = 1e-3       # 水的动力粘度 (Pa·s)
        
        # 计算拉梅常数
        self.lambda_ = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu_ = self.E / (2 * (1 + self.nu))

        # 几何参数
        self.depth = 10.0    # 钻孔深度 (m)
        self.height = 13.0   # 地基高度 (m)

        # 创建函数空间
        self.V_u = functionspace(msh, ('CG', 1, (msh.topology.dim,)))  # 位移空间
        self.V_p = functionspace(msh, ("CG", 1))                       # 压力空间

        # 初始化场变量
        self.u = Function(self.V_u)
        self.u_initial = Function(self.V_u)
        self.u_increment = Function(self.V_u)
        self.p = Function(self.V_p)
        
        self._init_displacement_field(initial_displacement)
        self._setup_simulation_domain()
        self._setup_boundary_conditions()

    def _init_displacement_field(self, initial_displacement):
        """初始化位移场"""
        if initial_displacement is not None:
            self.u.x.array[:] = initial_displacement.flatten()
            self.u_initial.x.array[:] = initial_displacement.flatten()
        else:
            self.u.x.array[:] = 0.0

    def _setup_simulation_domain(self):
        """设置计算域（排除钻孔区域）"""
        foundation_cells = np.where(self.cell_markers.values == 1)[0]
        if len(foundation_cells) == 0:
            raise ValueError("未找到标记为1的地基单元!")

        self.foundation_domain = self.cell_markers
        self.dx_foundation = ufl.Measure(
            "dx", domain=self.msh, subdomain_data=self.foundation_domain,
            metadata={"quadrature_degree": 2}
        )

    def _setup_boundary_conditions(self):
        """设置边界条件"""
        fdim = self.msh.topology.dim - 1
        unique_markers = np.unique(self.facet_markers.values)
        
        self.bcs_u = self._create_displacement_bcs(unique_markers, fdim)
        self.bcs_p = self._create_pressure_bcs(unique_markers, fdim)

    def _create_displacement_bcs(self, unique_markers, fdim):
        """创建位移边界条件"""
        bcs = []
        
        # 底部固定边界
        if 107 in unique_markers:
            facets_bottom = self.facet_markers.find(107)
            dofs = fem.locate_dofs_topological(self.V_u, fdim, facets_bottom)
            bcs.append(dirichletbc(PETSc.ScalarType((0, 0, 0)), dofs, self.V_u))

        # 四周边界约束
        boundary_constraints = [(103, 0), (104, 0), (105, 1), (106, 1)]
        for marker, component in boundary_constraints:
            if marker in unique_markers:
                facets = self.facet_markers.find(marker)
                dofs = fem.locate_dofs_topological(self.V_u.sub(component), fdim, facets)
                bcs.append(dirichletbc(PETSc.ScalarType(0), dofs, self.V_u.sub(component)))
        
        return bcs

    def _create_pressure_bcs(self, unique_markers, fdim):
        """创建压力边界条件"""
        bcs = []
        x = ufl.SpatialCoordinate(self.msh)
        
        # 顶部自由排水边界
        facets_top = self._find_top_boundary()
        if len(facets_top) > 0:
            dofs_top = fem.locate_dofs_topological(self.V_p, fdim, facets_top)
            p_top = Function(self.V_p)
            p_top.x.array[:] = 0.0
            bcs.append(dirichletbc(p_top, dofs_top))

        # 创建压力表达式
        water_pressure = self.rho_w * self.g * (self.height - x[2])
        water_pressure_expr = fem.Expression(water_pressure, self.V_p.element.interpolation_points())
        grout_pressure_expr = fem.Expression(Constant(self.msh, self.p_z), self.V_p.element.interpolation_points())

        # 四周边界施加水压力
        for marker in [103, 104, 105, 106]:
            if marker in unique_markers:
                self._apply_pressure_on_boundary(marker, water_pressure_expr, bcs, fdim)

        # 钻孔边界施加注浆压力
        if 101 in unique_markers:
            self._apply_grouting_pressure(grout_pressure_expr, bcs, fdim)
        
        return bcs

    def _find_top_boundary(self):
        """识别顶部边界"""
        top_facets = []
        fdim = self.msh.topology.dim - 1
        facet_geom = self.msh.geometry.x
        facet_to_vertex = self.msh.topology.connectivity(fdim, 0)
        
        for facet in np.where(self.facet_markers.values >= 0)[0]:
            vertices = facet_to_vertex.links(facet)
            avg_z = np.mean(facet_geom[vertices, 2])
            if abs(avg_z - self.height) < 1e-6:
                top_facets.append(facet)
        
        return np.array(top_facets, dtype=np.int32)

    def _apply_pressure_on_boundary(self, marker, pressure_expr, bcs, fdim):
        """在边界上施加压力"""
        facets = self.facet_markers.find(marker)
        dofs = fem.locate_dofs_topological(self.V_p, fdim, facets)
        p_func = Function(self.V_p)
        p_func.interpolate(pressure_expr)
        bcs.append(dirichletbc(p_func, dofs))

    def _apply_grouting_pressure(self, pressure_expr, bcs, fdim):
        """在钻孔边界施加注浆压力"""
        facets_drill = self.facet_markers.find(101)
        facet_geom = self.msh.geometry.x
        facet_to_vertex = self.msh.topology.connectivity(fdim, 0)
        
        # 钻孔压力施加深度
        drill_depths = [
            self.height - self.depth + 1.6,
            self.height - self.depth + 1.2,
            self.height - self.depth + 0.8,
            self.height - self.depth + 0.4,
            self.height - self.depth
        ]
        
        selected_facets = []
        for facet in facets_drill:
            vertices = facet_to_vertex.links(facet)
            center = np.mean(facet_geom[vertices], axis=0)
            z = center[2]
            if any(abs(z - depth) < 0.02 for depth in drill_depths):
                selected_facets.append(facet)
        
        if selected_facets:
            selected_facets = np.array(selected_facets, dtype=np.int32)
            dofs = fem.locate_dofs_topological(self.V_p, fdim, selected_facets)
            p_drill = Function(self.V_p)
            p_drill.interpolate(pressure_expr)
            bcs.append(dirichletbc(p_drill, dofs))
        
    def solve_darcy_law(self):
        """求解达西定律得到压力场"""
        # 测试函数和试验函数
        p_trial = ufl.TrialFunction(self.V_p)
        p_test = ufl.TestFunction(self.V_p)
        
        K = self.k / self.mu  # 水力传导系数
        
        # 变分形式
        a = fem.form(K * ufl.dot(ufl.grad(p_trial), ufl.grad(p_test)) * self.dx_foundation(1))
        L = fem.form(ufl.inner(fem.Constant(self.msh, PETSc.ScalarType(0.0)), p_test) * self.dx_foundation(1))
        
        # 组装矩阵和向量
        A = fem.petsc.assemble_matrix(a, bcs=self.bcs_p)
        A.assemble()
        b = fem.petsc.assemble_vector(L)
        
        # 应用边界条件
        fem.petsc.apply_lifting(b, [a], [self.bcs_p])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, self.bcs_p)
        
        # 求解压力场
        solver = PETSc.KSP().create(self.msh.comm)
        solver.setOperators(A)
        solver.setType("cg")
        solver.getPC().setType("hypre")
        solver.setTolerances(rtol=1e-10, atol=1e-10, max_it=1000)
        solver.solve(b, self.p.x.petsc_vec)
        
        self.p.x.scatter_forward()
        solver.destroy()

    def solve_equilibrium(self):
        """求解平衡方程得到位移场"""
        # 定义应变和应力
        def epsilon(u):
            return ufl.sym(ufl.grad(u))
        
        def sigma(u):
            return self.lambda_ * ufl.tr(epsilon(u)) * ufl.Identity(self.msh.topology.dim) + 2 * self.mu_ * epsilon(u)
        
        # 测试函数和试验函数
        u_trial = ufl.TrialFunction(self.V_u)
        u_test = ufl.TestFunction(self.V_u)
        
        # 体力项
        soil_density = 2020
        f = Constant(self.msh, PETSc.ScalarType((0.0, 0.0, -soil_density * self.g)))
        
        # 变分形式
        a = fem.form(ufl.inner(sigma(u_trial), epsilon(u_test)) * self.dx_foundation(1))
        L = fem.form(ufl.dot(f, u_test) * self.dx_foundation(1) + self.p * ufl.div(u_test) * self.dx_foundation(1))
        
        # 组装矩阵和向量
        A = petsc.assemble_matrix(a, bcs=self.bcs_u)
        A.assemble()
        b = petsc.assemble_vector(L)
        
        # 应用边界条件
        petsc.apply_lifting(b, [a], [self.bcs_u])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, self.bcs_u)
        
        # 求解位移场
        solver = PETSc.KSP().create(self.msh.comm)
        solver.setOperators(A)
        solver.setType("cg")
        solver.getPC().setType("hypre")
        solver.setTolerances(rtol=1e-8, atol=1e-8, max_it=1000)
        solver.solve(b, self.u.x.petsc_vec)
        
        self.u.x.scatter_forward()
        solver.destroy()

    def run_simulation(self):
        """运行灌浆模拟"""
        output_dir = "GroutingSimulation_3D/results/grouting_simulation"
        if self.msh.comm.rank == 0 and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存初始状态
        self.save_results("initial", output_dir)
        self.print_statistics("initial")
        
        # 求解过程
        print("\n=== 灌浆过程 ===")
        self.solve_darcy_law()
        self.solve_equilibrium()
        
        # 计算位移增量
        self.u_increment.x.array[:] = self.u.x.array[:] - self.u_initial.x.array[:]
        
        # 保存结果
        self.save_results("final", output_dir)
        self.print_statistics("final")
        
        # 可视化
        if self.msh.comm.rank == 0:
            self._visualize_results(output_dir)
        
        print(f"\n灌浆模拟完成! 结果保存在 {output_dir}")

    def save_results(self, step, output_dir):
        """保存场变量"""
        with io.XDMFFile(self.msh.comm, f"{output_dir}/displacement_{step}.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.msh)
            xdmf.write_function(self.u)
        
        with io.XDMFFile(self.msh.comm, f"{output_dir}/pressure_{step}.xdmf", "w") as xdmf:
            xdmf.write_mesh(self.msh)
            xdmf.write_function(self.p)

    def print_statistics(self, step):
        """打印场变量统计信息"""
        displacement = self.u.x.array.reshape(-1, 3)
        pressure = self.p.x.array
        
        if self.msh.comm.rank == 0:
            print(f"{step} 状态统计:")
            print(f"  位移范围: [{np.min(displacement[:, 2]):.6e}, {np.max(displacement[:, 2]):.6e}] m")
            print(f"  压力范围: [{np.min(pressure):.2f}, {np.max(pressure):.2f}] Pa")
            print(f"  最大沉降: {-np.min(displacement[:, 2]):.6e} m")

    def _visualize_results(self, output_dir):
        """创建结果可视化"""
        try:
            self._create_slice_visualizations(output_dir)
            self._create_analysis_curves(output_dir)
        except Exception as e:
            print(f"可视化失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_slice_visualizations(self, output_dir):
        """创建四分之一内部场可视化（交互式）"""
        try:
            # ========== 位移场可视化 ==========
            # 创建位移网格
            topology_u, cell_types_u, geometry_u = plot.vtk_mesh(self.V_u)
            grid_u_full = pyvista.UnstructuredGrid(topology_u, cell_types_u, geometry_u)
            disp_z = self.u_increment.x.array.reshape(geometry_u.shape[0], 3)[:, 2]
            grid_u_full.point_data["Displacement_Z"] = disp_z

            # 提取地基单元（标记为1）
            foundation_cells = np.where(self.cell_markers.values == 1)[0]
            grid_u_foundation = grid_u_full.extract_cells(foundation_cells)

            # 裁剪出内部四分之一：x ∈ [0,2], y ∈ [0,2]
            grid_u_inner = grid_u_foundation.clip(normal='x', origin=[2.0, 0, 0], invert=True)   # x ≤ 2
            grid_u_inner = grid_u_inner.clip(normal='y', origin=[0, 2.0, 0], invert=True)        # y ≤ 2

            # 创建位移窗口
            plotter_u = pyvista.Plotter(window_size=[1200, 900])
            plotter_u.add_mesh(grid_u_inner,
                            scalars="Displacement_Z",
                            cmap="rainbow",
                            show_scalar_bar=True,
                            scalar_bar_args={
                            'title': 'Displacement(m)',
                            'vertical': True,
                            })
            plotter_u.add_axes()
            plotter_u.view_isometric()
            plotter_u.camera.zoom(1.2)

            # 显示交互窗口
            plotter_u.show()

            # ========== 压力场可视化 ==========
            # 创建压力网格
            topology_p, cell_types_p, geometry_p = plot.vtk_mesh(self.V_p)
            grid_p_full = pyvista.UnstructuredGrid(topology_p, cell_types_p, geometry_p)
            grid_p_full.point_data["Pressure"] = self.p.x.array[:]  # 单位 Pa

            # 提取地基单元
            grid_p_foundation = grid_p_full.extract_cells(foundation_cells)

            # 裁剪内部四分之一
            grid_p_inner = grid_p_foundation.clip(normal='x', origin=[2.0, 0, 0], invert=True)
            grid_p_inner = grid_p_inner.clip(normal='y', origin=[0, 2.0, 0], invert=True)

            # 创建压力窗口
            plotter_p = pyvista.Plotter(window_size=[1200, 900])
            plotter_p.add_mesh(grid_p_inner,
                            scalars="Pressure",
                            cmap="rainbow",
                            show_scalar_bar=True,
                            scalar_bar_args={'title': 'Pressure (Pa)',
                                             'vertical': True,
                            })
            plotter_p.add_axes()
            plotter_p.view_isometric()
            plotter_p.camera.zoom(1.2)

            # 显示交互窗口
            plotter_p.show()

        except Exception as e:
            print(f"四分之一内部场可视化失败: {e}")
            import traceback
            traceback.print_exc()
    '''
    def _create_slice_visualizations(self, output_dir):
        """创建切片可视化"""
        # 位移场切片
        topology, cell_types, geometry = plot.vtk_mesh(self.V_u)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        displacement = self.u_increment.x.array.reshape(geometry.shape[0], 3)
        grid.point_data["Displacement_Z"] = displacement[:, 2]
        
        # x=2切片
        slice_x = grid.slice(normal='x', origin=[2.0, 0, 0])
        plotter = pyvista.Plotter(off_screen=True, window_size=[1200, 900])
        plotter.add_mesh(slice_x, scalars="Displacement_Z", cmap="rainbow", show_scalar_bar=True, scalar_bar_args={
                     'title': 'Displacement(m)',
                     'vertical': True,
                     # 设置自定义刻度和标签
                 })
        #plotter.add_title("Final Displacement Field - Slice at x=2.0")
        plotter.add_axes()
        plotter.view_vector((1, 0, 0))
        plotter.camera.zoom(1.5)
        plotter.screenshot(f"{output_dir}/displacement_slice_x2.png")
        plotter.close()
        
        # y=2切片
        slice_y = grid.slice(normal='y', origin=[0, 2.0, 0])
        plotter2 = pyvista.Plotter(off_screen=True, window_size=[1200, 900])
        plotter2.add_mesh(slice_y, scalars="Displacement_Z", cmap="rainbow", show_scalar_bar=True)
        plotter2.add_title("Final Displacement Field - Slice at y=2.0")
        plotter2.add_axes()
        plotter2.view_vector((0, 1, 0))
        plotter2.camera.zoom(1.5)
        plotter2.screenshot(f"{output_dir}/displacement_slice_y2.png")
        plotter2.close()
        
        # 压力场切片
        topology_p, cell_types_p, geometry_p = plot.vtk_mesh(self.V_p)
        grid_p = pyvista.UnstructuredGrid(topology_p, cell_types_p, geometry_p)
        grid_p.point_data["Pressure"] = self.p.x.array[:] / 1000.0
        
        slice_x_p = grid_p.slice(normal='x', origin=[2.0, 0, 0])
        plotter3 = pyvista.Plotter(off_screen=True, window_size=[1200, 900])
        plotter3.add_mesh(slice_x_p, scalars="Pressure", cmap="rainbow", show_scalar_bar=True, scalar_bar_args={
                     'title': 'Pressure(kPa)',
                     'vertical': True,
                     # 设置自定义刻度和标签
                 })
        #plotter3.add_title("Final Pressure Field - Slice at x=2.0")
        plotter3.add_axes()
        plotter3.view_vector((1, 0, 0))
        plotter3.camera.zoom(1.5)
        plotter3.screenshot(f"{output_dir}/Pressure.png")
        plotter3.close()
    '''
    def _create_analysis_curves(self, output_dir):
        """创建分析曲线"""
        # 创建位移场和压力场的网格
        topology_u, cell_types_u, geometry_u = plot.vtk_mesh(self.V_u)
        grid_u = pyvista.UnstructuredGrid(topology_u, cell_types_u, geometry_u)
        displacement = self.u_increment.x.array.reshape(geometry_u.shape[0], 3)
        grid_u.point_data["Displacement_Z"] = displacement[:, 2]
        
        topology_p, cell_types_p, geometry_p = plot.vtk_mesh(self.V_p)
        grid_p = pyvista.UnstructuredGrid(topology_p, cell_types_p, geometry_p)
        grid_p.point_data["Pressure"] = self.p.x.array[:]
        
        # 获取切片
        slice_x_u = grid_u.slice(normal='x', origin=[2.0, 0, 0])
        slice_x_p = grid_p.slice(normal='x', origin=[2.0, 0, 0])
        
        # 1. 注浆孔高度处的压力变化曲线
        pressure_points, pressure_values = self._extract_data_at_height(
            slice_x_p, self.height - self.depth + 0.8, "Pressure", tolerance=0.05
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(pressure_points[:, 1], pressure_values / 1000.0, 'b-', linewidth=2, label='Pressure')
        plt.xlabel('Y Coordinate (m)')
        plt.ylabel('Pressure (kPa)')
        #plt.title('Pressure Variation at red line (Slice x=2.0)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pressure_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 地面高度处的位移曲线
        disp_points, disp_values = self._extract_data_at_height(
            slice_x_u, self.height, "Displacement_Z", tolerance=0.05
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(disp_points[:, 1], disp_values, 'r-', linewidth=2, label='Z-Displacement')
        plt.xlabel('Y Coordinate (m)')
        plt.ylabel('Z-Displacement (m)')
        #plt.title('Foundation Heave (Z-Displacement) at ground (Slice x=2.0)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/displacement_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _extract_data_at_height(self, slice_mesh, target_z, data_field, tolerance=0.05):
        """从切片网格中提取特定高度处的数据"""
        points, values = [], []
        
        for i in range(slice_mesh.n_points):
            point = slice_mesh.points[i]
            value = slice_mesh.point_data[data_field][i]
            if abs(point[2] - target_z) < tolerance:
                points.append(point)
                values.append(value)
        
        # 按y坐标排序
        sorted_indices = np.argsort([p[1] for p in points])
        return np.array([points[i] for i in sorted_indices]), np.array([values[i] for i in sorted_indices])


def read_initial_displacement(h5_filename, msh):
    """从HDF5文件读取初始位移场"""
    with h5py.File(h5_filename, 'r') as h5f:
        displacement_data = h5f['/Function/f/0'][:]
    
    if displacement_data.shape[0] != msh.geometry.x.shape[0]:
        print(f"警告: 位移数据节点数 ({displacement_data.shape[0]}) 与网格节点数 ({msh.geometry.x.shape[0]}) 不匹配!")
        return None
    
    return displacement_data


def main():
    comm = MPI.COMM_WORLD
    
    # 读取网格
    print("读取网格...")
    msh, cell_markers, facet_markers = gmshio.read_from_msh(
        "GroutingSimulation_3D/results/MeshCreate/foundation_drilling_model.msh", 
        comm, rank=0, gdim=3
    )
    
    print(f"网格导入: {msh.topology.index_map(3).size_local} 个单元")
    print(f"单元标记: {np.unique(cell_markers.values)}")
    
    # 读取初始位移场
    h5_filename = "GroutingSimulation_3D/results/drilling_simulation/final_displacement.h5"
    initial_displacement = None
    
    if os.path.exists(h5_filename):
        initial_displacement = read_initial_displacement(h5_filename, msh)
    else:
        print(f"警告: 初始位移场文件 {h5_filename} 不存在，将使用零位移初始条件")
    
    # 初始化并运行灌浆模拟
    grouting_sim = GroutingSimulation(
        msh, cell_markers, facet_markers, 
        initial_displacement=initial_displacement,
        grout_density=1800
    )
    
    grouting_sim.run_simulation()


if __name__ == "__main__":
    main()