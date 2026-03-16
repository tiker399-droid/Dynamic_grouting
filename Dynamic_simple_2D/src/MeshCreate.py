import gmsh
import math
import sys
import numpy as np
import os
from mpi4py import MPI
from dolfinx import mesh, io, plot
from dolfinx.io import gmshio
import pyvista as pv

def create_foundation_2d_with_drill_hole():
    """创建二维xz平面带钻孔的地基网格，钻孔为空洞，边界标记侧面和底面"""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Geometry.Tolerance", 1e-6)
    gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-6)

    gmsh.model.add("foundation_2d")

    # 参数
    foundation_size = 4.0
    foundation_height = 13.0
    hole_diameter = 0.1
    hole_depth = 10.0
    hole_radius = hole_diameter / 2

    center_x = foundation_size / 2
    hole_xmin = center_x - hole_radius
    hole_xmax = center_x + hole_radius
    hole_zmin = foundation_height - hole_depth   # 3.0 m
    hole_zmax = foundation_height                # 13.0 m


    # 创建几何
    foundation = gmsh.model.occ.addRectangle(0, 0, 0, foundation_size, foundation_height)
    drill_hole = gmsh.model.occ.addRectangle(hole_xmin, hole_zmin, 0, hole_diameter, hole_depth)

    gmsh.model.occ.synchronize()

    # 切割，得到带孔的面
    out, _ = gmsh.model.occ.cut([(2, foundation)], [(2, drill_hole)])
    gmsh.model.occ.synchronize()

    # 获取面
    surfaces = gmsh.model.getEntities(2)
    print(f"Created {len(surfaces)} surfaces (should be 1)")
    
    # 获取切割后的面（带孔的面）
    foundation_surface = surfaces[0][1] if surfaces else None
    
    # 获取这个面的所有边界曲线
    if foundation_surface:
        boundary_curves = gmsh.model.getBoundary([(2, foundation_surface)], False, True)
        print(f"Found {len(boundary_curves)} boundary curves for the surface")
        for bc in boundary_curves:
            print(f"  Boundary curve tag: {bc[1]}, dim: {bc[0]}")

    # 物理组：整个面标记为1
    foundation_surface_tags = [surf[1] for surf in surfaces]
    gmsh.model.addPhysicalGroup(2, foundation_surface_tags, 1)
    gmsh.model.setPhysicalName(2, 1, "Foundation")

    # 获取所有曲线
    curves = gmsh.model.getEntities(1)

    # 边界容器
    hole_side = []      # 钻孔左右侧面，标记101
    hole_bottom = []    # 钻孔底面，标记102
    hole_top = []       # 钻孔顶面，标记106
    left_boundary = []  # 地基左边界 (x=0)，标记103
    right_boundary = [] # 地基右边界 (x=4)，标记104
    bottom_boundary = [] # 地基底边界 (z=0)，标记107

    tol = 1e-1          # 坐标容差
    tol_dir = 1e-2      # 方向判断容差

    # 打印边界曲线信息
    print("\n--- All boundary curves ---")
    all_curve_tags = []
    for bc in boundary_curves:
        tag = abs(bc[1])  # 取绝对值
        all_curve_tags.append(tag)
        bbox = gmsh.model.getBoundingBox(1, tag)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox
        dx = xmax - xmin
        dy = ymax - ymin
        print(f"Curve {tag}: x=[{xmin:.4f},{xmax:.4f}], y=[{ymin:.4f},{ymax:.4f}], dx={dx:.4f}, dy={dy:.4f}")
    print("--- End of boundary curves ---\n")

    print("\n--- Curve identification (fixed for xy plane) ---")
    for curve in curves:
        tag = curve[1]
        # 跳过不在边界曲线列表中的曲线（内部曲线）
        if tag not in all_curve_tags and abs(tag) not in all_curve_tags:
            continue
            
        bbox = gmsh.model.getBoundingBox(curve[0], tag)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox
        dx = xmax - xmin
        dy = ymax - ymin
        
        # 在xy平面中：垂直线dy大，dx小；水平线dx大，dy小
        # 原来的z坐标现在对应y坐标
        
        # 判断方向 - 垂直线 (dx 很小，dy 较大)
        if dx < tol_dir and dy > tol_dir:
            x_pos = xmin  # 垂直线的 x 坐标
            
            # 检查是否是钻孔侧面 (x = hole_xmin 或 hole_xmax)
            if (abs(x_pos - hole_xmin) < tol or abs(x_pos - hole_xmax) < tol):
                # 确保 y 范围与钻孔深度相交
                if ymax > hole_zmin and ymin < hole_zmax:
                    hole_side.append(tag)
                    print(f"Curve {tag}: vertical at x={x_pos:.4f}, y∈[{ymin:.4f},{ymax:.4f}] -> Hole side")
                    continue
            
            # 检查是否是地基左右边界
            if abs(x_pos - 0) < tol:
                left_boundary.append(tag)
                print(f"Curve {tag}: vertical at x=0, y∈[{ymin:.4f},{ymax:.4f}] -> Left boundary")
            elif abs(x_pos - foundation_size) < tol:
                right_boundary.append(tag)
                print(f"Curve {tag}: vertical at x={foundation_size}, y∈[{ymin:.4f},{ymax:.4f}] -> Right boundary")
            else:
                print(f"Curve {tag}: vertical at x={x_pos:.4f}, unmarked")

        # 判断方向 - 水平线 (dy 很小，dx 较大)
        elif dy < tol_dir and dx > tol_dir:
            y_pos = ymin  # 水平线的 y 坐标
            
            # 检查是否是钻孔底面 (y = hole_zmin)
            if abs(y_pos - hole_zmin) < tol:
                # 确保 x 范围与钻孔宽度相交
                if xmax > hole_xmin and xmin < hole_xmax:
                    hole_bottom.append(tag)
                    print(f"Curve {tag}: horizontal at y={y_pos:.4f}, x∈[{xmin:.4f},{xmax:.4f}] -> Hole bottom")
                    continue
            
            # 检查是否是钻孔顶面 (y = hole_zmax)
            if abs(y_pos - hole_zmax) < tol:
                # 确保 x 范围与钻孔宽度相交
                if xmax > hole_xmin and xmin < hole_xmax:
                    hole_top.append(tag)
                    print(f"Curve {tag}: horizontal at y={y_pos:.4f}, x∈[{xmin:.4f},{xmax:.4f}] -> Hole top")
                    continue
            
            # 检查是否是地基底边 (y = 0)
            if abs(y_pos - 0) < tol:
                bottom_boundary.append(tag)
                print(f"Curve {tag}: horizontal at y=0, x∈[{xmin:.4f},{xmax:.4f}] -> Bottom boundary")
            elif abs(y_pos - foundation_height) < tol:
                print(f"Curve {tag}: horizontal at y={foundation_height} (top, unmarked)")
            else:
                print(f"Curve {tag}: horizontal at y={y_pos:.4f}, unmarked")

        else:
            # 点
            if dx < tol_dir and dy < tol_dir:
                print(f"Curve {tag}: point-like, ignored")
            else:
                print(f"Curve {tag}: diagonal or complex, dx={dx:.4f}, dy={dy:.4f}")
    print("--- End of identification ---\n")

    # 添加物理组
    if hole_side:
        gmsh.model.addPhysicalGroup(1, hole_side, 101)
        gmsh.model.setPhysicalName(1, 101, "HoleSide")
        print(f"Hole side curves: {hole_side}")
    if hole_bottom:
        gmsh.model.addPhysicalGroup(1, hole_bottom, 102)
        gmsh.model.setPhysicalName(1, 102, "HoleBottom")
        print(f"Hole bottom curves: {hole_bottom}")
    if hole_top:
        gmsh.model.addPhysicalGroup(1, hole_top, 106)
        gmsh.model.setPhysicalName(1, 106, "HoleTop")
        print(f"Hole top curves: {hole_top}")
    if left_boundary:
        gmsh.model.addPhysicalGroup(1, left_boundary, 103)
        gmsh.model.setPhysicalName(1, 103, "Left")
        print(f"Left boundary curves: {left_boundary}")
    if right_boundary:
        gmsh.model.addPhysicalGroup(1, right_boundary, 104)
        gmsh.model.setPhysicalName(1, 104, "Right")
        print(f"Right boundary curves: {right_boundary}")
    if bottom_boundary:
        gmsh.model.addPhysicalGroup(1, bottom_boundary, 107)
        gmsh.model.setPhysicalName(1, 107, "Bottom")
        print(f"Bottom boundary curves: {bottom_boundary}")

    # 设置网格尺寸
    mesh_size_coarse = 0.5  # 增大粗网格尺寸
    mesh_size_fine = 0.05   # 适当减小细网格尺寸

    # 钻孔边界细化
    drill_curves = hole_side + hole_bottom
    if drill_curves:
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", drill_curves)
        gmsh.model.mesh.field.setNumber(distance_field, "Sampling", 100)

        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", mesh_size_fine)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", mesh_size_coarse)
        # 增大细化距离范围，从0.2米到1.0米范围内逐渐过渡
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.2)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 1.0)

        gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
        print(f"Mesh refinement based on curves: {drill_curves}")
        print(f"Refinement: LcMin={mesh_size_fine}, LcMax={mesh_size_coarse}, DistMin=0.2, DistMax=1.0")
    else:
        print("Warning: No drill curves found for refinement!")

    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay for 2D

    # 生成网格
    try:
        gmsh.model.mesh.generate(2)
        print("2D mesh generation successful")
    except Exception as e:
        print(f"2D mesh generation failed: {e}")
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)

    gmsh.model.mesh.optimize("Netgen")

    # 保存网格（确保目录存在）
    output_dir = "/root/shared/Dynamic_simple_2D/meshes"  # 注意：如需适应新文件夹名，请手动修改
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "foundation_drilling_2d.msh")
    gmsh.write(output_file)
    print(f"Mesh saved to {output_file}")

    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()

def import_mesh_to_fenicsx():
    """将Gmsh网格导入FEniCSx"""
    comm = MPI.COMM_WORLD
    # 注意：路径必须与保存路径一致！
    msh, cell_markers, facet_markers = gmshio.read_from_msh(
        "/root/shared/Dynamic_simple_2D/meshes/foundation_drilling_2d.msh",  # 如需适应新文件夹，请手动修改
        comm,
        rank=0,
        gdim=2
    )
    return msh, cell_markers, facet_markers

def visualize_with_pyvista(msh, cell_markers, facet_markers):
    """使用PyVista可视化二维网格"""
    # 单元网格
    topology, cell_types, geometry = plot.vtk_mesh(msh, msh.topology.dim)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    if cell_markers is not None:
        grid.cell_data["CellMarkers"] = cell_markers.values

    plotter = pv.Plotter(window_size=[1000, 800])
    plotter.add_mesh(grid, show_edges=True, color="lightblue", opacity=0.8)

    # 边界网格
    if facet_markers is not None:
        facet_topology, facet_cell_types, facet_geometry = plot.vtk_mesh(msh, msh.topology.dim - 1)
        facet_grid = pv.UnstructuredGrid(facet_topology, facet_cell_types, facet_geometry)
        num_facets = facet_grid.n_cells
        facet_values = np.zeros(num_facets, dtype=facet_markers.values.dtype)
        facet_values[facet_markers.indices] = facet_markers.values
        facet_grid.cell_data["FacetMarkers"] = facet_values

        # 为不同标记设置颜色
        colors = np.zeros(num_facets, dtype=int)
        colors[facet_values == 101] = 1   # 钻孔侧面
        colors[facet_values == 102] = 2   # 钻孔底面
        colors[facet_values == 103] = 3   # 左边界
        colors[facet_values == 104] = 4   # 右边界
        colors[facet_values == 106] = 5   # 钻孔顶面
        colors[facet_values == 107] = 6   # 底边界
        facet_grid.cell_data["ColorIndex"] = colors
        
        # 使用自定义颜色列表
        from matplotlib.colors import ListedColormap
        custom_cmap = ListedColormap(['gray', 'red', 'blue', 'orange', 'purple', 'green', 'brown'])
        plotter.add_mesh(facet_grid, scalars="ColorIndex", cmap=custom_cmap, 
                         show_scalar_bar=False, line_width=3)

    plotter.add_axes()
    plotter.camera_position = 'xy'
    plotter.show()

def print_boundary_info(facet_markers):
    if facet_markers is not None:
        unique_markers = np.unique(facet_markers.values)
        print(f"Boundary markers: {unique_markers}")
        for marker in unique_markers:
            count = np.sum(facet_markers.values == marker)
            if marker == 101:
                print(f"  Hole Side (101): {count} facets")
            elif marker == 102:
                print(f"  Hole Bottom (102): {count} facets")
            elif marker == 103:
                print(f"  Foundation Left (103): {count} facets")
            elif marker == 104:
                print(f"  Foundation Right (104): {count} facets")
            elif marker == 106:
                print(f"  Hole Top (106): {count} facets")
            elif marker == 107:
                print(f"  Foundation Bottom (107): {count} facets")
            else:
                print(f"  Unknown marker {marker}: {count} facets")

if __name__ == "__main__":
    # 1. 创建网格
    create_foundation_2d_with_drill_hole()

    # 2. 导入网格
    msh, cell_markers, facet_markers = import_mesh_to_fenicsx()
    print(f"Mesh imported: {msh.topology.index_map(msh.topology.dim).size_local} cells")
    if cell_markers is not None:
        print(f"Cell markers: {np.unique(cell_markers.values)}")
    print_boundary_info(facet_markers)

    # 3. 可视化
    visualize_with_pyvista(msh, cell_markers, facet_markers)