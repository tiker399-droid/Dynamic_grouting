import gmsh
import sys
import numpy as np
import os
from mpi4py import MPI
from dolfinx import mesh, io, plot
from dolfinx.io import gmshio
import pyvista as pv


def create_foundation_axisymmetric():
    """
    创建轴对称注浆模型的 r-z 半平面网格。

    坐标约定：
        x → r（径向，向右为正）
        y → z（竖向，向上为正）

    计算域为 L 形区域（全矩形 减去 钻孔矩形）：
        全矩形：r ∈ [0, r_far],   z ∈ [0, z_total]
        钻孔：  r ∈ [0, r_hole],  z ∈ [z_hole, z_total]  （从地表向下10m）

    边界标记：
        101  钻孔壁面   r = r_hole, z ∈ [z_hole, z_total]   → 注浆压力边界
        102  钻孔底面   z = z_hole, r ∈ [0,      r_hole]    → 零法向通量（自然BC）
        103  对称轴     r = 0,      z ∈ [0,      z_hole]    → u_r = 0，零通量
        104  远场边界   r = r_far                            → 静水压力
        106  地表       z = z_total, r ∈ [r_hole, r_far]    → 自由排水 p = 0
        107  底面       z = 0                                → 全固定 u = 0
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Geometry.Tolerance", 1e-9)
    gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-9)

    gmsh.model.add("foundation_axisymmetric")

    # ------------------------------------------------------------------
    # 几何参数
    # ------------------------------------------------------------------
    r_far   = 4.0    # 远场半径 (m)
    z_total = 13.0   # 地基总高度 (m)
    r_hole  = 0.05   # 钻孔半径 (m)
    z_hole  = 3.0    # 钻孔底距地基底面的高度 (m)，即钻孔深度 = z_total - z_hole = 10 m

    # ------------------------------------------------------------------
    # 构建 L 形计算域
    # ------------------------------------------------------------------
    # 全矩形
    full_rect = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, r_far, z_total)
    # 钻孔矩形（将被切除）
    hole_rect = gmsh.model.occ.addRectangle(0.0, z_hole, 0.0, r_hole, z_total - z_hole)
    gmsh.model.occ.synchronize()

    # 布尔差集：full_rect - hole_rect = L 形域
    out, _ = gmsh.model.occ.cut([(2, full_rect)], [(2, hole_rect)])
    gmsh.model.occ.synchronize()

    surfaces = gmsh.model.getEntities(2)
    assert len(surfaces) == 1, f"期望1个面，实际得到 {len(surfaces)} 个"
    domain_tag = surfaces[0][1]

    # 物理组：计算域
    gmsh.model.addPhysicalGroup(2, [domain_tag], 1)
    gmsh.model.setPhysicalName(2, 1, "Domain")

    # ------------------------------------------------------------------
    # 获取边界曲线并打印调试信息
    # ------------------------------------------------------------------
    boundary_curves = gmsh.model.getBoundary([(2, domain_tag)], oriented=False, combined=False)
    curve_tags = [abs(bc[1]) for bc in boundary_curves]

    print("\n===== 边界曲线信息 =====")
    for tag in curve_tags:
        bbox = gmsh.model.getBoundingBox(1, tag)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox
        print(f"  曲线 {tag:3d}: r∈[{xmin:.4f},{xmax:.4f}], z∈[{ymin:.4f},{ymax:.4f}]")
    print("========================\n")

    # ------------------------------------------------------------------
    # 边界识别（基于包围盒坐标）
    # ------------------------------------------------------------------
    tol = 1e-3   # 坐标判断容差

    hole_wall   = []   # 101
    hole_bottom = []   # 102
    axis_sym    = []   # 103
    far_field   = []   # 104
    top_surface = []   # 106
    bottom_fix  = []   # 107

    for tag in curve_tags:
        bbox = gmsh.model.getBoundingBox(1, tag)
        xmin, ymin, _, xmax, ymax, _ = bbox
        dr = xmax - xmin   # r 方向跨度
        dz = ymax - ymin   # z 方向跨度

        is_vertical   = (dr < tol) and (dz > tol)
        is_horizontal = (dz < tol) and (dr > tol)

        if is_vertical:
            r_pos = xmin  # 竖直线的 r 坐标
            z_lo  = ymin
            z_hi  = ymax

            if abs(r_pos - r_hole) < tol and z_lo >= z_hole - tol:
                # 钻孔壁面：r = r_hole，z ∈ [z_hole, z_total]
                hole_wall.append(tag)
                print(f"  → 钻孔壁面 (101): 曲线 {tag}")

            elif abs(r_pos - 0.0) < tol and z_hi <= z_hole + tol:
                # 对称轴：r = 0，z ∈ [0, z_hole]
                axis_sym.append(tag)
                print(f"  → 对称轴   (103): 曲线 {tag}")

            elif abs(r_pos - r_far) < tol:
                # 远场边界：r = r_far
                far_field.append(tag)
                print(f"  → 远场边界 (104): 曲线 {tag}")

            else:
                print(f"  ？未识别竖直线 {tag}: r={r_pos:.4f}, z∈[{z_lo:.4f},{z_hi:.4f}]")

        elif is_horizontal:
            z_pos = ymin  # 水平线的 z 坐标
            r_lo  = xmin
            r_hi  = xmax

            if abs(z_pos - z_hole) < tol and r_hi <= r_hole + tol:
                # 钻孔底面：z = z_hole，r ∈ [0, r_hole]
                hole_bottom.append(tag)
                print(f"  → 钻孔底面 (102): 曲线 {tag}")

            elif abs(z_pos - z_total) < tol and r_lo >= r_hole - tol:
                # 地表：z = z_total，r ∈ [r_hole, r_far]
                top_surface.append(tag)
                print(f"  → 地表     (106): 曲线 {tag}")

            elif abs(z_pos - 0.0) < tol:
                # 底面：z = 0，r ∈ [0, r_far]
                bottom_fix.append(tag)
                print(f"  → 底面     (107): 曲线 {tag}")

            else:
                print(f"  ？未识别水平线 {tag}: z={z_pos:.4f}, r∈[{r_lo:.4f},{r_hi:.4f}]")

        else:
            print(f"  ？斜线或点状曲线 {tag}: dr={dr:.4f}, dz={dz:.4f}")

    # ------------------------------------------------------------------
    # 添加物理组
    # ------------------------------------------------------------------
    def add_group(curves, dim, tag, name):
        if curves:
            gmsh.model.addPhysicalGroup(dim, curves, tag)
            gmsh.model.setPhysicalName(dim, tag, name)
            print(f"物理组 {tag:3d} ({name:12s}): 曲线 {curves}")
        else:
            print(f"警告：{name} (标记 {tag}) 未找到任何曲线！")

    print("\n===== 物理组分配 =====")
    add_group(hole_wall,   1, 101, "HoleWall")
    add_group(hole_bottom, 1, 102, "HoleBottom")
    add_group(axis_sym,    1, 103, "AxisSym")
    add_group(far_field,   1, 104, "FarField")
    add_group(top_surface, 1, 106, "TopSurface")
    add_group(bottom_fix,  1, 107, "Bottom")
    print("======================\n")

    # ------------------------------------------------------------------
    # 网格尺寸设置：孔壁区域加密，向远场渐粗
    # ------------------------------------------------------------------
    mesh_fine   = 0.01   # 孔壁附近最小单元尺寸 (m)
    mesh_medium = 0.2    # 过渡区单元尺寸 (m)
    mesh_coarse = 0.5     # 远场最大单元尺寸 (m)

    refine_curves = hole_wall + hole_bottom

    if refine_curves:
        dist_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_field, "CurvesList", refine_curves)
        gmsh.model.mesh.field.setNumber(dist_field, "Sampling", 200)

        # 第一层过渡：孔壁→0.2 m
        thresh1 = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thresh1, "IField",  dist_field)
        gmsh.model.mesh.field.setNumber(thresh1, "LcMin",   mesh_fine)
        gmsh.model.mesh.field.setNumber(thresh1, "LcMax",   mesh_medium)
        gmsh.model.mesh.field.setNumber(thresh1, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(thresh1, "DistMax", 0.2)

        # 第二层过渡：0.2 m→1.5 m
        thresh2 = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thresh2, "IField",  dist_field)
        gmsh.model.mesh.field.setNumber(thresh2, "LcMin",   mesh_medium)
        gmsh.model.mesh.field.setNumber(thresh2, "LcMax",   mesh_coarse)
        gmsh.model.mesh.field.setNumber(thresh2, "DistMin", 0.2)
        gmsh.model.mesh.field.setNumber(thresh2, "DistMax", 1.5)

        # 取两层中的较小值
        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [thresh1, thresh2])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

        print(f"网格加密：孔壁区 {mesh_fine} m → 过渡区 {mesh_medium} m → 远场 {mesh_coarse} m")
    else:
        print("警告：未找到孔壁曲线，跳过网格加密！")

    # 使用 Frontal-Delaunay 算法（适合过渡网格）
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    # 对称轴上的点避免退化单元
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)

    # ------------------------------------------------------------------
    # 生成并保存网格
    # ------------------------------------------------------------------
    try:
        gmsh.model.mesh.generate(2)
        print("二维网格生成成功")
    except Exception as e:
        print(f"网格生成异常，尝试重新生成: {e}")
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)

    gmsh.model.mesh.optimize("Netgen")

    output_dir = "/root/shared/Dynamic_simple_2D/meshes"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "foundation_axisymmetric.msh")
    gmsh.write(output_file)
    print(f"网格已保存至: {output_file}")

    # 统计信息
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2)
    total_elements = sum(len(t) for t in elem_tags)
    node_tags, _, _ = gmsh.model.mesh.getNodes()
    print(f"单元总数: {total_elements}，节点总数: {len(node_tags)}")

    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()
    return output_file


def import_mesh_to_fenicsx(mesh_file=None):
    """将轴对称 Gmsh 网格导入 FEniCSx"""
    if mesh_file is None:
        mesh_file = "/root/shared/Dynamic_simple_2D/meshes/foundation_axisymmetric.msh"

    comm = MPI.COMM_WORLD
    msh, cell_markers, facet_markers = gmshio.read_from_msh(
        mesh_file,
        comm,
        rank=0,
        gdim=2      # 二维平面，x=r，y=z
    )
    return msh, cell_markers, facet_markers


def print_boundary_info(facet_markers):
    """打印各边界标记的面片数量"""
    marker_names = {
        101: "钻孔壁面  (HoleWall)",
        102: "钻孔底面  (HoleBottom)",
        103: "对称轴    (AxisSym)",
        104: "远场边界  (FarField)",
        106: "地表      (TopSurface)",
        107: "底面      (Bottom)",
    }
    if facet_markers is not None:
        unique = np.unique(facet_markers.values)
        print("\n===== 边界统计 =====")
        for m in unique:
            count = np.sum(facet_markers.values == m)
            name = marker_names.get(m, f"未知标记 {m}")
            print(f"  标记 {m:3d}  {name}: {count} 个面片")
        print("====================\n")


def visualize_with_pyvista(msh, cell_markers, facet_markers):
    """使用 PyVista 可视化轴对称网格和边界"""
    topology, cell_types, geometry = plot.vtk_mesh(msh, msh.topology.dim)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)

    #plotter = pv.Plotter(window_size=[1600, 2000], image_scale=1)
    plotter = pv.Plotter(window_size=[1200, 800])
    surface = grid.extract_surface()
    plotter.add_mesh(surface, show_edges=False, color="lightblue",
                     opacity=0.7, label="计算域")

    if facet_markers is not None:
        ftopo, fctypes, fgeom = plot.vtk_mesh(msh, msh.topology.dim - 1)
        fgrid = pv.UnstructuredGrid(ftopo, fctypes, fgeom)

        n_facets = fgrid.n_cells
        color_idx = np.zeros(n_facets, dtype=int)
        vals = np.zeros(n_facets, dtype=facet_markers.values.dtype)
        vals[facet_markers.indices] = facet_markers.values

        # 边界着色
        color_map = {101: 1, 102: 2, 103: 3, 104: 4, 106: 5, 107: 6}
        label_map = {
            101: "钻孔壁面 (101)",
            102: "钻孔底面 (102)",
            103: "对称轴   (103)",
            104: "远场边界 (104)",
            106: "地表     (106)",
            107: "底面     (107)",
        }
        for marker, cidx in color_map.items():
            color_idx[vals == marker] = cidx
        fgrid.cell_data["Boundary"] = color_idx

        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(
            ["gray", "red", "blue", "orange", "purple", "green", "brown"]
        )
        #plotter.add_mesh(fgrid, scalars="Boundary", cmap=cmap,show_scalar_bar=False, line_width=3)
        
    plotter.add_axes()
    #plotter.add_title("轴对称注浆模型 (r-z 半平面)", font_size=12)
    plotter.camera_position = 'xy'
    plotter.view_xy()
    plotter.hide_axes()
    plotter.show(screenshot='axisymmetric_mesh.png')


# ======================================================================
if __name__ == "__main__":
    # 1. 创建网格
    mesh_file = create_foundation_axisymmetric()

    # 2. 导入 FEniCSx
    msh, cell_markers, facet_markers = import_mesh_to_fenicsx(mesh_file)
    print(f"网格导入成功：{msh.topology.index_map(msh.topology.dim).size_local} 个单元")
    if cell_markers is not None:
        print(f"单元标记: {np.unique(cell_markers.values)}")

    # 3. 打印边界统计
    print_boundary_info(facet_markers)

    # 4. 可视化
    visualize_with_pyvista(msh, cell_markers, facet_markers)