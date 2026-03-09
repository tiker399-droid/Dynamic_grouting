import gmsh
import math
import sys
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, io, plot
from dolfinx.io import gmshio
import pyvista as pv
from dolfinx.fem import FunctionSpace
import ufl

def create_foundation_with_drill_hole():
    # 初始化Gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    
    # 设置几何容差
    gmsh.option.setNumber("Geometry.Tolerance", 1e-6)
    gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-6)
    
    # 创建新模型
    gmsh.model.add("foundation_drilling")
    
    # 参数设置
    foundation_size= 4.0      # 长宽 4 m
    foundation_height = 13.0  # z 方向
    hole_diameter = 0.1       # 钻孔直径 10 cm
    hole_depth = 10.0         # 钻孔深度 10 m
    hole_radius = hole_diameter / 2
    
    # 设置网格尺寸
    mesh_size_coarse = 0.4  # 粗网格尺寸
    mesh_size_fine = 0.02   # 钻孔附近细网格尺寸
    
    # 创建地基立方体
    foundation = gmsh.model.occ.addBox(0, 0, 0, 
                                      foundation_size, 
                                      foundation_size, 
                                      foundation_height)
    
    # 创建钻孔圆柱体（位于地基中心）
    center_x = foundation_size / 2
    center_y = foundation_size / 2
    
    # 钻孔圆柱体（从顶部到底部，稍微延长一点避免边界问题）
    drill_hole = gmsh.model.occ.addCylinder(center_x, center_y, foundation_height + 0.001, 
                                           0, 0, -hole_depth - 0.002, 
                                           hole_radius)
    
    
    # 同步几何模型
    gmsh.model.occ.synchronize()
    # --- 提取钻孔圆柱的侧壁（在 fragment 前！）---
    cyl_surfaces = gmsh.model.getBoundary([(3, drill_hole)], oriented=False, recursive=False)
    cylinder_lateral_surface = None
    cylinder_bottom_surface = None
    for dim, surf_tag in cyl_surfaces:
        edges = gmsh.model.getBoundary([(2, surf_tag)], oriented=False, recursive=False)
        if len(edges) == 2:  # 侧面有两个边界环
            cylinder_lateral_surface = surf_tag
        elif len(edges) == 1:
        # 底面或顶面：一个圆环
        # 判断 z 坐标
            com_surf = gmsh.model.occ.getCenterOfMass(2, surf_tag)
            if abs(com_surf[2] - (foundation_height - hole_depth)) < 0.1:
                cylinder_bottom_surface = surf_tag
            break
    
    # Fragment操作会保留所有体积但将它们分割开
    all_volumes, _ = gmsh.model.occ.fragment([(3, foundation)], [(3, drill_hole)])
    
    # 同步几何模型
    gmsh.model.occ.synchronize()
    
    # 获取所有体积
    volumes = gmsh.model.getEntities(3)
    print(f"Created {len(volumes)} volumes")
    
    # 识别地基和钻孔体积
    foundation_volumes = []
    drill_hole_volumes = []
    
    for vol in volumes:
        vol_tag = vol[1]
        volume = gmsh.model.occ.getMass(3, vol_tag)  # 获取体积
        expected_drill_vol = math.pi * hole_radius**2 * hole_depth
        if volume < expected_drill_vol * 1.2:  # 小于预期钻孔体积的 1.2 倍
            drill_hole_volumes.append(vol_tag)
        else:
            foundation_volumes.append(vol_tag)
    
    print(f"Foundation volumes: {len(foundation_volumes)}")
    print(f"Drill hole volumes: {len(drill_hole_volumes)}")
    
    # 设置物理组
    if foundation_volumes:
        foundation_physical = gmsh.model.addPhysicalGroup(3, foundation_volumes)
        gmsh.model.setPhysicalName(3, foundation_physical, "Foundation")
    
    if drill_hole_volumes:
        drill_hole_physical = gmsh.model.addPhysicalGroup(3, drill_hole_volumes)
        gmsh.model.setPhysicalName(3, drill_hole_physical, "DrillHole")
    
    # 标记边界表面
    # 获取所有表面
    surfaces = gmsh.model.getEntities(2)
    
    # 定义边界标记
    cylinder_wall_tag = 101  # 圆柱体侧面
    cylinder_bottom_tag = 102  # 圆柱体底面
    foundation_xmin_tag = 103  # 地基垂直于x方向的最小面 (x=0)
    foundation_xmax_tag = 104  # 地基垂直于x方向的最大面 (x=3)
    foundation_ymin_tag = 105  # 地基垂直于y方向的最小面 (y=0)
    foundation_ymax_tag = 106  # 地基垂直于y方向的最大面 (y=3)
    foundation_bottom_tag = 107  # 地基底面
    
    # 用于收集各个边界的表面
    cylinder_wall_surfaces = []
    cylinder_bottom_surfaces = []
    foundation_xmin_surfaces = []
    foundation_xmax_surfaces = []
    foundation_ymin_surfaces = []
    foundation_ymax_surfaces = []
    foundation_bottom_surfaces = []

    # 容差
    tol = 1e-3
    
    for surface in surfaces:
        com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
        '''
        # 检查是否为圆柱体侧面
        # 圆柱体侧面的点在中心轴附近，但z坐标在钻孔深度范围内
        
        if abs(distance_from_axis - hole_radius) < 0.04 and \
           foundation_size - hole_depth - tol <= com[2] <= foundation_size + tol:
            cylinder_wall_surfaces.append(surface[1])
        distance_from_axis = math.sqrt((com[0] - center_x)**2 + (com[1] - center_y)**2)
        # 检查是否为圆柱体底面
        # 圆柱体底面的点在钻孔半径内，且z坐标接近钻孔底部
        if distance_from_axis < hole_radius + tol and \
             abs(com[2] - (foundation_height - hole_depth)) < tol:
            cylinder_bottom_surfaces.append(surface[1])
        '''
        # 检查是否为地基底面 (z=0)
        if abs(com[2]) < tol:
            foundation_bottom_surfaces.append(surface[1])
        
        # 检查是否为垂直于x方向的面 (x=0 和 x=4)
        elif abs(com[0]) < tol and com[2] > 0 and com[2] < foundation_height:
            foundation_xmin_surfaces.append(surface[1])
        elif abs(com[0] - foundation_size) < tol and com[2] > 0 and com[2] < foundation_height:
            foundation_xmax_surfaces.append(surface[1])
        
        # 检查是否为垂直于y方向的面 (y=0 和 y=4)
        elif abs(com[1]) < tol and com[2] > 0 and com[2] < foundation_height:
            foundation_ymin_surfaces.append(surface[1])
        elif abs(com[1] - foundation_size) < tol and com[2] > 0 and com[2] < foundation_height:
            foundation_ymax_surfaces.append(surface[1])

    # 创建边界物理组
    '''
    if cylinder_wall_surfaces:
        gmsh.model.addPhysicalGroup(2, cylinder_wall_surfaces, cylinder_wall_tag)
        gmsh.model.setPhysicalName(2, cylinder_wall_tag, "CylinderWall")
        print(f"Cylinder wall surfaces: {len(cylinder_wall_surfaces)}")
    '''
    if cylinder_lateral_surface is not None:
        gmsh.model.addPhysicalGroup(2, [cylinder_lateral_surface], 101)
        gmsh.model.setPhysicalName(2, 101, "CylinderWall")
    else:
        print("⚠️ Warning: Cylinder wall not tagged.")
    
    if cylinder_bottom_surface is not None:
        gmsh.model.addPhysicalGroup(2, [cylinder_bottom_surface], 102)
        gmsh.model.setPhysicalName(2, 102, "CylinderWall")
    
    if foundation_xmin_surfaces:
        gmsh.model.addPhysicalGroup(2, foundation_xmin_surfaces, foundation_xmin_tag)
        gmsh.model.setPhysicalName(2, foundation_xmin_tag, "FoundationXmin")
        print(f"Foundation X-min surfaces: {len(foundation_xmin_surfaces)}")
    
    if foundation_xmax_surfaces:
        gmsh.model.addPhysicalGroup(2, foundation_xmax_surfaces, foundation_xmax_tag)
        gmsh.model.setPhysicalName(2, foundation_xmax_tag, "FoundationXmax")
        print(f"Foundation X-max surfaces: {len(foundation_xmax_surfaces)}")
    
    if foundation_ymin_surfaces:
        gmsh.model.addPhysicalGroup(2, foundation_ymin_surfaces, foundation_ymin_tag)
        gmsh.model.setPhysicalName(2, foundation_ymin_tag, "FoundationYmin")
        print(f"Foundation Y-min surfaces: {len(foundation_ymin_surfaces)}")
    
    if foundation_ymax_surfaces:
        gmsh.model.addPhysicalGroup(2, foundation_ymax_surfaces, foundation_ymax_tag)
        gmsh.model.setPhysicalName(2, foundation_ymax_tag, "FoundationYmax")
        print(f"Foundation Y-max surfaces: {len(foundation_ymax_surfaces)}")
    
    if foundation_bottom_surfaces:
        gmsh.model.addPhysicalGroup(2, foundation_bottom_surfaces, foundation_bottom_tag)
        gmsh.model.setPhysicalName(2, foundation_bottom_tag, "FoundationBottom")
        print(f"Foundation bottom surfaces: {len(foundation_bottom_surfaces)}")
    
    # 设置网格尺寸
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_coarse)
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_fine)
    
    # 在钻孔附近设置更细的网格
    distance_field = gmsh.model.mesh.field.add("Distance")
    # 选择钻孔相关的曲线（边界）
    curves = gmsh.model.getEntities(1)
    drill_curves = []
    for curve in curves:
        com = gmsh.model.occ.getCenterOfMass(curve[0], curve[1])
        distance_from_center = math.sqrt((com[0] - center_x)**2 + (com[1] - center_y)**2)
        if distance_from_center < hole_radius * 2:
            drill_curves.append(curve[1])
    
    if drill_curves:
        gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", drill_curves)
        gmsh.model.mesh.field.setNumber(distance_field, "Sampling", 100)
        
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", mesh_size_fine)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", mesh_size_coarse)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", hole_radius)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", hole_radius * 4)
        
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
    
    # 设置网格算法
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
    
    # 生成3D网格
    try:
        gmsh.model.mesh.generate(3)
        print("3D mesh generation successful")
    except Exception as e:
        print(f"3D mesh generation failed: {e}")
        # 尝试重新生成几何
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
    
    # 优化网格质量
    gmsh.model.mesh.optimize("Netgen", 0)
    
    # 保存网格文件
    gmsh.write("GroutingSimulation_3D/results/MeshCreate/foundation_drilling_model.msh")
    
    # 在GUI中显示网格（如果可用）
    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
    
    # 结束Gmsh
    gmsh.finalize()

def import_mesh_to_fenicsx():
    """将Gmsh网格导入FEniCSx"""
    # MPI通信器
    comm = MPI.COMM_WORLD
    
    # 读取Gmsh网格
    msh, cell_markers, facet_markers = gmshio.read_from_msh(
        "GroutingSimulation_3D/results/MeshCreate/foundation_drilling_model.msh", 
        comm, 
        rank=0, 
        gdim=3
    )
    
    return msh, cell_markers, facet_markers

def visualize_with_pyvista(msh, cell_markers, facet_markers):
    """使用PyVista可视化网格 - 简化版本"""
    # 创建PyVista网格
    topology, cell_types, geometry = plot.vtk_mesh(msh, msh.topology.dim)
    
    # 创建PyVista非结构化网格
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    
    # 添加单元标记数据
    if cell_markers is not None:
        grid.cell_data["CellMarkers"] = cell_markers.values
        
        # 创建颜色数组：圆柱体为黄色，地基为淡蓝色
        colors = np.zeros(len(cell_markers.values))
        
        # 标记2为圆柱体（黄色），标记1为地基（淡蓝色）
        colors[cell_markers.values == 2] = 1  # 黄色
        colors[cell_markers.values == 1] = 2  # 淡蓝色
        
        grid.cell_data["Colors"] = colors
    
    slice_x = grid.slice(normal='x', origin=[2.0, 0, 0])
    # 创建绘图器
    plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])
    
    # 添加体积网格
    if cell_markers is not None:
        # 使用自定义颜色映射
        cmap = ["yellow", "lightblue"]  # 索引0:黄色, 索引1:淡蓝色
        plotter.add_mesh(slice_x, scalars='Colors', show_edges=False,
                 cmap=cmap, opacity=1.0, clim=[1, 2],
                 scalar_bar_args={
                     'title': 'Cell Markers',
                     'n_labels': 2,           # 只显示两个标签
                     'vertical': True,
                     # 设置自定义刻度和标签
                 })
    else:
        plotter.add_mesh(grid, color="lightblue", show_edges=False, opacity=0.8)
    
    # 设置相机位置和视角
    #plotter.camera_position = 'iso'
    #plotter.camera.azimuth = 30
    #plotter.camera.elevation = 30
    
    # 添加坐标轴
    plotter.add_axes()
    plotter.view_vector((1, 0, 0))
    plotter.camera.zoom(1.5)
    # 添加标题
    # plotter.add_title("Foundation with Drill Hole")
    plotter.screenshot(f"GroutingSimulation_3D/results/MeshCreate")
    # 显示绘图
    plotter.show()
    
    return grid

def print_boundary_info(facet_markers):
    """打印边界信息"""
    if facet_markers is not None:
        unique_markers = np.unique(facet_markers.values)
        print(f"Boundary markers: {unique_markers}")
        
        marker_counts = {}
        for marker in unique_markers:
            count = np.sum(facet_markers.values == marker)
            marker_counts[marker] = count
        
        print("Boundary marker counts:")
        for marker, count in marker_counts.items():
            if marker == 101:
                print(f"  Cylinder Wall (101): {count} facets")
            elif marker == 102:
                print(f"  Cylinder Bottom (102): {count} facets")
            elif marker == 103:
                print(f"  Foundation X-min (103): {count} facets")
            elif marker == 104:
                print(f"  Foundation X-max (104): {count} facets")
            elif marker == 105:
                print(f"  Foundation Y-min (105): {count} facets")
            elif marker == 106:
                print(f"  Foundation Y-max (106): {count} facets")
            elif marker == 107:
                print(f"  Foundation Bottom (107): {count} facets")
            else:
                print(f"  Unknown marker {marker}: {count} facets")

if __name__ == "__main__":
    try:
        # 尝试创建完整模型
        # create_foundation_with_drill_hole()
        
        # 导入网格到FEniCSx
        msh, cell_markers, facet_markers = import_mesh_to_fenicsx()
        
        print(f"Mesh imported: {msh.topology.index_map(3).size_local} cells")
        if cell_markers is not None:
            print(f"Cell markers: {np.unique(cell_markers.values)}")
        
        # 打印边界信息
        print_boundary_info(facet_markers)
        
        # 使用PyVista可视化
        grid = visualize_with_pyvista(msh, cell_markers, facet_markers)
        
    except Exception as e:
        print(f"Error with complex model: {e}")