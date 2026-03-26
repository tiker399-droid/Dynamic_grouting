"""
绘制最终时刻 y=3.8m 处的压力分布曲线
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from dolfinx import mesh
from dolfinx.io import gmshio
from mpi4py import MPI

# ==================== 用户修改区域 ====================
# 结果文件路径
H5_FILE = "/root/shared/Dynamic_simple_2D/results/results/main_results.h5"
MESH_FILE = "/root/shared/Dynamic_simple_2D/meshes/foundation_drilling_2d.msh"
# 目标高度 (y 坐标)
TARGET_Y = 3.8
# 容差
TOLERANCE = 0.05
# 输出图片文件名
OUTPUT_IMAGE = "pressure_at_y3.8.png"
# =====================================================

def get_nodes_at_y(mesh_file, target_y, tolerance):
    """读取网格，返回 y 坐标接近 target_y 的节点索引和 x 坐标"""
    comm = MPI.COMM_WORLD
    msh, _, _ = gmshio.read_from_msh(mesh_file, comm, rank=0, gdim=2)
    points = msh.geometry.x
    y_coords = points[:, 1]
    indices = np.where(np.abs(y_coords - target_y) < tolerance)[0]
    if len(indices) == 0:
        raise RuntimeError(f"未找到 y ≈ {target_y} 的节点")
    x_coords = points[indices, 0]
    return indices, x_coords

def extract_pressure_at_nodes(h5_file, node_indices):
    """从 H5 文件中读取最后一个时间步的压力，返回指定节点处的压力值"""
    with h5py.File(h5_file, 'r') as f:
        # 查找压力数据集（通常位于 /Function/pressure/ 下）
        if 'Function' not in f or 'pressure' not in f['Function']:
            raise RuntimeError("H5 文件中未找到 Function/pressure 组")
        press_group = f['Function/pressure']
        # 按数字排序数据集名称，取最后一个（最终时刻）
        dataset_names = sorted(press_group.keys(), key=int)
        if not dataset_names:
            raise RuntimeError("未找到压力数据集")
        last_name = dataset_names[-1]
        print(f"使用最后一步压力数据集: {last_name}")
        data = press_group[last_name][()]  # 形状 (num_nodes,) 或 (num_nodes,1)
        # 如果是二维数组，取第一列（假设单分量）
        if data.ndim == 2:
            pressure = data[:, 0]
        else:
            pressure = data
        # 提取指定节点的压力
        return pressure[node_indices]

def main():
    # 获取目标高度处的节点
    print("读取网格...")
    try:
        node_indices, x_coords = get_nodes_at_y(MESH_FILE, TARGET_Y, TOLERANCE)
        print(f"找到 {len(node_indices)} 个节点，x 范围 [{x_coords.min():.3f}, {x_coords.max():.3f}]")
    except Exception as e:
        print(f"网格读取失败: {e}")
        return

    # 提取压力值
    print("提取压力数据...")
    try:
        pressure = extract_pressure_at_nodes(H5_FILE, node_indices)
        print(f"压力范围: [{pressure.min():.3f}, {pressure.max():.3f}] Pa")
    except Exception as e:
        print(f"数据提取失败: {e}")
        return

    # 按 x 坐标排序，以便绘图
    sort_idx = np.argsort(x_coords)
    x_sorted = x_coords[sort_idx]
    p_sorted = pressure[sort_idx] / 1000  # 转换为 kPa

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(x_sorted, p_sorted, 'b-', linewidth=2)
    plt.ylim(bottom=0)
    plt.xlabel('x 坐标 (m)')
    plt.ylabel('压力 (kPa)')
    plt.title(f'最终时刻压力分布 (y = {TARGET_Y} m)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"曲线已保存至 {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()