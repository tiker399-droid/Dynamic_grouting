"""
绘制最后时刻的地面抬升曲线（横坐标 x，纵坐标竖向位移）
纵坐标从 0 开始，不显示数据点
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from dolfinx import mesh
from dolfinx.io import gmshio
from mpi4py import MPI
import os

# ==================== 用户修改区域 ====================
H5_FILE = "/root/shared/Dynamic_simple_2D/results/results/main_results.h5"
MESH_FILE = "/root/shared/Dynamic_simple_2D/meshes/foundation_drilling_2d.msh"
OUTPUT_IMAGE = "/root/shared/Dynamic_simple_2D/results/final_settlement_curve.png"
GROUND_HEIGHT = 8.0          # 地面高度（y 坐标）
TOLERANCE = 0.05               # 判断地面节点的容差
# =====================================================

def get_ground_nodes_and_x(mesh_file, ground_height, tolerance):
    """
    读取网格，返回地面节点的 x 坐标和索引
    """
    comm = MPI.COMM_WORLD
    msh, _, _ = gmshio.read_from_msh(mesh_file, comm, rank=0, gdim=2)
    points = msh.geometry.x
    y_coords = points[:, 1]
    ground_indices = np.where(np.abs(y_coords - ground_height) < tolerance)[0]
    if len(ground_indices) == 0:
        raise RuntimeError(f"未找到地面节点 (y ≈ {ground_height})")
    x_coords = points[ground_indices, 0]   # x 坐标
    return ground_indices, x_coords

def extract_final_displacement(h5_file, ground_indices):
    """
    从 H5 文件中读取最后一个时间步的位移，提取地面节点的竖向位移
    """
    with h5py.File(h5_file, 'r') as f:
        if 'Function' not in f or 'displacement' not in f['Function']:
            raise RuntimeError("H5 文件中未找到 Function/displacement 组")
        disp_group = f['Function/displacement']
        # 按数字排序数据集名称，取最后一个
        dataset_names = sorted(disp_group.keys(), key=int)
        if not dataset_names:
            raise RuntimeError("未找到位移数据集")
        last_name = dataset_names[-1]
        print(f"使用最后一步数据集: {last_name}")
        data = disp_group[last_name][()]   # 形状 (num_nodes, 3)
        # 取地面节点的 y 方向位移（第二分量）
        vertical_disp = data[ground_indices, 1]
        return vertical_disp

def main():
    # 检查文件存在
    for f in [H5_FILE, MESH_FILE]:
        if not os.path.exists(f):
            print(f"错误：文件不存在 {f}")
            return

    # 获取地面节点和 x 坐标
    print("读取网格...")
    try:
        ground_indices, x_coords = get_ground_nodes_and_x(MESH_FILE, GROUND_HEIGHT, TOLERANCE)
        print(f"地面节点数: {len(ground_indices)}")
    except Exception as e:
        print(f"网格读取失败: {e}")
        return

    # 提取最后时刻的竖向位移
    print("提取最后时刻位移...")
    try:
        vertical_disp = extract_final_displacement(H5_FILE, ground_indices)
    except Exception as e:
        print(f"数据提取失败: {e}")
        return

    # 按 x 坐标排序，便于绘图
    sort_idx = np.argsort(x_coords)
    x_sorted = x_coords[sort_idx]
    disp_sorted = vertical_disp[sort_idx]

    # 绘图
    plt.figure(figsize=(10, 6))
    # 只画线条，不画数据点标记
    plt.plot(x_sorted, disp_sorted, 'b-', linewidth=2)
    # 设置纵坐标下限为 0
    plt.ylim(bottom=0)
    plt.ylim(top=0.017)
    plt.xlabel('x 坐标 (m)')
    plt.ylabel('竖向位移 (m)')
    plt.title('最后时刻地面抬升曲线')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"曲线已保存至 {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()