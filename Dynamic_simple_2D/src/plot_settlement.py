"""
地面抬升时程曲线绘制脚本（基于 H5 文件，使用时间步索引作为横坐标）
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
OUTPUT_IMAGE = "/root/shared/Dynamic_simple_2D/results/settlement_curve.png"
GROUND_HEIGHT = 13.0
TOLERANCE = 0.05
# =====================================================

def get_ground_nodes(mesh_file, ground_height, tolerance):
    """读取网格文件，返回地面节点索引"""
    comm = MPI.COMM_WORLD
    msh, _, _ = gmshio.read_from_msh(mesh_file, comm, rank=0, gdim=2)
    points = msh.geometry.x
    y_coords = points[:, 1]
    ground_indices = np.where(np.abs(y_coords - ground_height) < tolerance)[0]
    if len(ground_indices) == 0:
        raise RuntimeError(f"未找到地面节点 (y ≈ {ground_height})")
    return ground_indices, points.shape[0]

def extract_settlement(h5_file, ground_indices):
    """从 H5 文件中读取所有时间步的位移，提取地面节点的竖向位移最大值"""
    with h5py.File(h5_file, 'r') as f:
        if 'Function' not in f or 'displacement' not in f['Function']:
            raise RuntimeError("H5 文件中未找到 Function/displacement 组")
        disp_group = f['Function/displacement']
        # 按数字排序数据集名称
        dataset_names = sorted(disp_group.keys(), key=int)
        print(f"找到 {len(dataset_names)} 个位移数据集")

        max_settlement = []
        for name in dataset_names:
            ds = disp_group[name]
            data = ds[()]  # 形状 (num_nodes, 3)
            # 取地面节点的 y 方向位移（第二分量）
            vertical_disp = data[ground_indices, 1]
            max_val = np.max(vertical_disp)
            max_settlement.append(max_val)

    # 横坐标使用步索引
    times = list(range(len(max_settlement)))
    return times, max_settlement

def main():
    # 检查文件存在
    for f in [H5_FILE, MESH_FILE]:
        if not os.path.exists(f):
            print(f"错误：文件不存在 {f}")
            return

    # 获取地面节点
    print("读取网格...")
    try:
        ground_nodes, total_nodes = get_ground_nodes(MESH_FILE, GROUND_HEIGHT, TOLERANCE)
        print(f"网格节点总数: {total_nodes}")
        print(f"地面节点数: {len(ground_nodes)}")
    except Exception as e:
        print(f"网格读取失败: {e}")
        return

    # 提取沉降数据
    print("提取沉降数据...")
    try:
        times, max_settlement = extract_settlement(H5_FILE, ground_nodes)
        if not max_settlement:
            print("未提取到任何数据")
            return
    except Exception as e:
        print(f"数据提取失败: {e}")
        return

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(times, max_settlement, 'b-', linewidth=2, label='最大竖向位移')
    plt.xlabel('时间步索引')
    plt.ylabel('竖向位移 (m)')
    plt.title('地面抬升时程曲线')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"曲线已保存至 {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()