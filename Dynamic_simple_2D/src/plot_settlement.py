"""
地面最大抬升时程曲线绘制脚本
直接从 XDMF 文件解析时间-数据映射关系，再从 H5 读取位移数值。
横坐标为真实物理时间 (s)，与 visualize.py 保持一致。
"""
 
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict
 
# try:
#     plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
#     plt.rcParams['axes.unicode_minus'] = False
# except Exception:
#     pass
 
# ==================== 用户修改区域 ====================
XDMF_FILE     = "/root/shared/Dynamic_simple_2D/results/results/main_results.xdmf"
OUTPUT_IMAGE  = "/root/shared/Dynamic_simple_2D/results/settlement_curve.png"
GROUND_HEIGHT = 13.0      # 地面 y 坐标 (m)
TOLERANCE     = 0.05      # y 坐标容差 (m)
# =====================================================
 
 
# ─────────────────────── XDMF 解析（与 visualize.py 相同） ─────────────────────── #
 
def _strip_ns(tag: str) -> str:
    return tag.split('}')[-1] if '}' in tag else tag
 
 
def _dataitem_to_h5key(di_elem) -> str | None:
    text = (di_elem.text or '').strip()
    if not text:
        return None
    _, key = text.split(':', 1) if ':' in text else ('', text)
    return key.lstrip('/')
 
 
def _resolve_dataitem(di_elem, h5_path: Path) -> np.ndarray | None:
    key = _dataitem_to_h5key(di_elem)
    if key is None:
        return None
    try:
        with h5py.File(h5_path, 'r') as f:
            return np.array(f[key])
    except Exception:
        return None
 
 
def parse_xdmf(xdmf_path: Path):
    """
    解析 XDMF，返回:
      coords         : np.ndarray (n_nodes, n_dim)
      field_records  : dict[str, list[{'time': float, 'h5_path': str}]]  已按时间升序
      h5_path        : Path
    """
    tree = ET.parse(xdmf_path)
    root = tree.getroot()
    h5_path = xdmf_path.with_suffix('.h5')
 
    coords = None
    field_records: dict[str, list] = defaultdict(list)
 
    def iter_grids(node):
        for child in node:
            if _strip_ns(child.tag) == 'Grid':
                yield child
                yield from iter_grids(child)
            else:
                yield from iter_grids(child)
 
    for grid in iter_grids(root):
        if grid.get('GridType', 'Uniform') not in ('Uniform', ''):
            continue
 
        time_val = None
        for child in grid:
            if _strip_ns(child.tag) == 'Time':
                raw = child.get('Value')
                if raw is not None:
                    time_val = float(raw)
                break
 
        if coords is None:
            for child in grid:
                if _strip_ns(child.tag) == 'Geometry':
                    for di in child:
                        if _strip_ns(di.tag) == 'DataItem':
                            coords = _resolve_dataitem(di, h5_path)
                    break
 
        for child in grid:
            if _strip_ns(child.tag) == 'Attribute':
                attr_name = child.get('Name', 'unknown')
                for di in child:
                    if _strip_ns(di.tag) == 'DataItem':
                        h5_key = _dataitem_to_h5key(di)
                        if h5_key and time_val is not None:
                            field_records[attr_name].append(
                                {'time': time_val, 'h5_path': h5_key}
                            )
                        break
 
    for name in field_records:
        field_records[name].sort(key=lambda r: r['time'])
 
    if coords is None:
        raise RuntimeError("XDMF 文件中未能解析到网格坐标 (Geometry)")
 
    return coords, dict(field_records), h5_path
 
 
# ─────────────────────────── 主逻辑 ─────────────────────────── #
 
def main():
    xdmf_path = Path(XDMF_FILE)
    if not xdmf_path.exists():
        raise FileNotFoundError(f"XDMF 文件不存在: {xdmf_path}")
 
    # ── 解析 XDMF ──
    print("解析 XDMF 文件...")
    coords, field_records, h5_path = parse_xdmf(xdmf_path)
    print(f"网格节点数: {coords.shape[0]},  坐标维度: {coords.shape[1]}D")
 
    if 'displacement' not in field_records:
        raise RuntimeError("XDMF/HDF5 中未找到 'displacement' 字段")
 
    records = field_records['displacement']
    print(f"位移场时间步数: {len(records)},  "
          f"t ∈ [{records[0]['time']:.4f}, {records[-1]['time']:.4f}] s")
 
    # ── 选取地面节点 ──
    y = coords[:, 1]
    mask = np.abs(y - GROUND_HEIGHT) < TOLERANCE
    if mask.sum() < 3:
        dist = np.abs(y - GROUND_HEIGHT)
        tol_auto = np.sort(dist)[min(100, len(dist)) - 1] * 1.01
        mask = dist <= tol_auto
        print(f"容差自动扩大至 {tol_auto:.4f} m，选中 {mask.sum()} 个地面节点")
    else:
        print(f"地面节点数 (y ≈ {GROUND_HEIGHT} ± {TOLERANCE} m): {mask.sum()}")
    ground_indices = np.where(mask)[0]
 
    # ── 逐时间步提取最大竖向位移 ──
    print("提取各时间步地面最大抬升量...")
    times, max_uplift = [], []
    with h5py.File(h5_path, 'r') as f:
        for rec in records:
            data = np.array(f[rec['h5_path']])   # (n_nodes, n_dim) 或 (n_nodes*n_dim,)
            if data.ndim == 1:
                # 若数据是 flat 存储，按节点数重塑
                n_nodes = coords.shape[0]
                n_dim   = data.size // n_nodes
                data    = data.reshape(n_nodes, n_dim)
            vert_disp = data[ground_indices, 1]   # y 分量
            times.append(rec['time'])
            max_uplift.append(float(np.max(vert_disp)))
 
    times      = np.array(times)
    max_uplift = np.array(max_uplift)
 
    print(f"最大抬升量: {max_uplift.max():.6f} m  "
          f"(t = {times[np.argmax(max_uplift)]:.4f} s)")
 
    # ── 绘图 ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, max_uplift * 1e3, 'b-', linewidth=2, label='Maximum ground uplift')
    # 在 x=60 处画一条竖直直线
    plt.axvline(x=60, color='red', linestyle='--', linewidth=2, label='t = 60s, reach the target pressure')
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Vertical displacement (mm)', fontsize=14)
    ax.set_title('Maximum ground uplift vs time', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
 
    out = Path(OUTPUT_IMAGE)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"曲线已保存至: {out}")
    plt.show()
 
 
if __name__ == "__main__":
    main()