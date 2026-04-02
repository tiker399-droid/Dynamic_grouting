"""
可视化分析模块 - 用于后处理模拟结果
直接从 XDMF 文件解析时间-数据映射关系，再按需从 H5 读取数值。
 
修复说明：
  原代码通过 H5 路径顺序猜测时间对应关系，导致时间步错位。
  新代码解析 XDMF XML，每个 <Grid> 节点同时包含
    <Time Value="..."/> 和 <DataItem .../> (→ H5 路径)，
  因此时间与数据的对应关系完全由 XDMF 自身保证，不会错乱。
"""
 
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict
 
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass
 
 
# ─────────────────────────── XDMF 解析工具 ─────────────────────────── #
 
def _strip_ns(tag: str) -> str:
    """去掉 XML 命名空间前缀，例如 '{http://...}Grid' → 'Grid'"""
    return tag.split('}')[-1] if '}' in tag else tag
 
 
def parse_xdmf(xdmf_path: Path):
    """
    解析 FEniCSx 生成的 XDMF 文件。
 
    Returns
    -------
    coords : np.ndarray, shape (n_nodes, n_dim)
        网格节点坐标（来自第一个时间步的 Geometry）
    field_records : dict[str, list[dict]]
        { 字段名 : [ { 'time': float, 'h5_path': str }, ... ] }
        列表已按时间升序排列
    h5_path : Path
        H5 文件路径
    """
    tree = ET.parse(xdmf_path)
    root = tree.getroot()
 
    h5_path = xdmf_path.with_suffix('.h5')
 
    coords = None
    field_records: dict[str, list] = defaultdict(list)
 
    def iter_grids(node):
        """递归遍历所有 Grid 节点"""
        for child in node:
            tag = _strip_ns(child.tag)
            if tag == 'Grid':
                yield child
                yield from iter_grids(child)
            else:
                yield from iter_grids(child)
 
    for grid in iter_grids(root):
        # 只处理 Uniform 网格（单个时间步）
        if grid.get('GridType', 'Uniform') not in ('Uniform', ''):
            continue
 
        # 读取本 Grid 的时间值
        time_val = None
        for child in grid:
            if _strip_ns(child.tag) == 'Time':
                raw = child.get('Value')
                if raw is not None:
                    time_val = float(raw)
                break
 
        # 读取 Geometry（坐标），仅取第一次出现的
        if coords is None:
            for child in grid:
                if _strip_ns(child.tag) == 'Geometry':
                    for di in child:
                        if _strip_ns(di.tag) == 'DataItem':
                            ref = _resolve_dataitem(di, h5_path)
                            if ref is not None:
                                coords = ref
                    break
 
        # 读取 Attribute（场数据）
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
                        break   # 只取第一个 DataItem
 
    # 按时间升序排列
    for name in field_records:
        field_records[name].sort(key=lambda r: r['time'])
 
    if coords is None:
        raise RuntimeError("XDMF 文件中未能解析到网格坐标 (Geometry)")
 
    return coords, dict(field_records), h5_path
 
 
def _dataitem_to_h5key(di_elem) -> str | None:
    """从 DataItem 元素提取 HDF5 内部路径（去掉文件名前缀）"""
    text = (di_elem.text or '').strip()
    if not text:
        return None
    # 格式通常为 "filename.h5:/internal/path" 或 ":/internal/path"
    if ':' in text:
        _, key = text.split(':', 1)
    else:
        key = text
    return key.lstrip('/')
 
 
def _resolve_dataitem(di_elem, h5_path: Path) -> np.ndarray | None:
    """直接读取 DataItem 所引用的 H5 数据集并返回 numpy 数组"""
    key = _dataitem_to_h5key(di_elem)
    if key is None:
        return None
    try:
        with h5py.File(h5_path, 'r') as f:
            return np.array(f[key])
    except Exception:
        return None
 
 
# ─────────────────────────── 主类 ─────────────────────────── #
 
class ResultVisualizer:
    """
    模拟结果可视化分析器
 
    核心改动：时间–数据对应关系直接取自 XDMF XML，
    彻底避免原来依赖 H5 路径顺序猜测时间步导致的错位问题。
    """
 
    def __init__(self, results_dir: str | Path):
        self.results_dir = Path(results_dir)
        self.xdmf_path = self.results_dir / "main_results.xdmf"
        self.h5_path   = self.results_dir / "main_results.h5"
 
        if not self.xdmf_path.exists():
            raise FileNotFoundError(f"XDMF 文件不存在: {self.xdmf_path}")
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 文件不存在: {self.h5_path}")
 
        self._load_metadata()
 
    # ──────────────── 内部方法 ──────────────── #
 
    def _load_metadata(self):
        """解析 XDMF，建立坐标和场记录表"""
        self.coords, self._field_records, _ = parse_xdmf(self.xdmf_path)
 
        print("===== ResultVisualizer =====")
        print(f"网格节点数 : {self.coords.shape[0]}")
        print(f"坐标维度   : {self.coords.shape[1]}D")
        for fname, records in self._field_records.items():
            ts = [r['time'] for r in records]
            print(f"场 '{fname}' : {len(records)} 步, "
                  f"t ∈ [{ts[0]:.4f}, {ts[-1]:.4f}] s")
        print("============================\n")
 
    def _field_times(self, field_name: str) -> np.ndarray:
        return np.array([r['time'] for r in self._field_records[field_name]])
 
    def _read_field_step(self, field_name: str, step_index: int) -> np.ndarray:
        """读取指定场、指定时间步的数值（flat array）"""
        record = self._field_records[field_name][step_index]
        with h5py.File(self.h5_path, 'r') as f:
            return np.array(f[record['h5_path']]).ravel()
 
    def _select_nodes_at_y(self, y_target: float, tol: float):
        """选取 y ≈ y_target 处的节点，返回 (x_vals, indices, y_vals)"""
        y = self.coords[:, 1]
        mask = np.abs(y - y_target) < tol
 
        if mask.sum() < 3:
            dist = np.abs(y - y_target)
            n_want = min(100, len(dist))
            tol_auto = np.sort(dist)[n_want - 1] * 1.01
            mask = dist <= tol_auto
            print(f"容差自动扩大至 {tol_auto:.4f} m，选中 {mask.sum()} 个节点")
        else:
            print(f"y = {y_target} ± {tol} m 选中 {mask.sum()} 个节点"
                  f" (y 范围 [{y[mask].min():.4f}, {y[mask].max():.4f}])")
 
        indices = np.where(mask)[0]
        x_vals  = self.coords[indices, 0]
        order   = np.argsort(x_vals)
        return x_vals[order], indices[order], y[mask][order]
 
    # ──────────────── 绘图方法 ──────────────── #
 
    def plot_pressure_at_y(
        self,
        times_to_plot,
        y_target: float = 3.8,
        tol: float = 0.05,
        unit: str = 'kPa',
        save: bool = True,
        figsize: tuple = (13, 7),
    ):
        """
        绘制 y = y_target 高度处指定时刻的压力沿 x 方向分布曲线。
 
        参数
        ----
        times_to_plot    要绘制的时刻，单位 s。
                         可以是单个数值，例如 0.05，
                         也可以是列表，例如 [0.01, 0.03, 0.05]。
                         程序自动匹配最近的时间步。
        y_target         目标 y 坐标 (m)
        tol              y 坐标容差 (m)
        unit             压力单位：'Pa' / 'kPa' / 'MPa'
        save             是否保存图片
        figsize          图片尺寸
        """
        scale = {'Pa': 1.0, 'kPa': 1e-3, 'MPa': 1e-6}.get(unit, 1e-3)
 
        if 'pressure' not in self._field_records:
            raise RuntimeError("XDMF/HDF5 中未找到 'pressure' 字段")
 
        # 统一转为列表
        if np.isscalar(times_to_plot):
            times_to_plot = [times_to_plot]
 
        x_line, node_indices, _ = self._select_nodes_at_y(y_target, tol)
        n_total = len(self._field_records['pressure'])
        times   = self._field_times('pressure')
 
        # ── 将输入时刻匹配到最近时间步，保持输入顺序 ──
        step_indices = []
        for t_req in times_to_plot:
            idx = int(np.argmin(np.abs(times - t_req)))
            step_indices.append(idx)
            print(f"请求 t = {t_req:.6f} s → 匹配到索引 {idx}, 实际时间 {times[idx]:.6f} s")
 
        if not step_indices:
            raise ValueError("没有有效的时间步被选中")
 
        # ── 绘图 ──
        fig, ax = plt.subplots(figsize=figsize)
        colors = plt.cm.coolwarm(np.linspace(0.05, 0.95, len(step_indices)))
 
        print("\n选中的时间步及对应时间：")
        for k, i in enumerate(step_indices):
            t = times[i]
            print(f"  {k+1}: 索引={i:3d}, 时间={t:.6f} s")
 
            p_line = self._read_field_step('pressure', i)[node_indices] * scale
 
            # 凿孔段压力置零（x ∈ [1.95, 2.05]）
            p_line[(x_line >= 1.95) & (x_line <= 2.05)] = 0.0
 
            ax.plot(x_line, p_line,
                    linestyle='-', linewidth=1.8,
                    color=colors[k], label=f't = {t:.3f} s')
 
        # 图例
        n_entries = len(step_indices)
        if n_entries > 8:
            ax.legend(fontsize=9, loc='upper left',
                      bbox_to_anchor=(1.02, 1), ncol=1,
                      framealpha=0.9, edgecolor='gray')
            plt.subplots_adjust(right=0.85)
        else:
            ax.legend(fontsize=10, loc='best',
                      ncol=min(3, n_entries),
                      framealpha=0.9, edgecolor='gray')
 
        ax.set_xlabel('x (m)', fontsize=14)
        ax.set_ylabel(f'Pressure ({unit})', fontsize=14)
        ax.set_title(f'Pressure distribution at y = {y_target:.2f} m', fontsize=16)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=12)
        plt.tight_layout()
 
        if save:
            out = self.results_dir / f"pressure_at_y{y_target:.1f}.png"
            fig.savefig(out, dpi=200, bbox_inches='tight')
            print(f"图片已保存: {out}")
 
        plt.show()
        return fig, ax
 
    # ──────────────── 调试辅助 ──────────────── #
 
    def print_time_table(self, field_name: str = 'pressure', max_rows: int = 20):
        """打印时间步–H5路径对照表，方便验证对应关系是否正确"""
        records = self._field_records.get(field_name, [])
        if not records:
            print(f"未找到字段 '{field_name}'")
            return
        print(f"\n{'索引':>5}  {'时间 (s)':>14}  H5 路径")
        print("-" * 70)
        show = records if len(records) <= max_rows else (
            records[:max_rows // 2] + [None] + records[-max_rows // 2:]
        )
        for i, r in enumerate(records):
            if r is None:
                print("  ...")
                continue
            print(f"  {i:>3}  {r['time']:>14.6f}  {r['h5_path']}")
 
 
# ─────────────────────────── 入口 ─────────────────────────── #
 
if __name__ == "__main__":
    results_dir = "/root/shared/Dynamic_simple_2D/results/results"
    viz = ResultVisualizer(results_dir)
 
    # 可选：打印时间步-路径对照表，验证时间步是否正确
    #viz.print_time_table('pressure')
 
    # 绘制指定时刻的压力曲线（单个时刻或列表均可）
    viz.plot_pressure_at_y(
        times_to_plot=[30, 60, 90,120],   # 修改为你需要的时刻 (s)
        y_target=3.8,
        unit='kPa',
        save=True,
    )