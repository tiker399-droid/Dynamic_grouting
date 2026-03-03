"""
输出管理器 - 处理模拟结果输出
支持主要物理场和衍生场的 XDMF 格式输出
"""

from pathlib import Path
from dolfinx import io, fem
from mpi4py import MPI
import logging
import numpy as np
from typing import Dict, Any, Optional, List


class OutputManager:
    """专业的输出管理器，负责模拟结果的 I/O 操作"""

    def __init__(self, comm, config: Dict[str, Any], mesh, output_dir: Path):
        """
        初始化输出管理器

        Args:
            comm: MPI 通信器
            config: 完整配置字典
            mesh: 计算网格
            output_dir: 输出目录路径（Path 对象）
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.mesh = mesh
        self.output_dir = Path(output_dir)
        self.config = config
        self.logger = logging.getLogger(f"OutputManager_rank{self.rank}")

        # 从配置提取输出参数
        output_config = config.get('output', {})
        self.save_frequency = output_config.get('write_frequency', 10)
        self.fields_to_save = output_config.get('fields', [])
        self.output_format = output_config.get('format', 'xdmf')

        # 文件对象
        self.main_file = None
        self.derived_file = None

        # 创建输出目录（仅主进程）
        if self.rank == 0:
            results_dir = self.output_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"输出目录: {results_dir}")

        # 同步所有进程
        self.comm.Barrier()

        # 初始化输出文件
        self._initialize_files()

    def _initialize_files(self):
        """初始化 XDMF 输出文件"""
        import os
        try:
            # 主文件：主要物理场
            main_path = self.output_dir / "results" / "main_results.xdmf"
            h5_main_path = self.output_dir / "results" / "main_results.h5"
            
            # 衍生文件：衍生量
            derived_path = self.output_dir / "results" / "derived_fields.xdmf"
            h5_derived_path = self.output_dir / "results" / "derived_fields.h5"
            
            # 确保文件不存在（清理旧的）
            for fpath in [main_path, h5_main_path, derived_path, h5_derived_path]:
                if fpath.exists():
                    try:
                        fpath.unlink()
                    except Exception:
                        pass
            
            # 确保没有锁文件
            for fpath in [h5_main_path, h5_derived_path]:
                lock_path = Path(str(fpath) + ".lock")
                if lock_path.exists():
                    try:
                        lock_path.unlink()
                    except Exception:
                        pass
            
            # 同步所有进程
            self.comm.Barrier()
            
            # 创建 XDMF 文件
            self.main_file = io.XDMFFile(self.comm, str(main_path), "w")
            self.main_file.write_mesh(self.mesh)

            self.derived_file = io.XDMFFile(self.comm, str(derived_path), "w")
            self.derived_file.write_mesh(self.mesh)

            if self.rank == 0:
                self.logger.info(f"输出文件已创建: {main_path}")

        except Exception as e:
            self.logger.error(f"初始化输出文件失败: {e}")
            raise

    def _get_field_index(self, field_name: str) -> Optional[int]:
        """
        根据字段名获取在混合空间中的索引

        Args:
            field_name: 字段名

        Returns:
            索引或 None
        """
        # 4场系统：[位移, 压力, 孔隙度, 浓度]
        field_mapping = {
            'displacement': 0,
            'pressure': 1,
            'porosity': 2,
            'concentration': 3
        }
        return field_mapping.get(field_name)

    def write_timestep(
        self,
        time: float,
        time_step: int,
        solution: fem.Function,
        derived_fields: Optional[Dict[str, Any]] = None
    ):
        """
        写入一个时间步的结果

        Args:
            time: 当前时间
            time_step: 当前时间步编号
            solution: 混合函数（包含所有场）
            derived_fields: 衍生场字典，键为字段名，值为 Function 或可插值对象
        """
        # 检查保存频率
        if time_step % self.save_frequency != 0:
            return

        try:
            # 写入主要物理场
            for field_name in self.fields_to_save:
                idx = self._get_field_index(field_name)
                if idx is not None:
                    # 提取子函数并 collapse 到独立空间
                    field_func = solution.sub(idx).collapse()
                    field_func.name = field_name
                    self.main_file.write_function(field_func, time)

            # 写入衍生场
            if derived_fields and self.derived_file:
                for name, field_data in derived_fields.items():
                    # 如果 field_data 是 UFL 表达式，需先转换为 Function
                    # 假设调用者已转换为 Function 或提供可写的对象
                    if hasattr(field_data, 'write'):
                        field_data.name = name
                        self.derived_file.write_function(field_data, time)
                    else:
                        self.logger.warning(
                            f"衍生场 '{name}' 不是可写的 Function 对象，跳过"
                        )

            # 定期日志（可选）
            if self.rank == 0 and time_step % (self.save_frequency * 10) == 0:
                self.logger.debug(f"已写入时间步 {time_step}, t={time:.2f}s")

        except Exception as e:
            self.logger.warning(f"写入时间步 {time_step} 时出错: {e}")

    def close(self):
        """关闭所有输出文件"""
        try:
            if self.main_file:
                self.main_file.close()
            if self.derived_file:
                self.derived_file.close()
            if self.rank == 0:
                self.logger.info("输出文件已关闭")
        except Exception as e:
            self.logger.warning(f"关闭文件时出错: {e}")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，自动关闭文件"""
        self.close()