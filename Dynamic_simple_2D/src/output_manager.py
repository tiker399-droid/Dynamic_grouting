"""
输出管理器 - 处理模拟结果输出
支持主要物理场和衍生场的 XDMF 格式输出
简化两场系统 [位移, 压力]，独立函数空间版本
"""

from pathlib import Path
from dolfinx import io, fem
from mpi4py import MPI
import logging
import numpy as np
from typing import Dict, Any, Optional, List
import basix.ufl


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

        # 用于存储线性向量空间（供位移场插值使用）
        self._P1_vec_space = None
        
        # 存储初始位移场（第一步的解），用于计算位移增量
        self._u_initial = None
        self._u_initial_p1 = None
        self._is_first_step = True

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

    def write_timestep(
        self,
        time: float,
        time_step: int,
        u: fem.Function,
        p: fem.Function,
        derived_fields: Optional[Dict[str, Any]] = None
    ):
        """
        写入一个时间步的结果

        Args:
            time: 当前时间
            time_step: 当前时间步编号
            u: 位移函数
            p: 压力函数
            derived_fields: 衍生场字典，键为字段名，值为 Function 或可插值对象
        """
        # 检查保存频率
        if time_step % self.save_frequency != 0:
            return

        try:
            # 写入位移（如果需要）
            if 'displacement' in self.fields_to_save:
                # 创建线性向量空间（若尚未创建）
                if self._P1_vec_space is None:
                    gdim = self.mesh.geometry.dim
                    cell_type = self.mesh.topology.cell_name()
                    P1_vec = basix.ufl.element("Lagrange", cell_type, 1, shape=(gdim,))
                    self._P1_vec_space = fem.functionspace(self.mesh, P1_vec)

                # 插值到线性空间
                u_p1 = fem.Function(self._P1_vec_space)
                u_p1.interpolate(u)
                
                # 如果是第一步，保存初始位移
                if self._is_first_step:
                    self._u_initial_p1 = fem.Function(self._P1_vec_space)
                    self._u_initial_p1.interpolate(u)
                    self._is_first_step = False
                
                # 计算位移增量（当前位移 - 初始位移）
                u_increment = fem.Function(self._P1_vec_space)
                u_increment.x.array[:] = u_p1.x.array[:] - self._u_initial_p1.x.array[:]
                u_increment.name = "displacement"
                self.main_file.write_function(u_increment, time)

            # 写入压力（如果需要）
            if 'pressure' in self.fields_to_save:
                p.name = "pressure"
                self.main_file.write_function(p, time)

            # 写入衍生场
            if derived_fields and self.derived_file:
                for name, field_data in derived_fields.items():
                    # 如果 field_data 是 UFL 表达式，需先转换为 Function
                    if hasattr(field_data, 'write'):
                        field_data.name = name
                        self.derived_file.write_function(field_data, time)
                    else:
                        pass

            # 定期日志（可选）
            if self.rank == 0 and time_step % (self.save_frequency * 10) == 0:
                print(f"已写入时间步 {time_step}, t={time:.2f}s")

        except Exception as e:
            print(f"写入时间步 {time_step} 时出错: {e}")

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