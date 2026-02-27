class OutputManager:
    """专业的输出管理器"""
    
    def __init__(self, comm, config, mesh, output_dir):
        self.comm = comm
        self.config = config
        self.mesh = mesh
        self.output_dir = Path(output_dir)
        self.rank = comm.Get_rank()
        
        # 创建输出目录
        if self.rank == 0:
            (self.output_dir / "results").mkdir(exist_ok=True)
        comm.Barrier()
        
        # 输出频率
        self.save_frequency = config['output'].get('write_frequency', 10)
        self.fields_to_save = config['output'].get('fields', [])
        
        # 文件格式：主文件 + 衍生文件
        self.main_file = None
        self.derived_file = None
        self._initialize_files()
    
    def _initialize_files(self):
        """初始化输出文件"""
        # 主文件：包含主要物理场
        main_path = self.output_dir / "results/main_results.xdmf"
        self.main_file = io.XDMFFile(self.comm, str(main_path), "w")
        self.main_file.write_mesh(self.mesh)
        
        # 衍生文件：包含渗透率、粘度等衍生量
        derived_path = self.output_dir / "results/derived_fields.xdmf"
        self.derived_file = io.XDMFFile(self.comm, str(derived_path), "w")
        self.derived_file.write_mesh(self.mesh)
    
    def write_timestep(self, time, time_step, solution, derived_fields=None):
        """写入时间步结果"""
        if time_step % self.save_frequency != 0:
            return
        
        # 保存主要物理场到主文件
        for field_name in self.fields_to_save:
            field_idx = self._get_field_index(field_name)
            if field_idx is not None:
                field_func = solution.sub(field_idx).collapse()
                field_func.name = field_name
                self.main_file.write_function(field_func, time)
        
        # 保存衍生量
        if derived_fields:
            for field_name, field_data in derived_fields.items():
                # 假设field_data是Function或可插值对象
                self.derived_file.write_function(field_data, time)
    
    def _get_field_index(self, field_name):
        """根据字段名获取在混合空间中的索引"""
        field_mapping = {
            'displacement': 0,
            'pressure': 1,
            'porosity': 2,
            'concentration': 3,
            'darcy_velocity': 4
        }
        return field_mapping.get(field_name)
    
    def close(self):
        """关闭所有文件"""
        if self.main_file:
            self.main_file.close()
        if self.derived_file:
            self.derived_file.close()