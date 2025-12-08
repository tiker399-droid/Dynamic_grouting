"""
多物理场耦合注浆模拟主程序
"""

import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from .core import MultiphysicsGroutingSimulation


def main():
    """主函数"""
    # 配置文件路径
    config_file = "config/parameters.yaml"
    mesh_file = "meshes/foundation_drilling_model.msh"
    output_dir = "results/multiphysics_simulation"
    
    # 检查文件是否存在
    if not Path(config_file).exists():
        print(f"Error: Config file not found: {config_file}")
        return 1
    
    if not Path(mesh_file).exists():
        print(f"Error: Mesh file not found: {mesh_file}")
        return 1
    
    try:
        # 创建模拟器
        print("=" * 60)
        print("MULTIPHYSICS GROUTING SIMULATION")
        print("Based on Acta Geotechnica (2023) 18:553-571")
        print("=" * 60)
        
        simulator = MultiphysicsGroutingSimulation(
            config_file=config_file,
            mesh_file=mesh_file,
            output_dir=output_dir
        )
        
        # 运行模拟
        print("\nStarting simulation...")
        simulator.run()
        
        # 后处理
        print("\nPost-processing...")
        simulator.post_process()
        
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETED SUCCESSFULLY")
        print(f"Results saved to: {output_dir}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())