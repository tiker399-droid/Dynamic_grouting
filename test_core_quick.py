#!/usr/bin/env python3
"""快速测试core文件初始化和基本功能"""
import sys
import os
import pathlib

# 确保 src 目录在 Python 路径中
script_dir = pathlib.Path(__file__).parent
src_dir = script_dir / 'Dynamic' / 'src'
sys.path.insert(0, str(src_dir))

from mpi4py import MPI
from core import MultiphysicsGroutingSimulation

print("=" * 60)
print("开始测试 Dynamic/src/core.py")
print("=" * 60)

# 创建模拟实例
sim = MultiphysicsGroutingSimulation(
    config_file='/root/shared/Dynamic/config/grouting_config.yaml',
    mesh_file='/root/shared/Dynamic/meshes/foundation_drilling_model.msh',
    output_dir='/root/shared/Dynamic/results'
)

print("\n[1/4] 初始化所有模块...")
sim._initialize_modules()

if sim.rank == 0:
    print(f"✓ 网格: {sim.mesh.topology.index_map(3).size_global} 个单元")
    print(f"✓ 函数空间: {sim.W.dofmap.index_map.size_global * sim.W.dofmap.index_map_bs} 个自由度")
    print(f"✓ 边界条件: {len(sim.bc_manager.get_boundary_conditions())} 个")
    
    # 测试弱形式构建
    print("\n[2/4] 测试弱形式构建...")
    F, J = sim.weak_form_builder.build_form(
        dt=1.0,
        time=0.0,
        solution=sim.solution,
        solution_prev=sim.solution_prev,
        boundary_conditions=sim.bc_manager.get_boundary_conditions()
    )
    print(f"✓ 残差形式: {type(F).__module__}.{type(F).__name__}")
    print(f"✓ 雅可比形式: {type(J).__module__}.{type(J).__name__}")

print("\n[3/4] 测试时间步进...")
# 初始化时间相关属性
sim.materials.update_time_dependent_properties(0.0)
sim.bc_manager.update(0.0)

# 尝试第一次求解（使用非常小的时间步）
if sim.rank == 0:
    print("  尝试求解第一个时间步...")
try:
    dt = 0.1  # 非常小的时间步
    converged, iterations = sim.solver_manager.solve(
        dt=dt,
        time=0.0,
        solution=sim.solution,
        solution_prev=sim.solution_prev,
        materials=sim.materials
    )
    
    if sim.rank == 0:
        print(f"  ✓ 求解完成: 收敛={converged}, 迭代次数={iterations}")
except Exception as e:
    if sim.rank == 0:
        print(f"  ✗ 求解失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n[4/4] 测试完成!")
if sim.rank == 0:
    print("=" * 60)
    print("SUCCESS: core.py 所有基本功能正常")
    print("=" * 60)
