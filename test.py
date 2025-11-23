import numpy as np
import pandas as pd

class ReservoirFloodControl:
    def __init__(self):
        # 水库特征曲线
        self.water_levels = [147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 159.5]
        self.storages = [23.38, 24.78, 26.22, 27.71, 29.25, 30.31, 32.41, 34.05, 35.72, 37.43, 39.18, 40.98, 42.81, 43.75]
        self.discharge_capacities = [1800, 2150, 2600, 3130, 3690, 4370, 5510, 6200, 6910, 7920, 8730, 9540, 10340, 10750]
        
        # 洪水过程数据
        self.inflow_data = [
            # 7月31日
            (7, 31, 0, 6780), (7, 31, 3, 7500), (7, 31, 6, 7950), (7, 31, 9, 9850),
            (7, 31, 12, 11100), (7, 31, 15, 13220), (7, 31, 18, 16500), (7, 31, 21, 13600),
            # 8月1日
            (8, 1, 0, 12950), (8, 1, 3, 12700), (8, 1, 6, 12200), (8, 1, 9, 11300),
            (8, 1, 12, 9950), (8, 1, 15, 9400), (8, 1, 18, 8500), (8, 1, 21, 7000),
            # 8月2日
            (8, 2, 0, 5200)
        ]
        
        # 初始条件
        self.initial_conditions = {
            'month': 7, 'day': 31, 'hour': 0,
            'water_level': 153.72,  # 从第一阶段计算结果
            'storage': 33.592,      # 从第一阶段计算结果
            'outflow': 600          # 从第一阶段计算结果
        }
        
        # 泄流控制规则
        self.discharge_rules = {
            'low': 600,     # 入库流量 < 7950 m³/s
            'medium': 1500, # 7950 ≤ 入库流量 ≤ 9850 m³/s
            'high': 'capacity'  # 入库流量 > 9850 m³/s
        }
        
        # 特殊规则：一旦入库流量超过9850，后续全部按泄流能力泄水
        self.special_rule = True
        
    def interpolate_water_level(self, storage):
        """根据库容插值计算水位"""
        if storage <= self.storages[0]:
            return self.water_levels[0]
        if storage >= self.storages[-1]:
            # 外推
            return self.water_levels[-1] + (storage - self.storages[-1]) / (self.storages[-1] - self.storages[-2]) * (self.water_levels[-1] - self.water_levels[-2])
        
        for i in range(len(self.storages)-1):
            if self.storages[i] <= storage <= self.storages[i+1]:
                ratio = (storage - self.storages[i]) / (self.storages[i+1] - self.storages[i])
                return self.water_levels[i] + ratio * (self.water_levels[i+1] - self.water_levels[i])
        
        return self.water_levels[-1]
    
    def interpolate_discharge_capacity(self, water_level):
        """根据水位插值计算泄流能力"""
        if water_level <= self.water_levels[0]:
            return self.discharge_capacities[0]
        if water_level >= self.water_levels[-1]:
            # 外推
            return self.discharge_capacities[-1] + (water_level - self.water_levels[-1]) / (self.water_levels[-1] - self.water_levels[-2]) * (self.discharge_capacities[-1] - self.discharge_capacities[-2])
        
        for i in range(len(self.water_levels)-1):
            if self.water_levels[i] <= water_level <= self.water_levels[i+1]:
                ratio = (water_level - self.water_levels[i]) / (self.water_levels[i+1] - self.water_levels[i])
                return self.discharge_capacities[i] + ratio * (self.discharge_capacities[i+1] - self.discharge_capacities[i])
        
        return self.discharge_capacities[-1]
    
    def calculate_discharge_by_rule(self, inflow, use_capacity=False):
        """根据泄流规则计算下泄流量"""
        if use_capacity:
            return 'capacity'
        elif inflow < 7950:
            return self.discharge_rules['low']
        elif 7950 <= inflow <= 9850:
            return self.discharge_rules['medium']
        else:
            return self.discharge_rules['high']
    
    def trial_calculation(self, current_storage, current_outflow, avg_inflow, target_outflow, time_step=10800, tolerance=10, max_iterations=10):
        """试算法计算时段末状态"""
        for iteration in range(max_iterations):
            # 计算时段平均下泄流量
            avg_outflow = (current_outflow + target_outflow) / 2
            
            # 计算库容变化 (单位: 10^8 m³)
            delta_storage = (avg_inflow - avg_outflow) * time_step / 1e8
            
            # 计算时段末库容
            end_storage = current_storage + delta_storage
            
            # 计算时段末水位
            end_water_level = self.interpolate_water_level(end_storage)
            
            # 计算泄流能力
            discharge_capacity = self.interpolate_discharge_capacity(end_water_level)
            
            # 计算误差
            error = discharge_capacity - target_outflow
            
            print(f"  迭代 {iteration+1}: 假设O={target_outflow}, O_avg={avg_outflow:.0f}, "
                  f"ΔV={delta_storage:.4f}, V₂={end_storage:.3f}, Z={end_water_level:.2f}, "
                  f"Q={discharge_capacity:.0f}, 误差={error:.0f}")
            
            # 检查收敛
            if abs(error) <= tolerance:
                return target_outflow, end_storage, end_water_level, True
            
            # 调整假设值
            if iteration < max_iterations - 1:
                target_outflow += error * 0.5  # 使用较小的调整步长
        
        print(f"  警告: 在{max_iterations}次迭代后未收敛")
        return target_outflow, end_storage, end_water_level, False
    
    def calculate_flood_control(self):
        """执行防洪调节计算"""
        results = []
        
        # 初始状态
        current_month = self.initial_conditions['month']
        current_day = self.initial_conditions['day']
        current_hour = self.initial_conditions['hour']
        current_water_level = self.initial_conditions['water_level']
        current_storage = self.initial_conditions['storage']
        current_outflow = self.initial_conditions['outflow']
        
        # 特殊规则标志
        use_capacity_rule = False
        
        # 遍历洪水过程
        for i, (month, day, hour, inflow) in enumerate(self.inflow_data):
            print(f"\n计算时段: {month}/{day} {hour}时")
            
            # 计算时段平均入库流量
            if i == 0:
                # 第一个时段，使用初始出流和当前入库流量
                avg_inflow = (self.initial_conditions['outflow'] + inflow) / 2
            else:
                # 使用前一时段入库流量和当前入库流量
                prev_inflow = self.inflow_data[i-1][3]
                avg_inflow = (prev_inflow + inflow) / 2
            
            print(f"  入库流量: {inflow}, 平均入库: {avg_inflow:.0f}")
            
            # 检查是否需要启用特殊规则
            if not use_capacity_rule and inflow > 9850:
                use_capacity_rule = True
                print("  触发特殊规则: 此后所有时段按泄流能力泄水")
            
            # 确定下泄流量规则
            discharge_rule = self.calculate_discharge_by_rule(inflow, use_capacity_rule)
            
            if discharge_rule == 'capacity':
                # 需要试算法
                print("  使用试算法确定下泄流量:")
                # 初始假设值
                initial_guess = self.interpolate_discharge_capacity(current_water_level)
                target_outflow, end_storage, end_water_level, converged = self.trial_calculation(
                    current_storage, current_outflow, avg_inflow, initial_guess)
                
                if not converged:
                    print("  注意: 试算法未完全收敛，使用最后迭代结果")
                
            else:
                # 直接使用规则确定的下泄流量
                target_outflow = discharge_rule
                # 计算时段平均下泄流量
                avg_outflow = (current_outflow + target_outflow) / 2
                # 计算库容变化
                delta_storage = (avg_inflow - avg_outflow) * 10800 / 1e8
                # 计算时段末库容
                end_storage = current_storage + delta_storage
                # 计算时段末水位
                end_water_level = self.interpolate_water_level(end_storage)
                
                print(f"  直接计算: O={target_outflow}, O_avg={avg_outflow:.0f}, "
                      f"ΔV={delta_storage:.4f}, V₂={end_storage:.3f}, Z={end_water_level:.2f}")
            
            # 记录结果
            results.append({
                'month': month,
                'day': day,
                'hour': hour,
                'inflow': inflow,
                'outflow': target_outflow,
                'storage': end_storage,
                'water_level': end_water_level,
                'avg_inflow': avg_inflow
            })
            
            # 更新当前状态
            current_storage = end_storage
            current_water_level = end_water_level
            current_outflow = target_outflow
        
        return results
    
    def print_results(self, results):
        """打印计算结果"""
        print("\n" + "="*80)
        print("防洪调节计算结果")
        print("="*80)
        print(f"{'时间':<10} {'入库流量':<10} {'下泄流量':<10} {'库容':<10} {'水位':<10}")
        print("-"*50)
        
        for result in results:
            time_str = f"{result['month']}/{result['day']} {result['hour']}时"
            print(f"{time_str:<10} {result['inflow']:<10.0f} {result['outflow']:<10.0f} "
                  f"{result['storage']:<10.3f} {result['water_level']:<10.2f}")
        
        # 统计信息
        max_water_level = max(r['water_level'] for r in results)
        max_outflow = max(r['outflow'] for r in results)
        max_water_level_time = next(r for r in results if r['water_level'] == max_water_level)
        
        print("\n统计信息:")
        print(f"设计洪水位: {max_water_level:.2f} m "
              f"(发生在{max_water_level_time['month']}/{max_water_level_time['day']} {max_water_level_time['hour']}时)")
        print(f"最大下泄流量: {max_outflow:.0f} m³/s")

# 执行计算
if __name__ == "__main__":
    calculator = ReservoirFloodControl()
    results = calculator.calculate_flood_control()
    calculator.print_results(results)
