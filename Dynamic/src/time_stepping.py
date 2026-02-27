"""
时间步进管理器 - 地基注浆模拟（简化版）
集成注浆过程监控、屏浆模拟和基础自适应时间步长控制
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Callable
import logging
from enum import Enum
from dataclasses import dataclass

@dataclass
class GroutingStageStatus:
    """注浆阶段状态（简化版）"""
    stage_name: str = "before_grouting"
    start_time: float = 0.0
    duration: float = 0.0
    current_pressure: float = 0.0
    target_pressure: float = 0.0
    is_complete: bool = False

@dataclass
class PressureHoldingStatus:
    """屏浆状态（简化版）"""
    is_active: bool = False
    start_time: float = 0.0
    current_duration: float = 0.0
    target_duration: float = 1200.0  # 20分钟
    recent_injection_rates: List[float] = None  # 只保留最近的注入率
    average_injection_rate: float = 0.0
    is_complete: bool = False
    
    def __post_init__(self):
        if self.recent_injection_rates is None:
            self.recent_injection_rates = []

@dataclass
class TerminationCriteria:
    """终止条件（简化版）"""
    max_simulation_time: float = 3600.0
    max_time_steps: int = 10000
    min_time_step: float = 0.01
    grouting_complete: bool = False

class GroutingStage(Enum):
    """注浆阶段枚举（简化版）"""
    BEFORE_GROUTING = "before_grouting"
    PRESSURE_RISING = "pressure_rising"
    PRESSURE_HOLDING = "pressure_holding"
    COMPLETED = "completed"

class TimeIntegrationMethod(Enum):
    """时间积分方法（简化版）"""
    BACKWARD_EULER = "backward_euler"
    CRANK_NICOLSON = "crank_nicolson"

class AdaptiveTimeStepStrategy(Enum):
    """自适应时间步策略（简化版）"""
    FIXED = "fixed"
    BASIC = "basic"

class TimeStepManager:
    """时间步管理器 - 集成注浆监控"""
    
    def __init__(self, config: Dict[str, Any], comm=None):
        """
        初始化时间步管理器
        
        Args:
            config: 配置字典
            comm: MPI通信器（可选）
        """
        # 通信设置
        self.comm = comm
        self.rank = comm.Get_rank() if comm is not None else 0
        
        # 设置日志
        self.logger = logging.getLogger(f"TimeStepManager_rank{self.rank}")
        
        # 从配置中提取参数
        self._extract_configuration(config)
        
        # 初始化时间状态
        self.time = self.initial_time
        self.time_step = 0
        self.converged_steps = 0
        self.failed_steps = 0
        
        # 时间步长控制
        self.dt = self.initial_dt
        self.previous_dt = self.dt
        self.dt_history = [self.dt]
        
        # 时间历史记录
        self.time_history = [self.time]
        self.convergence_history = []
        
        # 注浆阶段管理
        self.grouting_stage = GroutingStage.BEFORE_GROUTING
        self.grouting_stages = {}
        self._initialize_grouting_stages()
        
        # 屏浆状态
        self.pressure_holding = PressureHoldingStatus(
            target_duration=self.pressure_holding_duration
        )
        
        # 终止条件
        self.termination_criteria = TerminationCriteria(
            max_simulation_time=self.total_time,
            max_time_steps=self.max_steps,
            min_time_step=self.min_dt
        )
        
        # 注入率计算函数（由求解器提供）
        self.injection_rate_calculator = None
        
        # 简化的统计信息
        self.pressure_history = []
        self.current_injection_rate = 0.0
        
        if self.rank == 0:
            self.logger.info("=" * 60)
            self.logger.info("时间步管理器初始化完成")
            self.logger.info(f"总模拟时间: {self.total_time}秒")
            self.logger.info(f"初始时间步长: {self.initial_dt}秒")
            self.logger.info(f"时间积分方法: {self.time_integration_method.value}")
            self.logger.info(f"自适应策略: {self.adaptive_strategy.value}")
            self.logger.info(f"屏浆时间: {self.pressure_holding_duration/60:.1f}分钟")
            self.logger.info("=" * 60)
    
    def _extract_configuration(self, config: Dict[str, Any]):
        """从配置中提取参数"""
        sim_config = config.get('simulation', {})
        
        # 基础时间参数
        self.initial_time = sim_config.get('initial_time', 0.0)
        self.total_time = sim_config.get('total_time', 7200.0)  # 2小时
        self.initial_dt = sim_config.get('dt_initial', 1.0)
        self.min_dt = sim_config.get('dt_min', 0.01)
        self.max_dt = sim_config.get('dt_max', 60.0)
        
        # 步数限制
        self.max_steps = sim_config.get('max_steps', 10000)
        self.max_consecutive_failures = sim_config.get('max_consecutive_failures', 5)
        
        # 收敛容差
        self.tolerance = sim_config.get('tolerance', 1e-6)
        self.relative_tolerance = sim_config.get('relative_tolerance', 1e-4)
        
        # 时间积分方法
        method_name = sim_config.get('time_integration_method', 'backward_euler')
        try:
            self.time_integration_method = TimeIntegrationMethod(method_name)
        except ValueError:
            self.time_integration_method = TimeIntegrationMethod.BACKWARD_EULER
        
        # 自适应时间步策略
        strategy_name = sim_config.get('adaptive_strategy', 'basic')
        try:
            self.adaptive_strategy = AdaptiveTimeStepStrategy(strategy_name)
        except ValueError:
            self.adaptive_strategy = AdaptiveTimeStepStrategy.BASIC
        
        # 自适应参数
        self.adaptive_grow_factor = sim_config.get('adaptive_grow_factor', 1.2)
        self.adaptive_reduce_factor = sim_config.get('adaptive_reduce_factor', 0.5)
        self.max_dt_change_ratio = sim_config.get('max_dt_change_ratio', 10.0)
        
        # 注浆参数
        grout_config = config.get('materials', {}).get('grout', {})
        self.max_grouting_pressure = grout_config.get('pressure', 220e3)
        self.grouting_duration = grout_config.get('duration', 600.0)
        self.pressure_rise_time = grout_config.get('rise_time', 60.0)
        
        # 屏浆参数
        termination_config = config.get('grouting_termination', {})
        self.pressure_holding_duration = termination_config.get('holding_time', 1200.0)  # 20分钟
        self.injection_rate_threshold = termination_config.get('injection_rate_threshold', 2.0)  # L/min
        self.holding_rate_threshold = termination_config.get('holding_rate_threshold', 2.0)  # L/min
        
        # 输出频率
        output_config = config.get('output', {})
        self.save_frequency = output_config.get('write_frequency', 10)
        self.monitor_frequency = output_config.get('monitor_frequency', 100)
    
    def _initialize_grouting_stages(self):
        """初始化注浆阶段（简化版）"""
        # 阶段1: 注浆前
        self.grouting_stages[GroutingStage.BEFORE_GROUTING] = GroutingStageStatus(
            stage_name="before_grouting",
            target_pressure=0.0
        )
        
        # 阶段2: 压力上升期（包含最大压力期）
        self.grouting_stages[GroutingStage.PRESSURE_RISING] = GroutingStageStatus(
            stage_name="pressure_rising",
            target_pressure=self.max_grouting_pressure
        )
        
        # 阶段3: 屏浆期
        self.grouting_stages[GroutingStage.PRESSURE_HOLDING] = GroutingStageStatus(
            stage_name="pressure_holding",
            target_pressure=self.max_grouting_pressure
        )
        
        # 阶段4: 完成
        self.grouting_stages[GroutingStage.COMPLETED] = GroutingStageStatus(
            stage_name="completed",
            target_pressure=0.0,
            is_complete=True
        )
    
    def set_injection_rate_calculator(self, calculator: Callable):
        """
        设置注入率计算函数
        solve函数中得到
        Args:
            calculator: 注入率计算函数，返回L/min
        """
        self.injection_rate_calculator = calculator
        if self.rank == 0:
            self.logger.debug("注入率计算函数已设置")
    
    def calculate_injection_rate(self, *args, **kwargs) -> float:
        """
        计算当前注入率
        
        Returns:
            注入率 (L/min)
        """
        if self.injection_rate_calculator is not None:
            try:
                return self.injection_rate_calculator(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"注入率计算失败: {e}")
                return 0.0
        return 0.0
    
    def update_grouting_status(self, current_pressure: float, injection_rate: float,
                              _grout_volume: Optional[float] = None):
        """
        更新注浆状态（简化版）
        
        Args:
            current_pressure: 当前注浆压力 (Pa)
            injection_rate: 当前注入率 (L/min)
            _grout_volume: 当前浆液体积 (m³，未使用，保留接口兼容性)
        """
        # 更新历史记录
        self.pressure_history.append((self.time, current_pressure))
        self.current_injection_rate = injection_rate
        
        # 更新当前阶段状态
        current_stage = self.grouting_stages[self.grouting_stage]
        current_stage.current_pressure = current_pressure
        current_stage.duration = self.time - current_stage.start_time
        
        # 判断是否应该进入下一阶段
        self._update_grouting_stage(current_pressure, injection_rate)
        
        # 更新屏浆状态（如果处于屏浆期）
        if self.grouting_stage == GroutingStage.PRESSURE_HOLDING:
            self._update_pressure_holding_status(injection_rate)
        
        # 定期输出状态
        if self.time_step % self.monitor_frequency == 0 and self.rank == 0:
            self._log_grouting_status(current_pressure, injection_rate, None)
    
    def _update_grouting_stage(self, current_pressure: float, injection_rate: float):
        """更新注浆阶段（简化版）"""
        old_stage = self.grouting_stage
        
        # 状态转移逻辑
        if self.grouting_stage == GroutingStage.BEFORE_GROUTING:
            if current_pressure > 0.01 * self.max_grouting_pressure:
                self._transition_to_stage(GroutingStage.PRESSURE_RISING)
        
        elif self.grouting_stage == GroutingStage.PRESSURE_RISING:
            # 当压力接近最大值且注入率降到阈值以下时，开始屏浆
            if (current_pressure >= 0.95 * self.max_grouting_pressure and 
                injection_rate <= self.injection_rate_threshold):
                self._transition_to_stage(GroutingStage.PRESSURE_HOLDING)
        
        elif self.grouting_stage == GroutingStage.PRESSURE_HOLDING:
            # 屏浆完成条件在_update_pressure_holding_status中处理
            pass
        
        # 记录阶段变化
        if old_stage != self.grouting_stage and self.rank == 0:
            self.logger.info(f"注浆阶段变化: {old_stage.value} → {self.grouting_stage.value}")
    
    def _transition_to_stage(self, new_stage: GroutingStage):
        """过渡到新阶段"""
        old_stage = self.grouting_stage
        old_stage_status = self.grouting_stages[old_stage]
        old_stage_status.is_complete = True
        
        # 更新当前阶段
        self.grouting_stage = new_stage
        new_stage_status = self.grouting_stages[new_stage]
        new_stage_status.start_time = self.time
        new_stage_status.is_complete = False
        
        # 特殊处理屏浆阶段
        if new_stage == GroutingStage.PRESSURE_HOLDING:
            self.pressure_holding.is_active = True
            self.pressure_holding.start_time = self.time
            self.pressure_holding.recent_injection_rates.clear()
            if self.rank == 0:
                self.logger.info(f"开始屏浆期，目标持续时间: {self.pressure_holding_duration/60:.1f}分钟")
        
        # 记录阶段过渡
        if self.rank == 0:
            self.logger.info(f"进入阶段: {new_stage_status.stage_name}")
    
    def _update_pressure_holding_status(self, injection_rate: float):
        """更新屏浆状态（简化版）"""
        if not self.pressure_holding.is_active:
            return
        
        # 记录最近的注入率（只保留最近100个值）
        self.pressure_holding.recent_injection_rates.append(injection_rate)
        if len(self.pressure_holding.recent_injection_rates) > 100:
            self.pressure_holding.recent_injection_rates.pop(0)
        
        # 计算屏浆持续时间
        self.pressure_holding.current_duration = self.time - self.pressure_holding.start_time
        
        # 计算屏浆期间的平均注入率
        if len(self.pressure_holding.recent_injection_rates) > 0:
            self.pressure_holding.average_injection_rate = np.mean(self.pressure_holding.recent_injection_rates)
        
        # 检查屏浆是否完成
        self._check_pressure_holding_completion()
    
    def _check_pressure_holding_completion(self):
        """检查屏浆是否完成"""
        # 屏浆完成条件：
        # 1. 屏浆时间达到20分钟
        # 2. 屏浆期间平均注入率不大于2L/min
        
        duration_condition = self.pressure_holding.current_duration >= self.pressure_holding.target_duration
        rate_condition = self.pressure_holding.average_injection_rate <= self.holding_rate_threshold
        
        if duration_condition and rate_condition:
            self.pressure_holding.is_complete = True
            self._transition_to_stage(GroutingStage.AFTER_GROUTING)
            
            if self.rank == 0:
                self.logger.info("=" * 60)
                self.logger.info("屏浆完成!")
                self.logger.info(f"屏浆持续时间: {self.pressure_holding.current_duration/60:.1f}分钟")
                self.logger.info(f"平均注入率: {self.pressure_holding.average_injection_rate:.2f} L/min")
                self.logger.info("=" * 60)
    
    def _log_grouting_status(self, pressure: float, injection_rate: float, _volume: Optional[float]):
        """记录注浆状态"""
        stage_info = self.grouting_stages[self.grouting_stage]
        
        status_msg = (
            f"时间步 {self.time_step}: "
            f"t={self.time:.1f}s, "
            f"dt={self.dt:.2f}s, "
            f"阶段={stage_info.stage_name}, "
            f"压力={pressure/1000:.1f}kPa, "
            f"注入率={injection_rate:.2f}L/min"
        )
        
        if self.grouting_stage == GroutingStage.PRESSURE_HOLDING:
            holding_time = self.pressure_holding.current_duration
            avg_rate = self.pressure_holding.average_injection_rate
            status_msg += f", 屏浆{holding_time/60:.1f}min, 平均注入率={avg_rate:.2f}L/min"
        
        self.logger.info(status_msg)
    
    def advance(self, converged: bool = True, iterations: int = 0, 
                _residual: float = 0.0) -> Tuple[float, bool]:
        """
        推进到下一个时间步
        
        Args:
            converged: 当前步是否收敛
            iterations: 迭代次数
            residual: 残差
            
        Returns:
            (时间步长, 是否应该继续)
        """
        # 检查终止条件
        if not self.should_continue():
            return 0.0, False
        
        # 调整时间步长
        if converged:
            self.converged_steps += 1
            self.consecutive_failures = 0
            self.dt = self._adjust_time_step(converged=True, iterations=iterations)
        else:
            self.failed_steps += 1
            self.consecutive_failures += 1
            self.dt = self._adjust_time_step(converged=False, iterations=iterations)
            
            # 检查连续失败次数
            if self.consecutive_failures > self.max_consecutive_failures:
                self.logger.error(f"连续{self.consecutive_failures}次求解失败，终止模拟")
                return 0.0, False
        
        # 确保时间步长在合理范围内
        self.dt = max(self.min_dt, min(self.dt, self.max_dt))
        
        # 确保不会超过总时间
        if self.time + self.dt > self.total_time:
            self.dt = self.total_time - self.time
        
        # 更新时间
        self.time += self.dt
        self.time_step += 1
        
        # 记录历史
        self.time_history.append(self.time)
        self.dt_history.append(self.dt)
        
        return self.dt, True
    
    def _adjust_time_step(self, converged: bool, iterations: int = 0) -> float:
        """
        根据收敛情况调整时间步长（简化版）
        
        Args:
            converged: 是否收敛
            iterations: 迭代次数
            
        Returns:
            调整后的时间步长
        """
        old_dt = self.dt
        
        if self.adaptive_strategy == AdaptiveTimeStepStrategy.FIXED:
            return old_dt
        
        if not converged:
            # 收敛失败，减小步长
            new_dt = old_dt * self.adaptive_reduce_factor
            if self.rank == 0:
                self.logger.debug(f"收敛失败，减小步长: {old_dt:.3e}s → {new_dt:.3e}s")
            return new_dt
        
        # 根据策略调整步长
        if self.adaptive_strategy == AdaptiveTimeStepStrategy.BASIC:
            new_dt = self._basic_adaptive_adjustment(old_dt, iterations)
        
        else:
            new_dt = old_dt
        
        # 限制变化幅度
        max_change = old_dt * self.max_dt_change_ratio
        min_change = old_dt / self.max_dt_change_ratio
        new_dt = max(min_change, min(new_dt, max_change))
        
        # 记录显著的步长变化
        if abs(new_dt - old_dt) / old_dt > 0.2 and self.rank == 0:
            self.logger.debug(f"时间步长调整: {old_dt:.3e}s → {new_dt:.3e}s")
        
        return new_dt
    
    def _basic_adaptive_adjustment(self, old_dt: float, iterations: int) -> float:
        """基础自适应调整"""
        if iterations < 5:
            # 迭代次数少，增大步长
            return old_dt * self.adaptive_grow_factor
        elif iterations > 15:
            # 迭代次数多，减小步长
            return old_dt * self.adaptive_reduce_factor
        else:
            return old_dt
    

    
    def should_continue(self) -> bool:
        """
        检查是否应该继续计算
        
        Returns:
            是否继续
        """
        # 检查时间限制
        if self.time >= self.total_time - 1e-10:
            self.termination_criteria.max_simulation_time = True
            if self.rank == 0:
                self.logger.info(f"达到总模拟时间: {self.total_time}s")
            return False
        
        # 检查步数限制
        if self.time_step >= self.max_steps:
            self.termination_criteria.max_time_steps = True
            if self.rank == 0:
                self.logger.warning(f"达到最大时间步数: {self.max_steps}")
            return False
        
        # 检查时间步长是否过小
        if self.dt < self.min_dt * 0.1:
            if self.rank == 0:
                self.logger.warning(f"时间步长过小: dt={self.dt:.3e}s")
            return False
        
        # 检查注浆是否完成
        if self.grouting_stage == GroutingStage.COMPLETED:
            self.termination_criteria.grouting_complete = True
            if self.rank == 0:
                self.logger.info("注浆过程完成，继续模拟固结过程")
        
        return True
    
    def get_time_integration_coefficients(self) -> Dict[str, float]:
        """
        获取时间积分系数
        
        Returns:
            积分系数字典
        """
        if self.time_integration_method == TimeIntegrationMethod.BACKWARD_EULER:
            return {'alpha_n': 1.0, 'alpha_np1': 1.0, 'beta': 1.0}
        
        elif self.time_integration_method == TimeIntegrationMethod.CRANK_NICOLSON:
            return {'alpha_n': 0.5, 'alpha_np1': 0.5, 'beta': 1.0}
        
        else:
            return {'alpha_n': 1.0, 'alpha_np1': 1.0, 'beta': 1.0}
    
    def get_progress(self) -> float:
        """
        获取模拟进度百分比
        
        Returns:
            进度百分比 (0-100)
        """
        return min(self.time / self.total_time * 100, 100.0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息（简化版）
        
        Returns:
            统计信息字典
        """
        stats = {
            'current_time': self.time,
            'time_step': self.time_step,
            'current_dt': self.dt,
            'progress_percent': self.get_progress(),
            'converged_steps': self.converged_steps,
            'failed_steps': self.failed_steps,
            'success_rate': self.converged_steps / max(self.time_step, 1),
            'grouting_stage': self.grouting_stage.value,
            'pressure_holding_active': self.pressure_holding.is_active,
            'pressure_holding_duration': self.pressure_holding.current_duration,
            'average_injection_rate': self.pressure_holding.average_injection_rate,
            'current_injection_rate': self.current_injection_rate,
            'should_terminate': not self.should_continue()
        }
        
        return stats
    
