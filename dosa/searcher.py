from abc import ABC, abstractmethod
import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Tuple
import random
from .utils import OptimizationLogger, get_divisors, derive_minimal_hardware
from .space import SearchSpace


class BaseSearcher(ABC):
    """抽象基类，定义所有搜索器的通用接口"""
    
    def __init__(self, graph, hw_params, mapping, fusion_params, perf_model, config, logger=None, recorder=None):
        """
        初始化搜索器
        
        Args:
            graph: ComputationGraph实例
            hw_params: HardwareParameters实例
            mapping: FineGrainedMapping实例
            fusion_params: FusionParameters实例
            perf_model: HighFidelityPerformanceModel实例
            config: 配置对象
            logger: StructuredLogger实例
        """
        self.graph = graph
        self.hw_params = hw_params
        # Number of PEs is now derived from ``min_hw`` and kept fixed during
        # optimization. Disable gradient updates for this parameter to avoid
        # treating it as a learnable variable.
        self.hw_params.log_num_pes.requires_grad = False
        self.mapping = mapping
        self.fusion_params = fusion_params
        self.perf_model = perf_model
        self.config = config
        self.logger = logger
        # 主调用方可选地传入 Recorder，用于记录每一步试验信息和最佳结果
        self.recorder = recorder
        
        # 创建搜索空间实例
        self.space = SearchSpace(graph)
        
        # 记录最佳结果
        self.best_loss = float('inf')
        self.best_params = None
        self.best_metrics = None
        
        # 损失策略配置
        self.loss_strategy = getattr(config, 'LOSS_STRATEGY', 'log_edp_plus_area')
        self.loss_weights = getattr(config, 'LOSS_WEIGHTS', {
            'area_weight': getattr(config, 'AREA_WEIGHT', 0.1),
            'mismatch_penalty_weight': getattr(config, 'MISMATCH_PENALTY_WEIGHT', 0.1),
            'compatibility_penalty_weight': getattr(config, 'COMPATIBILITY_PENALTY_WEIGHT', 100.0),
            'edp_weight': 1.0
        })
    
    @abstractmethod
    def search(self, num_trials: int) -> Dict[str, Any]:
        """
        执行搜索算法
        
        Args:
            num_trials: 评估次数
            
        Returns:
            包含最佳结果的字典
        """
        pass
    
    def evaluate(self, flat_params: List[float]) -> Tuple[float, Dict[str, float]]:
        """
        统一的目标函数接口，评估给定参数的性能
        
        Args:
            flat_params: 扁平化的参数列表
            
        Returns:
            (loss, metrics): 损失值和性能指标字典
        """
        # 将扁平化参数转换为结构化字典
        params_dict = self.space.from_flat(flat_params)
        
        # 将参数设置到模型中
        self._set_params_from_dict(params_dict)

        # Derive minimal hardware and fix the number of PEs accordingly.
        # The PE count is treated as a deterministic value from ``min_hw``
        # instead of a differentiable parameter.
        min_hw = derive_minimal_hardware(self.mapping, self.config)
        if getattr(self.config, "APPLY_MIN_HW_BOUNDS", True):
            self._apply_min_hw_bounds(min_hw, reset=False)
        else:
            print("[DEBUG] Skipping minimal hardware bounds in evaluation")

        # 调用性能模型
        latency, energy, area, mismatch_loss, compatibility_penalty = self.perf_model(
            self.graph, self.hw_params, self.mapping, self.fusion_params
        )

        if self.logger is not None:
            self.logger.event(
                "fusion_decisions",
                decisions=self.fusion_params.get_fusion_decisions_serializable(self.graph),
            )
        
        # 使用统一的损失计算方法
        loss = self._compute_loss(latency, energy, area, mismatch_loss, compatibility_penalty)
        
        # 构建性能指标字典
        metrics = {
            'latency_sec': latency.item(),
            'energy_pj': energy.item(),
            'area_mm2': area.item(),
            'edp': (latency * energy).item(),
            'log_edp': (torch.log(latency + 1e-9) + torch.log(energy + 1e-9)).item(),
            'mismatch_loss': mismatch_loss.item()
        }
        
        # 存储loss breakdown用于后续的update_best_result调用
        self._last_loss_breakdown = self._compute_loss_breakdown(latency, energy, area, mismatch_loss, compatibility_penalty, step_count=0)
        
        return loss.item(), metrics
    
    def _compute_loss(self, latency, energy, area, mismatch_loss, compatibility_penalty, step_count=0):
        """
        计算总损失 - 完整复现原始run.py中的损失计算逻辑，并集成面积预算惩罚项
        
        Args:
            latency: 延迟张量
            energy: 能耗张量
            area: 面积张量
            mismatch_loss: 不匹配损失张量
            compatibility_penalty: 兼容性惩罚张量
            step_count: 当前训练步数，用于权重调度
            
        Returns:
            总损失张量
        """
        # 确保所有输入都是标量张量
        latency = latency.squeeze() if latency.dim() > 0 else latency
        energy = energy.squeeze() if energy.dim() > 0 else energy
        area = area.squeeze() if area.dim() > 0 else area
        mismatch_loss = mismatch_loss.squeeze() if mismatch_loss.dim() > 0 else mismatch_loss
        compatibility_penalty = compatibility_penalty.squeeze() if compatibility_penalty.dim() > 0 else compatibility_penalty
        
        # 获取兼容性惩罚权重
        comp_penalty_weight = self.loss_weights.get('compatibility_penalty_weight', 100.0)
        comp_penalty = comp_penalty_weight * compatibility_penalty
        
        # 计算面积预算惩罚项
        area_budget_penalty = self._compute_area_budget_penalty(area, step_count)
        
        # 根据损失策略计算损失
        if self.loss_strategy == 'strategy_A':
            # Strategy A: 复杂的对数损失计算
            edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
            area_loss = self.loss_weights['area_weight'] * area
            mismatch_penalty = torch.log(1.0 + mismatch_loss * self.loss_weights['mismatch_penalty_weight'])
            loss = edp_loss + area_loss + mismatch_penalty + comp_penalty + area_budget_penalty
            
        elif self.loss_strategy == 'strategy_B':
            # Strategy B: 加权EDP损失计算
            edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
            area_loss = self.loss_weights['area_weight'] * area
            mismatch_penalty = mismatch_loss * self.loss_weights['mismatch_penalty_weight']
            loss = (self.loss_weights['edp_weight'] * edp_loss +
                   area_loss + mismatch_penalty + comp_penalty + area_budget_penalty)
            
        elif self.loss_strategy == 'log_edp_plus_area':
            # 标准策略：log(EDP) + 面积惩罚
            log_edp = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
            area_penalty = self.loss_weights['area_weight'] * area
            mismatch_penalty = mismatch_loss * self.loss_weights.get('mismatch_penalty_weight', 0.1)
            loss = log_edp + area_penalty + mismatch_penalty + comp_penalty + area_budget_penalty
            
        elif self.loss_strategy == 'edp_plus_area':
            # EDP + 面积惩罚
            edp = latency * energy
            area_penalty = self.loss_weights['area_weight'] * area
            mismatch_penalty = mismatch_loss * self.loss_weights.get('mismatch_penalty_weight', 0.1)
            loss = edp + area_penalty + mismatch_penalty + comp_penalty + area_budget_penalty

        elif self.loss_strategy == 'pure_edp':
            # Pure EDP optimisation without area or PE penalties
            edp = latency * energy
            mismatch_penalty = mismatch_loss * self.loss_weights.get('mismatch_penalty_weight', 0.1)
            loss = edp + mismatch_penalty + comp_penalty + area_budget_penalty

        else:
            # 默认策略：与log_edp_plus_area相同
            log_edp = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
            area_penalty = self.loss_weights['area_weight'] * area
            mismatch_penalty = mismatch_loss * self.loss_weights.get('mismatch_penalty_weight', 0.1)
            loss = log_edp + area_penalty + mismatch_penalty + comp_penalty + area_budget_penalty
        
        # 确保返回标量张量
        return loss.squeeze() if loss.dim() > 0 else loss

    def _apply_min_hw_bounds(self, min_hw: Dict[str, float], reset: bool = False):
        """Apply minimal hardware constraints.

        Args:
            min_hw: Dictionary returned by ``derive_minimal_hardware``.
            reset: If True, hardware parameters are reset exactly to the
                minimal values. If False, existing parameters are only clamped
                to be no smaller than the minima.
        """
        device = self.hw_params.log_num_pes.device

        min_num_pes = torch.tensor(float(min_hw.get('num_pes', 1)), device=device)
        current_pes = torch.exp(self.hw_params.log_num_pes.data)
        new_pes = min_num_pes if reset else torch.maximum(current_pes, min_num_pes)
        self.hw_params.log_num_pes.data = torch.log(new_pes)

        for level, param in self.hw_params.log_buffer_sizes_kb.items():
            if level not in min_hw:
                continue
            min_size = torch.tensor(float(min_hw[level]), device=param.device)
            current_size = torch.exp(param.data)
            new_size = min_size if reset else torch.maximum(current_size, min_size)
            param.data = torch.log(new_size)
    
    def _set_params_from_dict(self, params: Dict[str, Any]):
        """
        将扁平化的参数字典设置到模型实例中
        
        Args:
            params: 包含所有参数的扁平化字典
        """
        # 设置硬件参数（PE数量固定为min_hw推导值，不再从参数设置）
        for level in ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad']:
            key = f'{level.lower()}_size_kb'
            if key in params:
                device = self.hw_params.log_buffer_sizes_kb[level].device
                self.hw_params.log_buffer_sizes_kb[level].data = torch.log(torch.tensor(params[key], device=device))
        
        # 设置映射参数 - 只为实际存在的on-chip buffer层级设置参数
        on_chip_levels = ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad']
        for dim_name in self.graph.problem_dims.keys():
            for level_name in on_chip_levels:
                # 确保level_name在mapping.factors中存在
                if level_name in self.mapping.factors:
                    temporal_key = f'{dim_name}_{level_name}_temporal'
                    spatial_key = f'{dim_name}_{level_name}_spatial'
                    
                    if temporal_key in params:
                        device = self.mapping.factors[level_name][dim_name]['temporal'].device
                        self.mapping.factors[level_name][dim_name]['temporal'].data = torch.log(torch.tensor(params[temporal_key], device=device))
                    if spatial_key in params:
                        device = self.mapping.factors[level_name][dim_name]['spatial'].device
                        self.mapping.factors[level_name][dim_name]['spatial'].data = torch.log(torch.tensor(params[spatial_key], device=device))
        
        # 设置融合参数
        if 'fusion_logits' in params:
            fusion_logits = params['fusion_logits']
            if isinstance(fusion_logits, list):
                fusion_logits = torch.tensor(fusion_logits, device=self.fusion_params.fusion_logits.device).unsqueeze(1)
            else:
                fusion_logits = fusion_logits.to(self.fusion_params.fusion_logits.device)
            self.fusion_params.fusion_logits.data = fusion_logits
    
    def _get_params_as_dict(self) -> Dict[str, Any]:
        """
        将当前模型参数转换为扁平化字典
        
        Returns:
            扁平化的参数字典
        """
        params = {}
        
        # 硬件参数
        params['num_pes'] = self.hw_params.get_projected_num_pes().item()
        for level in ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad']:
            key = f'{level.lower()}_size_kb'
            params[key] = self.hw_params.get_buffer_size_kb(level).item()
        
        # 映射参数
        mapping_factors = self.mapping.get_all_factors()
        for dim_name, dim_factors in mapping_factors.items():
            for level_name, level_factors in dim_factors.items():
                params[f'{dim_name}_{level_name}_temporal'] = level_factors['temporal'].item()
                params[f'{dim_name}_{level_name}_spatial'] = level_factors['spatial'].item()
        
        # 融合参数
        fusion_logits = self.fusion_params.fusion_logits.squeeze()
        if fusion_logits.dim() == 0:  # 标量情况
            params['fusion_logits'] = [fusion_logits.item()]
        else:
            params['fusion_logits'] = fusion_logits.tolist()
        
        return params
    
    def _compute_loss_breakdown(self, latency, energy, area, mismatch_loss, compatibility_penalty, step_count=0):
        """
        计算loss的详细组成部分，包括面积预算惩罚项
        
        Args:
            latency: 延迟张量
            energy: 能耗张量
            area: 面积张量
            mismatch_loss: 不匹配损失张量
            compatibility_penalty: 兼容性惩罚张量
            step_count: 当前训练步数，用于权重调度
            
        Returns:
            包含loss详细组成的字典
        """
        # 确保所有输入都是标量张量
        latency = latency.squeeze() if latency.dim() > 0 else latency
        energy = energy.squeeze() if energy.dim() > 0 else energy
        area = area.squeeze() if area.dim() > 0 else area
        mismatch_loss = mismatch_loss.squeeze() if mismatch_loss.dim() > 0 else mismatch_loss
        compatibility_penalty = compatibility_penalty.squeeze() if compatibility_penalty.dim() > 0 else compatibility_penalty
        
        # 获取兼容性惩罚权重
        comp_penalty_weight = self.loss_weights.get('compatibility_penalty_weight', 100.0)
        comp_penalty = comp_penalty_weight * compatibility_penalty
        
        # 计算面积预算惩罚项
        area_budget_penalty = self._compute_area_budget_penalty(area, step_count)
        
        breakdown = {
            'strategy': self.loss_strategy,
            'latency': latency.item(),
            'energy': energy.item(),
            'area': area.item(),
            'area_budget_penalty': area_budget_penalty.item()
        }
        
        # 根据损失策略计算各组成部分
        if self.loss_strategy == 'strategy_A':
            edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
            area_loss = self.loss_weights['area_weight'] * area
            mismatch_penalty = torch.log(1.0 + mismatch_loss * self.loss_weights['mismatch_penalty_weight'])
            breakdown.update({
                'log_edp': edp_loss.item(),
                'area_penalty': area_loss.item(),
                'mismatch_penalty': mismatch_penalty.item(),
                'compatibility_penalty': comp_penalty.item()
            })
            
        elif self.loss_strategy == 'strategy_B':
            edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
            area_loss = self.loss_weights['area_weight'] * area
            mismatch_penalty = mismatch_loss * self.loss_weights['mismatch_penalty_weight']
            weighted_edp = self.loss_weights['edp_weight'] * edp_loss
            breakdown.update({
                'weighted_log_edp': weighted_edp.item(),
                'area_penalty': area_loss.item(),
                'mismatch_penalty': mismatch_penalty.item(),
                'compatibility_penalty': comp_penalty.item()
            })
            
        elif self.loss_strategy == 'log_edp_plus_area':
            log_edp = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
            area_penalty = self.loss_weights['area_weight'] * area
            mismatch_penalty = mismatch_loss * self.loss_weights.get('mismatch_penalty_weight', 0.1)
            breakdown.update({
                'log_edp': log_edp.item(),
                'area_penalty': area_penalty.item(),
                'mismatch_penalty': mismatch_penalty.item(),
                'compatibility_penalty': comp_penalty.item()
            })
            
        elif self.loss_strategy == 'edp_plus_area':
            edp = latency * energy
            area_penalty = self.loss_weights['area_weight'] * area
            mismatch_penalty = mismatch_loss * self.loss_weights.get('mismatch_penalty_weight', 0.1)
            breakdown.update({
                'edp': edp.item(),
                'area_penalty': area_penalty.item(),
                'mismatch_penalty': mismatch_penalty.item(),
                'compatibility_penalty': comp_penalty.item()
            })
            
        elif self.loss_strategy == 'pure_edp':
            edp = latency * energy
            mismatch_penalty = mismatch_loss * self.loss_weights.get('mismatch_penalty_weight', 0.1)
            breakdown.update({
                'edp': edp.item(),
                'mismatch_penalty': mismatch_penalty.item(),
                'compatibility_penalty': comp_penalty.item(),
                'area_not_in_loss': area.item()  # 面积不计入loss但显示
            })
            
        else:
            # 默认策略：与log_edp_plus_area相同
            log_edp = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
            area_penalty = self.loss_weights['area_weight'] * area
            mismatch_penalty = mismatch_loss * self.loss_weights.get('mismatch_penalty_weight', 0.1)
            breakdown.update({
                'log_edp': log_edp.item(),
                'area_penalty': area_penalty.item(),
                'mismatch_penalty': mismatch_penalty.item(),
                'compatibility_penalty': comp_penalty.item()
            })
        
        return breakdown
    
    def _compute_area_budget_penalty(self, area, step_count=0):
        """
        计算面积预算惩罚项
        
        Args:
            area: 当前面积 (mm²)
            step_count: 当前训练步数，用于权重调度
            
        Returns:
            面积预算惩罚值 (torch.Tensor)
        """
        from .config import Config
        config = Config.get_instance()
        
        # 如果未启用面积预算或预算为None，返回0
        if not config.ENABLE_AREA_BUDGET or config.AREA_BUDGET_MM2 is None:
            return torch.tensor(0.0, device=area.device, dtype=area.dtype)
        
        budget = config.AREA_BUDGET_MM2
        tolerance = config.AREA_BUDGET_TOLERANCE
        strategy = config.AREA_BUDGET_PENALTY_STRATEGY
        
        # 计算当前权重（支持权重调度）
        base_weight = config.AREA_BUDGET_PENALTY_WEIGHT
        if config.AREA_BUDGET_WEIGHT_SCHEDULE['enable']:
            schedule_config = config.AREA_BUDGET_WEIGHT_SCHEDULE
            initial_weight = schedule_config['initial_weight']
            final_weight = schedule_config['final_weight']
            warmup_steps = schedule_config['warmup_steps']
            schedule_type = schedule_config['schedule_type']
            
            if step_count < warmup_steps:
                if schedule_type == 'linear':
                    progress = step_count / warmup_steps
                    current_weight = initial_weight + (final_weight - initial_weight) * progress
                elif schedule_type == 'exponential':
                    progress = step_count / warmup_steps
                    current_weight = initial_weight * ((final_weight / initial_weight) ** progress)
                else:
                    current_weight = initial_weight
            else:
                current_weight = final_weight
        else:
            current_weight = base_weight
        
        # 计算预算边界
        lower_bound = budget * (1 - tolerance)
        upper_bound = budget * (1 + tolerance)
        
        # 在容忍区间内不施加惩罚
        if lower_bound <= area <= upper_bound:
            return torch.tensor(0.0, device=area.device, dtype=area.dtype)
        
        # 计算偏离量
        if area < lower_bound:
            deviation = lower_bound - area
        else:  # area > upper_bound
            deviation = area - upper_bound
        
        # 归一化偏离量（相对于预算的百分比）
        normalized_deviation = deviation / budget
        
        # 根据策略计算惩罚
        if strategy == 'quadratic':
            penalty = normalized_deviation ** 2
        elif strategy == 'linear':
            penalty = normalized_deviation
        elif strategy == 'huber':
            delta = config.AREA_BUDGET_HUBER_DELTA
            if normalized_deviation <= delta:
                penalty = 0.5 * (normalized_deviation ** 2)
            else:
                penalty = delta * (normalized_deviation - 0.5 * delta)
        elif strategy == 'exponential':
            penalty = torch.exp(normalized_deviation) - 1
        else:
            # 默认使用二次惩罚
            penalty = normalized_deviation ** 2
        
        # 应用权重
        final_penalty = current_weight * penalty
        
        return final_penalty

    def update_best_result(self, loss: float, params: Dict[str, Any], metrics: Dict[str, float], trial: int, loss_breakdown: Dict[str, Any] = None):
        """
        更新最佳结果
        
        Args:
            loss: 当前损失值
            params: 当前参数
            metrics: 当前性能指标
            trial: 当前试验次数
            loss_breakdown: loss的详细组成部分（可选）
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = params.copy()
            self.best_metrics = metrics.copy()
            # 若集成 Recorder，则同步更新
            if self.recorder is not None:
                self.recorder.update_best(metrics, key="edp")
            # 使用StructuredLogger记录新的最佳结果事件
            if self.logger:
                event_data = {"loss": loss, **metrics}
                if loss_breakdown:
                    event_data["loss_breakdown"] = loss_breakdown
                self.logger.event("new_best", step=trial, metrics=event_data)
    
    from typing import Optional

    def log_trial(self, trial: int, loss: float, metrics: Dict[str, float], params: Dict[str, Any], is_best: Optional[bool] = None):
        """
        记录试验结果
        
        Args:
            trial: 试验次数
            loss: 损失值
            metrics: 性能指标
            params: 参数字典
            is_best: 是否为最佳结果
        """
        if self.logger:
            num_pes_val = params.get('num_pes', self.hw_params.get_projected_num_pes().item())
            trial_data = {
                'searcher_type': self.__class__.__name__,
                'loss': loss,
                'metrics': {
                    'loss': loss,
                    'edp': metrics['edp'],
                    'latency_sec': metrics['latency_sec'],
                    'energy_pj': metrics['energy_pj'],
                    'area_mm2': metrics['area_mm2']
                },
                'hardware_params': {
                    'num_pes': num_pes_val,
                    'l0_size_kb': params.get('l0_registers_size_kb', 0),
                    'l1_size_kb': params.get('l1_accumulator_size_kb', 0),
                    'l2_size_kb': params.get('l2_scratchpad_size_kb', 0)
                },
                'fusion_decisions': self.fusion_params.get_fusion_decisions_serializable(self.graph),
                'best_so_far': is_best if is_best is not None else (loss <= self.best_loss)
            }

            self.logger.trial(trial, trial_data)
        
        # ------ Recorder 集成 ------
        if self.recorder is not None:
            trial_row = {
                "trial": trial,
                "loss": loss,
                **metrics
            }
            self.recorder.record_trial(trial_row)


def get_random_valid_divisor(dim_size: int) -> int:
    """
    获取给定维度大小的随机有效约数
    
    Args:
        dim_size: 维度大小
        
    Returns:
        随机选择的有效约数
    """
    divisors = get_divisors(dim_size)
    return int(divisors[torch.randint(0, len(divisors), (1,)).item()].item())


class FADOSASearcher(BaseSearcher):
    """
    FA-DOSA搜索器：基于梯度的交替优化
    """
    
    def __init__(self, graph, hw_params, mapping, fusion_params, perf_model, config, logger=None, recorder=None):
        super().__init__(graph, hw_params, mapping, fusion_params, perf_model, config, logger, recorder)
        
        # FA-DOSA特定参数
        self.num_outer_steps = getattr(config, 'NUM_OUTER_STEPS', 5)
        self.num_mapping_steps = getattr(config, 'NUM_MAPPING_STEPS', 50)
        self.num_hardware_steps = getattr(config, 'NUM_HARDWARE_STEPS', 50)
        self.lr_mapping = getattr(config, 'LR_MAPPING', 0.01)
        self.lr_hardware = getattr(config, 'LR_HARDWARE', 0.01)
    
    def update_loss_weights(self, new_weights: dict):
        """Update loss weights dynamically for Pareto frontier scanning.
        
        Args:
            new_weights: Dictionary containing new weight values
        """
        self.loss_weights.update(new_weights)
        if self.logger:
            self.logger.console(f"Updated loss weights: {self.loss_weights}")

    def _snapshot_mapping(self):
        """Capture current mapping factors as plain floats for change tracking."""
        with torch.no_grad():
            factors = self.mapping.get_all_factors()
        snapshot = {}
        for dim, levels in factors.items():
            snapshot[dim] = {}
            for level, facs in levels.items():
                snapshot[dim][level] = {k: float(v.item()) for k, v in facs.items()}
        return snapshot

    def _snapshot_fusion(self):
        """Capture current fusion decisions for change tracking."""
        decisions = self.fusion_params.get_fusion_decisions_serializable(self.graph)
        snapshot = {}
        for d in decisions:
            group = d["group"]
            key = "|".join(group) if isinstance(group, list) else str(group)
            snapshot[key] = d["fused"]
        return snapshot

    def _diff_mapping(self, prev, curr, tol: float = 1e-6, limit: int = 50):
        """Return a summary of mapping factor changes between two snapshots."""
        changes = []
        for dim, levels in curr.items():
            for level, facs in levels.items():
                for k, v in facs.items():
                    prev_v = prev.get(dim, {}).get(level, {}).get(k)
                    if prev_v is None or abs(v - prev_v) > tol:
                        if prev_v is None:
                            changes.append(f"{dim}.{level}.{k}: {v:.2f}")
                        else:
                            changes.append(f"{dim}.{level}.{k}: {prev_v:.2f}->{v:.2f}")
        if len(changes) > limit:
            return changes[:limit] + [f"... (+{len(changes) - limit} more)"]
        return changes

    def _diff_fusion(self, prev, curr, limit: int = 5):
        """Return a summary of fusion decision changes between two snapshots."""
        changes = []
        for group, fused in curr.items():
            prev_fused = prev.get(group)
            if prev_fused is None or fused != prev_fused:
                if prev_fused is None:
                    changes.append(f"{group}: {fused}")
                else:
                    changes.append(f"{group}: {prev_fused}->{fused}")
        if len(changes) > limit:
            return changes[:limit] + [f"... (+{len(changes) - limit} more)"]
        return changes
    
    def search(self, num_trials: int) -> Dict[str, Any]:
        """
        执行FA-DOSA的交替优化搜索
        
        Args:
            num_trials: 这里对应于外层优化步数
            
        Returns:
            最佳结果字典
        """
        import os
        from .utils import save_configuration_to_json
        
        # -------- 设备同步 --------
        device = self.config.DEVICE
        self.hw_params.to(device)
        self.mapping.to(device)
        self.fusion_params.to(device)

        if self.logger:
            self.logger.event("search_start", searcher_type="FA-DOSA", outer_steps=self.num_outer_steps)
            self.logger.console(f"Starting FA-DOSA search with {self.num_outer_steps} outer steps...")

        # 确保output目录存在
        os.makedirs('output', exist_ok=True)

        trial_count = 0

        # Snapshots for tracking mapping and fusion changes across outer steps
        prev_mapping_state = self._snapshot_mapping()
        prev_fusion_state = self._snapshot_fusion()

        # 记录基线约束下的 requires_grad 状态，以防被后续阶段覆盖
        hw_params_list = list(self.hw_params.parameters())
        mapping_params_list = list(self.mapping.parameters())
        fusion_params_list = list(self.fusion_params.parameters())
        init_hw_requires_grad = [p.requires_grad for p in hw_params_list]
        init_mapping_requires_grad = [p.requires_grad for p in mapping_params_list]
        init_fusion_requires_grad = [p.requires_grad for p in fusion_params_list]
        
        # 交替优化循环
        for outer_step in range(self.num_outer_steps):
            if self.logger:
                self.logger.event("outer_step_start", index=outer_step + 1, total=self.num_outer_steps)
                
            
            # Phase A: 优化映射和融合参数（冻结硬件参数）
            if self.logger:
                self.logger.event("phase_start", phase="mapping_fusion")
                # Removed duplicate phase console

            # 冻结硬件参数
            for p in hw_params_list:
                p.requires_grad = False
            # 恢复映射和融合参数在基线约束下的 requires_grad 状态
            for p, flag in zip(mapping_params_list, init_mapping_requires_grad):
                p.requires_grad = flag
            for p, flag in zip(fusion_params_list, init_fusion_requires_grad):
                p.requires_grad = flag

            # 收集可训练的映射和融合参数
            map_opt_params = [p for p in mapping_params_list + fusion_params_list if p.requires_grad]
            if map_opt_params:
                optimizer_map = optim.Adam(map_opt_params, lr=self.lr_mapping)

                for i in range(self.num_mapping_steps):
                    optimizer_map.zero_grad()

                    # 直接计算损失（保持梯度图）
                    latency, energy, area, mismatch_loss, compatibility_penalty = self.perf_model(
                        self.graph, self.hw_params, self.mapping, self.fusion_params
                    )

                    if self.logger is not None:
                        self.logger.event(
                            "fusion_decisions",
                            decisions=self.fusion_params.get_fusion_decisions_serializable(self.graph),
                        )

                    # 使用统一的损失计算方法
                    loss = self._compute_loss(latency, energy, area, mismatch_loss, compatibility_penalty, step_count=trial_count)

                    # 反向传播
                    loss.backward()

                    # ---- 调试日志记录（Phase A） ----
                    if self.recorder is not None:
                        try:
                            first_map_grad = next((p.grad for p in self.mapping.parameters() if p.grad is not None), None)
                            mapping_grad_mean = float(first_map_grad.abs().mean().item()) if first_map_grad is not None else 0.0
                        except StopIteration:
                            mapping_grad_mean = 0.0
                        fusion_grad = self.fusion_params.fusion_logits.grad
                        fusion_grad_mean = float(fusion_grad.abs().mean().item()) if fusion_grad is not None else 0.0
                        debug_snapshot = {
                            "trial": trial_count + 1,
                            "phase": "A_Mapping_Fusion",
                            "outer_step": outer_step,
                            "inner_step": i,
                            "loss": loss.item(),
                            "loss_breakdown": {
                                "log_edp": (torch.log(latency + 1e-9) + torch.log(energy + 1e-9)).item(),
                                "area_penalty": (self.loss_weights['area_weight'] * area).item(),
                                "mismatch_penalty": mismatch_loss.item(),
                                "compatibility_penalty": compatibility_penalty.item()
                            },
                            "learning_rate": self.lr_mapping,
                            "gradients": {
                                "mapping_sample_grad_mean_abs": mapping_grad_mean,
                                "fusion_logits_grad_mean_abs": fusion_grad_mean
                            }
                        }
                        self.recorder.log_coopt_debug_step(debug_snapshot)

                    optimizer_map.step()

                    # 计算指标用于记录
                    with torch.no_grad():
                        current_params = self._get_params_as_dict()
                        flat_params = self.space.to_flat(current_params)
                        _, metrics = self.evaluate(flat_params)
                
                    # 添加缺失的日志记录和loss详细组成打印
                    if i % 10 == 0:
                        print(f"\n[DEBUG] Phase A - Outer Step {outer_step+1}, Inner Step {i+1}:")
                        
                        # 计算并显示loss的详细组成部分
                        comp_penalty_weight = self.loss_weights.get('compatibility_penalty_weight', 100.0)
                        comp_penalty = comp_penalty_weight * compatibility_penalty
                        
                        if self.loss_strategy == 'strategy_A':
                            edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
                            area_loss = self.loss_weights['area_weight'] * area
                            mismatch_penalty = torch.log(1.0 + mismatch_loss * self.loss_weights['mismatch_penalty_weight'])
                            print(f"[DEBUG] Loss详细组成 (strategy_A): 总计={loss.item():.6f}")
                            print(f"[DEBUG]   - Log(EDP): {edp_loss.item():.6f}")
                            print(f"[DEBUG]   - Area惩罚: {area_loss.item():.6f} (面积: {area.item():.2f} mm²)")
                            print(f"[DEBUG]   - Mismatch惩罚: {mismatch_penalty.item():.6f}")
                            print(f"[DEBUG]   - Compatibility惩罚: {comp_penalty.item():.6f}")
                            
                        elif self.loss_strategy == 'strategy_B':
                            edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
                            area_loss = self.loss_weights['area_weight'] * area
                            mismatch_penalty = mismatch_loss * self.loss_weights['mismatch_penalty_weight']
                            weighted_edp = self.loss_weights['edp_weight'] * edp_loss
                            print(f"[DEBUG] Loss详细组成 (strategy_B): 总计={loss.item():.6f}")
                            print(f"[DEBUG]   - 加权Log(EDP): {weighted_edp.item():.6f}")
                            print(f"[DEBUG]   - Area惩罚: {area_loss.item():.6f} (面积: {area.item():.2f} mm²)")
                            print(f"[DEBUG]   - Mismatch惩罚: {mismatch_penalty.item():.6f}")
                            print(f"[DEBUG]   - Compatibility惩罚: {comp_penalty.item():.6f}")
                            
                        elif self.loss_strategy == 'log_edp_plus_area':
                            log_edp = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
                            area_penalty = self.loss_weights['area_weight'] * area
                            mismatch_penalty = mismatch_loss * self.loss_weights.get('mismatch_penalty_weight', 0.1)
                            print(f"[DEBUG] Loss详细组成 (log_edp_plus_area): 总计={loss.item():.6f}")
                            print(f"[DEBUG]   - Log(EDP): {log_edp.item():.6f}")
                            print(f"[DEBUG]   - Area惩罚: {area_penalty.item():.6f} (面积: {area.item():.2f} mm²)")
                            print(f"[DEBUG]   - Mismatch惩罚: {mismatch_penalty.item():.6f}")
                            print(f"[DEBUG]   - Compatibility惩罚: {comp_penalty.item():.6f}")
                            
                        elif self.loss_strategy == 'edp_plus_area':
                            edp = latency * energy
                            area_penalty = self.loss_weights['area_weight'] * area
                            mismatch_penalty = mismatch_loss * self.loss_weights.get('mismatch_penalty_weight', 0.1)
                            print(f"[DEBUG] Loss详细组成 (edp_plus_area): 总计={loss.item():.6f}")
                            print(f"[DEBUG]   - EDP: {edp.item():.6f}")
                            print(f"[DEBUG]   - Area惩罚: {area_penalty.item():.6f} (面积: {area.item():.2f} mm²)")
                            print(f"[DEBUG]   - Mismatch惩罚: {mismatch_penalty.item():.6f}")
                            print(f"[DEBUG]   - Compatibility惩罚: {comp_penalty.item():.6f}")
                            
                        elif self.loss_strategy == 'pure_edp':
                            edp = latency * energy
                            mismatch_penalty = mismatch_loss * self.loss_weights.get('mismatch_penalty_weight', 0.1)
                            print(f"[DEBUG] Loss详细组成 (pure_edp): 总计={loss.item():.6f}")
                            print(f"[DEBUG]   - EDP: {edp.item():.6f}")
                            print(f"[DEBUG]   - Mismatch惩罚: {mismatch_penalty.item():.6f}")
                            print(f"[DEBUG]   - Compatibility惩罚: {comp_penalty.item():.6f}")
                            print(f"[DEBUG]   - 面积: {area.item():.2f} mm² (未计入loss)")
                            
                        else:
                            # 默认策略
                            log_edp = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
                            area_penalty = self.loss_weights['area_weight'] * area
                            mismatch_penalty = mismatch_loss * self.loss_weights.get('mismatch_penalty_weight', 0.1)
                            print(f"[DEBUG] Loss详细组成 (默认策略): 总计={loss.item():.6f}")
                            print(f"[DEBUG]   - Log(EDP): {log_edp.item():.6f}")
                            print(f"[DEBUG]   - Area惩罚: {area_penalty.item():.6f} (面积: {area.item():.2f} mm²)")
                            print(f"[DEBUG]   - Mismatch惩罚: {mismatch_penalty.item():.6f}")
                            print(f"[DEBUG]   - Compatibility惩罚: {comp_penalty.item():.6f}")
                        
                        # 显示基础性能指标
                        print(f"[DEBUG] 基础指标: 延迟={latency.item():.2e}s, 能耗={energy.item():.2e}pJ")
                        
                        trial_count += 1
                        self.log_trial(trial_count, loss.item(), metrics, current_params)

                    # 退火温度
                    self.mapping.anneal_tau()

                    # 更新最佳结果
                    trial_count += 1
                    old_best_loss = self.best_loss
                    # 计算loss的详细组成部分
                    loss_breakdown = self._compute_loss_breakdown(latency, energy, area, mismatch_loss, compatibility_penalty, step_count=trial_count)
                    self.update_best_result(loss.item(), current_params, metrics, trial_count, loss_breakdown)

                    # 质量驱动的触发：当找到新的全局最优解时保存配置
                    if loss.item() < old_best_loss:
                        self._save_validation_config(trial_count, "quality_driven")

                    # 多样性驱动的触发：周期性保存配置
                    if i % 50 == 0:
                        self._save_validation_config(trial_count, "diversity_driven")

                    # 记录日志
                    if i % 10 == 0:
                        self.log_trial(trial_count, loss.item(), metrics, current_params)

            # Restore best parameters from Phase A before hardware optimization
            self._set_params_from_dict(self.best_params)
            if self.logger:
                self.logger.console("Restored best parameters from Phase A before hardware optimization.")

            # 根据当前映射推导最小硬件规模，作为硬件优化的起点
            with torch.no_grad():
                # 恢复Phase A中的最佳映射/融合配置，确保后续硬件搜索基于最优映射
                if self.best_params is not None:
                    print("[DEBUG] Phase A结束 - 恢复最佳映射/融合配置")
                    self._set_params_from_dict(self.best_params)
                else:
                    print("[DEBUG] Phase A结束 - 无可恢复的最佳配置，使用当前参数")

                # 记录当前硬件参数（Phase A结束时）
                current_hw_before = {
                    'num_pes': self.hw_params.get_projected_num_pes().item(),
                    'L0_size_kb': self.hw_params.get_buffer_size_kb('L0_Registers').item(),
                    'L1_size_kb': self.hw_params.get_buffer_size_kb('L1_Accumulator').item(),
                    'L2_size_kb': self.hw_params.get_buffer_size_kb('L2_Scratchpad').item()
                }

                min_hw = derive_minimal_hardware(self.mapping, self.config)
                print(f"\n[DEBUG] Phase A结束 - 推导的最小硬件需求: {min_hw}")
                print(f"[DEBUG] Phase A结束 - 当前硬件配置: {current_hw_before}")
                
                # Apply minimal hardware bounds only if configured to do so
                if self.config.APPLY_MIN_HW_BOUNDS:
                    print(f"[DEBUG] 应用最小硬件约束 (reset={self.config.RESET_TO_MIN_HW})")
                    # Reset hardware to the minimal configuration if configured to do so.
                    # The number of PEs is deterministically determined by ``min_hw`` when reset=True.
                    self._apply_min_hw_bounds(min_hw, reset=self.config.RESET_TO_MIN_HW)
                    
                    # 记录应用约束后的硬件参数
                    current_hw_after = {
                        'num_pes': self.hw_params.get_projected_num_pes().item(),
                        'L0_size_kb': self.hw_params.get_buffer_size_kb('L0_Registers').item(),
                        'L1_size_kb': self.hw_params.get_buffer_size_kb('L1_Accumulator').item(),
                        'L2_size_kb': self.hw_params.get_buffer_size_kb('L2_Scratchpad').item()
                    }
                    print(f"[DEBUG] 应用约束后硬件配置: {current_hw_after}")
                    
                    # 检查是否有参数发生变化
                    changed_params = []
                    for key in current_hw_before:
                        if abs(current_hw_before[key] - current_hw_after[key]) > 1e-6:
                            changed_params.append(f"{key}: {current_hw_before[key]:.2f} -> {current_hw_after[key]:.2f}")
                    
                    if changed_params:
                        print(f"[DEBUG] ⚠️  硬件参数发生变化: {', '.join(changed_params)}")
                    else:
                        print(f"[DEBUG] ✓ 硬件参数未发生变化")
                else:
                    print(f"[DEBUG] 跳过最小硬件约束应用 (APPLY_MIN_HW_BOUNDS=False)")
                    current_hw_after = {
                        'num_pes': self.hw_params.get_projected_num_pes().item(),
                        'L0_size_kb': self.hw_params.get_buffer_size_kb('L0_Registers').item(),
                        'L1_size_kb': self.hw_params.get_buffer_size_kb('L1_Accumulator').item(),
                        'L2_size_kb': self.hw_params.get_buffer_size_kb('L2_Scratchpad').item()
                    }
                    assert all(abs(current_hw_before[k] - current_hw_after[k]) < 1e-6 for k in current_hw_before), (
                        "Hardware parameters changed despite APPLY_MIN_HW_BOUNDS=False"
                    )
                    print("[DEBUG] ✓ 硬件参数未发生变化 (APPLY_MIN_HW_BOUNDS=False)")

            # Report mapping and fusion parameter changes
            current_mapping_state = self._snapshot_mapping()
            mapping_changes = self._diff_mapping(prev_mapping_state, current_mapping_state)
            if mapping_changes:
                print(f"[DEBUG] ⚠️ 映射参数变化: {', '.join(mapping_changes)}")
            else:
                print(f"[DEBUG] ✓ 映射参数未变化")
            prev_mapping_state = current_mapping_state

            current_fusion_state = self._snapshot_fusion()
            fusion_changes = self._diff_fusion(prev_fusion_state, current_fusion_state)
            if fusion_changes:
                print(f"[DEBUG] ⚠️ 融合决策变化: {', '.join(fusion_changes)}")
            else:
                print(f"[DEBUG] ✓ 融合决策未变化")
            prev_fusion_state = current_fusion_state

            # Phase B: 优化硬件参数（冻结映射和融合参数）
            if self.logger:
                self.logger.event("phase_start", phase="hardware")
                # Removed duplicate phase console

            # 冻结映射和融合参数
            for p in mapping_params_list + fusion_params_list:
                p.requires_grad = False
            # 恢复硬件参数在基线约束下的 requires_grad 状态
            for p, flag in zip(hw_params_list, init_hw_requires_grad):
                p.requires_grad = flag

            # 收集可训练的硬件参数
            hw_opt_params = [p for p in hw_params_list if p.requires_grad]
            if hw_opt_params:
                optimizer_hw = optim.Adam(hw_opt_params, lr=self.lr_hardware)

                for i in range(self.num_hardware_steps):
                    optimizer_hw.zero_grad()

                    # 直接计算损失（保持梯度图）
                    latency, energy, area, mismatch_loss, compatibility_penalty = self.perf_model(
                        self.graph, self.hw_params, self.mapping, self.fusion_params
                    )

                    if self.logger is not None:
                        self.logger.event(
                            "fusion_decisions",
                            decisions=self.fusion_params.get_fusion_decisions_serializable(self.graph),
                        )

                    # 使用统一的损失计算方法
                    loss = self._compute_loss(latency, energy, area, mismatch_loss, compatibility_penalty, step_count=trial_count)

                    # 反向传播
                    loss.backward()

                    # ---- 调试日志记录（Phase B） ----
                    if self.recorder is not None:
                        log_num_pes_grad = self.hw_params.log_num_pes.grad
                        l0_grad = self.hw_params.log_buffer_sizes_kb['L0_Registers'].grad
                        l1_grad = self.hw_params.log_buffer_sizes_kb['L1_Accumulator'].grad
                        l2_grad = self.hw_params.log_buffer_sizes_kb['L2_Scratchpad'].grad
                        debug_snapshot = {
                            "trial": trial_count + 1,
                            "phase": "B_Hardware",
                            "outer_step": outer_step,
                            "inner_step": i,
                            "loss": loss.item(),
                            "loss_breakdown": {
                                "log_edp": (torch.log(latency + 1e-9) + torch.log(energy + 1e-9)).item(),
                                "area_penalty": (self.loss_weights['area_weight'] * area).item(),
                                "mismatch_penalty": mismatch_loss.item(),
                                "compatibility_penalty": compatibility_penalty.item()
                            },
                            "learning_rate": self.lr_hardware,
                            "hardware_params_log_space": {
                                "log_num_pes": self.hw_params.log_num_pes.item(),
                                "log_l0_kb": self.hw_params.log_buffer_sizes_kb['L0_Registers'].item(),
                                "log_l1_kb": self.hw_params.log_buffer_sizes_kb['L1_Accumulator'].item(),
                                "log_l2_kb": self.hw_params.log_buffer_sizes_kb['L2_Scratchpad'].item()
                            },
                            "gradients": {
                                "log_num_pes_grad": float(log_num_pes_grad.item()) if log_num_pes_grad is not None else 0.0,
                                "log_l0_kb_grad": float(l0_grad.item()) if l0_grad is not None else 0.0,
                                "log_l1_kb_grad": float(l1_grad.item()) if l1_grad is not None else 0.0,
                                "log_l2_kb_grad": float(l2_grad.item()) if l2_grad is not None else 0.0
                            }
                        }
                        self.recorder.log_coopt_debug_step(debug_snapshot)

                    # 记录优化前的硬件参数
                    hw_before_step = {
                        'num_pes': self.hw_params.get_projected_num_pes().item(),
                        'L0_size_kb': self.hw_params.get_buffer_size_kb('L0_Registers').item(),
                        'L1_size_kb': self.hw_params.get_buffer_size_kb('L1_Accumulator').item(),
                        'L2_size_kb': self.hw_params.get_buffer_size_kb('L2_Scratchpad').item()
                    }
                    
                    optimizer_hw.step()
                    
                    # 记录优化后的硬件参数
                    hw_after_step = {
                        'num_pes': self.hw_params.get_projected_num_pes().item(),
                        'L0_size_kb': self.hw_params.get_buffer_size_kb('L0_Registers').item(),
                        'L1_size_kb': self.hw_params.get_buffer_size_kb('L1_Accumulator').item(),
                        'L2_size_kb': self.hw_params.get_buffer_size_kb('L2_Scratchpad').item()
                    }

                    # Enforce minimal hardware as lower bounds after the update
                    with torch.no_grad():
                        self._apply_min_hw_bounds(min_hw, reset=False)
                        
                        # 记录应用约束后的硬件参数
                        hw_after_bounds = {
                            'num_pes': self.hw_params.get_projected_num_pes().item(),
                            'L0_size_kb': self.hw_params.get_buffer_size_kb('L0_Registers').item(),
                            'L1_size_kb': self.hw_params.get_buffer_size_kb('L1_Accumulator').item(),
                            'L2_size_kb': self.hw_params.get_buffer_size_kb('L2_Scratchpad').item()
                        }
                        
                        # 每10步打印一次详细的参数变化
                        if i % 10 == 0:
                            print(f"\n[DEBUG] Phase B - Outer Step {outer_step+1}, Inner Step {i+1}:")
                            
                            # 计算并显示loss的详细组成部分
                            comp_penalty_weight = self.loss_weights.get('compatibility_penalty_weight', 100.0)
                            comp_penalty = comp_penalty_weight * compatibility_penalty
                            
                            if self.loss_strategy == 'strategy_A':
                                edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
                                area_loss = self.loss_weights['area_weight'] * area
                                mismatch_penalty = torch.log(1.0 + mismatch_loss * self.loss_weights['mismatch_penalty_weight'])
                                print(f"[DEBUG] Loss详细组成 (strategy_A): 总计={loss.item():.6f}")
                                print(f"[DEBUG]   - Log(EDP): {edp_loss.item():.6f}")
                                print(f"[DEBUG]   - Area惩罚: {area_loss.item():.6f} (面积: {area.item():.2f} mm²)")
                                print(f"[DEBUG]   - Mismatch惩罚: {mismatch_penalty.item():.6f}")
                                print(f"[DEBUG]   - Compatibility惩罚: {comp_penalty.item():.6f}")
                                
                            elif self.loss_strategy == 'strategy_B':
                                edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
                                area_loss = self.loss_weights['area_weight'] * area
                                mismatch_penalty = mismatch_loss * self.loss_weights['mismatch_penalty_weight']
                                weighted_edp = self.loss_weights['edp_weight'] * edp_loss
                                print(f"[DEBUG] Loss详细组成 (strategy_B): 总计={loss.item():.6f}")
                                print(f"[DEBUG]   - 加权Log(EDP): {weighted_edp.item():.6f}")
                                print(f"[DEBUG]   - Area惩罚: {area_loss.item():.6f} (面积: {area.item():.2f} mm²)")
                                print(f"[DEBUG]   - Mismatch惩罚: {mismatch_penalty.item():.6f}")
                                print(f"[DEBUG]   - Compatibility惩罚: {comp_penalty.item():.6f}")
                                
                            elif self.loss_strategy == 'log_edp_plus_area':
                                log_edp = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
                                area_penalty = self.loss_weights['area_weight'] * area
                                mismatch_penalty = mismatch_loss * self.loss_weights.get('mismatch_penalty_weight', 0.1)
                                print(f"[DEBUG] Loss详细组成 (log_edp_plus_area): 总计={loss.item():.6f}")
                                print(f"[DEBUG]   - Log(EDP): {log_edp.item():.6f}")
                                print(f"[DEBUG]   - Area惩罚: {area_penalty.item():.6f} (面积: {area.item():.2f} mm²)")
                                print(f"[DEBUG]   - Mismatch惩罚: {mismatch_penalty.item():.6f}")
                                print(f"[DEBUG]   - Compatibility惩罚: {comp_penalty.item():.6f}")
                                
                            elif self.loss_strategy == 'edp_plus_area':
                                edp = latency * energy
                                area_penalty = self.loss_weights['area_weight'] * area
                                mismatch_penalty = mismatch_loss * self.loss_weights.get('mismatch_penalty_weight', 0.1)
                                print(f"[DEBUG] Loss详细组成 (edp_plus_area): 总计={loss.item():.6f}")
                                print(f"[DEBUG]   - EDP: {edp.item():.6f}")
                                print(f"[DEBUG]   - Area惩罚: {area_penalty.item():.6f} (面积: {area.item():.2f} mm²)")
                                print(f"[DEBUG]   - Mismatch惩罚: {mismatch_penalty.item():.6f}")
                                print(f"[DEBUG]   - Compatibility惩罚: {comp_penalty.item():.6f}")
                                
                            elif self.loss_strategy == 'pure_edp':
                                edp = latency * energy
                                mismatch_penalty = mismatch_loss * self.loss_weights.get('mismatch_penalty_weight', 0.1)
                                print(f"[DEBUG] Loss详细组成 (pure_edp): 总计={loss.item():.6f}")
                                print(f"[DEBUG]   - EDP: {edp.item():.6f}")
                                print(f"[DEBUG]   - Mismatch惩罚: {mismatch_penalty.item():.6f}")
                                print(f"[DEBUG]   - Compatibility惩罚: {comp_penalty.item():.6f}")
                                print(f"[DEBUG]   - 面积: {area.item():.2f} mm² (未计入loss)")
                                
                            else:
                                # 默认策略
                                log_edp = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
                                area_penalty = self.loss_weights['area_weight'] * area
                                mismatch_penalty = mismatch_loss * self.loss_weights.get('mismatch_penalty_weight', 0.1)
                                print(f"[DEBUG] Loss详细组成 (默认策略): 总计={loss.item():.6f}")
                                print(f"[DEBUG]   - Log(EDP): {log_edp.item():.6f}")
                                print(f"[DEBUG]   - Area惩罚: {area_penalty.item():.6f} (面积: {area.item():.2f} mm²)")
                                print(f"[DEBUG]   - Mismatch惩罚: {mismatch_penalty.item():.6f}")
                                print(f"[DEBUG]   - Compatibility惩罚: {comp_penalty.item():.6f}")
                            
                            # 显示基础性能指标
                            print(f"[DEBUG] 基础指标: 延迟={latency.item():.2e}s, 能耗={energy.item():.2e}pJ")
                            
                            # 检查optimizer.step()造成的变化
                            step_changes = []
                            for key in hw_before_step:
                                if abs(hw_before_step[key] - hw_after_step[key]) > 1e-6:
                                    step_changes.append(f"{key}: {hw_before_step[key]:.2f} -> {hw_after_step[key]:.2f}")
                            
                            if step_changes:
                                print(f"[DEBUG] Optimizer步骤变化: {', '.join(step_changes)}")
                            
                            # 检查应用最小硬件约束造成的变化
                            bounds_changes = []
                            for key in hw_after_step:
                                if abs(hw_after_step[key] - hw_after_bounds[key]) > 1e-6:
                                    bounds_changes.append(f"{key}: {hw_after_step[key]:.2f} -> {hw_after_bounds[key]:.2f}")
                            
                            if bounds_changes:
                                print(f"[DEBUG] 最小约束调整: {', '.join(bounds_changes)}")
                            
                            if not step_changes and not bounds_changes:
                                print(f"[DEBUG] ✓ 硬件参数无变化")

                    # 计算指标用于记录（避免再次调用 evaluate(flat_params) 造成的二次完整前向）
                    with torch.no_grad():
                        current_params = self._get_params_as_dict()
                        latency2, energy2, area2, mismatch2, compat2 = self.perf_model(
                            self.graph, self.hw_params, self.mapping, self.fusion_params
                        )
                        metrics = {
                            'latency_sec': latency2.item(),
                            'energy_pj': energy2.item(),
                            'area_mm2': area2.item(),
                            'edp': (latency2 * energy2).item(),
                            'log_edp': (torch.log(latency2 + 1e-9) + torch.log(energy2 + 1e-9)).item(),
                            'mismatch_loss': mismatch2.item()
                        }
                    
                    # 在每个hardware step中检查并更新最佳结果
                    trial_count += 1
                    old_best_loss = self.best_loss
                    self.update_best_result(loss.item(), current_params, metrics, trial_count)
                    
                    # 质量驱动的触发：当找到新的全局最优解时保存配置
                    if loss.item() < old_best_loss:
                        self._save_validation_config(trial_count, "quality_driven")
                    
                    # 添加进度输出
                    if i % 2 == 0 or i == self.num_hardware_steps - 1:
                        if self.logger:
                            self.logger.console(
                                f"  Hardware Step {i+1}/{self.num_hardware_steps}: Loss={loss.item():.4f}, EDP={metrics['edp']:.2e}, Area={metrics['area_mm2']:.2f}mm²"
                            )
                
                # Phase B结束后的最终记录（最佳结果已在每个hardware step中更新）
                # 记录日志
                if self.num_hardware_steps % 10 == 0:
                    self.log_trial(trial_count, loss.item(), metrics, current_params)
        
        return {
            'best_loss': self.best_loss,
            'best_params': self.best_params,
            'best_metrics': self.best_metrics,
            'total_trials': trial_count
        }
    
    def _save_validation_config(self, trial_count: int, trigger_type: str):
        """
        保存当前配置到验证数据集
        
        Args:
            trial_count: 当前试验次数
            trigger_type: 触发类型（"quality_driven" 或 "diversity_driven"）
        """
        from .utils import save_configuration_to_json
        
        try:
            # 获取当前完整的映射信息
            projected_mapping = self.mapping.get_all_factors()
            
            # 获取融合决策
            fusion_decisions = self.fusion_params.get_fusion_decisions_serializable(self.graph)
            
            # 生成文件路径
            file_path = f"output/validation_config_trial_{trial_count}.json"
            
            # 保存配置
            save_configuration_to_json(
                hw_params=self.hw_params,
                projected_mapping=projected_mapping,
                fusion_decisions=fusion_decisions,
                file_path=file_path
            )
            
            if self.logger:
                self.logger.event("validation_config_saved", trigger=trigger_type, file_path=file_path)
            
        except Exception as e:
            print(f"Warning: Failed to save validation config at trial {trial_count}: {e}")


class RandomSearcher(BaseSearcher):
    """
    随机搜索器：随机采样参数空间
    """
    
    def __init__(self, graph, hw_params, mapping, fusion_params, perf_model, config, logger=None):
        super().__init__(graph, hw_params, mapping, fusion_params, perf_model, config, logger)
    
    def search(self, num_trials: int) -> Dict[str, Any]:
        """
        执行随机搜索
        
        Args:
            num_trials: 随机试验次数
            
        Returns:
            最佳结果字典
        """
        print(f"Starting Random Search with {num_trials} trials...")
        
        for trial in range(num_trials):
            # 使用SearchSpace随机采样参数
            random_params_dict = self.space.sample()
            
            # 转换为扁平化格式
            flat_params = self.space.to_flat(random_params_dict)
            
            # 评估当前配置
            loss, metrics = self.evaluate(flat_params)
            
            # 更新最佳结果
            self.update_best_result(loss, random_params_dict, metrics, trial + 1)
            
            # 记录日志
            if (trial + 1) % 10 == 0 or trial == 0:
                best_edp = self.best_metrics['edp'] if self.best_metrics else float('inf')
                print(f"Trial {trial + 1}: Loss={loss:.4f}, EDP={metrics['edp']:.2e}, Best EDP={best_edp:.2e}")
            
            self.log_trial(trial + 1, loss, metrics, random_params_dict)
        
        return {
            'best_loss': self.best_loss,
            'best_params': self.best_params,
            'best_metrics': self.best_metrics,
            'total_trials': num_trials
        }


class BayesianOptimizationSearcher(BaseSearcher):
    """
    贝叶斯优化搜索器：基于 scikit-optimize 的高效黑盒优化
    """
    
    def __init__(self, graph, hw_params, mapping, fusion_params, perf_model, config, logger=None):
        super().__init__(graph, hw_params, mapping, fusion_params, perf_model, config, logger)
        
        # 定义 scikit-optimize 搜索空间
        self.skopt_space = self._define_search_space()
    
    def _define_search_space(self):
        """
        将 SearchSpace 转换为 scikit-optimize 格式的搜索空间
        
        Returns:
            scikit-optimize 的 space 对象列表
        """
        from skopt.space import Real, Integer, Categorical
        
        skopt_dimensions = []
        
        # 遍历 SearchSpace 中定义的所有维度，确保顺序一致
        for dim in self.space.dimensions:
            dim_type = dim['type']
            name = dim['name']
            
            if dim_type == 'integer_square':
                # 平方数参数：使用sqrt范围
                min_sqrt, max_sqrt = dim['range']
                skopt_dimensions.append(
                    Integer(low=min_sqrt, high=max_sqrt, name=name)
                )
            
            elif dim_type == 'log_uniform':
                # 对数均匀分布
                min_val, max_val = dim['range']
                skopt_dimensions.append(
                    Real(low=min_val, high=max_val, 
                         prior='log-uniform', name=name)
                )
            
            elif dim_type == 'categorical':
                # 类别类型参数：使用Categorical维度
                categories = dim['categories']
                skopt_dimensions.append(
                    Categorical(categories=categories, name=name)
                )
            
            else:
                raise ValueError(f"Unknown dimension type: {dim_type}")
        
        return skopt_dimensions
    
    def search(self, num_trials: int) -> Dict[str, Any]:
        """
        执行贝叶斯优化搜索
        
        Args:
            num_trials: 评估次数
            
        Returns:
            最佳结果字典
        """
        from skopt import gp_minimize
        
        print(f"Starting Bayesian Optimization with {num_trials} trials...")
        
        # 定义目标函数
        def objective(flat_params: list) -> float:
            """
            贝叶斯优化的目标函数
            
            Args:
                flat_params: scikit-optimize 传入的扁平化参数列表
                
            Returns:
                损失值（需要最小化）
            """
            # 评估参数配置
            loss, metrics = self.evaluate(flat_params)
            
            # 处理无效的损失值
            import numpy as np
            if np.isnan(loss) or np.isinf(loss) or loss > 1e15:
                # 对于无效值，使用一个大的有限值
                loss = 1e15
                # 同时修正metrics中的无效值
                for key, value in metrics.items():
                    if np.isnan(value) or np.isinf(value):
                        metrics[key] = 1e15
            
            # 将扁平化参数转换为结构化字典用于记录
            params_dict = self.space.from_flat(flat_params)
            
            # 更新最佳结果（只有当损失值有效时）
            trial_num = len(objective.trial_history) + 1
            if loss < 1e15:  # 只有有效的损失值才更新最佳结果
                loss_breakdown = getattr(self, '_last_loss_breakdown', None)
                self.update_best_result(loss, params_dict, metrics, trial_num, loss_breakdown)
            
            # 记录试验历史
            objective.trial_history.append({
                'loss': loss,
                'metrics': metrics,
                'params': params_dict
            })
            
            # 记录日志
            if trial_num % 10 == 0 or trial_num == 1:
                best_edp = self.best_metrics['edp'] if self.best_metrics else float('inf')
                print(f"BO Trial {trial_num}: Loss={loss:.4f}, EDP={metrics['edp']:.2e}, Best EDP={best_edp:.2e}")
            
            self.log_trial(trial_num, loss, metrics, params_dict)
            
            return loss
        
        # 初始化试验历史
        objective.trial_history = []
        
        # 执行贝叶斯优化
        result = gp_minimize(
            func=objective,
            dimensions=self.skopt_space,
            n_calls=num_trials,
            n_initial_points=min(20, num_trials // 2),  # 初始随机采样点数
            random_state=42,  # 固定随机种子保证可复现性
            acq_func='EI',  # 期望改进采集函数
            n_jobs=1  # 单线程执行
        )
        
        # 处理优化结果
        best_flat_params = result.x
        best_loss = result.fun
        
        # 将最优参数转换为结构化字典
        best_params_dict = self.space.from_flat(best_flat_params)
        
        print(f"\nBayesian Optimization completed!")
        print(f"Best loss: {best_loss:.4f}")
        if self.best_metrics is not None:
            print(f"Best EDP: {self.best_metrics['edp']:.2e}")
        else:
            print("No valid solutions found during optimization.")
        
        return {
            'best_loss': self.best_loss,
            'best_params': self.best_params,
            'best_metrics': self.best_metrics or {},  # 如果为None则返回空字典
            'total_trials': num_trials,
            'skopt_result': result  # 保存完整的 scikit-optimize 结果
        }


class GeneticAlgorithmSearcher(BaseSearcher):
    """
    遗传算法搜索器（基于DEAP实现）
    """
    
    def __init__(self, graph, hw_params, mapping, fusion_params, perf_model, config, logger=None):
        super().__init__(graph, hw_params, mapping, fusion_params, perf_model, config, logger)
        
        # 遗传算法参数
        self.population_size = getattr(config, 'GA_POPULATION_SIZE', 50)
        self.mutation_rate = getattr(config, 'GA_MUTATION_RATE', 0.1)
        self.crossover_rate = getattr(config, 'GA_CROSSOVER_RATE', 0.8)
        
        # 初始化DEAP
        self._setup_deap()
    
    def _setup_deap(self):
        """
        设置DEAP遗传算法框架
        """
        from deap import base, creator, tools
        
        # 创建适应度类型（最小化）
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        
        # 创建个体类型
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # 创建工具箱
        self.toolbox = base.Toolbox()
        
        # 注册基因生成函数
        self.toolbox.register("attr_item", self._sample_attribute)
        
        # 注册个体和种群生成函数
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attr_item)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # 注册演化算子
        self.toolbox.register("evaluate", self._deap_evaluate_wrapper)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._deap_mutate, indpb=self.mutation_rate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def _sample_attribute(self) -> list:
        """
        从搜索空间中随机采样一个扁平化的参数列表
        
        Returns:
            扁平化的参数列表
        """
        # 从SearchSpace随机采样
        params_dict = self.space.sample()
        # 转换为扁平化列表
        return self.space.to_flat(params_dict)
    
    def _deap_evaluate_wrapper(self, individual: list) -> tuple:
        """
        DEAP评估函数包装器
        
        Args:
            individual: 个体（扁平化参数列表）
            
        Returns:
            适应度元组
        """
        # 评估个体
        loss, metrics = self.evaluate(individual)
        
        # 转换为结构化字典用于记录
        params_dict = self.space.from_flat(individual)
        
        # 更新最佳结果
        trial_num = getattr(self, '_current_trial', 0) + 1
        self._current_trial = trial_num
        loss_breakdown = getattr(self, '_last_loss_breakdown', None)
        self.update_best_result(loss, params_dict, metrics, trial_num, loss_breakdown)
        
        # 记录日志
        if trial_num % 10 == 0 or trial_num == 1:
            best_edp = self.best_metrics['edp'] if self.best_metrics else float('inf')
            print(f"GA Trial {trial_num}: Loss={loss:.4f}, EDP={metrics['edp']:.2e}, Best EDP={best_edp:.2e}")
        
        self.log_trial(trial_num, loss, metrics, params_dict)
        
        # DEAP需要返回元组
        return (loss,)
    
    def _deap_mutate(self, individual: list, indpb: float) -> tuple:
        """
        自定义变异算子
        
        Args:
            individual: 个体（扁平化参数列表）
            indpb: 每个基因的变异概率
            
        Returns:
            变异后的个体
        """
        import random
        
        # 遍历个体中的每个基因
        for i in range(len(individual)):
            # 以indpb概率决定是否变异
            if random.random() < indpb:
                # 获取对应的维度定义
                dim = self.space.dimensions[i]
                dim_type = dim['type']
                
                if dim_type == 'integer_square':
                    # 平方数参数：重新采样sqrt值
                    min_sqrt, max_sqrt = dim['range']
                    individual[i] = float(random.randint(min_sqrt, max_sqrt))
                    
                elif dim_type == 'log_uniform':
                    # 对数均匀分布：重新采样
                    min_val, max_val = dim['range']
                    import numpy as np
                    log_min, log_max = np.log(min_val), np.log(max_val)
                    individual[i] = float(np.exp(random.uniform(log_min, log_max)))
                    
                elif dim_type == 'categorical':
                    # 类别参数：重新采样索引
                    num_categories = len(dim['categories'])
                    individual[i] = float(random.randint(0, num_categories - 1))
                    
                else:
                    raise ValueError(f"Unknown dimension type: {dim_type}")
        
        return (individual,)
    
    def search(self, num_trials: int) -> Dict[str, Any]:
        """
        执行遗传算法搜索
        
        Args:
            num_trials: 评估次数（对应于代数 * 种群大小）
            
        Returns:
            最佳结果字典
        """
        from deap import algorithms, tools
        import random
        
        # 计算代数
        generations = max(1, num_trials // self.population_size)
        actual_trials = generations * self.population_size
        
        print(f"Starting Genetic Algorithm with {generations} generations, population size {self.population_size}")
        print(f"Total evaluations: {actual_trials}")
        
        # 初始化试验计数器
        self._current_trial = 0
        
        # 设置随机种子
        random.seed(42)
        
        # 初始化种群
        pop = self.toolbox.population(n=self.population_size)
        
        # 设置统计信息
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", lambda x: sum(x) / len(x))
        stats.register("min", min)
        stats.register("max", max)
        
        # 名人堂（保存最优个体）
        hof = tools.HallOfFame(1)
        
        # 运行演化算法
        print("\nStarting evolution...")
        
        # 评估初始种群
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # 更新名人堂
        hof.update(pop)
        
        # 记录初始统计信息
        record = stats.compile(pop)
        print(f"Generation 0: {record}")
        
        # 演化循环
        for generation in range(1, generations + 1):
            print(f"\n--- Generation {generation}/{generations} ---")
            
            # 选择下一代的父代
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # 交叉
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # 变异
            for mutant in offspring:
                if random.random() < self.mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # 评估无效个体
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # 替换种群
            pop[:] = offspring
            
            # 更新名人堂
            hof.update(pop)
            
            # 记录统计信息
            record = stats.compile(pop)
            print(f"Generation {generation}: {record}")
        
        # 获取最优个体
        best_individual = hof[0]
        best_loss = best_individual.fitness.values[0]
        
        # 转换最优参数为结构化字典
        best_params_dict = self.space.from_flat(list(best_individual))
        
        print(f"\nGenetic Algorithm completed!")
        print(f"Best loss: {best_loss:.4f}")
        print(f"Best EDP: {self.best_metrics['edp']:.2e}")
        print(f"Total evaluations: {self._current_trial}")
        
        return {
            'best_loss': self.best_loss,
            'best_params': self.best_params,
            'best_metrics': self.best_metrics,
            'total_trials': self._current_trial,
            'generations': generations,
            'population_size': self.population_size,
            'hall_of_fame': hof  # 保存名人堂
        }