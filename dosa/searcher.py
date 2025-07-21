from abc import ABC, abstractmethod
import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Tuple
import random
from .utils import OptimizationLogger, get_divisors


class BaseSearcher(ABC):
    """抽象基类，定义所有搜索器的通用接口"""
    
    def __init__(self, graph, hw_params, mapping, fusion_params, perf_model, config, logger=None):
        """
        初始化搜索器
        
        Args:
            graph: ComputationGraph实例
            hw_params: HardwareParameters实例
            mapping: FineGrainedMapping实例
            fusion_params: FusionParameters实例
            perf_model: HighFidelityPerformanceModel实例
            config: 配置对象
            logger: OptimizationLogger实例
        """
        self.graph = graph
        self.hw_params = hw_params
        self.mapping = mapping
        self.fusion_params = fusion_params
        self.perf_model = perf_model
        self.config = config
        self.logger = logger or OptimizationLogger()
        
        # 记录最佳结果
        self.best_loss = float('inf')
        self.best_params = None
        self.best_metrics = None
        
        # 损失策略配置
        self.loss_strategy = getattr(config, 'LOSS_STRATEGY', 'original')
        self.loss_weights = getattr(config, 'LOSS_WEIGHTS', {
            'area_weight': 0.1,
            'mismatch_penalty_weight': 10.0,
            'pe_penalty_weight_phase_a': 0.1,
            'pe_penalty_weight_phase_b': 0.01,
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
    
    def evaluate(self, params: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        统一的目标函数接口，评估给定参数的性能
        
        Args:
            params: 扁平化的参数字典
            
        Returns:
            (loss, metrics): 损失值和性能指标字典
        """
        # 将扁平化参数设置到模型中
        self._set_params_from_dict(params)
        
        # 调用性能模型
        latency, energy, area, mismatch_loss = self.perf_model(
            self.graph, self.hw_params, self.mapping
        )
        
        # 计算PE惩罚
        pe_square_penalty = self.hw_params.get_pe_square_penalty()
        
        # 根据策略计算损失
        if self.loss_strategy == 'strategy_A':
            edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
            area_loss = self.loss_weights['area_weight'] * area
            mismatch_penalty = torch.log(1.0 + mismatch_loss * self.loss_weights['mismatch_penalty_weight'])
            loss = edp_loss + area_loss + mismatch_penalty + pe_square_penalty * self.loss_weights['pe_penalty_weight_phase_a']
        elif self.loss_strategy == 'strategy_B':
            edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
            area_loss = self.loss_weights['area_weight'] * area
            loss = self.loss_weights['edp_weight'] * edp_loss + area_loss + mismatch_loss * self.loss_weights['mismatch_penalty_weight'] + pe_square_penalty * self.loss_weights['pe_penalty_weight_phase_a']
        else:  # 原始策略
            edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
            area_loss = self.loss_weights['area_weight'] * area
            loss = edp_loss + area_loss + mismatch_loss * self.loss_weights['mismatch_penalty_weight'] + pe_square_penalty * self.loss_weights['pe_penalty_weight_phase_a']
        
        # 构建性能指标字典
        metrics = {
            'latency_sec': latency.item(),
            'energy_pj': energy.item(),
            'area_mm2': area.item(),
            'edp': (latency * energy).item(),
            'log_edp': (torch.log(latency + 1e-9) + torch.log(energy + 1e-9)).item(),
            'mismatch_loss': mismatch_loss.item(),
            'pe_penalty': pe_square_penalty.item()
        }
        
        return loss.item(), metrics
    
    def _compute_loss(self, latency, energy, area, mismatch_loss, pe_square_penalty):
        """
        计算总损失
        
        Args:
            latency: 延迟张量
            energy: 能耗张量
            area: 面积张量
            mismatch_loss: 不匹配损失张量
            pe_square_penalty: PE平方惩罚张量
            
        Returns:
            总损失张量
        """
        # 确保所有输入都是标量张量
        latency = latency.squeeze() if latency.dim() > 0 else latency
        energy = energy.squeeze() if energy.dim() > 0 else energy
        area = area.squeeze() if area.dim() > 0 else area
        mismatch_loss = mismatch_loss.squeeze() if mismatch_loss.dim() > 0 else mismatch_loss
        pe_square_penalty = pe_square_penalty.squeeze() if pe_square_penalty.dim() > 0 else pe_square_penalty
        
        # 使用与原始实现相同的损失策略
        if self.config.LOSS_STRATEGY == 'log_edp_plus_area':
            log_edp = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
            area_penalty = self.config.AREA_WEIGHT * area
            loss = log_edp + area_penalty + mismatch_loss + pe_square_penalty
        elif self.config.LOSS_STRATEGY == 'edp_plus_area':
            edp = latency * energy
            area_penalty = self.config.AREA_WEIGHT * area
            loss = edp + area_penalty + mismatch_loss + pe_square_penalty
        else:
            # 默认策略
            log_edp = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
            area_penalty = self.config.AREA_WEIGHT * area
            loss = log_edp + area_penalty + mismatch_loss + pe_square_penalty
        
        # 确保返回标量张量
        return loss.squeeze() if loss.dim() > 0 else loss
    
    def _set_params_from_dict(self, params: Dict[str, Any]):
        """
        将扁平化的参数字典设置到模型实例中
        
        Args:
            params: 包含所有参数的扁平化字典
        """
        # 设置硬件参数
        if 'num_pes' in params:
            # 确保num_pes是平方数
            sqrt_pes = int(np.sqrt(params['num_pes']))
            actual_pes = sqrt_pes * sqrt_pes
            self.hw_params.log_num_pes.data = torch.log(torch.tensor(float(actual_pes)))
        
        for level in ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad']:
            key = f'{level.lower()}_size_kb'
            if key in params:
                self.hw_params.log_buffer_sizes_kb[level].data = torch.log(torch.tensor(params[key]))
        
        # 设置映射参数 - 只为实际存在的on-chip buffer层级设置参数
        on_chip_levels = ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad']
        for dim_name in self.graph.problem_dims.keys():
            for level_name in on_chip_levels:
                # 确保level_name在mapping.factors中存在
                if level_name in self.mapping.factors:
                    temporal_key = f'{dim_name}_{level_name}_temporal'
                    spatial_key = f'{dim_name}_{level_name}_spatial'
                    
                    if temporal_key in params:
                        self.mapping.factors[level_name][dim_name]['temporal'].data = torch.log(torch.tensor(params[temporal_key]))
                    if spatial_key in params:
                        self.mapping.factors[level_name][dim_name]['spatial'].data = torch.log(torch.tensor(params[spatial_key]))
        
        # 设置融合参数
        if 'fusion_logits' in params:
            fusion_logits = params['fusion_logits']
            if isinstance(fusion_logits, list):
                fusion_logits = torch.tensor(fusion_logits).unsqueeze(1)
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
        params['fusion_logits'] = self.fusion_params.fusion_logits.squeeze().tolist()
        
        return params
    
    def update_best_result(self, loss: float, params: Dict[str, Any], metrics: Dict[str, float], trial: int):
        """
        更新最佳结果
        
        Args:
            loss: 当前损失值
            params: 当前参数
            metrics: 当前性能指标
            trial: 当前试验次数
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = params.copy()
            self.best_metrics = metrics.copy()
            
            print(f"Trial {trial}: New best found! Loss={loss:.4f}, EDP={metrics['edp']:.2e}")
    
    def log_trial(self, trial: int, loss: float, metrics: Dict[str, float], params: Dict[str, Any]):
        """
        记录试验结果
        
        Args:
            trial: 试验次数
            loss: 损失值
            metrics: 性能指标
            params: 参数字典
        """
        log_data = {
            'searcher_type': self.__class__.__name__,
            'trial_number': trial,
            'loss_total': loss,
            'current_edp': metrics['edp'],
            'best_edp_so_far': self.best_metrics['edp'] if self.best_metrics else metrics['edp'],
            'performance_metrics': {
                'latency_sec': metrics['latency_sec'],
                'energy_pj': metrics['energy_pj'],
                'area_mm2': metrics['area_mm2'],
                'log_edp': metrics['log_edp']
            },
            'hardware_params': {
                'num_pes': params.get('num_pes', 0),
                'l0_size_kb': params.get('l0_registers_size_kb', 0),
                'l1_size_kb': params.get('l1_accumulator_size_kb', 0),
                'l2_size_kb': params.get('l2_scratchpad_size_kb', 0)
            },
            'fusion_decisions': self.fusion_params.get_fusion_decisions_serializable(self.graph)
        }
        
        self.logger.log_step(log_data)


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


def sample_random_params(graph, hw_param_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    """
    随机采样参数空间
    
    Args:
        graph: ComputationGraph实例
        hw_param_ranges: 硬件参数范围字典
        
    Returns:
        随机采样的参数字典
    """
    params = {}
    
    # 随机采样硬件参数
    # num_pes: 确保是平方数
    min_sqrt = int(np.sqrt(hw_param_ranges['num_pes'][0]))
    max_sqrt = int(np.sqrt(hw_param_ranges['num_pes'][1]))
    sqrt_pes = random.randint(min_sqrt, max_sqrt)
    params['num_pes'] = sqrt_pes * sqrt_pes
    
    # buffer sizes: 对数均匀采样
    for level in ['l0_registers', 'l1_accumulator', 'l2_scratchpad']:
        key = f'{level}_size_kb'
        if key in hw_param_ranges:
            min_val, max_val = hw_param_ranges[key]
            log_min, log_max = np.log(min_val), np.log(max_val)
            params[key] = np.exp(random.uniform(log_min, log_max))
    
    # 随机采样映射参数
    for dim_name, dim_size in graph.problem_dims.items():
        for level_name in ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad', 'L3_DRAM']:
            # 随机选择有效的temporal和spatial因子
            temporal_factor = get_random_valid_divisor(dim_size)
            remaining_size = dim_size // temporal_factor
            spatial_factor = get_random_valid_divisor(remaining_size)
            
            params[f'{dim_name}_{level_name}_temporal'] = float(temporal_factor)
            params[f'{dim_name}_{level_name}_spatial'] = float(spatial_factor)
    
    # 随机采样融合参数
    num_fusion_groups = len(graph.fusion_groups)
    params['fusion_logits'] = [random.choice([-2.0, 2.0]) for _ in range(num_fusion_groups)]
    
    return params


class FADOSASearcher(BaseSearcher):
    """
    FA-DOSA搜索器：基于梯度的交替优化
    """
    
    def __init__(self, graph, hw_params, mapping, fusion_params, perf_model, config, logger=None):
        super().__init__(graph, hw_params, mapping, fusion_params, perf_model, config, logger)
        
        # FA-DOSA特定参数
        self.num_outer_steps = getattr(config, 'NUM_OUTER_STEPS', 5)
        self.num_mapping_steps = getattr(config, 'NUM_MAPPING_STEPS', 50)
        self.num_hardware_steps = getattr(config, 'NUM_HARDWARE_STEPS', 50)
        self.lr_mapping = getattr(config, 'LR_MAPPING', 0.01)
        self.lr_hardware = getattr(config, 'LR_HARDWARE', 0.01)
    
    def search(self, num_trials: int) -> Dict[str, Any]:
        """
        执行FA-DOSA的交替优化搜索
        
        Args:
            num_trials: 这里对应于外层优化步数
            
        Returns:
            最佳结果字典
        """
        print(f"Starting FA-DOSA search with {self.num_outer_steps} outer steps...")
        
        trial_count = 0
        
        # 交替优化循环
        for outer_step in range(self.num_outer_steps):
            print(f"\n--- Outer Step {outer_step + 1}/{self.num_outer_steps} ---")
            
            # Phase A: 优化映射和融合参数（冻结硬件参数）
            print("--- Phase A: Optimizing Mapping & Fusion ---")
            
            # 冻结硬件参数
            for p in self.hw_params.parameters():
                p.requires_grad = False
            # 解冻映射和融合参数
            for p in list(self.mapping.parameters()) + list(self.fusion_params.parameters()):
                p.requires_grad = True
            
            # 创建映射和融合参数的优化器
            optimizer_map = optim.Adam(
                list(self.mapping.parameters()) + list(self.fusion_params.parameters()), 
                lr=self.lr_mapping
            )
            
            for i in range(self.num_mapping_steps):
                optimizer_map.zero_grad()
                
                # 直接计算损失（保持梯度图）
                latency, energy, area, mismatch_loss = self.perf_model(
                    self.graph, self.hw_params, self.mapping
                )
                
                # 简化损失计算，先只使用基本项
                edp = latency * energy
                loss = torch.log(edp + 1e-8)  # 只使用log EDP
                
                # 反向传播
                loss.backward()
                optimizer_map.step()
                
                # 计算指标用于记录
                with torch.no_grad():
                    current_params = self._get_params_as_dict()
                    _, metrics = self.evaluate(current_params)
                
                # 退火温度
                self.mapping.anneal_tau()
                
                # 更新最佳结果
                trial_count += 1
                self.update_best_result(loss.item(), current_params, metrics, trial_count)
                
                # 记录日志
                if i % 10 == 0:
                    print(f"[Map] Iter {i}: Loss={loss.item():.4f}, EDP={metrics['edp']:.2e}, Area={metrics['area_mm2']:.2f}mm²")
                    self.log_trial(trial_count, loss.item(), metrics, current_params)
            
            # Phase B: 优化硬件参数（冻结映射和融合参数）
            print("--- Phase B: Optimizing Hardware ---")
            
            # 冻结映射和融合参数
            for p in list(self.mapping.parameters()) + list(self.fusion_params.parameters()):
                p.requires_grad = False
            # 解冻硬件参数
            for p in self.hw_params.parameters():
                p.requires_grad = True
            
            # 创建硬件参数的优化器
            optimizer_hw = optim.Adam(self.hw_params.parameters(), lr=self.lr_hardware)
            
            for i in range(self.num_hardware_steps):
                optimizer_hw.zero_grad()
                
                # 直接计算损失（保持梯度图）
                latency, energy, area, mismatch_loss = self.perf_model(
                    self.graph, self.hw_params, self.mapping
                )
                
                # 简化损失计算，先只使用基本项
                edp = latency * energy
                loss = torch.log(edp + 1e-8)  # 只使用log EDP
                
                # 反向传播
                loss.backward()
                optimizer_hw.step()
                
                # 计算指标用于记录
                with torch.no_grad():
                    current_params = self._get_params_as_dict()
                    _, metrics = self.evaluate(current_params)
                
                # 更新最佳结果
                trial_count += 1
                self.update_best_result(loss.item(), current_params, metrics, trial_count)
                
                # 记录日志
                if i % 10 == 0:
                    print(f"[HW] Iter {i}: Loss={loss.item():.4f}, EDP={metrics['edp']:.2e}, Area={metrics['area_mm2']:.2f}mm²")
                    self.log_trial(trial_count, loss.item(), metrics, current_params)
        
        return {
            'best_loss': self.best_loss,
            'best_params': self.best_params,
            'best_metrics': self.best_metrics,
            'total_trials': trial_count
        }


class RandomSearcher(BaseSearcher):
    """
    随机搜索器：随机采样参数空间
    """
    
    def __init__(self, graph, hw_params, mapping, fusion_params, perf_model, config, logger=None):
        super().__init__(graph, hw_params, mapping, fusion_params, perf_model, config, logger)
        
        # 定义参数空间范围
        self.hw_param_ranges = {
            'num_pes': (16, 1024),  # PE数量范围
            'l0_registers_size_kb': (0.1, 10.0),  # L0缓存大小范围
            'l1_accumulator_size_kb': (0.5, 50.0),  # L1缓存大小范围
            'l2_scratchpad_size_kb': (1.0, 100.0)  # L2缓存大小范围
        }
    
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
            # 随机采样参数
            random_params = sample_random_params(self.graph, self.hw_param_ranges)
            
            # 评估当前配置
            loss, metrics = self.evaluate(random_params)
            
            # 更新最佳结果
            self.update_best_result(loss, random_params, metrics, trial + 1)
            
            # 记录日志
            if (trial + 1) % 10 == 0 or trial == 0:
                print(f"Trial {trial + 1}: Loss={loss:.4f}, EDP={metrics['edp']:.2e}, Best EDP={self.best_metrics['edp']:.2e}")
            
            self.log_trial(trial + 1, loss, metrics, random_params)
        
        return {
            'best_loss': self.best_loss,
            'best_params': self.best_params,
            'best_metrics': self.best_metrics,
            'total_trials': num_trials
        }


class BayesianOptimizationSearcher(BaseSearcher):
    """
    贝叶斯优化搜索器（框架实现）
    """
    
    def __init__(self, graph, hw_params, mapping, fusion_params, perf_model, config, logger=None):
        super().__init__(graph, hw_params, mapping, fusion_params, perf_model, config, logger)
        
        # TODO: 集成scikit-optimize
        # self.space = self._define_search_space()
    
    def search(self, num_trials: int) -> Dict[str, Any]:
        """
        执行贝叶斯优化搜索（待实现）
        
        Args:
            num_trials: 评估次数
            
        Returns:
            最佳结果字典
        """
        print(f"Bayesian Optimization Searcher (Framework) - {num_trials} trials")
        print("TODO: Implement scikit-optimize integration")
        
        # 临时实现：回退到随机搜索
        random_searcher = RandomSearcher(
            self.graph, self.hw_params, self.mapping, 
            self.fusion_params, self.perf_model, self.config, self.logger
        )
        return random_searcher.search(num_trials)
    
    def _define_search_space(self):
        """
        定义贝叶斯优化的搜索空间（待实现）
        
        Returns:
            scikit-optimize的space对象
        """
        # TODO: 使用skopt.space定义搜索空间
        pass


class GeneticAlgorithmSearcher(BaseSearcher):
    """
    遗传算法搜索器（框架实现）
    """
    
    def __init__(self, graph, hw_params, mapping, fusion_params, perf_model, config, logger=None):
        super().__init__(graph, hw_params, mapping, fusion_params, perf_model, config, logger)
        
        # TODO: 集成DEAP
        self.population_size = getattr(config, 'GA_POPULATION_SIZE', 50)
        self.mutation_rate = getattr(config, 'GA_MUTATION_RATE', 0.1)
        self.crossover_rate = getattr(config, 'GA_CROSSOVER_RATE', 0.8)
    
    def search(self, num_trials: int) -> Dict[str, Any]:
        """
        执行遗传算法搜索（待实现）
        
        Args:
            num_trials: 评估次数（对应于代数 * 种群大小）
            
        Returns:
            最佳结果字典
        """
        print(f"Genetic Algorithm Searcher (Framework) - {num_trials} trials")
        print("TODO: Implement DEAP integration")
        
        # 临时实现：回退到随机搜索
        random_searcher = RandomSearcher(
            self.graph, self.hw_params, self.mapping, 
            self.fusion_params, self.perf_model, self.config, self.logger
        )
        return random_searcher.search(num_trials)