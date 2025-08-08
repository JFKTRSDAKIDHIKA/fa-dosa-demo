import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from functools import reduce

# Forward-declare for type hinting
class ComputationGraph:
    pass

class HardwareParameters:
    pass

class FineGrainedMapping:
    pass

class Config:
    pass

class BaseDMT(nn.Module, ABC):
    """Abstract Base Class for Differentiable Mapping Templates (DMTs)."""

    @abstractmethod
    def forward(self, group: list, graph: ComputationGraph, hw_params: HardwareParameters, 
                mapping: FineGrainedMapping, config: Config, direct_mapping_table: dict = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        DMT前向传播方法，支持验证快车道。
        
        Args:
            group: 融合组层名列表
            graph: 计算图
            hw_params: 硬件参数
            mapping: FineGrainedMapping实例（用于训练路径）
            config: 配置对象
            direct_mapping_table: （可选）直接映射表，用于验证模式
        
        Returns:
            (latency, energy, buffer_mismatch_loss, compatibility_penalty, detailed_metrics)
        
        Estimates the performance and energy for a given fusion group.

        Args:
            group (list): A list of layer names forming the fusion group.
            graph (ComputationGraph): The overall computation graph.
            hw_params (HardwareParameters): The hardware parameters.
            mapping (FineGrainedMapping): The fine-grained mapping configuration.
            config (Config): The general configuration.

        Returns:
            A tuple containing:
            - latency (torch.Tensor): The estimated latency for the group.
            - energy (torch.Tensor): The estimated energy consumption for the group.
            - buffer_mismatch_loss (torch.Tensor): The buffer mismatch loss for the group.
            - compatibility_penalty (torch.Tensor): The compatibility penalty for the group.
            - detailed_metrics (dict): Detailed performance metrics.
        """
        pass

class InPlaceFusionDMT(BaseDMT):
    """DMT for in-place fusion patterns like Conv -> ReLU.
    
    重构后的InPlaceFusionDMT作为任务分派器，将融合组的计算任务
    分派给HighFidelityPerformanceModel核心引擎进行统一的PPA计算。
    """
    
    def __init__(self, debug_latency: bool = False, debug_output_path: str = None):
        super().__init__()
        self.debug_latency = debug_latency
        self.debug_output_path = debug_output_path

    def forward(self, group: list, graph: ComputationGraph, hw_params: HardwareParameters, 
                mapping: FineGrainedMapping, config: Config, direct_mapping_table: dict = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        
        # 步骤1: 解析输入 - 识别生产者层（决定性能的关键层）
        producer_name = group[0]  # 对于Conv->ReLU，Conv是生产者
        consumers = group[1:]     # ReLU等消费者层
        
        # 步骤2: 准备输入 - 创建只包含生产者层的伪图
        # 核心洞察：原位融合的性能完全由生产者层决定
        pseudo_graph = type('PseudoGraph', (), {
            'layers': {producer_name: graph.layers[producer_name]},
            'fusion_groups': [[producer_name]]  # 单层融合组
        })()
        
        # 步骤3: 调用核心引擎 - 实例化并调用HighFidelityPerformanceModel
        from dosa.performance_model import HighFidelityPerformanceModel
        perf_model = HighFidelityPerformanceModel(config, debug_latency=self.debug_latency)
        
        # 调用核心引擎的forward方法进行统一PPA计算
        latency, energy, area, buffer_mismatch_loss, compatibility_penalty = perf_model.forward(
            graph=pseudo_graph,
            hw_params=hw_params,
            mapping=mapping,
            direct_mapping_table=direct_mapping_table,
            debug_output_path=self.debug_output_path
        )
        
        # 步骤4: 返回结果 - 直接返回核心引擎的计算结果
        # 对于原位融合，不需要额外的组合逻辑
        
        # 构建详细指标（保持接口兼容性）
        detailed_metrics = {
            'energy_breakdown_pj': {
                'compute': float(energy.item()) if hasattr(energy, 'item') else float(energy),
                'intra_level': {
                    'L0_Registers': 0.0,
                    'L1_Accumulator': 0.0,
                    'L2_Scratchpad': 0.0
                },
                'inter_level': {
                    'L3_DRAM': 0.0
                }
            },
            'access_counts': {
                'intra_level': {},
                'inter_level': {}
            },
            'fusion_type': 'in_place',
            'producer_layer': producer_name,
            'consumer_layers': consumers
        }
        
        return latency, energy, buffer_mismatch_loss, compatibility_penalty, detailed_metrics


class SkipConnectionDMT(BaseDMT):
    """DMT for ResNet skip connection patterns like Conv -> BN -> ReLU -> Add.
    
    重构后的SkipConnectionDMT作为任务分派器，处理残差连接的并行计算逻辑：
    - 主路径和旁路并行执行，延迟取较大者
    - 能耗为两路径的叠加
    """

    def forward(self, group: list, graph: ComputationGraph, hw_params: HardwareParameters, 
                mapping: FineGrainedMapping, config: Config, direct_mapping_table: dict = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        
        # 步骤1: 解析输入 - 识别主路径和旁路
        add_layer_name = group[-1]  # Add操作通常是最后一个
        main_path = group[:-1]      # 主计算路径（Conv->BN->ReLU等）
        
        # 识别跳跃连接的源头
        add_layer_inputs = graph.get_layer_inputs(add_layer_name)
        skip_path_source = None
        for input_layer in add_layer_inputs:
            if input_layer not in group:
                skip_path_source = input_layer
                break
        
        # 步骤2: 准备输入 - 为主路径创建伪图
        if main_path:
            # 为主路径创建只包含生产者层的伪图
            producer_name = main_path[0]  # 主路径的生产者层
            main_pseudo_graph = type('MainPseudoGraph', (), {
                'layers': {producer_name: graph.layers[producer_name]},
                'fusion_groups': [[producer_name]]  # 单层融合组
            })()
        else:
            main_pseudo_graph = None
        
        # 步骤3: 调用核心引擎 - 计算主路径PPA
        from dosa.performance_model import HighFidelityPerformanceModel
        
        if main_pseudo_graph:
            perf_model_main = HighFidelityPerformanceModel(config, debug_latency=True)
            latency_main, energy_main, area_main, buffer_loss_main, compat_penalty_main = perf_model_main.forward(
                graph=main_pseudo_graph,
                hw_params=hw_params,
                mapping=mapping,
                direct_mapping_table=direct_mapping_table,
                debug_output_path="debug_performance_model.json"
            )
        else:
            latency_main = torch.tensor(0.0, device=config.DEVICE)
            energy_main = torch.tensor(0.0, device=config.DEVICE)
            buffer_loss_main = torch.tensor(0.0, device=config.DEVICE)
        
        # 步骤4: 建模旁路数据搬运成本
        if skip_path_source:
            skip_layer = graph.layers[skip_path_source]
            # 旁路张量大小（字节）
            skip_tensor_elements = reduce(lambda x, y: x * y, skip_layer['dims'].values(), 1)
            skip_tensor_bytes = skip_tensor_elements * config.BYTES_PER_ELEMENT
            
            # 旁路延迟：从L2读取的时间
            from dosa.performance_model import calculate_bandwidth_bytes_per_cycle
            l2_bandwidth_bytes_per_cycle = calculate_bandwidth_bytes_per_cycle(
                'L2_Scratchpad', hw_params.get_num_pes(), config)
            latency_skip_cycles = skip_tensor_bytes / l2_bandwidth_bytes_per_cycle
            latency_skip = latency_skip_cycles / (config.CLOCK_FREQUENCY_MHZ * 1e6)
            
            # 旁路能耗：L2读取能耗
            l2_size_kb = hw_params.get_buffer_size_kb('L2_Scratchpad')
            l2_epa = config.L2_SPM_BASE_EPA_PJ
            energy_skip = (skip_tensor_elements * l2_epa)
        else:
            latency_skip = torch.tensor(0.0, device=config.DEVICE)
            energy_skip = torch.tensor(0.0, device=config.DEVICE)
        
        # 步骤5: Add操作成本建模
        add_layer = graph.layers[add_layer_name]
        add_elements = reduce(lambda x, y: x * y, add_layer['dims'].values(), 1)
        
        # Add操作延迟（简化为单周期操作）
        add_cycles = add_elements / hw_params.get_num_pes()
        latency_add_op = add_cycles / (config.CLOCK_FREQUENCY_MHZ * 1e6)
        
        # Add操作能耗（简化为基础运算能耗）
        energy_add_op = add_elements * config.PE_MAC_EPA_PJ * 0.1  # Add比MAC能耗低
        
        # 步骤6: 组合结果
        # 延迟：并行执行取最大值，再加Add操作延迟
        total_latency = torch.maximum(latency_main, latency_skip) + latency_add_op
        
        # 能耗：所有路径能耗叠加
        total_energy = energy_main + energy_skip + energy_add_op
        
        # 缓冲区不匹配损失：主路径的损失（旁路数据通常已在缓存中）
        total_buffer_mismatch_loss = buffer_loss_main
        
        # 兼容性惩罚：暂时设为0
        total_compatibility_penalty = torch.tensor(0.0, device=config.DEVICE)
        
        # 构建详细指标（保持接口兼容性）
        detailed_metrics = {
            'energy_breakdown_pj': {
                'compute': float(total_energy.item()) if hasattr(total_energy, 'item') else float(total_energy),
                'intra_level': {
                    'L0_Registers': 0.0,
                    'L1_Accumulator': 0.0,
                    'L2_Scratchpad': 0.0
                },
                'inter_level': {
                    'L3_DRAM': 0.0
                }
            },
            'access_counts': {
                'intra_level': {},
                'inter_level': {}
            },
            'fusion_type': 'skip_connection',
            'main_path': main_path,
            'skip_path_source': skip_path_source,
            'add_layer': add_layer_name,
            'latency_breakdown': {
                'main_path': float(latency_main.item()) if hasattr(latency_main, 'item') else float(latency_main),
                'skip_path': float(latency_skip.item()) if hasattr(latency_skip, 'item') else float(latency_skip),
                'add_operation': float(latency_add_op.item()) if hasattr(latency_add_op, 'item') else float(latency_add_op)
            },
            'energy_breakdown': {
                'main_path': float(energy_main.item()) if hasattr(energy_main, 'item') else float(energy_main),
                'skip_path': float(energy_skip.item()) if hasattr(energy_skip, 'item') else float(energy_skip),
                'add_operation': float(energy_add_op.item()) if hasattr(energy_add_op, 'item') else float(energy_add_op)
            }
        }
        
        return total_latency, total_energy, total_buffer_mismatch_loss, total_compatibility_penalty, detailed_metrics


class SequentialConvDMT(BaseDMT):
    """示例DMT类，展示基于Orojenesis/FFMT思想的兼容性惩罚计算逻辑。"""

    def forward(self, group: list, graph: ComputationGraph, hw_params: HardwareParameters, 
                mapping: FineGrainedMapping, config: Config, direct_mapping_table: dict = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        
        # 简化的性能建模（可以不完整）
        latency = torch.tensor(1.0, device=config.DEVICE)
        energy = torch.tensor(1.0, device=config.DEVICE)
        buffer_mismatch_loss = torch.tensor(0.0, device=config.DEVICE)
        
        detailed_metrics = {
            'energy_breakdown_pj': {'compute': 1.0, 'intra_level': {}, 'inter_level': {}},
            'access_counts': {'intra_level': {}, 'inter_level': {}}
        }
        
        # 兼容性惩罚计算逻辑（基于Orojenesis/FFMT思想）
        compatibility_penalty = torch.tensor(0.0, device=config.DEVICE)
        
        if group:  # 确保融合链非空
            # 检查融合链中第一个Conv层（生产者）的映射
            producer_name = group[0]
            producer_layer = graph.layers[producer_name]
            
            # 检查是否为卷积层
            if 'conv' in producer_name.lower() or producer_layer.get('type', '').lower() == 'conv':
                all_factors = mapping.get_all_factors()
                
                # 检查所有归约维度(C, R, S)在L3_DRAM上的时间分块因子
                reduction_dims = ['C', 'R', 'S']
                
                for dim in reduction_dims:
                    # 构造DRAM时间分块因子的键名
                    dram_temporal_key = f'{dim}_L3_DRAM_temporal'
                    
                    if dram_temporal_key in all_factors:
                        dram_temporal_factor = all_factors[dram_temporal_key]
                        
                        # 如果DRAM时间分块因子大于1，说明有部分和溢出到DRAM
                        # 这违反了融合约束，需要施加惩罚
                        penalty_contribution = torch.relu(dram_temporal_factor - 1.0)
                        compatibility_penalty += penalty_contribution
        
        return latency, energy, buffer_mismatch_loss, compatibility_penalty, detailed_metrics


def convert_tensors_to_native(obj):
    """将Tensor对象转换为Python原生数据类型，确保detailed_metrics可以被正确处理。"""
    if isinstance(obj, torch.Tensor):
        return float(obj.item())
    elif isinstance(obj, dict):
        return {k: convert_tensors_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors_to_native(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_tensors_to_native(item) for item in obj)
    else:
        return obj