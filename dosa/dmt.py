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
                mapping: FineGrainedMapping, config: Config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
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
    """DMT for in-place fusion patterns like Conv -> ReLU."""

    def forward(self, group: list, graph: ComputationGraph, hw_params: HardwareParameters, 
                mapping: FineGrainedMapping, config: Config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        
        # Initialize detailed metrics dictionary - 精确复现DOSA源码结构
        detailed_metrics = {
            # 1. 能耗分解 (单位: pJ) - 三段式能量模型
            'energy_breakdown_pj': {
                'compute': 0.0,  # 计算能耗
                'intra_level': {  # 层内能耗 (L0, L1, L2)
                    'L0_Registers': 0.0,
                    'L1_Accumulator': 0.0,
                    'L2_Scratchpad': 0.0
                },
                'inter_level': {  # 层间能耗 (仅DRAM)
                    'L3_DRAM': 0.0
                }
            },
            # 2. 访问次数分解
            'access_counts': {
                'intra_level': {},  # calculate_intra_level_accesses 结果
                'inter_level': {}   # calculate_per_level_accesses 结果
            }
        }
        


        # 1. Identify Producer and Consumers
        producer_name = group[0]
        consumers = group[1:]
        producer_layer = graph.layers[producer_name]

        # 2. Latency Calculation: Dominated by the producer
        # We can reuse the logic from the main performance model for a single layer
        producer_macs = reduce(lambda x, y: x * y, producer_layer['dims'].values(), 1)
        num_pes = hw_params.get_projected_num_pes()
        compute_latency = producer_macs / (num_pes * config.CLOCK_FREQUENCY_MHZ * 1e6 + 1e-9)

        # 创建性能模型实例
        from dosa.performance_model import HighFidelityPerformanceModel
        perf_model = HighFidelityPerformanceModel(config)
        perf_model.hw_params = hw_params  # Temporarily attach hw_params
        
        all_factors = mapping.get_all_factors()
        producer_accesses = perf_model.calculate_per_level_accesses(producer_layer['dims'], all_factors)
        
        memory_latencies = []
        for interface, accesses in producer_accesses.items():
            upper_level_name = interface.split('_to_')[0]
            level_info = next((level for level in config.MEMORY_HIERARCHY if level['name'] == upper_level_name), None)
            if upper_level_name in ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad']:
                bandwidth_gb_s = (2 * torch.sqrt(num_pes) * config.BYTES_PER_ELEMENT * config.CLOCK_FREQUENCY_MHZ * 1e6) / 1e9
            else:
                bandwidth_gb_s = level_info['bandwidth_gb_s']
            memory_latencies.append(accesses / (bandwidth_gb_s * 1e9 + 1e-9))

        if memory_latencies:
            latency = torch.maximum(compute_latency, torch.max(torch.stack(memory_latencies)))
        else:
            latency = compute_latency

        # 3. Energy Calculation
        total_energy = torch.tensor(0.0, device=config.DEVICE)

        # 3.1. Compute Energy: 融合组内所有操作的计算能耗
        group_macs = 0
        for layer_name in group:
            layer_dims = graph.layers[layer_name]['dims']
            # 只计算实际的MAC操作，ReLU等激活函数不产生MAC
            if 'conv' in layer_name.lower():
                # 对于卷积层：N * K * P * Q * C * R * S
                layer_macs = (layer_dims.get('N', 1) * layer_dims.get('K', 1) * 
                             layer_dims.get('P', 1) * layer_dims.get('Q', 1) * 
                             layer_dims.get('C', 1) * layer_dims.get('R', 1) * 
                             layer_dims.get('S', 1))
                group_macs += layer_macs
            # ReLU等激活函数不计入MAC运算
        
        # 计算能耗：MAC运算 - 精确复现DOSA源码
        compute_energy = group_macs * config.PE_MAC_EPA_PJ
        total_energy += compute_energy
        
        # 存储计算能耗到详细指标中
        if isinstance(compute_energy, torch.Tensor):
            detailed_metrics['energy_breakdown_pj']['compute'] = float(compute_energy.item())
        else:
            detailed_metrics['energy_breakdown_pj']['compute'] = float(compute_energy)

        # 3.2. 层内能耗 (Intra-level Energy) - 精确复现dosa源码实现
        # 调用 calculate_intra_level_accesses 来获取包含L0高估访问在内的次数
        intra_level_accesses = perf_model.calculate_intra_level_accesses(
            producer_layer['dims'], all_factors, num_pes)
        
        # 将访问次数详情存入metrics
        detailed_metrics['access_counts']['intra_level'] = convert_tensors_to_native(intra_level_accesses)
        
        # 累加L0, L1, L2的层内能耗
        for level_name, tensors in intra_level_accesses.items():
            level_energy = torch.tensor(0.0, device=config.DEVICE)
            for tensor_type, operations in tensors.items():
                for op_type, access_count in operations.items():
                    access_bytes = access_count * config.BYTES_PER_ELEMENT
                    
                    energy_contribution = torch.tensor(0.0, device=config.DEVICE)
                    if level_name == 'L0_Registers':
                        energy_contribution = access_bytes * config.L0_REG_BASE_EPA_PJ
                    elif level_name == 'L1_Accumulator':
                        size_kb = hw_params.get_buffer_size_kb(level_name)
                        epa = config.L1_ACCUM_BASE_EPA_PJ + config.L1_ACCUM_CAPACITY_COEFF_PJ_PER_KB * (size_kb / torch.sqrt(num_pes))
                        energy_contribution = access_bytes * epa
                    elif level_name == 'L2_Scratchpad':
                        size_kb = hw_params.get_buffer_size_kb(level_name)
                        epa = config.L2_SPM_BASE_EPA_PJ + config.L2_SPM_CAPACITY_COEFF_PJ_PER_KB * size_kb
                        energy_contribution = access_bytes * epa
                    
                    level_energy += energy_contribution
                    total_energy += energy_contribution
            
            detailed_metrics['energy_breakdown_pj']['intra_level'][level_name] = float(level_energy.item())
        
        # 3.3. 层间能耗 (Inter-level Energy) - 只计算DRAM部分
        inter_level_accesses = producer_accesses
        detailed_metrics['access_counts']['inter_level'] = {k: v.item() for k, v in inter_level_accesses.items()}
        
        dram_energy = torch.tensor(0.0, device=config.DEVICE)
        for interface, accesses_bytes in inter_level_accesses.items():
            if 'DRAM' in interface:
                energy_contribution = (accesses_bytes / config.BYTES_PER_ELEMENT) * config.L3_DRAM_EPA_PJ
                dram_energy += energy_contribution
                total_energy += energy_contribution
        
        detailed_metrics['energy_breakdown_pj']['inter_level']['L3_DRAM'] = float(dram_energy.item())

        # 4. Buffer Mismatch Loss Calculation (approximated for the producer)
        group_buffer_mismatch_loss = torch.tensor(0.0, device=config.DEVICE)
        for i, level in enumerate(config.MEMORY_HIERARCHY):
            if level['type'] == 'buffer':
                # In a fused group, the buffer must hold the producer's data.
                # We approximate the requirement based on the producer layer.
                required_kb = perf_model.calculate_buffer_req_kb(producer_layer['dims'], all_factors, i)
                available_kb = hw_params.get_buffer_size_kb(level['name'])
                buffer_deficit = torch.relu(required_kb - available_kb)
                level_mismatch_loss = torch.pow(buffer_deficit, 2)
                group_buffer_mismatch_loss += level_mismatch_loss

        # 5. Compatibility Penalty - 对于InPlaceFusionDMT，直接返回0
        compatibility_penalty = torch.tensor(0.0, device=config.DEVICE)
        
        return latency, total_energy, group_buffer_mismatch_loss, compatibility_penalty, detailed_metrics


class SkipConnectionDMT(BaseDMT):
    """DMT for ResNet skip connection patterns like Conv -> BN -> ReLU -> Add."""

    def forward(self, group: list, graph: ComputationGraph, hw_params: HardwareParameters, 
                mapping: FineGrainedMapping, config: Config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        
        # Initialize detailed metrics
        detailed_metrics = {
            'energy_breakdown_pj': {
                'compute': 0.0,
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
            }
        }
        
        # 1. 路径识别：解析Add节点的两个输入源
        add_layer_name = group[-1]  # Add操作通常是最后一个
        main_path = group[:-1]  # 主计算路径（Conv->BN->ReLU等）
        
        # 获取Add层的输入，识别主路径和旁路
        add_layer_inputs = graph.get_layer_inputs(add_layer_name)
        
        # 假设主路径是group中的最后一个非Add层的输出
        main_path_output = main_path[-1] if main_path else None
        skip_path_source = None
        
        # 识别跳跃连接的源头
        for input_layer in add_layer_inputs:
            if input_layer not in group:
                skip_path_source = input_layer
                break
        
        # 2. 延迟建模
        num_pes = hw_params.get_projected_num_pes()
        
        # 2.1 主路径延迟（复用InPlaceFusionDMT逻辑）
        if main_path:
            producer_layer = graph.layers[main_path[0]]
            producer_macs = reduce(lambda x, y: x * y, producer_layer['dims'].values(), 1)
            latency_main = producer_macs / (num_pes * config.CLOCK_FREQUENCY_MHZ * 1e6 + 1e-9)
        else:
            latency_main = torch.tensor(0.0, device=config.DEVICE)
        
        # 2.2 旁路数据获取延迟
        if skip_path_source:
            skip_layer = graph.layers[skip_path_source]
            # 假设旁路张量需要从L2_Scratchpad读取
            skip_tensor_size = reduce(lambda x, y: x * y, skip_layer['dims'].values(), 1) * config.BYTES_PER_ELEMENT
            l2_bandwidth_gb_s = next((level['bandwidth_gb_s'] for level in config.MEMORY_HIERARCHY 
                                    if level['name'] == 'L2_Scratchpad'), 10.0)  # 默认带宽
            latency_skip = skip_tensor_size / (l2_bandwidth_gb_s * 1e9 + 1e-9)
        else:
            latency_skip = torch.tensor(0.0, device=config.DEVICE)
        
        # 2.3 Add操作延迟（简化估计）
        add_layer = graph.layers[add_layer_name]
        add_ops = reduce(lambda x, y: x * y, add_layer['dims'].values(), 1)
        latency_add_op = add_ops / (num_pes * config.CLOCK_FREQUENCY_MHZ * 1e6 + 1e-9)
        
        # 总延迟
        latency_total = torch.maximum(latency_main, latency_skip) + latency_add_op
        
        # 3. 能耗建模
        total_energy = torch.tensor(0.0, device=config.DEVICE)
        
        # 3.1 主路径能耗（复用InPlaceFusionDMT逻辑）
        group_macs = 0
        for layer_name in main_path:
            layer_dims = graph.layers[layer_name]['dims']
            if 'conv' in layer_name.lower():
                layer_macs = (layer_dims.get('N', 1) * layer_dims.get('K', 1) * 
                             layer_dims.get('P', 1) * layer_dims.get('Q', 1) * 
                             layer_dims.get('C', 1) * layer_dims.get('R', 1) * 
                             layer_dims.get('S', 1))
                group_macs += layer_macs
        
        energy_main = group_macs * config.PE_MAC_EPA_PJ
        total_energy += energy_main
        
        # 3.2 旁路数据读取能耗
        if skip_path_source:
            skip_layer = graph.layers[skip_path_source]
            skip_tensor_size = reduce(lambda x, y: x * y, skip_layer['dims'].values(), 1)
            # 使用L2_Scratchpad的EPA
            l2_size_kb = hw_params.get_buffer_size_kb('L2_Scratchpad')
            l2_epa = config.L2_SPM_BASE_EPA_PJ + config.L2_SPM_CAPACITY_COEFF_PJ_PER_KB * l2_size_kb
            energy_skip = skip_tensor_size * config.BYTES_PER_ELEMENT * l2_epa
            total_energy += energy_skip
        
        # 3.3 Add操作能耗
        add_layer = graph.layers[add_layer_name]
        add_ops = reduce(lambda x, y: x * y, add_layer['dims'].values(), 1)
        energy_add_op = add_ops * config.PE_MAC_EPA_PJ  # 简化为MAC能耗
        total_energy += energy_add_op
        
        detailed_metrics['energy_breakdown_pj']['compute'] = float(total_energy.item())
        
        # 4. Buffer需求建模
        buffer_mismatch_loss = torch.tensor(0.0, device=config.DEVICE)
        
        if main_path and skip_path_source:
            # 创建性能模型实例
            from dosa.performance_model import HighFidelityPerformanceModel
            perf_model = HighFidelityPerformanceModel(config)
            perf_model.hw_params = hw_params
            
            all_factors = mapping.get_all_factors()
            
            # 计算主路径输出Tile和旁路输入Tile的总Buffer需求
            for i, level in enumerate(config.MEMORY_HIERARCHY):
                if level['type'] == 'buffer':
                    # 主路径输出Tile需求
                    main_output_layer = graph.layers[main_path[-1]]
                    main_required_kb = perf_model.calculate_buffer_req_kb(main_output_layer['dims'], all_factors, i)
                    
                    # 旁路输入Tile需求
                    skip_layer = graph.layers[skip_path_source]
                    skip_required_kb = perf_model.calculate_buffer_req_kb(skip_layer['dims'], all_factors, i)
                    
                    # 总需求是两者之和（因为需要同时存在于片上缓存中）
                    total_required_kb = main_required_kb + skip_required_kb
                    available_kb = hw_params.get_buffer_size_kb(level['name'])
                    buffer_deficit = torch.relu(total_required_kb - available_kb)
                    level_mismatch_loss = torch.pow(buffer_deficit, 2)
                    buffer_mismatch_loss += level_mismatch_loss
        
        # 5. 兼容性惩罚（对于SkipConnectionDMT暂时返回0）
        compatibility_penalty = torch.tensor(0.0, device=config.DEVICE)
        
        return latency_total, total_energy, buffer_mismatch_loss, compatibility_penalty, detailed_metrics


class SequentialConvDMT(BaseDMT):
    """示例DMT类，展示基于Orojenesis/FFMT思想的兼容性惩罚计算逻辑。"""

    def forward(self, group: list, graph: ComputationGraph, hw_params: HardwareParameters, 
                mapping: FineGrainedMapping, config: Config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        
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