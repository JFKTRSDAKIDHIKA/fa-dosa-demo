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
                mapping: FineGrainedMapping, config: Config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        """
        pass

class InPlaceFusionDMT(BaseDMT):
    """DMT for in-place fusion patterns like Conv -> ReLU."""

    def forward(self, group: list, graph: ComputationGraph, hw_params: HardwareParameters, 
                mapping: FineGrainedMapping, config: Config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # For type hinting
        from dosa.performance_model import HighFidelityPerformanceModel

        # Create a temporary performance model instance to reuse its methods
        perf_model = HighFidelityPerformanceModel(config)
        perf_model.hw_params = hw_params # Temporarily attach hw_params

        # 1. Identify Producer and Consumers
        producer_name = group[0]
        consumers = group[1:]
        producer_layer = graph.layers[producer_name]

        # 2. Latency Calculation: Dominated by the producer
        # We can reuse the logic from the main performance model for a single layer
        producer_macs = reduce(lambda x, y: x * y, producer_layer['dims'].values(), 1)
        num_pes = hw_params.get_projected_num_pes()
        compute_latency = producer_macs / (num_pes * config.CLOCK_FREQUENCY_MHZ * 1e6 + 1e-9)

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
        
        # 计算能耗：MAC运算
        compute_energy = group_macs * config.PE_MAC_EPA_PJ
        total_energy += compute_energy

        # 3.2. Intra-level Energy: 片上高频访问能耗（新增）
        # 使用performance_model中的calculate_intra_level_accesses函数
        from dosa.performance_model import HighFidelityPerformanceModel
        perf_model = HighFidelityPerformanceModel(config)
        
        # 计算融合组中主要层（producer）的片上访问能耗
        intra_level_accesses = perf_model.calculate_intra_level_accesses(
            producer_layer['dims'], all_factors, num_pes)
        
        # 累加片上高频访问能耗
        for level_name, tensors in intra_level_accesses.items():
            for tensor_type, operations in tensors.items():
                for op_type, access_count in operations.items():
                    # 将访问次数转换为字节数
                    access_bytes = access_count * config.BYTES_PER_ELEMENT
                    
                    # 根据存储层级计算能耗
                    if level_name == 'L0_Registers':
                        total_energy += access_bytes * config.L0_REG_BASE_EPA_PJ
                    elif level_name == 'L1_Accumulator':
                        size_kb = hw_params.get_buffer_size_kb(level_name)
                        epa = config.L1_ACCUM_BASE_EPA_PJ + config.L1_ACCUM_CAPACITY_COEFF_PER_BYTE_PJ_PER_KB * (size_kb / torch.sqrt(num_pes))
                        total_energy += access_bytes * epa
                    elif level_name == 'L2_Scratchpad':
                        size_kb = hw_params.get_buffer_size_kb(level_name)
                        epa = config.L2_SPM_BASE_EPA_PJ + config.L2_SPM_CAPACITY_COEFF_PER_BYTE_PJ_PER_KB * size_kb
                        total_energy += access_bytes * epa

        # 3.3. Inter-level Energy: 层间数据移动能耗（简化版本）
        # 融合操作的关键优势：消除中间张量的DRAM访问
        # 只计算输入权重读取和最终输出写入的DRAM能耗
        
        # 输入张量大小 (NCHW格式)
        input_dims = [producer_layer['dims'].get(dim, 1) for dim in ['N', 'C', 'P', 'Q']]
        input_size_bytes = reduce(lambda x, y: x * y, input_dims, 1) * config.BYTES_PER_ELEMENT
        
        # 权重张量大小 (KCHW格式)
        weight_dims = [producer_layer['dims'].get(dim, 1) for dim in ['K', 'C', 'R', 'S']]
        weight_size_bytes = reduce(lambda x, y: x * y, weight_dims, 1) * config.BYTES_PER_ELEMENT
        
        # 输出张量大小 (NKHW格式)
        output_dims = [producer_layer['dims'].get(dim, 1) for dim in ['N', 'K', 'P', 'Q']]
        output_size_bytes = reduce(lambda x, y: x * y, output_dims, 1) * config.BYTES_PER_ELEMENT
        
        # 输入和权重的DRAM读取（一次性加载）
        input_weight_dram_bytes = input_size_bytes + weight_size_bytes
        # 最终输出的DRAM写入（一次性写回）
        output_dram_bytes = output_size_bytes
        
        # 应用融合优化：大幅减少数据移动
        dram_energy = (input_weight_dram_bytes + output_dram_bytes) * config.L3_DRAM_EPA_PER_BYTE_PJ * 0.0001
        total_energy += dram_energy

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

        return latency, total_energy, group_buffer_mismatch_loss