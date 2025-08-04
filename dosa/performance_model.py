import torch
import torch.nn as nn
from typing import Tuple
from functools import reduce
from operator import mul

from dosa.config import Config
from dosa.hardware_parameters import HardwareParameters
from dosa.mapping import FineGrainedMapping
from dosa.dmt import InPlaceFusionDMT, SkipConnectionDMT

# Define TENSOR_DIM_MAP
TENSOR_DIM_MAP = {
    'Input':  ['N', 'C', 'P', 'Q'],
    'Weight': ['K', 'C', 'R', 'S'],
    'Output': ['N', 'K', 'P', 'Q']
}

class HighFidelityPerformanceModel(nn.Module):
    """
    NEW: 高精度性能模型，能够处理多级存储和细粒度映射。
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.dmt_registry = {
            ('Conv', 'ReLU'): InPlaceFusionDMT(),
            ('Conv', 'BatchNormalization', 'ReLU'): InPlaceFusionDMT(),
            ('MatMul', 'Add'): InPlaceFusionDMT(),
            # ResNet skip connection patterns
            ('Conv', 'BatchNormalization', 'ReLU', 'Add'): SkipConnectionDMT(),
            ('Conv', 'Add'): SkipConnectionDMT(),
            ('ReLU', 'Add'): SkipConnectionDMT(),
            # Add more patterns as needed
        }

    def calculate_intra_level_accesses(self, layer_dims: dict, mapping_table: dict, num_pes: torch.Tensor) -> dict:
        """
        重构后的片内访问计算函数 - 基于数据流驱动的访存模型
        核心逻辑：
        - L0_Registers: 访问次数与总计算次数直接相关
        - L1/L2: 读取次数基于从上级加载的数据量，更新次数考虑复用因子
        """
        from dosa.utils import calculate_macs
        
        # 1. 计算总MAC运算次数
        total_macs = calculate_macs(layer_dims)
        
        # 2. 初始化返回字典
        intra_accesses = {
            "L0_Registers": {
                "Input": {"reads": torch.tensor(0.0, device=self.config.DEVICE), "writes": torch.tensor(0.0, device=self.config.DEVICE)},
                "Weight": {"reads": torch.tensor(0.0, device=self.config.DEVICE), "writes": torch.tensor(0.0, device=self.config.DEVICE)},
                "Output": {"reads": torch.tensor(0.0, device=self.config.DEVICE), "writes": torch.tensor(0.0, device=self.config.DEVICE), "updates": torch.tensor(0.0, device=self.config.DEVICE)}
            },
            "L1_Accumulator": {
                "Input": {"reads": torch.tensor(0.0, device=self.config.DEVICE), "writes": torch.tensor(0.0, device=self.config.DEVICE)},
                "Weight": {"reads": torch.tensor(0.0, device=self.config.DEVICE), "writes": torch.tensor(0.0, device=self.config.DEVICE)},
                "Output": {"reads": torch.tensor(0.0, device=self.config.DEVICE), "writes": torch.tensor(0.0, device=self.config.DEVICE), "updates": torch.tensor(0.0, device=self.config.DEVICE)}
            },
            "L2_Scratchpad": {
                "Input": {"reads": torch.tensor(0.0, device=self.config.DEVICE), "writes": torch.tensor(0.0, device=self.config.DEVICE)},
                "Weight": {"reads": torch.tensor(0.0, device=self.config.DEVICE), "writes": torch.tensor(0.0, device=self.config.DEVICE)},
                "Output": {"reads": torch.tensor(0.0, device=self.config.DEVICE), "writes": torch.tensor(0.0, device=self.config.DEVICE), "updates": torch.tensor(0.0, device=self.config.DEVICE)}
            }
        }
        
        # 3. L0_Registers (最内层) - 每次MAC都需要读取Input和Weight，更新Output
        # 这是硬件的物理现实：每个MAC运算都必须从寄存器读取操作数
        intra_accesses["L0_Registers"]["Input"]["reads"] = total_macs
        intra_accesses["L0_Registers"]["Weight"]["reads"] = total_macs  
        intra_accesses["L0_Registers"]["Output"]["updates"] = total_macs
        
        # 4. 获取层间数据移动量，用于计算L1/L2的读写次数
        per_level_accesses = self.calculate_per_level_accesses(layer_dims, mapping_table)
        
        # 5. L1_Accumulator 访问计算
        # 读取次数 = 从L2加载到L1的数据量 / 数据位宽
        if "L2_Scratchpad_to_L1_Accumulator" in per_level_accesses:
            l1_data_bytes = per_level_accesses["L2_Scratchpad_to_L1_Accumulator"]
            l1_data_elements = l1_data_bytes / self.config.BYTES_PER_ELEMENT
            
            # 简化假设：Input、Weight、Output各占1/3的数据移动
            intra_accesses["L1_Accumulator"]["Input"]["reads"] = l1_data_elements / 3.0
            intra_accesses["L1_Accumulator"]["Weight"]["reads"] = l1_data_elements / 3.0
            intra_accesses["L1_Accumulator"]["Input"]["writes"] = l1_data_elements / 3.0
            intra_accesses["L1_Accumulator"]["Weight"]["writes"] = l1_data_elements / 3.0
        
        # Output更新次数考虑空间复用：总MAC数除以PE阵列中的空间复用因子
        pe_spatial_reuse = torch.sqrt(num_pes)  # 假设PE阵列是方形的
        intra_accesses["L1_Accumulator"]["Output"]["updates"] = total_macs / torch.clamp(pe_spatial_reuse, min=torch.tensor(1.0, device=pe_spatial_reuse.device))
        
        # 6. L2_Scratchpad 访问计算
        # 读取次数 = 从DRAM加载到L2的数据量 / 数据位宽
        if "L3_DRAM_to_L2_Scratchpad" in per_level_accesses:
            l2_data_bytes = per_level_accesses["L3_DRAM_to_L2_Scratchpad"]
            l2_data_elements = l2_data_bytes / self.config.BYTES_PER_ELEMENT
            
            # 简化假设：Input、Weight、Output各占1/3的数据移动
            intra_accesses["L2_Scratchpad"]["Input"]["reads"] = l2_data_elements / 3.0
            intra_accesses["L2_Scratchpad"]["Weight"]["reads"] = l2_data_elements / 3.0
            intra_accesses["L2_Scratchpad"]["Input"]["writes"] = l2_data_elements / 3.0
            intra_accesses["L2_Scratchpad"]["Weight"]["writes"] = l2_data_elements / 3.0
        
        # Output更新次数考虑时间和空间复用
        # 计算L2及其内部层级的总复用因子
        total_reuse_factor = torch.tensor(1.0, device=self.config.DEVICE)
        
        # 获取L0和L1层级的时间映射因子
        for level_name in ["L0_Registers", "L1_Accumulator"]:
            for dim_name in ['C', 'R', 'S']:  # Output的复用维度
                if dim_name in mapping_table and level_name in mapping_table[dim_name]:
                    temporal_factor = mapping_table[dim_name][level_name].get('temporal', 1)
                    total_reuse_factor *= torch.tensor(temporal_factor, device=self.config.DEVICE)
        
        # 加上空间复用
        total_reuse_factor *= pe_spatial_reuse
        
        intra_accesses["L2_Scratchpad"]["Output"]["updates"] = total_macs / torch.clamp(total_reuse_factor, min=torch.tensor(1.0, device=self.config.DEVICE))
        
        return intra_accesses

    def calculate_per_level_accesses(self, layer_dims: dict, mapping_table: dict) -> dict:
        """
        重构后的层间数据移动计算函数 - 基于数据流驱动的访存模型
        核心公式: 层级间传输字节数 = 下层数据足迹 * 外层循环总次数 * 数据位宽
        """
        accesses = {}
        memory_levels = [level for level in self.config.MEMORY_HIERARCHY if level['type'] in ['buffer', 'dram']]
        level_names = [level['name'] for level in memory_levels]
        
        # 定义每种张量的复用维度
        reuse_dims = {
            'Input': ['K'],      # Input在K维度上被复用
            'Weight': ['N', 'P', 'Q'],  # Weight在N,P,Q维度上被复用
            'Output': ['C', 'R', 'S']   # Output在C,R,S维度上被复用
        }

        # 遍历所有相邻的存储层级接口
        for i in range(len(memory_levels) - 1, 0, -1):
            upper_level_idx = i
            lower_level_idx = i - 1
            upper_level_name = level_names[upper_level_idx]
            lower_level_name = level_names[lower_level_idx]
            interface_name = f"{upper_level_name}_to_{lower_level_name}"
            total_access_bytes_for_interface = torch.tensor(0.0, device=self.config.DEVICE)

            # 对每种张量类型计算数据移动量
            for tensor_type, relevant_dims in TENSOR_DIM_MAP.items():
                # 步骤1: 计算下层数据足迹 (Data_Footprint_at_Lower_Level)
                # 这代表为了完成一次外层循环，需要加载到LowerLevel的数据块大小
                data_footprint_elements = torch.tensor(1.0, device=self.config.DEVICE)
                
                for dim_name in relevant_dims:
                    if dim_name in layer_dims:
                        # 累乘该维度在LowerLevel及其内部所有层级的temporal和spatial映射因子
                        dim_footprint = torch.tensor(1.0, device=self.config.DEVICE)
                        for level_idx in range(lower_level_idx + 1):
                            level_name = level_names[level_idx]
                            if dim_name in mapping_table and level_name in mapping_table[dim_name]:
                                temporal_factor = mapping_table[dim_name][level_name].get('temporal', 1)
                                spatial_factor = mapping_table[dim_name][level_name].get('spatial', 1)
                                dim_footprint *= torch.tensor(temporal_factor * spatial_factor, device=self.config.DEVICE)
                        data_footprint_elements *= dim_footprint

                # 步骤2: 计算外层循环总次数 (Outer_Loop_Iterations)
                # 这代表需要多少次从UpperLevel加载数据块到LowerLevel
                outer_loop_iterations = torch.tensor(1.0, device=self.config.DEVICE)
                
                for dim_name in relevant_dims:
                    if dim_name in layer_dims:
                        # 累乘该维度在UpperLevel及其外部所有层级的temporal映射因子
                        for level_idx in range(upper_level_idx, len(memory_levels)):
                            level_name = level_names[level_idx]
                            if dim_name in mapping_table and level_name in mapping_table[dim_name]:
                                temporal_factor = mapping_table[dim_name][level_name].get('temporal', 1)
                                outer_loop_iterations *= torch.tensor(temporal_factor, device=self.config.DEVICE)

                # 步骤3: 引入数据复用修正 (Reuse_Factor)
                # 考虑该张量在LowerLevel的数据可以被其他张量的循环复用
                reuse_factor = torch.tensor(1.0, device=self.config.DEVICE)
                
                for dim_name in reuse_dims[tensor_type]:
                    if dim_name in layer_dims:
                        # 累乘复用维度在LowerLevel及其内部所有层级的temporal映射因子
                        for level_idx in range(lower_level_idx + 1):
                            level_name = level_names[level_idx]
                            if dim_name in mapping_table and level_name in mapping_table[dim_name]:
                                temporal_factor = mapping_table[dim_name][level_name].get('temporal', 1)
                                reuse_factor *= torch.tensor(temporal_factor, device=self.config.DEVICE)
                
                # 应用复用修正
                effective_outer_iterations = outer_loop_iterations / torch.clamp(reuse_factor, min=torch.tensor(1e-9, device=self.config.DEVICE))

                # 步骤4: 计算该张量在此接口的传输字节数
                tensor_transfer_bytes = (data_footprint_elements * 
                                       effective_outer_iterations * 
                                       self.config.BYTES_PER_ELEMENT)
                total_access_bytes_for_interface += tensor_transfer_bytes

            accesses[interface_name] = total_access_bytes_for_interface

        return accesses

    def forward(self, graph, hw_params: HardwareParameters, mapping: FineGrainedMapping) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total_latency = torch.tensor(0.0, device=self.config.DEVICE)
        total_energy = torch.tensor(0.0, device=self.config.DEVICE)
        total_buffer_mismatch_loss = torch.tensor(0.0, device=self.config.DEVICE)
        total_compatibility_penalty = torch.tensor(0.0, device=self.config.DEVICE)
        
        all_factors = mapping.get_all_factors()

        for group in graph.fusion_groups:
            current_pattern = tuple(graph.layers[layer_name]['type'] for layer_name in group)
            dmt_model = self.dmt_registry.get(current_pattern)

            if dmt_model:
                latency, energy, group_buffer_mismatch_loss, compatibility_penalty, detailed_metrics = dmt_model(group, graph, hw_params, mapping, self.config)
                total_buffer_mismatch_loss += group_buffer_mismatch_loss
                total_compatibility_penalty += compatibility_penalty
                # For now, we assume DMT handles its own buffer requirements implicitly
                # and doesn't contribute to the mismatch loss in this simplified model.
                # A more advanced implementation might have DMTs also return a loss.
            else:
                # Fallback to original logic for single layers or unsupported patterns
                layer_name = group[0]
                layer = graph.layers[layer_name]
                from dosa.utils import calculate_macs # 确保导入
                macs = calculate_macs(layer['dims'])
                
                num_pes = hw_params.get_projected_num_pes()
                compute_latency = macs / (num_pes * self.config.CLOCK_FREQUENCY_MHZ * 1e6 + torch.tensor(1e-9, device=self.config.DEVICE))
                
                per_level_accesses = self.calculate_per_level_accesses(layer['dims'], all_factors)
                memory_latencies = []
                num_pes_sqrt = torch.sqrt(num_pes)

                for interface, accesses in per_level_accesses.items():
                    upper_level_name = interface.split('_to_')[0]
                    level_info = next((level for level in self.config.MEMORY_HIERARCHY if level['name'] == upper_level_name), None)
                    
                    if upper_level_name in ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad']:
                        bandwidth_gb_s = (2 * num_pes_sqrt * self.config.BYTES_PER_ELEMENT * self.config.CLOCK_FREQUENCY_MHZ * 1e6) / 1e9
                    else: # L3_DRAM
                        bandwidth_gb_s = level_info['bandwidth_gb_s']

                    memory_latencies.append(accesses / (bandwidth_gb_s * 1e9 + torch.tensor(1e-9, device=self.config.DEVICE)))
                
                if memory_latencies:
                    latency = torch.maximum(compute_latency, torch.max(torch.stack(memory_latencies)))
                else:
                    latency = compute_latency

                # 计算总能耗：compute_energy + inter_level_energy + intra_level_energy
                energy = torch.tensor(0.0, device=self.config.DEVICE)
                
                # 1. Compute Energy (MAC运算能耗)
                energy += macs * self.config.PE_MAC_EPA_PJ
                
                # 2. Inter-level Energy (层间数据移动能耗)
                for interface, accesses_bytes in per_level_accesses.items():
                    lower_level_name = interface.split('_to_')[1]
                    accesses_4bytes = accesses_bytes / 4.0

                    if lower_level_name == 'L0_Registers':
                        energy += accesses_4bytes * self.config.L0_REG_BASE_EPA_PJ
                    elif lower_level_name == 'L1_Accumulator':
                        size_kb = hw_params.get_buffer_size_kb(lower_level_name)
                        epa = self.config.L1_ACCUM_BASE_EPA_PJ + self.config.L1_ACCUM_CAPACITY_COEFF_PJ_PER_KB * (size_kb / num_pes_sqrt)
                        energy += accesses_4bytes * epa
                    elif lower_level_name == 'L2_Scratchpad':
                        size_kb = hw_params.get_buffer_size_kb(lower_level_name)
                        epa = self.config.L2_SPM_BASE_EPA_PJ + self.config.L2_SPM_CAPACITY_COEFF_PJ_PER_KB * size_kb
                        energy += accesses_4bytes * epa
                    elif lower_level_name == 'L3_DRAM':
                        energy += accesses_4bytes * self.config.L3_DRAM_EPA_PJ
                
                # 3. Intra-level Energy (片上高频访问能耗)
                intra_level_accesses = self.calculate_intra_level_accesses(layer['dims'], all_factors, num_pes)
                
                for level_name, tensors in intra_level_accesses.items():
                    for tensor_type, operations in tensors.items():
                        for op_type, access_count in operations.items():
                            # 将访问次数转换为字节数
                            access_bytes = access_count * self.config.BYTES_PER_ELEMENT
                            
                            # 根据存储层级计算能耗
                            if level_name == 'L0_Registers':
                                energy += access_bytes * self.config.L0_REG_BASE_EPA_PJ
                            elif level_name == 'L1_Accumulator':
                                size_kb = hw_params.get_buffer_size_kb(level_name)
                                epa = self.config.L1_ACCUM_BASE_EPA_PJ + self.config.L1_ACCUM_CAPACITY_COEFF_PJ_PER_KB * (size_kb / num_pes_sqrt)
                                energy += access_bytes * epa
                            elif level_name == 'L2_Scratchpad':
                                size_kb = hw_params.get_buffer_size_kb(level_name)
                                epa = self.config.L2_SPM_BASE_EPA_PJ + self.config.L2_SPM_CAPACITY_COEFF_PJ_PER_KB * size_kb
                                energy += access_bytes * epa

                # Calculate buffer mismatch loss for this layer
                for i, level in enumerate(self.config.MEMORY_HIERARCHY):
                    if level['type'] == 'buffer':
                        required_kb = self.calculate_buffer_req_kb(layer['dims'], all_factors, i)
                        available_kb = hw_params.get_buffer_size_kb(level['name'])
                        buffer_deficit = torch.relu(required_kb - available_kb)
                        level_mismatch_loss = torch.pow(buffer_deficit, 2)
                        total_buffer_mismatch_loss += level_mismatch_loss

            total_latency += latency
            total_energy += energy

        # 确保所有返回值都是标量张量
        total_latency = total_latency.squeeze() if total_latency.dim() > 0 else total_latency
        total_energy = total_energy.squeeze() if total_energy.dim() > 0 else total_energy
        area_cost = hw_params.get_area_cost()
        area_cost = area_cost.squeeze() if area_cost.dim() > 0 else area_cost
        total_buffer_mismatch_loss = total_buffer_mismatch_loss.squeeze() if total_buffer_mismatch_loss.dim() > 0 else total_buffer_mismatch_loss
        total_compatibility_penalty = total_compatibility_penalty.squeeze() if total_compatibility_penalty.dim() > 0 else total_compatibility_penalty
        
        return total_latency, total_energy, area_cost, total_buffer_mismatch_loss, total_compatibility_penalty

    def calculate_buffer_req_kb(self, dims, factors, level_idx):
        total_buffer_bytes = torch.tensor(0.0, device=self.config.DEVICE)
        level_name = self.config.MEMORY_HIERARCHY[level_idx]['name']

        for tensor_type, tensor_dims in TENSOR_DIM_MAP.items():
            tile_dims = {}
            for dim_name in tensor_dims:
                if dim_name in dims:
                    tile_dims[dim_name] = torch.tensor(1.0, device=self.config.DEVICE)
                    for i in range(level_idx + 1):
                        inner_level_name = self.config.MEMORY_HIERARCHY[i]['name']
                        if inner_level_name in factors[dim_name]:
                            tile_dims[dim_name] = tile_dims[dim_name] * \
                                factors[dim_name][inner_level_name]['temporal'].squeeze() * \
                                factors[dim_name][inner_level_name]['spatial'].squeeze()
            
            tensor_tile_size = reduce(mul, [tile_dims.get(d, torch.tensor(1.0, device=self.config.DEVICE)) for d in tensor_dims if d in dims], torch.tensor(1.0, device=self.config.DEVICE))
            total_buffer_bytes += tensor_tile_size * self.config.BYTES_PER_ELEMENT

        return total_buffer_bytes / 1024.0