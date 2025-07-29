import torch
import torch.nn as nn
from typing import Tuple
from functools import reduce
from operator import mul

from dosa.config import Config
from dosa.hardware_parameters import HardwareParameters
from dosa.mapping import FineGrainedMapping
from dosa.dmt import InPlaceFusionDMT

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
            # Add more patterns as needed
        }

    def calculate_per_level_accesses(self, layer_dims: dict, mapping_table: dict) -> dict:
        """
        Physically accurate model for data movement between memory levels.
        Access_Bytes = Data_Footprint_at_Lower_Level * Outer_Loop_Iterations * Bytes_Per_Element
        """
        accesses = {}
        memory_levels = [level for level in self.config.MEMORY_HIERARCHY if level['type'] in ['buffer', 'dram']]
        level_names = [level['name'] for level in memory_levels]
        
        # Define reuse dimensions for each tensor type
        reuse_dims = {
            'Input': ['K'],
            'Weight': ['N', 'P', 'Q'],
            'Output': ['C', 'R', 'S']
        }

        # Iterate through interfaces from outermost to innermost
        for i in range(len(memory_levels) - 1, 0, -1):
            upper_level_idx = i
            lower_level_idx = i - 1
            upper_level_name = level_names[upper_level_idx]
            lower_level_name = level_names[lower_level_idx]
            interface_name = f"{upper_level_name}_to_{lower_level_name}"
            total_access_bytes_for_interface = torch.tensor(0.0, device=self.config.DEVICE)

            for tensor_type, relevant_dims in TENSOR_DIM_MAP.items():
                # Step A: Calculate data_footprint_elements_at_lower_level
                data_footprint_elements_at_lower_level = torch.tensor(1.0, device=self.config.DEVICE)
                
                for dim_name in relevant_dims:
                    if dim_name in layer_dims:
                        # Product of all temporal and spatial factors for this dimension
                        # at lower_level and all levels inside it (from lower_level inwards)
                        dim_footprint = torch.tensor(1.0, device=self.config.DEVICE)
                        for level_idx in range(lower_level_idx + 1):
                            level_name = level_names[level_idx]
                            if dim_name in mapping_table and level_name in mapping_table[dim_name]:
                                dim_footprint *= mapping_table[dim_name][level_name]['temporal']
                                dim_footprint *= mapping_table[dim_name][level_name]['spatial']
                        data_footprint_elements_at_lower_level *= dim_footprint

                # Step B: Calculate num_outer_iterations (before reuse adjustment)
                # This represents the total iterations of loops that execute at upper_level and beyond
                num_outer_iterations = torch.tensor(1.0, device=self.config.DEVICE)
                
                for dim_name in relevant_dims:
                    if dim_name in layer_dims:
                        # Only consider temporal factors at upper_level and beyond
                        for level_idx in range(upper_level_idx, len(memory_levels)):
                            level_name = level_names[level_idx]
                            if dim_name in mapping_table and level_name in mapping_table[dim_name]:
                                num_outer_iterations *= mapping_table[dim_name][level_name]['temporal']

                # Step C: Incorporate Data Reuse
                # The reuse factor is the product of temporal loop sizes for the reuse dimensions
                # that are executed at the lower_level and below.
                reuse_factor = torch.tensor(1.0, device=self.config.DEVICE)
                
                for dim_name in reuse_dims[tensor_type]:
                    if dim_name in layer_dims:
                        # Product of all temporal factors for this reuse dimension
                        # from lower_level down to the innermost level
                        for level_idx in range(lower_level_idx, -1, -1):
                            level_name = level_names[level_idx]
                            if dim_name in mapping_table and level_name in mapping_table[dim_name]:
                                reuse_factor *= mapping_table[dim_name][level_name]['temporal']
                
                refined_num_outer_iterations = num_outer_iterations / (reuse_factor + 1e-9)

                # Step D: Calculate tensor_access_bytes
                tensor_access_bytes = (data_footprint_elements_at_lower_level * 
                                     refined_num_outer_iterations * 
                                     self.config.BYTES_PER_ELEMENT)
                total_access_bytes_for_interface += tensor_access_bytes

            accesses[interface_name] = total_access_bytes_for_interface

        return accesses

    def forward(self, graph, hw_params: HardwareParameters, mapping: FineGrainedMapping) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total_latency = torch.tensor(0.0, device=self.config.DEVICE)
        total_energy = torch.tensor(0.0, device=self.config.DEVICE)
        total_buffer_mismatch_loss = torch.tensor(0.0, device=self.config.DEVICE)
        
        all_factors = mapping.get_all_factors()

        for group in graph.fusion_groups:
            current_pattern = tuple(graph.layers[layer_name]['type'] for layer_name in group)
            dmt_model = self.dmt_registry.get(current_pattern)

            if dmt_model:
                latency, energy, group_buffer_mismatch_loss = dmt_model(group, graph, hw_params, mapping, self.config)
                total_buffer_mismatch_loss += group_buffer_mismatch_loss
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
                compute_latency = macs / (num_pes * self.config.CLOCK_FREQUENCY_MHZ * 1e6 + 1e-9)
                
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

                    memory_latencies.append(accesses / (bandwidth_gb_s * 1e9 + 1e-9))
                
                if memory_latencies:
                    latency = torch.maximum(compute_latency, torch.max(torch.stack(memory_latencies)))
                else:
                    latency = compute_latency

                energy = torch.tensor(0.0, device=self.config.DEVICE)
                energy += macs * self.config.PE_MAC_EPA_PJ
                
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
        
        return total_latency, total_energy, area_cost, total_buffer_mismatch_loss

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