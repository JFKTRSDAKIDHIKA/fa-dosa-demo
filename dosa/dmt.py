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

        # 3.1. Compute Energy: Sum of all layers in the group
        group_macs = 0
        for layer_name in group:
            group_macs += reduce(lambda x, y: x * y, graph.layers[layer_name]['dims'].values(), 1)
        total_energy += group_macs * config.PE_MAC_EPA_PJ

        # 3.2. Data Movement Energy
        # Producer's input and weight energy
        # This is tricky because calculate_per_level_accesses combines all tensors.
        # For simplicity, we calculate total producer energy and subtract the output-to-DRAM part.
        
        # Full producer energy calculation (reusing logic)
        producer_energy = torch.tensor(0.0, device=config.DEVICE)
        for interface, accesses_bytes in producer_accesses.items():
            lower_level_name = interface.split('_to_')[1]
            accesses_4bytes = accesses_bytes / 4.0
            if lower_level_name == 'L0_Registers':
                producer_energy += accesses_4bytes * config.L0_REG_BASE_EPA_PJ
            elif lower_level_name == 'L1_Accumulator':
                size_kb = hw_params.get_buffer_size_kb(lower_level_name)
                epa = config.L1_ACCUM_BASE_EPA_PJ + config.L1_ACCUM_CAPACITY_COEFF_PJ_PER_KB * (size_kb / torch.sqrt(num_pes))
                producer_energy += accesses_4bytes * epa
            elif lower_level_name == 'L2_Scratchpad':
                size_kb = hw_params.get_buffer_size_kb(lower_level_name)
                epa = config.L2_SPM_BASE_EPA_PJ + config.L2_SPM_CAPACITY_COEFF_PJ_PER_KB * size_kb
                producer_energy += accesses_4bytes * epa
            elif lower_level_name == 'L3_DRAM':
                producer_energy += accesses_4bytes * config.L3_DRAM_EPA_PJ

        # Identify the intermediate tensor (producer's output)
        # Approximate intermediate tensor size using producer's output dimensions
        output_dims = [dim for dim_name, dim in producer_layer['dims'].items() if dim_name in ['N', 'K', 'P', 'Q']]
        intermediate_tensor_size_bytes = reduce(lambda x, y: x * y, output_dims, 1) * config.BYTES_PER_ELEMENT

        # Subtract producer's output write to DRAM and add its write to Scratchpad
        # This is an approximation. A more accurate model would recalculate accesses.
        dram_to_spm_interface = f"{config.MEMORY_HIERARCHY[-1]['name']}_to_{config.MEMORY_HIERARCHY[-2]['name']}"
        if dram_to_spm_interface in producer_accesses:
            # This is complex. Let's simplify: assume output is written to DRAM and read back.
            # We remove that energy and add scratchpad access energy.
            dram_epa = config.L3_DRAM_EPA_PJ
            spm_epa = config.L2_SPM_BASE_EPA_PJ + config.L2_SPM_CAPACITY_COEFF_PJ_PER_KB * hw_params.get_buffer_size_kb('L2_Scratchpad')

            # Energy to write output to DRAM and read it back
            dram_access_energy = (intermediate_tensor_size_bytes / 4.0) * dram_epa * 2 # write + read
            
            # Energy to write output to SPM and read it back
            spm_access_energy = (intermediate_tensor_size_bytes / 4.0) * spm_epa * 2 # write + read

            # Adjust total energy
            total_energy += producer_energy - dram_access_energy + spm_access_energy
        else:
            # If no DRAM access, just add the producer energy
            total_energy += producer_energy

        # Add consumers' energy (compute is already added, this is for potential other inputs)
        # For simple InPlaceFusion (e.g., ReLU), consumers have no weights and their input is the intermediate tensor.
        # So, no extra energy needed for consumers' data movement.

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