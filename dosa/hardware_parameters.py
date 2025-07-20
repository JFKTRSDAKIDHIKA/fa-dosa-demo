import torch
import torch.nn as nn
from dosa.config import Config

class HardwareParameters(nn.Module):
    """硬件参数，现在支持多级Buffer。"""
    def __init__(self, initial_num_pes=128.0, initial_l0_kb=2.0, initial_l1_kb=4.0, initial_l2_kb=256.0):
        super().__init__()
        self.log_num_pes = nn.Parameter(torch.log(torch.tensor(float(initial_num_pes))))
        
        # NEW: 为每个可学习的Buffer创建一个参数
        self.log_buffer_sizes_kb = nn.ParameterDict({
            'L0_Registers': nn.Parameter(torch.log(torch.tensor(float(initial_l0_kb)))),
            'L1_Accumulator': nn.Parameter(torch.log(torch.tensor(float(initial_l1_kb)))),
            'L2_Scratchpad': nn.Parameter(torch.log(torch.tensor(float(initial_l2_kb))))
        })
        
    def get_num_pes(self):
        return torch.exp(self.log_num_pes)

    def get_projected_num_pes(self):
        continuous_pes = self.get_num_pes()
        projected_num_pes = torch.round(torch.sqrt(continuous_pes)) ** 2
        return continuous_pes + (projected_num_pes - continuous_pes).detach()
        
    def get_buffer_size_kb(self, level_name: str):
        return torch.exp(self.log_buffer_sizes_kb[level_name])
        
    def get_area_cost(self) -> torch.Tensor:
        """Calculate total hardware area based on provisioned parameters.
        
        Returns:
            torch.Tensor: Total hardware area in mm²
        """
        config = Config.get_instance()
        pe_area = self.get_projected_num_pes() * config.AREA_PER_PE_MM2
        
        l0_buffer_area = self.get_buffer_size_kb('L0_Registers') * config.AREA_PER_KB_L1_MM2 # Use L1 cost for registers
        l1_buffer_area = self.get_buffer_size_kb('L1_Accumulator') * config.AREA_PER_KB_L1_MM2 # Use L1 cost for accumulators
        l2_buffer_area = self.get_buffer_size_kb('L2_Scratchpad') * config.AREA_PER_KB_L2_MM2
        
        return config.AREA_BASE_MM2 + pe_area + l0_buffer_area + l1_buffer_area + l2_buffer_area