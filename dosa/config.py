import torch
import torch.nn as nn

class Config:
    """全局配置类，已更新为支持多级存储层次结构。"""
    _instance = None
    def __init__(self):
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.BYTES_PER_ELEMENT = 4
        self.CLOCK_FREQUENCY_MHZ = 1000
        self.DRAM_BANDWIDTH_GB_S = 100 # Corresponds to 8 words/cycle at 1GHz with 4-byte words

        # --- NEW: 定义显式的多级存储层次 ---
        self.MEMORY_HIERARCHY = [
            # Level 0: Registers
            {'name': 'L0_Registers', 'type': 'buffer', 'size_kb': nn.Parameter(torch.log(torch.tensor(2.0)))},
            # Level 1: Accumulator
            {'name': 'L1_Accumulator', 'type': 'buffer', 'size_kb': nn.Parameter(torch.log(torch.tensor(4.0)))},
            # Level 2: Scratchpad
            {'name': 'L2_Scratchpad', 'type': 'buffer', 'size_kb': nn.Parameter(torch.log(torch.tensor(256.0)))},
            # Level 3: DRAM
            {'name': 'L3_DRAM', 'type': 'dram', 'bandwidth_gb_s': self.DRAM_BANDWIDTH_GB_S}
        ]
        
        # 能量模型（单位：pJ）
        self.PE_MAC_EPA_PJ = 12.68 
        # 单位能耗（pJ/access）
        self.L0_REG_BASE_EPA_PJ = 0.009
        self.L1_ACCUM_BASE_EPA_PJ = 1.94 * 1e6
        self.L1_ACCUM_CAPACITY_COEFF_PJ_PER_KB = 0.1005 * 1e6
        self.L2_SPM_BASE_EPA_PJ = 0.49 * 1e6
        self.L2_SPM_CAPACITY_COEFF_PJ_PER_KB = 0.025 * 1e6
        self.L3_DRAM_EPA_PJ = 100 * 1e6
        
        # 面积模型参数（注：以下参数均为经验估计值，未来需要通过实际硬件测量进行校准）
        self.AREA_PER_PE_MM2 = 0.015
        self.AREA_PER_KB_L1_MM2 = 0.008 # L1通常更贵
        self.AREA_PER_KB_L2_MM2 = 0.005 # L2相对便宜
        self.AREA_BASE_MM2 = 1.0
        self.PENALTY_WEIGHT = 1e3
        self.MISMATCH_PENALTY_WEIGHT = 1.0
        
        # 损失策略配置
        self.LOSS_STRATEGY = 'log_edp_plus_area'  # 可选: 'log_edp_plus_area', 'edp_plus_area'
        self.AREA_WEIGHT = 1e-3  # 面积权重

    @staticmethod
    def get_instance():
        if Config._instance is None:
            Config._instance = Config()
        return Config._instance