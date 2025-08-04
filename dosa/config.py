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
        self.PE_MAC_EPA_PJ = 0.845  # 0.845 pJ per MAC operation
        # 单位能耗（pJ/access）
        self.L0_REG_BASE_EPA_PJ = 9.4394
        
        # EPA = 1.94 + 0.1005 * (C1 / sqrt(C_PE))
        self.L1_ACCUM_BASE_EPA_PJ = 1.94
        self.L1_ACCUM_CAPACITY_COEFF_PJ_PER_KB = 0.1005
        
        self.L2_SPM_BASE_EPA_PJ = 31.0663
        self.L2_SPM_CAPACITY_COEFF_PJ_PER_KB = 0.3  # 300 * 1e-3 pJ -> 0.3 pJ
        self.L3_DRAM_EPA_PJ = 2048.0 / 16.0 # 128.0 pJ per word access
        
        # 损失策略配置
        self.LOSS_STRATEGY = 'log_edp_plus_area'  # 可选: 'log_edp_plus_area', 'edp_plus_area'
        self.AREA_WEIGHT = 1e-3  # 面积权重
        self.COMPATIBILITY_PENALTY_WEIGHT = 100.0  # 兼容性惩罚权重
        
        # 面积模型配置（单位：mm²）
        self.AREA_BASE_MM2 = 1.0  # 基础面积
        self.AREA_PER_PE_MM2 = 0.01  # 每个PE的面积
        self.AREA_PER_KB_L1_MM2 = 0.1  # L1缓存每KB的面积
        self.AREA_PER_KB_L2_MM2 = 0.05  # L2缓存每KB的面积

    @staticmethod
    def get_instance():
        if Config._instance is None:
            Config._instance = Config()
        return Config._instance