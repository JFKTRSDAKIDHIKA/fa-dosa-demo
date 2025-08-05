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

# 定义维度集合 D_t (与张量 t 相关的维度)
D_W = {'K', 'C', 'R', 'S'}  # Weight tensor dimensions
D_I = {'N', 'C', 'P', 'Q'}  # Input tensor dimensions  
D_O = {'N', 'K', 'P', 'Q'}  # Output tensor dimensions
D_ALL = {'R', 'S', 'P', 'Q', 'C', 'K', 'N'}  # All problem dimensions

# 张量维度映射
TENSOR_DIMS = {
    'W': D_W,
    'I': D_I, 
    'O': D_O
}

# STORAGE_MATRIX (B_i,t): 1 表示张量 t 存储在层级 i
# 严格对齐 DOSA 论文 Table 4 for Gemmini-like architecture
STORAGE_MATRIX = {
    # Level: {'W': val, 'I': val, 'O': val}
    0: {'W': 1, 'I': 1, 'O': 0},  # L0_Registers: Stores Weights, Inputs, but NOT Outputs
    1: {'W': 0, 'I': 0, 'O': 1},  # L1_Accumulator: Stores ONLY Outputs
    2: {'W': 1, 'I': 1, 'O': 1},  # L2_Scratchpad: Stores all tensors
    3: {'W': 1, 'I': 1, 'O': 1}   # L3_DRAM: Stores all tensors
}

def calculate_bandwidth_gb_s(level_name: str, num_pes: torch.Tensor, config: Config) -> torch.Tensor:
    """
    计算指定存储层级的带宽（GB/s）。
    
    Args:
        level_name: 存储层级名称 (如 'L0_Registers', 'L1_Accumulator', 'L2_Scratchpad', 'L3_DRAM')
        num_pes: 处理单元数量
        config: 全局配置对象
    
    Returns:
        torch.Tensor: 带宽值（GB/s）
    """
    # 根据新的带宽模型计算 words/cycle
    if level_name == 'L0_Registers':
        bandwidth_words_per_cycle = 2 * num_pes
    elif level_name in ['L1_Accumulator', 'L2_Scratchpad']:
        bandwidth_words_per_cycle = 2 * torch.sqrt(num_pes)
    elif level_name == 'L3_DRAM':
        bandwidth_words_per_cycle = 8
    else:
        # 对于未知层级，使用默认值
        bandwidth_words_per_cycle = 2 * torch.sqrt(num_pes)
    
    # 转换为 GB/s
    bandwidth_gb_s = (bandwidth_words_per_cycle * config.BYTES_PER_ELEMENT * config.CLOCK_FREQUENCY_MHZ * 1e6) / 1e9
    
    return bandwidth_gb_s

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
        
        return intra_accesses
    
    def calculate_traffic_formula_native(self, layer_dims: dict, mapping_table: dict, num_pes: torch.Tensor) -> dict:
        """
        基于学术论文公式的高保真度内存访问量计算方法。
        严格实现 Equations 6, 8, 9, 10, 11 中定义的公式体系。
        
        Args:
            layer_dims: 层的维度信息 {R, S, P, Q, C, K, N}
            mapping_table: 映射表 {dim: {level: {temporal: x, spatial: y}}}
            num_pes: 处理单元数量
            
        Returns:
            dict: 层级间数据传输量 {'L3_DRAM_to_L2_Scratchpad': bytes, ...}
        """
        from dosa.utils import calculate_macs
        
        # 计算总MAC运算次数
        total_macs = calculate_macs(layer_dims)
        
        # 内存层级定义 (0=Registers, 1=Accumulator, 2=Scratchpad, 3=DRAM)
        memory_levels = ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad', 'L3_DRAM']
        M = len(memory_levels) - 1  # 最高层级索引
        
        # 初始化结果字典
        accesses = {}
        
        # 为每个内存层级计算 Reads, Writes, Updates
        for i in range(M + 1):  # i = 0, 1, 2, 3
            level_name = memory_levels[i]
            
            # 对每种张量类型计算访问量
            for tensor_type in ['W', 'I', 'O']:
                
                # 检查该张量是否存储在当前层级
                if not STORAGE_MATRIX[i][tensor_type]:
                    continue
                    
                # 计算 C_i,t: 在层级 i 存储的张量 t 的数据块大小
                C_i_t = self._calculate_data_block_size(i, tensor_type, layer_dims, mapping_table)
                
                # 计算 Writes_t(i) - Equation 6
                writes = self._calculate_writes(i, tensor_type, layer_dims, mapping_table, C_i_t, M)
                
                # 计算 Reads_t(i) - Equation 11  
                reads = self._calculate_reads(i, tensor_type, layer_dims, mapping_table, total_macs, M)
                
                # 计算 Updates_O(i) - Equation 9 (仅对输出张量)
                updates = torch.tensor(0.0, device=self.config.DEVICE)
                if tensor_type == 'O':
                    updates = self._calculate_updates(i, layer_dims, mapping_table, total_macs, M)
                
                # 累加到对应的层级间接口
                if i < M:  # 不是最高层级
                    interface_name = f"{memory_levels[i+1]}_to_{level_name}"
                    if interface_name not in accesses:
                        accesses[interface_name] = torch.tensor(0.0, device=self.config.DEVICE)
                    
                    # 总访问量 = B_i,t × (Reads_t(i) + Writes_t(i)) + B_i,O × Updates_O(i)
                    tensor_accesses = reads + writes
                    if tensor_type == 'O':
                        tensor_accesses += updates
                    
                    # 转换为字节
                    accesses[interface_name] += tensor_accesses * self.config.BYTES_PER_ELEMENT
        
        return accesses
    
    def _calculate_data_block_size(self, i: int, tensor_type: str, layer_dims: dict, mapping_table: dict) -> torch.Tensor:
        """
        计算在层级 i 必须存储的张量 tensor_type 的数据块大小 C_i,t (以word为单位)
        严格按照论文公式实现：Equations 2, 3, 4
        """
        memory_levels = ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad', 'L3_DRAM']
        
        if tensor_type == 'W':
            # C_i,W (Equation 2 for Weights):
            # C_i,W = Π(f_k,j,d) 其中 k ∈ {S, T}, j ∈ {0, ..., i-1}, d ∈ D_W = {R, S, C, K}
            size = torch.tensor(1.0, device=self.config.DEVICE)
            relevant_dims = D_W  # {R, S, C, K}
            
            # 遍历从 0 到 i-1 的所有内层层级
            for j in range(i):
                level_name = memory_levels[j]
                # 对每个层级，遍历 D_W 中的所有维度，累乘其 temporal 和 spatial 因子
                for dim_name in relevant_dims:
                    if dim_name in layer_dims and dim_name in mapping_table and level_name in mapping_table[dim_name]:
                        temporal_factor = mapping_table[dim_name][level_name].get('temporal', 1)
                        spatial_factor = mapping_table[dim_name][level_name].get('spatial', 1)
                        size *= torch.tensor(temporal_factor * spatial_factor, device=self.config.DEVICE)
            
            return size
            
        elif tensor_type == 'O':
            # C_i,O (Equation 4 for Outputs):
            # C_i,O = Π(f_k,j,d) 其中 k ∈ {S, T}, j ∈ M (所有内存层级), d ∈ D_O = {P, Q, K, N}
            size = torch.tensor(1.0, device=self.config.DEVICE)
            relevant_dims = D_O  # {P, Q, K, N}
            
            # 遍历所有内存层级
            for j in range(len(memory_levels)):
                level_name = memory_levels[j]
                # 对每个层级，遍历 D_O 中的所有维度，累乘其 temporal 和 spatial 因子
                for dim_name in relevant_dims:
                    if dim_name in layer_dims and dim_name in mapping_table and level_name in mapping_table[dim_name]:
                        temporal_factor = mapping_table[dim_name][level_name].get('temporal', 1)
                        spatial_factor = mapping_table[dim_name][level_name].get('spatial', 1)
                        size *= torch.tensor(temporal_factor * spatial_factor, device=self.config.DEVICE)
            
            return size
            
        elif tensor_type == 'I':
            # C_i,I (Equation 3 for Inputs - 最复杂):
            # Inner(i, d) = Π(f_k,j,d) (其中 k,j 范围同 C_i,W)
            # C_i,I = Π_{d∈{C,N}} (f_k,j,d) × (P_stride × (Inner(i,P) - 1) + Inner(i,R)) × (Q_stride × (Inner(i,Q) - 1) + Inner(i,S))
            
            def calculate_inner(dim_name):
                """计算 Inner(i, d) - 某个维度 d 在层级 i 及其内层的总 tile size"""
                inner_size = torch.tensor(1.0, device=self.config.DEVICE)
                for j in range(i):
                    level_name = memory_levels[j]
                    if dim_name in layer_dims and dim_name in mapping_table and level_name in mapping_table[dim_name]:
                        temporal_factor = mapping_table[dim_name][level_name].get('temporal', 1)
                        spatial_factor = mapping_table[dim_name][level_name].get('spatial', 1)
                        inner_size *= torch.tensor(temporal_factor * spatial_factor, device=self.config.DEVICE)
                return inner_size
            
            # 假设 P_stride 和 Q_stride 为 1
            P_stride = 1
            Q_stride = 1
            
            # 计算各个维度的 Inner 值
            inner_P = calculate_inner('P')
            inner_R = calculate_inner('R')
            inner_Q = calculate_inner('Q')
            inner_S = calculate_inner('S')
            
            # 计算 C 和 N 维度在所有内层层级的因子连乘积
            cn_factors = torch.tensor(1.0, device=self.config.DEVICE)
            for j in range(i):
                level_name = memory_levels[j]
                for dim_name in ['C', 'N']:
                    if dim_name in layer_dims and dim_name in mapping_table and level_name in mapping_table[dim_name]:
                        temporal_factor = mapping_table[dim_name][level_name].get('temporal', 1)
                        spatial_factor = mapping_table[dim_name][level_name].get('spatial', 1)
                        cn_factors *= torch.tensor(temporal_factor * spatial_factor, device=self.config.DEVICE)
            
            # 根据公式组合这些部分
            size = cn_factors * (P_stride * (inner_P - 1) + inner_R) * (Q_stride * (inner_Q - 1) + inner_S)
            
            return size
            
        else:
            # 对于未知张量类型，返回默认值
            return torch.tensor(1.0, device=self.config.DEVICE)
    
    def _calculate_writes(self, i: int, tensor_type: str, layer_dims: dict, mapping_table: dict, C_i_t: torch.Tensor, M: int) -> torch.Tensor:
        """
        计算 Writes_t(i) - Equation 6
        Writes_t(i) = C_i,t × Π(f_k,j,d) 其中 j ∈ {i+1, ..., M}, d ∈ D_t
        """
        if i == M:  # 最高层级没有写入
            return torch.tensor(0.0, device=self.config.DEVICE)
        
        writes = C_i_t
        relevant_dims = TENSOR_DIMS[tensor_type]
        memory_levels = ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad', 'L3_DRAM']
        
        # 连乘所有外层于 i 的层级 (j ∈ {i+1, ..., M})
        for j in range(i + 1, M + 1):
            level_name = memory_levels[j]
            
            # 对所有与该张量相关的维度 d ∈ D_t
            for dim_name in relevant_dims:
                if dim_name in layer_dims and dim_name in mapping_table and level_name in mapping_table[dim_name]:
                    # 只累乘 temporal 因子，移除 spatial 因子的影响
                    # 根据原始方法论，外层循环的迭代次数仅由 temporal_factor 决定
                    # 因为空间tiling代表的是并行展开，不增加对上层存储的访问次数
                    temporal_factor = mapping_table[dim_name][level_name].get('temporal', 1)
                    writes *= torch.tensor(temporal_factor, device=self.config.DEVICE)
        
        return writes
    
    def _calculate_reads(self, i: int, tensor_type: str, layer_dims: dict, mapping_table: dict, total_macs: torch.Tensor, M: int) -> torch.Tensor:
        """
        计算 Reads_t(i) - Equation 11
        分段函数：
        - if i == innermost tensor t level: Reads_t(i) = MACs / F_S,t(i)
        - if i > innermost tensor t level: Reads_t(i) = Writes_t(i-1) / F_S,t(i)
        """
        # 计算空间复用修正项 F_S,t(i)
        F_S_t_i = self._calculate_spatial_reuse_factor(i, tensor_type, layer_dims, mapping_table)
        
        # 判断是否为该张量的最内层级
        innermost_level = self._find_innermost_level(tensor_type, mapping_table)
        
        if i == innermost_level:
            # 最内层级：Reads_t(i) = MACs / F_S,t(i)
            reads = total_macs / torch.clamp(F_S_t_i, min=torch.tensor(1e-9, device=self.config.DEVICE))
        else:
            # 非最内层级：Reads_t(i) = Writes_t(i-1) / F_S,t(i)
            if i > 0:
                C_i_minus_1_t = self._calculate_data_block_size(i-1, tensor_type, layer_dims, mapping_table)
                writes_i_minus_1 = self._calculate_writes(i-1, tensor_type, layer_dims, mapping_table, C_i_minus_1_t, M)
                reads = writes_i_minus_1 / torch.clamp(F_S_t_i, min=torch.tensor(1e-9, device=self.config.DEVICE))
            else:
                reads = torch.tensor(0.0, device=self.config.DEVICE)
        
        return reads
    
    def _calculate_updates(self, i: int, layer_dims: dict, mapping_table: dict, total_macs: torch.Tensor, M: int) -> torch.Tensor:
        """
        计算 Updates_O(i) - Equation 9 (仅对输出张量)
        分段函数：
        - if i == innermost output level: Updates_O(i) = MACs / F_S,O(i)
        - if i > innermost output level: Updates_O(i) = Writes_O(i-1) / F_S,O(i)
        """
        # 计算输出张量的空间复用修正项
        F_S_O_i = self._calculate_spatial_reuse_factor(i, 'O', layer_dims, mapping_table)
        
        # 判断是否为输出张量的最内层级
        innermost_output_level = self._find_innermost_level('O', mapping_table)
        
        if i == innermost_output_level:
            # 最内层级：Updates_O(i) = MACs / F_S,O(i)
            updates = total_macs / torch.clamp(F_S_O_i, min=torch.tensor(1e-9, device=self.config.DEVICE))
        else:
            # 非最内层级：Updates_O(i) = Writes_O(i-1) / F_S,O(i)
            if i > 0:
                C_i_minus_1_O = self._calculate_data_block_size(i-1, 'O', layer_dims, mapping_table)
                writes_O_i_minus_1 = self._calculate_writes(i-1, 'O', layer_dims, mapping_table, C_i_minus_1_O, M)
                updates = writes_O_i_minus_1 / torch.clamp(F_S_O_i, min=torch.tensor(1e-9, device=self.config.DEVICE))
            else:
                updates = torch.tensor(0.0, device=self.config.DEVICE)
        
        return updates
    
    def _calculate_spatial_reuse_factor(self, i: int, tensor_type: str, layer_dims: dict, mapping_table: dict) -> torch.Tensor:
        """
        计算空间复用修正项 F_S,t(i) - Equations 8 & 10
        F_S,t(i) = Π(f_S,i,d) 其中 d ∈ (D - D_t) (与张量 t 无关的维度)
        """
        F_S = torch.tensor(1.0, device=self.config.DEVICE)
        
        # 获取与该张量无关的维度 (D - D_t)
        relevant_dims = TENSOR_DIMS[tensor_type]
        irrelevant_dims = D_ALL - relevant_dims
        
        level_name = ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad', 'L3_DRAM'][i]
        
        # 对所有与张量无关的维度，累乘空间映射因子
        for dim_name in irrelevant_dims:
            if dim_name in layer_dims and dim_name in mapping_table and level_name in mapping_table[dim_name]:
                spatial_factor = mapping_table[dim_name][level_name].get('spatial', 1)
                F_S *= torch.tensor(spatial_factor, device=self.config.DEVICE)
        
        return F_S
    
    def _find_innermost_level(self, tensor_type: str, mapping_table: dict) -> int:
        """
        找到张量 tensor_type 的最内存储层级
        """
        # 逻辑说明：该函数通过从最内层（L0）开始查找，返回第一个发现有该张量相关维度映射
        # （即 temporal 或 spatial 因子 > 1）的层级作为 innermost_level。
        #
        # 设计假设：这是一个基于工程实践的合理推断 (engineering heuristic)，
        # 其核心假设是：一个张量如果在一个层级有 tiling 行为，那么该层级就是它在计算中活跃的最深层级。
        #
        # 潜在局限：在某些罕见的边缘映射情况下，此启发式可能与仿真器的行为存在偏差，
        # 但对于绝大多数典型数据流是有效且鲁棒的。
        
        memory_levels = ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad', 'L3_DRAM']
        relevant_dims = TENSOR_DIMS[tensor_type]
        
        # 从最内层开始检查
        for i in range(len(memory_levels)):
            level_name = memory_levels[i]
            
            # 检查该层级是否有该张量的任何维度映射
            has_mapping = False
            for dim_name in relevant_dims:
                if dim_name in mapping_table and level_name in mapping_table[dim_name]:
                    temporal = mapping_table[dim_name][level_name].get('temporal', 1)
                    spatial = mapping_table[dim_name][level_name].get('spatial', 1)
                    if temporal > 1 or spatial > 1:
                        has_mapping = True
                        break
            
            if has_mapping:
                return i
        
        # 默认返回最内层
        return 0
        

        
        return intra_accesses

    def calculate_per_level_accesses(self, layer_dims: dict, mapping_table: dict, return_detailed_info: bool = False):
        """
        重构后的层间数据移动计算函数 - 基于数据流驱动的访存模型
        核心公式: 层级间传输字节数 = 下层数据足迹 * 外层循环总次数 * 数据位宽

        Args:
            layer_dims (dict): 层的维度信息
            mapping_table (dict): 完整的映射表
            return_detailed_info (bool): 如果为True，则额外返回一个包含中间计算结果的字典

        Returns:
            - accesses (dict): 一个字典，键为接口名称，值为该接口的总访问字节数
            - detailed_info (dict, optional): 如果 return_detailed_info 为 True，则返回此字典
        """
        accesses = {}
        detailed_info = {}
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

            if return_detailed_info:
                detailed_info[interface_name] = {}

            # 对每种张量类型计算数据移动量
            for tensor_type, relevant_dims in TENSOR_DIM_MAP.items():
                # 步骤1: 计算下层数据足迹 (Data_Footprint_at_Lower_Level)
                # 这代表为了完成一次外层循环，需要加载到LowerLevel的数据块大小
                data_footprint_elements = torch.tensor(1.0, device=self.config.DEVICE)
                
                # 遍历所有内部层级（从最内层到LowerLevel）
                for level_idx in range(lower_level_idx + 1):
                    level_name = level_names[level_idx]
                    # 对于每个内部层级，累乘所有相关维度的temporal和spatial映射因子
                    for dim_name in relevant_dims:
                        if dim_name in layer_dims and dim_name in mapping_table and level_name in mapping_table[dim_name]:
                            temporal_factor = mapping_table[dim_name][level_name].get('temporal', 1)
                            spatial_factor = mapping_table[dim_name][level_name].get('spatial', 1)
                            data_footprint_elements *= torch.tensor(temporal_factor * spatial_factor, device=self.config.DEVICE)

                # 步骤2: 计算外层循环总次数 (Outer_Loop_Iterations)
                # 这代表需要多少次从UpperLevel加载数据块到LowerLevel
                # 外部循环次数 = 所有比UpperLevel更外层的相关维度temporal因子的乘积
                outer_loop_iterations = torch.tensor(1.0, device=self.config.DEVICE)
                
                # 遍历所有比UpperLevel更外的层级
                for level_idx in range(upper_level_idx + 1, len(memory_levels)):
                    level_name = level_names[level_idx]
                    # 对于每个更外层级，累乘所有相关维度的temporal映射因子
                    for dim_name in relevant_dims:
                        if dim_name in layer_dims and dim_name in mapping_table and level_name in mapping_table[dim_name]:
                            temporal_factor = mapping_table[dim_name][level_name].get('temporal', 1)
                            outer_loop_iterations *= torch.tensor(temporal_factor, device=self.config.DEVICE)

                # 步骤3: 引入数据复用修正 (Reuse_Factor)
                # 考虑该张量在LowerLevel的数据可以被其他张量的循环复用
                reuse_factor = torch.tensor(1.0, device=self.config.DEVICE)
                
                # 遍历所有内部层级（从最内层到LowerLevel）
                for level_idx in range(lower_level_idx + 1):
                    level_name = level_names[level_idx]
                    # 对于每个内部层级，累乘所有复用维度的temporal映射因子
                    for dim_name in reuse_dims[tensor_type]:
                        if dim_name in layer_dims and dim_name in mapping_table and level_name in mapping_table[dim_name]:
                            temporal_factor = mapping_table[dim_name][level_name].get('temporal', 1)
                            reuse_factor *= torch.tensor(temporal_factor, device=self.config.DEVICE)
                
                # 应用复用修正
                effective_outer_iterations = outer_loop_iterations / torch.clamp(reuse_factor, min=torch.tensor(1e-9, device=self.config.DEVICE))

                # 步骤4: 计算该张量在此接口的传输字节数
                tensor_transfer_bytes = (data_footprint_elements * 
                                       effective_outer_iterations * 
                                       self.config.BYTES_PER_ELEMENT)
                total_access_bytes_for_interface += tensor_transfer_bytes

                if return_detailed_info:
                    detailed_info[interface_name][tensor_type] = {
                        'data_footprint_elements': data_footprint_elements.clone().detach(),
                        'outer_loop_iterations': outer_loop_iterations.clone().detach(),
                        'reuse_factor': reuse_factor.clone().detach(),
                        'effective_outer_iterations': effective_outer_iterations.clone().detach(),
                        'tensor_transfer_bytes': tensor_transfer_bytes.clone().detach()
                    }

            accesses[interface_name] = total_access_bytes_for_interface

        if return_detailed_info:
            return accesses, detailed_info
        return accesses

    def forward(self, graph, hw_params: HardwareParameters, mapping: FineGrainedMapping, direct_mapping_table: dict = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        高保真度性能模型的前向传播方法。
        
        Args:
            graph: 计算图
            hw_params: 硬件参数
            mapping: FineGrainedMapping实例（用于训练路径）
            direct_mapping_table: （可选）一个离散的、嵌套的映射表字典。当提供此参数时，
                                模型将进入'验证模式'，直接使用此映射表，并完全绕过
                                mapping.get_all_factors()的可微投影逻辑。
                                格式: {dim: {level: {'temporal': val, 'spatial': val}}}
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                (total_latency, total_energy, total_buffer_mismatch_loss, total_compatibility_penalty, detailed_metrics)
        """
        total_latency = torch.tensor(0.0, device=self.config.DEVICE)
        total_energy = torch.tensor(0.0, device=self.config.DEVICE)
        total_buffer_mismatch_loss = torch.tensor(0.0, device=self.config.DEVICE)
        total_compatibility_penalty = torch.tensor(0.0, device=self.config.DEVICE)
        
        if direct_mapping_table:
            # [验证路径]：如果外部直接提供了映射表，则进入验证模式。
            # 我们信任调用者已确保该表的格式和物理有效性。
            all_factors = direct_mapping_table
            # 注：此路径下，mapping 对象中的可学习参数将不会被使用。
        else:
            # [训练路径]：如果没有提供直接映射，则保持原有的训练模式。
            # 通过可微投影仪从可学习参数中获取有效映射因子。
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
                # Refactored to Cycles Accumulation model based on DOSA paper Eq. (2)
                from dosa.utils import calculate_macs # 确保导入
                macs = calculate_macs(layer['dims'])
                
                num_pes = hw_params.get_projected_num_pes()
                
                # Calculate compute cycles
                compute_cycles = macs / num_pes
                
                # 使用新的高保真度公式原生计算方法
                per_level_accesses = self.calculate_traffic_formula_native(layer['dims'], all_factors, num_pes)
                
                # Calculate memory cycles for each interface
                memory_cycles_list = []
                num_pes_sqrt = torch.sqrt(num_pes)

                for interface, accesses in per_level_accesses.items():
                    upper_level_name = interface.split('_to_')[0]
                    
                    # 使用新的带宽计算函数
                    bandwidth_gb_s = calculate_bandwidth_gb_s(upper_level_name, num_pes, self.config)
                    
                    # Convert bandwidth from GB/s to bytes_per_cycle
                    bytes_per_cycle = bandwidth_gb_s * 1e9 / (self.config.CLOCK_FREQUENCY_MHZ * 1e6)
                    
                    # Calculate memory cycles for this interface
                    memory_cycles = accesses / (bytes_per_cycle + torch.tensor(1e-9, device=self.config.DEVICE))
                    memory_cycles_list.append(memory_cycles)
                
                # Identify memory bottleneck
                if memory_cycles_list:
                    bottleneck_memory_cycles = torch.max(torch.stack(memory_cycles_list))
                else:
                    bottleneck_memory_cycles = torch.tensor(0.0, device=self.config.DEVICE)
                
                # Calculate stall cycles
                stall_cycles = torch.relu(bottleneck_memory_cycles - compute_cycles)
                
                # Calculate total cycles and convert to latency
                total_cycles = compute_cycles + stall_cycles
                latency = total_cycles / (self.config.CLOCK_FREQUENCY_MHZ * 1e6)

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