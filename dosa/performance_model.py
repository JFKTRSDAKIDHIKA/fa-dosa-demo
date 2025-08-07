import torch
import torch.nn as nn
from typing import Tuple
from functools import reduce
from operator import mul

from dosa.config import Config
from dosa.hardware_parameters import HardwareParameters
from dosa.mapping import FineGrainedMapping
from dosa.dmt import InPlaceFusionDMT, SkipConnectionDMT

# 获取全局配置实例
_config = Config.get_instance()

# 从Config类获取物理概念体系定义
TENSOR_DIM_MAP = {
    'Input':  ['N', 'C', 'P', 'Q'],
    'Weight': ['K', 'C', 'R', 'S'],
    'Output': ['N', 'K', 'P', 'Q']
}

# 使用Config中的张量维度定义
D_W = _config.TENSOR_DIMENSIONS['W']  # Weight tensor dimensions
D_I = _config.TENSOR_DIMENSIONS['I']  # Input tensor dimensions  
D_O = _config.TENSOR_DIMENSIONS['O']  # Output tensor dimensions
D_ALL = _config.D_ALL  # All problem dimensions

# 张量维度映射 - 使用Config中的定义
TENSOR_DIMS = _config.TENSOR_DIMENSIONS

# 存储矩阵 - 使用Config中的定义
STORAGE_MATRIX = _config.STORAGE_MATRIX

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
        bandwidth_words_per_cycle = torch.tensor(8.0, device=config.DEVICE)
    else:
        # 对于未知层级，使用默认值
        bandwidth_words_per_cycle = 2 * torch.sqrt(num_pes)
    
    # 转换为 GB/s
    bandwidth_gb_s = (bandwidth_words_per_cycle * config.BYTES_PER_ELEMENT * config.CLOCK_FREQUENCY_MHZ * 1e6) / 1e9
    
    return bandwidth_gb_s

def calculate_bandwidth_bytes_per_cycle(level_name: str, num_pes: torch.Tensor, config: Config) -> torch.Tensor:
    """
    计算指定存储层级的带宽（bytes/cycle）。
    
    Args:
        level_name: 存储层级名称
        num_pes: 处理单元数量
        config: 全局配置对象
    
    Returns:
        torch.Tensor: 带宽值（bytes/cycle）
    """
    # 根据带宽模型计算 words/cycle
    if level_name == 'L0_Registers':
        bandwidth_words_per_cycle = 2 * num_pes
    elif level_name in ['L1_Accumulator', 'L2_Scratchpad']:
        bandwidth_words_per_cycle = 2 * torch.sqrt(num_pes)
    elif level_name == 'L3_DRAM':
        bandwidth_words_per_cycle = torch.tensor(8.0, device=config.DEVICE)
    else:
        bandwidth_words_per_cycle = 2 * torch.sqrt(num_pes)
    
    # 转换为 bytes/cycle
    bandwidth_bytes_per_cycle = bandwidth_words_per_cycle * torch.tensor(config.BYTES_PER_ELEMENT, device=config.DEVICE)
    
    return bandwidth_bytes_per_cycle

class HighFidelityPerformanceModel(nn.Module):
    """
    NEW: 高精度性能模型，能够处理多级存储和细粒度映射。
    """
    def __init__(self, config: Config, debug_latency: bool = False):
        super().__init__()
        self.config = config
        self.debug_latency = debug_latency
        self.dmt_registry = {
            ('Conv', 'ReLU'): InPlaceFusionDMT(debug_latency=self.debug_latency, debug_output_path="debug_performance_model.json"),
            ('Conv', 'BatchNormalization', 'ReLU'): InPlaceFusionDMT(debug_latency=self.debug_latency, debug_output_path="debug_performance_model.json"),
            ('MatMul', 'Add'): InPlaceFusionDMT(debug_latency=self.debug_latency, debug_output_path="debug_performance_model.json"),
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
    
    def calculate_traffic_formula_native(self, layer_dims: dict, mapping_table: dict, num_pes: torch.Tensor, debug_data: dict = None) -> dict:
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
        
        # 初始化调试数据收集
        if debug_data is not None:
            debug_data["detailed_traffic_formula_trace"] = {}
        
        # 为每个内存层级计算 Reads, Writes, Updates
        for i in range(M + 1):  # i = 0, 1, 2, 3
            level_name = memory_levels[i]
            
            # 初始化当前层级的调试数据
            if debug_data is not None:
                debug_data["detailed_traffic_formula_trace"][f"{level_name} (i={i})"] = {}
            
            # 对每种张量类型计算访问量
            for tensor_type in ['W', 'I', 'O']:
                
                # 检查该张量是否存储在当前层级
                if not STORAGE_MATRIX[i][tensor_type]:
                    continue
                    
                # 计算 C_i,t: 在层级 i 存储的张量 t 的数据块大小
                C_i_t = self._calculate_data_block_size(i, tensor_type, layer_dims, mapping_table)
                
                # 计算空间复用因子
                F_S_t_i = self._calculate_spatial_reuse_factor(i, tensor_type, layer_dims, mapping_table)
                
                # 计算最内层活跃层级
                innermost_level = self._find_innermost_level(tensor_type, mapping_table)
                
                # 计算 Writes_t(i) - Equation 6
                writes = self._calculate_writes(i, tensor_type, layer_dims, mapping_table, C_i_t, M)
                
                # 计算 Reads_t(i) - Equation 11  
                reads = self._calculate_reads(i, tensor_type, layer_dims, mapping_table, total_macs, M)
                
                # 计算 Updates_O(i) - Equation 9 (仅对输出张量)
                updates = torch.tensor(0.0, device=self.config.DEVICE)
                if tensor_type == 'O':
                    updates = self._calculate_updates(i, layer_dims, mapping_table, total_macs, M)
                
                # 收集详细的调试数据
                if debug_data is not None:
                    tensor_debug_data = {
                        "C_i,t": C_i_t.detach().cpu().item() if isinstance(C_i_t, torch.Tensor) else C_i_t,
                        "F_S,t(i)": F_S_t_i.detach().cpu().item() if isinstance(F_S_t_i, torch.Tensor) else F_S_t_i,
                        "innermost_level": innermost_level,
                        "Reads_t(i)": reads.detach().cpu().item() if isinstance(reads, torch.Tensor) else reads,
                        "Writes_t(i)": writes.detach().cpu().item() if isinstance(writes, torch.Tensor) else writes
                    }
                    if tensor_type == 'O':
                        tensor_debug_data["Updates_O(i)"] = updates.detach().cpu().item() if isinstance(updates, torch.Tensor) else updates
                    
                    debug_data["detailed_traffic_formula_trace"][f"{level_name} (i={i})"][tensor_type] = tensor_debug_data
                
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
        
        # === 第三步：为权重(W)实现真实的L1->L0层间填充流量计算 ===
        # 当i=1(L1)时，单独计算写往L0的权重填充流量
        if 'L1_Accumulator_to_L0_Registers' in accesses:
            # 重置L1->L0接口的流量，只包含真实的权重填充流量
            accesses['L1_Accumulator_to_L0_Registers'] = torch.tensor(0.0, device=self.config.DEVICE)
            
            # 计算权重在L0的数据足迹 (Data_Footprint_at_L0)
            weight_data_footprint_l0 = self._calculate_data_block_size(0, 'W', layer_dims, mapping_table)
            
            # 计算驱动L0数据重载的外层循环总次数 (参考calculate_per_level_accesses的思想)
            # 外层循环次数 = 所有比L0更外层的权重相关维度temporal因子的乘积
            outer_loop_iterations = torch.tensor(1.0, device=self.config.DEVICE)
            weight_dims = ['K', 'C', 'R', 'S']  # 权重张量的相关维度
            
            # 遍历L1及以上层级
            for level_idx in range(1, M + 1):
                level_name = memory_levels[level_idx]
                for dim_name in weight_dims:
                    if dim_name in layer_dims and dim_name in mapping_table and level_name in mapping_table[dim_name]:
                        temporal_factor = mapping_table[dim_name][level_name].get('temporal', 1)
                        outer_loop_iterations *= torch.tensor(temporal_factor, device=self.config.DEVICE)
            
            # 计算权重复用因子 (在L0层级的复用)
            weight_reuse_factor = torch.tensor(1.0, device=self.config.DEVICE)
            weight_reuse_dims = ['N', 'P', 'Q']  # 权重在这些维度上被复用
            
            # 只考虑L0层级的复用
            level_name = memory_levels[0]  # L0_Registers
            for dim_name in weight_reuse_dims:
                if dim_name in layer_dims and dim_name in mapping_table and level_name in mapping_table[dim_name]:
                    temporal_factor = mapping_table[dim_name][level_name].get('temporal', 1)
                    weight_reuse_factor *= torch.tensor(temporal_factor, device=self.config.DEVICE)
            
            # 应用复用修正
            effective_outer_iterations = outer_loop_iterations / torch.clamp(weight_reuse_factor, min=torch.tensor(1e-9, device=self.config.DEVICE))
            
            # 计算L1->L0的权重填充流量
            l1_to_l0_weight_fills_bytes = (weight_data_footprint_l0 * 
                                         effective_outer_iterations * 
                                         self.config.BYTES_PER_ELEMENT)
            
            # 更新L1->L0接口的流量（只包含权重填充流量）
            accesses['L1_Accumulator_to_L0_Registers'] = l1_to_l0_weight_fills_bytes
            
            # 收集调试数据
            if debug_data is not None:
                debug_data["L1_to_L0_weight_fills_calculation"] = {
                    "weight_data_footprint_l0": weight_data_footprint_l0.detach().cpu().item() if isinstance(weight_data_footprint_l0, torch.Tensor) else weight_data_footprint_l0,
                    "outer_loop_iterations": outer_loop_iterations.detach().cpu().item() if isinstance(outer_loop_iterations, torch.Tensor) else outer_loop_iterations,
                    "weight_reuse_factor": weight_reuse_factor.detach().cpu().item() if isinstance(weight_reuse_factor, torch.Tensor) else weight_reuse_factor,
                    "effective_outer_iterations": effective_outer_iterations.detach().cpu().item() if isinstance(effective_outer_iterations, torch.Tensor) else effective_outer_iterations,
                    "l1_to_l0_weight_fills_bytes": l1_to_l0_weight_fills_bytes.detach().cpu().item() if isinstance(l1_to_l0_weight_fills_bytes, torch.Tensor) else l1_to_l0_weight_fills_bytes
                }
        
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
            # --- 开始重构 ---
            if tensor_type == 'W':
                # 对于权重(W)，其在最内层(L0)的'Reads'是内部供给，不产生对上一级的物理读取流量。
                # 返回一个极小值或零，以确保它不贡献 L1->L0 的 traffic。
                reads = torch.tensor(0.0, device=self.config.DEVICE)
            else:
                # 对于其他张量（如输入I），其在最内层的'Reads'是真实的物理读取。
                # 维持原公式，因为它现在描述的是从其最内层存储（如L2 for I）到MAC的流量。
                reads = total_macs / torch.clamp(F_S_t_i, min=torch.tensor(1e-9, device=self.config.DEVICE))
            # --- 结束重构 ---
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
        # --- 开始重构 ---
        # 根据论文(表4)的设定，强制输入张量(I)的最内层存储为 L2 Scratchpad (index=2)
        if tensor_type == 'I':
            return 2
        # --- 结束重构 ---
        
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

    def forward(self, graph, hw_params: HardwareParameters, mapping: FineGrainedMapping, direct_mapping_table: dict = None, debug_output_path: str = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        高保真度性能模型的前向传播方法 - 实现完整的PPA计算逻辑。
        
        严格遵循DOSA论文的屋顶线时延模型（公式12）和三段式能耗模型（公式13）。
        
        Args:
            graph: 计算图
            hw_params: 硬件参数
            mapping: FineGrainedMapping实例（用于训练路径）
            direct_mapping_table: （可选）离散映射表，用于验证模式
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                (total_latency, total_energy, area_cost, total_buffer_mismatch_loss, total_compatibility_penalty)
        """
        total_latency = torch.tensor(0.0, device=self.config.DEVICE)
        total_energy = torch.tensor(0.0, device=self.config.DEVICE)
        total_buffer_mismatch_loss = torch.tensor(0.0, device=self.config.DEVICE)
        total_compatibility_penalty = torch.tensor(0.0, device=self.config.DEVICE)
        
        # 初始化调试数据容器
        debug_data = None
        if self.debug_latency and debug_output_path:
            debug_data = {
                "calculation_summary": {},
                "memory_interface_analysis": {},
                "detailed_traffic_formula_trace": {}
            }
        
        # 确定映射表来源
        if direct_mapping_table:
            all_factors = direct_mapping_table
        else:
            all_factors = mapping.get_all_factors()

        # 以计算组为单位进行PPA计算
        for group in graph.fusion_groups:
            current_pattern = tuple(graph.layers[layer_name]['type'] for layer_name in group)
            dmt_model = self.dmt_registry.get(current_pattern)

            if dmt_model:
                # 使用DMT处理融合模式
                latency, energy, group_buffer_mismatch_loss, compatibility_penalty, detailed_metrics = dmt_model(group, graph, hw_params, mapping, self.config)
                total_buffer_mismatch_loss += group_buffer_mismatch_loss
                total_compatibility_penalty += compatibility_penalty
            else:
                # 单层计算：实现完整的PPA计算逻辑
                layer_name = group[0]
                layer = graph.layers[layer_name]
                
                # 导入必要的工具函数
                from dosa.utils import calculate_macs
                
                # 基础参数计算
                total_macs = calculate_macs(layer['dims'])
                num_pes = hw_params.get_projected_num_pes()
                
                # ===== 2.1 时延模型实现 (屋顶线模型 - 公式12) =====
                
                # 步骤1: 计算实际利用的PE数量 (utilized_pes)
                utilized_pes = torch.tensor(1.0, device=self.config.DEVICE)
                
                # 从 all_factors 中提取所有 spatial 因子并累乘
                for dim_name, dim_mapping in all_factors.items():
                    for level_name, level_factors in dim_mapping.items():
                        if 'spatial' in level_factors:
                            spatial_factor = level_factors['spatial']
                            # 确保 spatial_factor 是张量类型
                            if not isinstance(spatial_factor, torch.Tensor):
                                spatial_factor = torch.tensor(float(spatial_factor), device=self.config.DEVICE)
                            utilized_pes *= spatial_factor
                
                # 边界条件检查：确保 utilized_pes 至少为 1，防止除以零错误
                effective_pes = torch.max(utilized_pes, torch.tensor(1.0, device=self.config.DEVICE))
                
                # 步骤2: 计算Compute_Cycles (使用实际利用的PE数量)
                compute_cycles = total_macs / effective_pes
                
                # 步骤3: 计算Memory_Cycles
                # 调用高保真访存计算引擎
                per_level_accesses = self.calculate_traffic_formula_native(layer['dims'], all_factors, num_pes, debug_data)
                
                memory_cycles_list = []
                for interface, accesses_bytes in per_level_accesses.items():
                    upper_level_name = interface.split('_to_')[0]
                    
                    # 计算该接口的带宽（bytes/cycle）
                    bandwidth_bytes_per_cycle = calculate_bandwidth_bytes_per_cycle(upper_level_name, num_pes, self.config)
                    
                    # 计算该接口的内存周期数
                    memory_cycles = accesses_bytes / (bandwidth_bytes_per_cycle + torch.tensor(1e-9, device=self.config.DEVICE))
                    memory_cycles_list.append(memory_cycles)
                    
                    # 收集内存接口分析数据
                    if debug_data is not None:
                        debug_data["memory_interface_analysis"][interface] = {
                            "accesses_bytes": accesses_bytes.detach().cpu().item() if isinstance(accesses_bytes, torch.Tensor) else accesses_bytes,
                            "bandwidth_bytes_per_cycle": bandwidth_bytes_per_cycle.detach().cpu().item() if isinstance(bandwidth_bytes_per_cycle, torch.Tensor) else bandwidth_bytes_per_cycle,
                            "memory_cycles": memory_cycles.detach().cpu().item() if isinstance(memory_cycles, torch.Tensor) else memory_cycles
                        }
                
                # === 第四步：建模L2_to_MAC接口 ===
                # 为从L2到MAC的输入数据流建立独立的成本核算
                
                # 计算L2_to_MAC的流量
                F_S_I_2 = self._calculate_spatial_reuse_factor(2, 'I', layer['dims'], all_factors)
                L2_to_MAC_traffic_elements = total_macs / torch.clamp(F_S_I_2, min=torch.tensor(1e-9, device=self.config.DEVICE))
                L2_to_MAC_traffic_bytes = L2_to_MAC_traffic_elements * self.config.BYTES_PER_ELEMENT
                
                # 根据论文，L2带宽与L1相同
                L2_to_MAC_bandwidth = calculate_bandwidth_bytes_per_cycle('L1_Accumulator', num_pes, self.config)
                l2_to_mac_cycles = L2_to_MAC_traffic_bytes / (L2_to_MAC_bandwidth + torch.tensor(1e-9, device=self.config.DEVICE))
                
                # 将L2_to_MAC周期加入到memory_cycles_list中
                memory_cycles_list.append(l2_to_mac_cycles)
                
                # 收集L2_to_MAC接口分析数据
                if debug_data is not None:
                    debug_data["memory_interface_analysis"]["L2_to_MAC"] = {
                        "F_S_I_2": F_S_I_2.detach().cpu().item() if isinstance(F_S_I_2, torch.Tensor) else F_S_I_2,
                        "L2_to_MAC_traffic_elements": L2_to_MAC_traffic_elements.detach().cpu().item() if isinstance(L2_to_MAC_traffic_elements, torch.Tensor) else L2_to_MAC_traffic_elements,
                        "accesses_bytes": L2_to_MAC_traffic_bytes.detach().cpu().item() if isinstance(L2_to_MAC_traffic_bytes, torch.Tensor) else L2_to_MAC_traffic_bytes,
                        "bandwidth_bytes_per_cycle": L2_to_MAC_bandwidth.detach().cpu().item() if isinstance(L2_to_MAC_bandwidth, torch.Tensor) else L2_to_MAC_bandwidth,
                        "memory_cycles": l2_to_mac_cycles.detach().cpu().item() if isinstance(l2_to_mac_cycles, torch.Tensor) else l2_to_mac_cycles
                    }
                
                # 步骤4: 确定瓶颈并计算总延迟
                if memory_cycles_list:
                    bottleneck_memory_cycles = torch.max(torch.stack(memory_cycles_list))
                else:
                    bottleneck_memory_cycles = torch.tensor(0.0, device=self.config.DEVICE)
                
                # 计算停滞周期
                stall_cycles = torch.relu(bottleneck_memory_cycles - compute_cycles)
                
                # 计算总周期数和延迟
                total_cycles = compute_cycles + stall_cycles
                latency = total_cycles / (self.config.CLOCK_FREQUENCY_MHZ * 1e6)
                
                # 收集计算总结数据
                if debug_data is not None:
                    debug_data["calculation_summary"] = {
                        "total_macs": total_macs.detach().cpu().item() if isinstance(total_macs, torch.Tensor) else total_macs,
                        "utilized_pes": utilized_pes.detach().cpu().item() if isinstance(utilized_pes, torch.Tensor) else utilized_pes,
                        "effective_pes": effective_pes.detach().cpu().item() if isinstance(effective_pes, torch.Tensor) else effective_pes,
                        "compute_cycles": compute_cycles.detach().cpu().item() if isinstance(compute_cycles, torch.Tensor) else compute_cycles,
                        "bottleneck_memory_cycles": bottleneck_memory_cycles.detach().cpu().item() if isinstance(bottleneck_memory_cycles, torch.Tensor) else bottleneck_memory_cycles,
                        "stall_cycles": stall_cycles.detach().cpu().item() if isinstance(stall_cycles, torch.Tensor) else stall_cycles,
                        "total_cycles": total_cycles.detach().cpu().item() if isinstance(total_cycles, torch.Tensor) else total_cycles,
                        "final_latency_s": latency.detach().cpu().item() if isinstance(latency, torch.Tensor) else latency
                    }
                
                # ===== 2.2 能耗模型实现 (三段式成本核算 - 公式13) =====
                
                energy = torch.tensor(0.0, device=self.config.DEVICE)
                
                # 步骤1: 计算Compute_Energy
                energy_compute = total_macs * self.config.PE_MAC_EPA_PJ
                energy += energy_compute
                
                # 步骤2: 计算Inter-level_Energy (层间能耗)
                energy_inter_level = torch.tensor(0.0, device=self.config.DEVICE)
                
                for interface, accesses_bytes in per_level_accesses.items():
                    lower_level_name = interface.split('_to_')[1]
                    accesses_words = accesses_bytes / self.config.BYTES_PER_ELEMENT
                    
                    # 根据目的地层级查找对应的EPA模型
                    if lower_level_name == 'L0_Registers':
                        epa = self.config.L0_REG_BASE_EPA_PJ
                    elif lower_level_name == 'L1_Accumulator':
                        size_kb = hw_params.get_buffer_size_kb(lower_level_name)
                        num_pes_sqrt = torch.sqrt(num_pes)
                        epa = self.config.L1_ACCUM_BASE_EPA_PJ + self.config.L1_ACCUM_CAPACITY_COEFF_PJ_PER_KB * (size_kb / num_pes_sqrt)
                    elif lower_level_name == 'L2_Scratchpad':
                        size_kb = hw_params.get_buffer_size_kb(lower_level_name)
                        epa = self.config.L2_SPM_BASE_EPA_PJ + self.config.L2_SPM_CAPACITY_COEFF_PJ_PER_KB * size_kb
                    elif lower_level_name == 'L3_DRAM':
                        epa = self.config.L3_DRAM_EPA_PJ
                    else:
                        epa = torch.tensor(0.0, device=self.config.DEVICE)
                    
                    energy_inter_level += accesses_words * epa
                
                energy += energy_inter_level
                
                # 步骤3: 计算Intra-level_Energy (层内能耗)
                energy_intra_level = torch.tensor(0.0, device=self.config.DEVICE)
                
                # 对于L0寄存器，reads和updates次数都等于Total_MACs
                accesses_reads_l0 = total_macs
                accesses_updates_l0 = total_macs
                
                energy_intra_level = (accesses_reads_l0 + accesses_updates_l0) * self.config.L0_REG_BASE_EPA_PJ
                energy += energy_intra_level
                
                # ===== 2.3 面积模型与其他损失项 =====
                
                # 计算缓冲区不匹配损失
                for i, level in enumerate(self.config.MEMORY_HIERARCHY):
                    if level['type'] == 'buffer':
                        required_kb = self.calculate_buffer_req_kb(layer['dims'], all_factors, i)
                        available_kb = hw_params.get_buffer_size_kb(level['name'])
                        buffer_deficit = torch.relu(required_kb - available_kb)
                        level_mismatch_loss = torch.pow(buffer_deficit, 2)
                        total_buffer_mismatch_loss += level_mismatch_loss

            total_latency += latency
            total_energy += energy

        # 面积成本
        area_cost = hw_params.get_area_cost()
        
        # 写入调试数据到JSON文件
        if debug_data is not None and debug_output_path is not None:
            import json
            with torch.no_grad():
                with open(debug_output_path, 'w', encoding='utf-8') as f:
                    json.dump(debug_data, f, indent=4, ensure_ascii=False)
        
        # 确保所有返回值都是标量张量
        total_latency = total_latency.squeeze() if total_latency.dim() > 0 else total_latency
        total_energy = total_energy.squeeze() if total_energy.dim() > 0 else total_energy
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
                            # 安全地处理不同数据类型
                            temporal_factor = factors[dim_name][inner_level_name]['temporal']
                            spatial_factor = factors[dim_name][inner_level_name]['spatial']
                            
                            # 确保因子是张量类型
                            if not isinstance(temporal_factor, torch.Tensor):
                                temporal_factor = torch.tensor(float(temporal_factor), device=self.config.DEVICE)
                            if not isinstance(spatial_factor, torch.Tensor):
                                spatial_factor = torch.tensor(float(spatial_factor), device=self.config.DEVICE)
                            
                            tile_dims[dim_name] = tile_dims[dim_name] * temporal_factor * spatial_factor
            
            tensor_tile_size = reduce(mul, [tile_dims.get(d, torch.tensor(1.0, device=self.config.DEVICE)) for d in tensor_dims if d in dims], torch.tensor(1.0, device=self.config.DEVICE))
            total_buffer_bytes += tensor_tile_size * self.config.BYTES_PER_ELEMENT

        return total_buffer_bytes / 1024.0