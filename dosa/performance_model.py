import torch
import torch.nn as nn
from typing import Tuple
from functools import reduce
from operator import mul

# --- Patch safe_tensor to suppress copy-construct warnings ---
if not hasattr(torch, "_original_tensor"):
    torch._original_tensor = torch.tensor  # type: ignore[attr-defined]
    
    def _safe_tensor(data, *args, **kwargs):
        """Replacement for torch.tensor that avoids copying when the input is already a Tensor."""
        if isinstance(data, torch.Tensor):
            device = kwargs.get("device")
            return data.to(device) if device is not None and data.device != device else data
        return torch._original_tensor(data, *args, **kwargs)  # type: ignore[attr-defined]

    torch.tensor = _safe_tensor  # type: ignore[assignment]
# --- End patch ---

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
        self._group_w_persist_enabled = False
        self._group_w_residency_level = 'L2_Scratchpad'
        self._group_w_first_load_done = False

        # === 2) 在类内新增一个方法（放在类中任意位置即可） ===
    def set_group_weight_residency(self, enabled: bool, residency_level: str = 'L2_Scratchpad'):
        """
        启用/关闭“融合组全权重常驻”模式。
        若 enabled=True，则本 perf_model 实例生命周期内仅第一次对指定 residency_level 的权重填充计入流量，
        后续层的权重填充记为 0（视作已驻留）。
        """
        self._group_w_persist_enabled = bool(enabled)
        self._group_w_residency_level = residency_level
        self._group_w_first_load_done = False

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
    
    def calculate_inter_level_fill_traffic(self, layer_dims: dict, mapping_table: dict, num_pes: torch.Tensor, hw_params: HardwareParameters, debug_data: dict = None) -> dict:
        """
        计算跨层级数据填充流量（Inter-Level Data Fill Traffic）。
        
        基于DATA_SUPPLY_MAP的新架构，遍历所有数据需求并追溯其供给来源，
        计算对应通路上的填充流量，支持跨级数据路径（如L2->L0）。
        
        Args:
            layer_dims: 层的维度信息 {R, S, P, Q, C, K, N}
            mapping_table: 映射表 {dim: {level: {temporal: x, spatial: y}}}
            num_pes: 处理单元数量
            debug_data: 调试数据收集容器
            
        Returns:
            dict: 详细的跨层级填充流量信息，包含按张量分解的流量和驱动因子
                 {
                     'L2_Scratchpad_to_L0_Registers': {
                         'total_bytes': float,
                         'breakdown': {'Weight': float, 'Input': float, 'Output': float},
                         'drivers': {'Weight': {'C_i,t': float, 'Outer_Temporal_Product': float}, ...}
                     },
                     ...
                 }
        """
        # 内存层级定义 (0=Registers, 1=Accumulator, 2=Scratchpad, 3=DRAM)
        memory_levels = ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad', 'L3_DRAM']
        M = len(memory_levels) - 1  # 最高层级索引
        
        # 初始化详细结果字典
        detailed_fill_traffic = {}
        
        # 初始化调试数据收集
        if debug_data is not None:
            debug_data["inter_level_fill_traffic_trace"] = {}
        
        # 基于数据需求的新逻辑：遍历所有存储层级和张量类型
        for i in range(M + 1):  # i = 0, 1, 2, 3
            destination_level_name = memory_levels[i]
            
            # 初始化当前层级的调试数据
            if debug_data is not None:
                debug_data["inter_level_fill_traffic_trace"][f"{destination_level_name} (i={i})"] = {}
            
            # 遍历所有张量类型
            for tensor_type in ['W', 'I']:  # 只处理需要填充的张量
                
                # 资格预审：检查STORAGE_MATRIX，确认层级i是否允许存储张量t
                if not STORAGE_MATRIX[i][tensor_type]:
                    continue
                
                # 映射张量类型到DATA_SUPPLY_MAP中使用的名称
                tensor_name_map = {'W': 'Weight', 'I': 'Input', 'O': 'Output'}
                tensor_name = tensor_name_map[tensor_type]

                # === GROUP RESIDENCY OVERRIDE: 全链权重常驻（只记一次填充） ===
                if self._group_w_persist_enabled and tensor_type == 'W':
                    # 仅对“驻留层”的权重填充做“只记一次”的处理
                    if destination_level_name == self._group_w_residency_level:
                        if self._group_w_first_load_done:
                            # 后续层的权重填充记 0——视作已驻留
                            tensor_fill_bytes = torch.tensor(0.0, device=self.config.DEVICE)
                            tensor_fill_accesses = torch.tensor(0.0, device=self.config.DEVICE)
                        else:
                            # 第一次遇到权重填充：正常计入，并标记完成
                            self._group_w_first_load_done = True
                
                # 查询供给来源：使用DATA_SUPPLY_MAP查询源层级
                if (destination_level_name not in self.config.DATA_SUPPLY_MAP or 
                    tensor_name not in self.config.DATA_SUPPLY_MAP[destination_level_name]):
                    continue
                
                source_level_name = self.config.DATA_SUPPLY_MAP[destination_level_name][tensor_name]
                
                # 跳过PE产生的数据（不需要填充）
                if source_level_name == 'PE':
                    continue
                
                # 获取源层级索引
                try:
                    j = memory_levels.index(source_level_name)
                except ValueError:
                    continue  # 无效的源层级名称
                
                # 计算 C_i,t: 在层级 i 存储的张量 t 的数据块大小
                C_i_t = self._calculate_data_block_size(i, tensor_type, layer_dims, mapping_table)
                
                # 计算最内层活跃层级（用于调试）
                innermost_level = self._find_innermost_level(tensor_type, mapping_table)
                
                # --- 开始重构区域：新的流量计算逻辑 ---
                
                # 计算 C_i_t = _calculate_data_block_size(i, tensor_type, ...)（已修正W/O截断后公式）
                # C_i_t 已在上面计算
                
                if tensor_type == 'W':
                    # 权重张量的新逻辑
                    persist_W = self._can_persist_W_at_level(i, layer_dims, mapping_table, hw_params)
                    tiles = self._tiles_above_for_W(i, layer_dims, mapping_table, persist_W)
                    
                elif tensor_type == 'I':
                    # 输入张量：tiles = ∏_{d∈D_I} ceil(total(d) / coverage_≤i(d))
                    tiles = torch.tensor(1.0, device=self.config.DEVICE)
                    for dim_name in D_I:  # {N, C, P, Q, R, S}
                        if dim_name in layer_dims:
                            total_dim_size = torch.tensor(layer_dims[dim_name], device=self.config.DEVICE)
                            coverage = self._coverage_upto(i, dim_name, mapping_table, layer_dims)
                            coverage = torch.clamp(coverage, min=1.0)  # 防0
                            tiles *= torch.ceil(total_dim_size / coverage)
                else:
                    # 其他张量类型，保持原逻辑
                    tiles = torch.tensor(1.0, device=self.config.DEVICE)
                
                # --- 结束重构区域 ---
                
                # 计算填充流量 = C_i,t × Tiles_above(i)
                tensor_fill_accesses = C_i_t * tiles
                
                # 收集详细的调试数据
                if debug_data is not None:
                    tensor_debug_data = {
                        "C_i,t": C_i_t.detach().cpu().item() if isinstance(C_i_t, torch.Tensor) else C_i_t,
                        "innermost_level": innermost_level,
                        "source_level": source_level_name,
                        "source_level_index": j,
                        "tiles": tiles.detach().cpu().item() if isinstance(tiles, torch.Tensor) else tiles,
                        "tensor_fill_accesses": tensor_fill_accesses.detach().cpu().item() if isinstance(tensor_fill_accesses, torch.Tensor) else tensor_fill_accesses
                    }
                    
                    # 为权重张量添加额外的调试信息
                    if tensor_type == 'W':
                        persist_W = self._can_persist_W_at_level(i, layer_dims, mapping_table, hw_params)
                        tensor_debug_data["persist_W"] = persist_W
                        
                        # 添加coverage快照
                        coverage_snapshot = {}
                        for dim_name in D_W:
                            if dim_name in layer_dims:
                                coverage_snapshot[dim_name] = self._coverage_upto(i, dim_name, mapping_table, layer_dims).detach().cpu().item()
                        tensor_debug_data["coverage_snapshot"] = coverage_snapshot
                    
                    debug_data["inter_level_fill_traffic_trace"][f"{destination_level_name} (i={i})"][tensor_type] = tensor_debug_data
                
                # 构建接口：动态构建接口名称
                interface_name = f"{source_level_name}_to_{destination_level_name}"
                
                # 初始化接口的详细信息结构
                if interface_name not in detailed_fill_traffic:
                    detailed_fill_traffic[interface_name] = {
                        'total_bytes': torch.tensor(0.0, device=self.config.DEVICE),
                        'breakdown': {'Weight': torch.tensor(0.0, device=self.config.DEVICE), 
                                    'Input': torch.tensor(0.0, device=self.config.DEVICE), 
                                    'Output': torch.tensor(0.0, device=self.config.DEVICE)},
                        'drivers': {'Weight': {}, 'Input': {}, 'Output': {}}
                    }
                
                # 计算该张量的填充流量（字节）
                tensor_fill_bytes = tensor_fill_accesses * self.config.BYTES_PER_ELEMENT
                
                # 映射张量类型到标准名称
                tensor_name_map = {'W': 'Weight', 'I': 'Input', 'O': 'Output'}
                tensor_name = tensor_name_map[tensor_type]
                
                # 更新分解信息
                detailed_fill_traffic[interface_name]['breakdown'][tensor_name] += tensor_fill_bytes
                detailed_fill_traffic[interface_name]['total_bytes'] += tensor_fill_bytes
                
                # 更新驱动因子信息
                detailed_fill_traffic[interface_name]['drivers'][tensor_name] = {
                    'C_i,t': C_i_t.detach().cpu().item() if isinstance(C_i_t, torch.Tensor) else C_i_t,
                    'Tiles_above': tiles.detach().cpu().item() if isinstance(tiles, torch.Tensor) else tiles
                }
        
        # 转换所有 torch.Tensor 为 float 以便 JSON 序列化
        result = {}
        for interface_name, interface_data in detailed_fill_traffic.items():
            result[interface_name] = {
                'total_bytes': interface_data['total_bytes'].detach().cpu().item() if isinstance(interface_data['total_bytes'], torch.Tensor) else interface_data['total_bytes'],
                'breakdown': {
                    tensor_name: tensor_bytes.detach().cpu().item() if isinstance(tensor_bytes, torch.Tensor) else tensor_bytes
                    for tensor_name, tensor_bytes in interface_data['breakdown'].items()
                },
                'drivers': interface_data['drivers']  # 驱动因子已经在上面转换为 float
            }
        
        return result
    
    def calculate_inter_level_writeback_traffic(self, layer_dims: dict, mapping_table: dict, num_pes: torch.Tensor, debug_data: dict = None) -> dict:
        """
        计算层级间写回流量（自下而上的数据移动）
        严格按照DOSA论文中的Updates公式实现：Updates_O(i) = Writes_O(i-1) / F_S,O(i)
        
        Args:
            layer_dims: 层维度信息
            mapping_table: 映射表
            num_pes: PE数量
            debug_data: 调试数据字典（可选）
            
        Returns:
            dict: 详细的写回流量信息
                 {
                     'interface_name': {
                         'total_bytes': float,
                         'breakdown': {'Weight': float, 'Input': float, 'Output': float},
                         'drivers': {'Weight': {}, 'Input': {}, 'Output': {}}
                     }
                 }
        """
        memory_levels = ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad', 'L3_DRAM']
        M = len(memory_levels) - 1  # 最高层级索引 (3)
        
        # 初始化详细写回流量字典
        detailed_writeback_traffic = {}
        
        # 只处理输出张量的写回流量
        tensor_type = 'O'
        
        # 初始化调试数据
        if debug_data is not None:
            debug_data["inter_level_writeback_traffic_trace"] = {}
        
        # 获取输出张量的最内层存储级别
        innermost_output_level = self._find_innermost_level(tensor_type, mapping_table)
        
        # 调试信息：记录关键参数
        if debug_data is not None:
            debug_data["writeback_debug_info"] = {
                "innermost_output_level": innermost_output_level,
                "M": M,
                "loop_range": f"range({innermost_output_level + 1}, {M + 1})",
                "storage_matrix_O": [STORAGE_MATRIX[i]['O'] for i in range(len(STORAGE_MATRIX))]
            }
        
        # 实现核心循环：遍历所有可能发生写回的层级
        # 写回是从最内层的上一级开始，一直到最外层
        for i in range(innermost_output_level + 1, M + 1):
            level_name = memory_levels[i]
            
            # 初始化当前层级的调试数据
            if debug_data is not None:
                debug_data["inter_level_writeback_traffic_trace"][f"{level_name} (i={i})"] = {}
            
            # 检查输出张量是否存储在当前层级
            if not STORAGE_MATRIX[i][tensor_type]:
                continue
            
            # 步骤 4.1: 获取代理变量 Writes_O(i-1)
            # 计算i-1层的数据块大小
            C_i_minus_1_O = self._calculate_data_block_size(i - 1, tensor_type, layer_dims, mapping_table)
            
            # 计算需要写入到i-1层的总数据量（作为代理变量）
            writes_O_i_minus_1 = self._calculate_writes(i - 1, tensor_type, layer_dims, mapping_table, C_i_minus_1_O, M)
            
            # 步骤 4.2: 计算空间复用因子 F_S,O(i)
            F_S_O_i = self._calculate_spatial_reuse_factor(i, tensor_type, layer_dims, mapping_table)
            
            # 步骤 4.3: 应用Updates公式
            # Updates_O(i) = Writes_O(i-1) / F_S,O(i)
            updates_O_i = writes_O_i_minus_1 / torch.clamp(F_S_O_i, min=1e-9)
            
            # 收集详细的调试数据
            if debug_data is not None:
                tensor_debug_data = {
                    "proxy_volume_Writes_O(i-1)": writes_O_i_minus_1.detach().cpu().item() if isinstance(writes_O_i_minus_1, torch.Tensor) else writes_O_i_minus_1,
                    "spatial_reuse_F_S_O(i)": F_S_O_i.detach().cpu().item() if isinstance(F_S_O_i, torch.Tensor) else F_S_O_i,
                    "final_updates_O(i)": updates_O_i.detach().cpu().item() if isinstance(updates_O_i, torch.Tensor) else updates_O_i,
                    "innermost_output_level": innermost_output_level,
                    "note": "Implemented using Updates formula: Updates_O(i) = Writes_O(i-1) / F_S,O(i)"
                }
                
                debug_data["inter_level_writeback_traffic_trace"][f"{level_name} (i={i})"][tensor_type] = tensor_debug_data
            
            # 填充返回字典
            # 确定正确的接口名称：从i-1层到i层的写回
            interface_name = f"{memory_levels[i-1]}_to_{memory_levels[i]}"
            
            # 初始化接口的详细信息结构
            if interface_name not in detailed_writeback_traffic:
                detailed_writeback_traffic[interface_name] = {
                    'total_bytes': torch.tensor(0.0, device=self.config.DEVICE),
                    'breakdown': {'Weight': torch.tensor(0.0, device=self.config.DEVICE), 
                                'Input': torch.tensor(0.0, device=self.config.DEVICE), 
                                'Output': torch.tensor(0.0, device=self.config.DEVICE)},
                    'drivers': {'Weight': {}, 'Input': {}, 'Output': {}}
                }
            
            # 将updates_O_i转换为字节
            tensor_writeback_bytes = updates_O_i * self.config.BYTES_PER_ELEMENT
            
            # 更新分解信息（只有输出张量有写回流量）
            detailed_writeback_traffic[interface_name]['breakdown']['Output'] += tensor_writeback_bytes
            detailed_writeback_traffic[interface_name]['total_bytes'] += tensor_writeback_bytes
            
            # 将关键中间值填充到drivers字典中
            detailed_writeback_traffic[interface_name]['drivers']['Output'] = {
                "proxy_volume_Writes_O(i-1)": writes_O_i_minus_1.detach().cpu().item() if isinstance(writes_O_i_minus_1, torch.Tensor) else writes_O_i_minus_1,
                "spatial_reuse_F_S_O(i)": F_S_O_i.detach().cpu().item() if isinstance(F_S_O_i, torch.Tensor) else F_S_O_i,
                "final_updates_O(i)": updates_O_i.detach().cpu().item() if isinstance(updates_O_i, torch.Tensor) else updates_O_i
            }
        
        # 转换所有 torch.Tensor 为 float 以便 JSON 序列化
        result = {}
        for interface_name, interface_data in detailed_writeback_traffic.items():
            result[interface_name] = {
                'total_bytes': interface_data['total_bytes'].detach().cpu().item() if isinstance(interface_data['total_bytes'], torch.Tensor) else interface_data['total_bytes'],
                'breakdown': {
                    tensor_name: tensor_bytes.detach().cpu().item() if isinstance(tensor_bytes, torch.Tensor) else tensor_bytes
                    for tensor_name, tensor_bytes in interface_data['breakdown'].items()
                },
                'drivers': interface_data['drivers']  # 驱动因子已经在上面转换为 float
            }
        
        return result
    
    def calculate_intra_level_consumption_accesses(self, layer_dims: dict, mapping_table: dict, num_pes: torch.Tensor, debug_data: dict = None) -> dict:
        """
        计算层级内部数据消耗访问（Intra-Level Data Consumption Accesses）。
        
        此函数专门负责计算每个存储层级发生的"层级内部"的"数据消耗"访问次数。
        严格遵循论文公式(11)中的路径B (i = innermost_level)，即访问次数与 total_macs 成正比。
        
        Args:
            layer_dims: 层的维度信息 {R, S, P, Q, C, K, N}
            mapping_table: 映射表 {dim: {level: {temporal: x, spatial: y}}}
            num_pes: 处理单元数量
            debug_data: 调试数据收集容器
            
        Returns:
            dict: 嵌套字典，描述每个层级、每种张量的"消耗性"读/写/更新次数
                 {'L2_Scratchpad': {'Input': {'reads': 1.15e8}}, 'L1_Accumulator': {'Output': {'updates': ...}}, ...}
        """
        from dosa.utils import calculate_macs
        
        # 计算总MAC运算次数
        total_macs = calculate_macs(layer_dims)
        
        # 内存层级定义 (0=Registers, 1=Accumulator, 2=Scratchpad, 3=DRAM)
        memory_levels = ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad', 'L3_DRAM']
        M = len(memory_levels) - 1  # 最高层级索引
        
        # 初始化结果字典
        consumption_accesses = {}
        
        # 初始化调试数据收集
        if debug_data is not None:
            debug_data["intra_level_consumption_trace"] = {}
        
        # 为每个内存层级计算消耗性访问
        for i in range(M + 1):  # i = 0, 1, 2, 3
            level_name = memory_levels[i]
            consumption_accesses[level_name] = {}
            
            # 初始化当前层级的调试数据
            if debug_data is not None:
                debug_data["intra_level_consumption_trace"][f"{level_name} (i={i})"] = {}
            
            # 对每种张量类型计算消耗性访问
            for tensor_type in ['W', 'I', 'O']:
                
                # 检查该张量是否存储在当前层级
                if not STORAGE_MATRIX[i][tensor_type]:
                    continue
                
                # 计算最内层活跃层级
                innermost_level = self._find_innermost_level(tensor_type, mapping_table)
                
                # 只有在 innermost_level 才会发生消耗性访问
                if i == innermost_level:
                    # 计算空间复用因子
                    F_S_t_i = self._calculate_spatial_reuse_factor(i, tensor_type, layer_dims, mapping_table)
                    
                    # 初始化张量访问字典
                    tensor_name = {'W': 'Weight', 'I': 'Input', 'O': 'Output'}[tensor_type]
                    consumption_accesses[level_name][tensor_name] = {}
                    
                    if tensor_type in ['W', 'I']:
                        # 对于权重和输入张量，计算消耗性读取
                        consumption_reads = total_macs / torch.clamp(F_S_t_i, min=torch.tensor(1e-9, device=self.config.DEVICE))
                        consumption_accesses[level_name][tensor_name]['reads'] = consumption_reads
                        
                        # 收集调试数据
                        if debug_data is not None:
                            debug_data["intra_level_consumption_trace"][f"{level_name} (i={i})"][tensor_type] = {
                                "F_S,t(i)": F_S_t_i.detach().cpu().item() if isinstance(F_S_t_i, torch.Tensor) else F_S_t_i,
                                "innermost_level": innermost_level,
                                "consumption_reads": consumption_reads.detach().cpu().item() if isinstance(consumption_reads, torch.Tensor) else consumption_reads
                            }
                    
                    elif tensor_type == 'O':
                        # 对于输出张量，计算消耗性更新
                        consumption_updates = total_macs / torch.clamp(F_S_t_i, min=torch.tensor(1e-9, device=self.config.DEVICE))
                        consumption_accesses[level_name][tensor_name]['updates'] = consumption_updates
                        
                        # 收集调试数据
                        if debug_data is not None:
                            debug_data["intra_level_consumption_trace"][f"{level_name} (i={i})"][tensor_type] = {
                                "F_S,t(i)": F_S_t_i.detach().cpu().item() if isinstance(F_S_t_i, torch.Tensor) else F_S_t_i,
                                "innermost_level": innermost_level,
                                "consumption_updates": consumption_updates.detach().cpu().item() if isinstance(consumption_updates, torch.Tensor) else consumption_updates
                            }
        
        # 清理空的层级
        consumption_accesses = {level: tensors for level, tensors in consumption_accesses.items() if tensors}
        
        return consumption_accesses
    
    def _coverage_upto(self, level_idx: int, dim: str, mapping_table: dict, layer_dims: dict) -> torch.Tensor:
        """
        计算维度 dim 在 ≤level_idx 的 temporal×spatial 乘积（并截断不超过 total(dim)）
        
        Args:
            level_idx: 层级索引 i
            dim: 维度名称
            mapping_table: 映射表
            layer_dims: 层维度信息
            
        Returns:
            coverage_≤i(d): 该维度在≤i层级的覆盖范围
        """
        memory_levels = ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad', 'L3_DRAM']
        
        cov = torch.tensor(1.0, device=self.config.DEVICE)
        for j in range(level_idx + 1):  # 0..i
            level_name = memory_levels[j]
            if dim in mapping_table and level_name in mapping_table[dim]:
                temporal_factor = mapping_table[dim][level_name].get('temporal', 1)
                spatial_factor = mapping_table[dim][level_name].get('spatial', 1)
                cov *= torch.tensor(temporal_factor * spatial_factor, device=self.config.DEVICE)
        
        # 关键：截断不超过 total
        if dim in layer_dims:
            total_dim_size = torch.tensor(layer_dims[dim], device=self.config.DEVICE)
            cov = torch.min(cov, total_dim_size)
        
        return torch.max(cov, torch.tensor(1.0, device=self.config.DEVICE))  # 防0
    
    def _can_persist_W_at_level(self, i: int, layer_dims: dict, mapping_table: dict, hw_params: HardwareParameters) -> bool:
        """
        判断权重W是否可以在层级i持久化
        
        条件：
        (A) 映射覆盖整张W：对d∈D_W满足coverage_≤i(d) ≥ total(d)
        (B) 容量允许：W_tile_words + buffer_overheads ≤ capacity_words(i)
        
        Args:
            i: 层级索引
            layer_dims: 层维度信息
            mapping_table: 映射表
            hw_params: 硬件参数
            
        Returns:
            bool: 是否可以持久化
        """
        memory_levels = ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad', 'L3_DRAM']
        
        # 条件A：映射覆盖整张W
        for dim_name in D_W:  # {K, C, R, S}
            if dim_name in layer_dims:
                total_dim_size = torch.tensor(layer_dims[dim_name], device=self.config.DEVICE)
                coverage = self._coverage_upto(i, dim_name, mapping_table, layer_dims)
                if coverage < total_dim_size:
                    return False
        
        # 条件B：容量允许
        level_name = memory_levels[i]
        capacity_words = hw_params.get_buffer_size_kb(level_name) * 1024 / self.config.BYTES_PER_ELEMENT
        
        # W_tile_words = 整张W的word数
        W_tile_words = torch.tensor(1.0, device=self.config.DEVICE)
        for dim_name in D_W:
            if dim_name in layer_dims:
                W_tile_words *= torch.tensor(layer_dims[dim_name], device=self.config.DEVICE)
        
        # 计算buffer_overheads
        input_ws_peak = self._calculate_data_block_size(i, 'I', layer_dims, mapping_table) if STORAGE_MATRIX[i].get('I', False) else torch.tensor(0.0, device=self.config.DEVICE)
        output_ws_peak = self._calculate_data_block_size(i, 'O', layer_dims, mapping_table) if STORAGE_MATRIX[i].get('O', False) else torch.tensor(0.0, device=self.config.DEVICE)
        
        # 双缓冲系数
        buf_mult = 2  # 默认双缓冲
        
        need = W_tile_words + buf_mult * input_ws_peak + output_ws_peak
        
        return need.item() <= capacity_words
    
    def _tiles_above_for_W(self, i: int, layer_dims: dict, mapping_table: dict, persist_W: bool) -> torch.Tensor:
        """
        计算权重W的外层tile数量 Tiles_above(i)
        
        公式：
        Tiles_above(i) = (∏_{d∈D_W} ceil(total/coverage_≤i)) × (persist_W ? 1 : ∏_{d∈{N,P,Q}} ceil(total/coverage_≤i))
        
        Args:
            i: 层级索引
            layer_dims: 层维度信息
            mapping_table: 映射表
            persist_W: 是否可以持久化
            
        Returns:
            torch.Tensor: 外层tile数量
        """
        # 计算与W相关维的tiles
        tiles_dep = torch.tensor(1.0, device=self.config.DEVICE)
        for dim_name in D_W:  # {K, C, R, S}
            if dim_name in layer_dims:
                total_dim_size = torch.tensor(layer_dims[dim_name], device=self.config.DEVICE)
                coverage = self._coverage_upto(i, dim_name, mapping_table, layer_dims)
                coverage = torch.clamp(coverage, min=1.0)  # 防0
                tiles_dep *= torch.ceil(total_dim_size / coverage)
        
        # 计算与W无关的维的tiles（N, P, Q）
        tiles_indep = torch.tensor(1.0, device=self.config.DEVICE)
        for dim_name in ['N', 'P', 'Q']:
            if dim_name in layer_dims:
                total_dim_size = torch.tensor(layer_dims[dim_name], device=self.config.DEVICE)
                coverage = self._coverage_upto(i, dim_name, mapping_table, layer_dims)
                coverage = torch.clamp(coverage, min=1.0)  # 防0
                tiles_indep *= torch.ceil(total_dim_size / coverage)
        
        return tiles_dep if persist_W else tiles_dep * tiles_indep

    def _calculate_data_block_size(self, i: int, tensor_type: str, layer_dims: dict, mapping_table: dict) -> torch.Tensor:
        """
        计算在层级 i 必须存储的张量 tensor_type 的数据块大小 C_i,t (以word为单位)
        严格按照论文公式实现：Equations 2, 3, 4
        修改：W/O分支加入截断逻辑，确保不超过总尺寸
        """
        memory_levels = ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad', 'L3_DRAM']
        
        if tensor_type == 'W':
            # C_i,W (修改后): ∏_{d∈D_W} min(total(d), coverage_≤i(d))
            size = torch.tensor(1.0, device=self.config.DEVICE)
            relevant_dims = D_W  # {R, S, C, K}
            
            for dim_name in relevant_dims:
                if dim_name in layer_dims:
                    total_dim_size = torch.tensor(layer_dims[dim_name], device=self.config.DEVICE)
                    coverage = self._coverage_upto(i, dim_name, mapping_table, layer_dims)
                    size *= torch.min(total_dim_size, coverage)
            
            return size
            
        elif tensor_type == 'O':
            # C_i,O (修改后): ∏_{d∈D_O} min(total(d), coverage_≤i(d))
            size = torch.tensor(1.0, device=self.config.DEVICE)
            relevant_dims = D_O  # {P, Q, K, N}
            
            for dim_name in relevant_dims:
                if dim_name in layer_dims:
                    total_dim_size = torch.tensor(layer_dims[dim_name], device=self.config.DEVICE)
                    coverage = self._coverage_upto(i, dim_name, mapping_table, layer_dims)
                    size *= torch.min(total_dim_size, coverage)
            
            return size
            
        elif tensor_type == 'I':
            # C_i,I (Equation 3 for Inputs - 最复杂):
            # Inner(i, d) = Π(f_k,j,d) (其中 k,j 范围同 C_i,W)
            # C_i,I = Π_{d∈{C,N}} (f_k,j,d) × (P_stride × (Inner(i,P) - 1) + Inner(i,R)) × (Q_stride × (Inner(i,Q) - 1) + Inner(i,S))
            
            def calculate_inner(dim_name):
                """计算 Inner(i, d) - 某个维度 d 在层级 i 及其内层的总 tile size"""
                inner_size = torch.tensor(1.0, device=self.config.DEVICE)
                for j in range(i + 1):
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
            for j in range(i + 1):
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
        F_S,t(i) = Π(f_S,0,d) 其中 d ∈ (D - D_t) (与张量 t 无关的维度)
        
        重要修正：空间并行性是由PE阵列的物理结构决定的全局效应，
        其大小由最内层L0_Registers的spatial映射唯一决定，
        并且这个效应会影响所有为PE阵列供给数据的上层存储。
        """
        F_S = torch.tensor(1.0, device=self.config.DEVICE)
        
        # 获取与该张量无关的维度 (D - D_t)
        relevant_dims = TENSOR_DIMS[tensor_type]
        irrelevant_dims = D_ALL - relevant_dims
        
        # 修正：空间映射的唯一来源层级 - 始终从最内层L0_Registers查找
        SPATIAL_MAPPING_LEVEL = 'L0_Registers'
        
        # 对所有与张量无关的维度，累乘空间映射因子
        for dim_name in irrelevant_dims:
            if dim_name in layer_dims and dim_name in mapping_table and SPATIAL_MAPPING_LEVEL in mapping_table[dim_name]:
                # 修正：始终在固定的SPATIAL_MAPPING_LEVEL中查找spatial因子
                spatial_factor = mapping_table[dim_name][SPATIAL_MAPPING_LEVEL].get('spatial', 1)
                F_S *= torch.tensor(spatial_factor, device=self.config.DEVICE)
        
        return F_S
    
    def _find_innermost_level(self, tensor_type: str, mapping_table: dict) -> int:
        """
        找到张量 tensor_type 的最内存储层级
        现在集成了 STORAGE_MATRIX 物理约束检查，确保返回的层级是物理上合法的存储位置
        """
        # --- 开始重构 ---
        # 根据论文(表4)的设定，强制输入张量(I)的最内层存储为 L2 Scratchpad (index=2)
        if tensor_type == 'I':
            return 2
        
        memory_levels = ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad', 'L3_DRAM']
        relevant_dims = TENSOR_DIMS[tensor_type]
        
        # 步骤 2: 核心循环，增加了"资格预审"机制
        for i in range(len(memory_levels)):
            
            # 【新增的核心逻辑】: 首先检查当前层级 i 是否合法
            is_legal_level = STORAGE_MATRIX[i].get(tensor_type, 0)
            if not is_legal_level:
                continue  # 如果不合法，直接跳过本轮循环
            
            # 如果合法，才继续进行后续的映射检查
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
                return i  # 找到了第一个有分块行为的"合法"层级
        
        # 步骤 3: 【新增的健壮默认值逻辑】
        # 如果遍历完所有合法层级都未找到分块行为，则返回第一个合法的层级索引
        for i in range(len(memory_levels)):
            if STORAGE_MATRIX[i].get(tensor_type, 0):
                return i  # 返回第一个允许存储该张量的层级索引
        
        # 步骤 4: 【新增的异常处理】
        raise ValueError(f"CRITICAL ERROR: No valid storage level found in STORAGE_MATRIX for tensor type: {tensor_type}")
        # --- 结束重构 ---

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

    def evaluate_group_depth_first(self, group_layers: list, graph, hw_params: HardwareParameters,
                                   mapping: FineGrainedMapping, direct_mapping_table: dict = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate a fusion group using depth-first scheduling and weight residency.

        This simplified model assumes:
        * All intermediate activations stay on-chip (depth-first execution).
        * Weights for all layers are loaded once if they fit in L2.

        Args:
            group_layers: List of layer names in the fusion group.
            graph: Computation graph providing layer metadata.
            hw_params: Hardware parameters instance.
            mapping: Mapping parameters (unused in simplified model).
            direct_mapping_table: Optional mapping table (unused).

        Returns:
            Tuple of (latency, energy, area_cost, buffer_mismatch_loss, compatibility_penalty).
        """

        # Gather layer information
        layers = [graph.layers[name] for name in group_layers]
        # Update mapping info for partial-sum detection
        all_factors = mapping.get_all_factors()

        # === 二道闸：禁止 partial sum ===
        if hasattr(mapping, 'has_partial_sums') and mapping.has_partial_sums():
            big = torch.tensor(1e6, device=self.config.DEVICE)
            return big, big, big, big, big

        # Compute total MACs across the group
        from dosa.utils import calculate_macs
        total_macs = torch.tensor(0.0, device=self.config.DEVICE)
        total_weight_bytes = torch.tensor(0.0, device=self.config.DEVICE)

        for layer in layers:
            total_macs += calculate_macs(layer['dims'])
            # weight size
            weight_elems = torch.tensor(1.0, device=self.config.DEVICE)
            for d in D_W:
                weight_elems *= torch.tensor(layer['dims'].get(d, 1), device=self.config.DEVICE)
            total_weight_bytes += weight_elems * self.config.BYTES_PER_ELEMENT

        # Determine if weights can stay resident in L2
        l2_bytes = hw_params.get_buffer_size_kb('L2_Scratchpad') * 1024
        if total_weight_bytes > l2_bytes:
            reload_factor = torch.ceil(total_weight_bytes / l2_bytes)
            weight_bytes_to_load = total_weight_bytes * reload_factor
        else:
            weight_bytes_to_load = total_weight_bytes

        # First input and final output bytes
        first_layer = layers[0]
        last_layer = layers[-1]
        input_elems = torch.tensor(1.0, device=self.config.DEVICE)
        for d in D_I:
            input_elems *= torch.tensor(first_layer['dims'].get(d, 1), device=self.config.DEVICE)
        output_elems = torch.tensor(1.0, device=self.config.DEVICE)
        for d in D_O:
            output_elems *= torch.tensor(last_layer['dims'].get(d, 1), device=self.config.DEVICE)

        input_bytes = input_elems * self.config.BYTES_PER_ELEMENT
        output_bytes = output_elems * self.config.BYTES_PER_ELEMENT

        # Handle extra traffic if partial sums exist
        extra_output_bytes = torch.tensor(0.0, device=self.config.DEVICE)
        partial_penalty = torch.tensor(0.0, device=self.config.DEVICE)
        if mapping.has_partial_sums():
            for tiles in mapping.get_partial_sum_tiles().values():
                extra_output_bytes += output_bytes * (tiles - 1) * 2
                partial_penalty += tiles - 1

        # Total DRAM traffic (weights + first input + final output + partial-sum overhead)
        dram_bytes = weight_bytes_to_load + input_bytes + output_bytes + extra_output_bytes

        num_pes = hw_params.get_projected_num_pes()

        # Compute cycles
        compute_cycles = total_macs / torch.clamp(num_pes, min=torch.tensor(1.0, device=num_pes.device))

        # Memory cycles based on DRAM bandwidth
        bandwidth = calculate_bandwidth_bytes_per_cycle('L3_DRAM', num_pes, self.config)
        memory_cycles = dram_bytes / torch.clamp(bandwidth, min=torch.tensor(1e-9, device=bandwidth.device))

        total_cycles = torch.max(compute_cycles, memory_cycles)
        latency = total_cycles / (self.config.CLOCK_FREQUENCY_MHZ * 1e6)

        # Energy estimation (compute + DRAM)
        compute_energy = total_macs * self.config.PE_MAC_EPA_PJ
        dram_energy = (dram_bytes / self.config.BYTES_PER_ELEMENT) * self.config.L2_SPM_BASE_EPA_PJ
        energy = compute_energy + dram_energy

        area_cost = hw_params.get_area_cost()

        zero = torch.tensor(0.0, device=self.config.DEVICE)
        return latency.squeeze(), energy.squeeze(), area_cost, zero, partial_penalty

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
                "detailed_traffic_formula_trace": {},
                "inter_level_fill_traffic_trace": {},
                "inter_level_writeback_traffic_trace": {}
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
                
                # 步骤3: 计算Memory_Cycles（基于双向跨层级数据流量）
                # 分别调用填充和写回流量计算引擎
                detailed_fill_traffic_info = self.calculate_inter_level_fill_traffic(layer['dims'], all_factors, num_pes, hw_params, debug_data)
                detailed_writeback_traffic_info = self.calculate_inter_level_writeback_traffic(layer['dims'], all_factors, num_pes, debug_data)
                
                memory_cycles_list = []
                for interface, interface_info in detailed_fill_traffic_info.items():
                    upper_level_name = interface.split('_to_')[0]
                    
                    # 从详细信息中提取总流量
                    fill_bytes_total = interface_info['total_bytes']
                    
                    # 计算该接口的带宽（bytes/cycle）
                    bandwidth_bytes_per_cycle = calculate_bandwidth_bytes_per_cycle(upper_level_name, num_pes, self.config)
                    
                    # 计算该接口的内存周期数（仅基于填充流量）
                    memory_cycles = fill_bytes_total / (bandwidth_bytes_per_cycle + torch.tensor(1e-9, device=self.config.DEVICE))
                    memory_cycles_list.append(memory_cycles)
                    
                    # 收集增强的内存接口分析数据
                    if debug_data is not None:
                        debug_data["memory_interface_analysis"][interface] = {
                            "fill_bytes_total": fill_bytes_total,
                            "bandwidth_bytes_per_cycle": bandwidth_bytes_per_cycle.detach().cpu().item() if isinstance(bandwidth_bytes_per_cycle, torch.Tensor) else bandwidth_bytes_per_cycle,
                            "memory_cycles": memory_cycles.detach().cpu().item() if isinstance(memory_cycles, torch.Tensor) else memory_cycles,
                            "fill_bytes_breakdown": interface_info['breakdown'],
                            "traffic_drivers": interface_info['drivers']
                        }
                
                # 处理写回流量的内存周期计算
                for interface, interface_info in detailed_writeback_traffic_info.items():
                    lower_level_name = interface.split('_to_')[0]  # 对于写回，源层级是下层
                    
                    # 从详细信息中提取总流量
                    writeback_bytes_total = interface_info['total_bytes']
                    
                    # 计算该接口的带宽（bytes/cycle）- 使用源层级的带宽
                    bandwidth_bytes_per_cycle = calculate_bandwidth_bytes_per_cycle(lower_level_name, num_pes, self.config)
                    
                    # 计算该接口的内存周期数（基于写回流量）
                    memory_cycles = writeback_bytes_total / (bandwidth_bytes_per_cycle + torch.tensor(1e-9, device=self.config.DEVICE))
                    memory_cycles_list.append(memory_cycles)
                    
                    # 收集写回流量的内存接口分析数据
                    if debug_data is not None:
                        debug_data["memory_interface_analysis"][interface] = {
                            "writeback_bytes_total": writeback_bytes_total,
                            "bandwidth_bytes_per_cycle": bandwidth_bytes_per_cycle.detach().cpu().item() if isinstance(bandwidth_bytes_per_cycle, torch.Tensor) else bandwidth_bytes_per_cycle,
                            "memory_cycles": memory_cycles.detach().cpu().item() if isinstance(memory_cycles, torch.Tensor) else memory_cycles,
                            "writeback_bytes_breakdown": interface_info['breakdown'],
                            "traffic_drivers": interface_info['drivers']
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
                
                # 步骤2: 计算Inter-level_Energy (双向层间数据流能耗)
                energy_inter_level = torch.tensor(0.0, device=self.config.DEVICE)
                
                # 步骤2.1: 计算填充能耗 (下行车道: W和I张量)
                energy_fill = torch.tensor(0.0, device=self.config.DEVICE)
                for interface, interface_info in detailed_fill_traffic_info.items():
                    lower_level_name = interface.split('_to_')[1]  # 目的地层级
                    fill_bytes = interface_info['total_bytes']
                    fill_words = fill_bytes / self.config.BYTES_PER_ELEMENT
                    
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
                    
                    energy_fill += fill_words * epa
                
                # 步骤2.2: 计算写回能耗 (上行车道: O张量)
                energy_writeback = torch.tensor(0.0, device=self.config.DEVICE)
                for interface, interface_info in detailed_writeback_traffic_info.items():
                    upper_level_name = interface.split('_to_')[1]  # 目的地层级（上层存储）
                    writeback_bytes = interface_info['total_bytes']
                    writeback_words = writeback_bytes / self.config.BYTES_PER_ELEMENT
                    
                    # 根据目的地层级（上层存储）查找对应的EPA模型
                    if upper_level_name == 'L0_Registers':
                        epa = self.config.L0_REG_BASE_EPA_PJ
                    elif upper_level_name == 'L1_Accumulator':
                        size_kb = hw_params.get_buffer_size_kb(upper_level_name)
                        num_pes_sqrt = torch.sqrt(num_pes)
                        epa = self.config.L1_ACCUM_BASE_EPA_PJ + self.config.L1_ACCUM_CAPACITY_COEFF_PJ_PER_KB * (size_kb / num_pes_sqrt)
                    elif upper_level_name == 'L2_Scratchpad':
                        size_kb = hw_params.get_buffer_size_kb(upper_level_name)
                        epa = self.config.L2_SPM_BASE_EPA_PJ + self.config.L2_SPM_CAPACITY_COEFF_PJ_PER_KB * size_kb
                    elif upper_level_name == 'L3_DRAM':
                        epa = self.config.L3_DRAM_EPA_PJ
                    else:
                        epa = torch.tensor(0.0, device=self.config.DEVICE)
                    
                    energy_writeback += writeback_words * epa
                
                # 汇总双向层间能耗
                energy_inter_level = energy_fill + energy_writeback
                energy += energy_inter_level
                
                # 步骤3: 计算Intra-level_Energy (层级内部消耗能耗)
                energy_intra_level = torch.tensor(0.0, device=self.config.DEVICE)
                
                # 调用层级内部消耗访问计算引擎
                intra_level_consumption_accesses = self.calculate_intra_level_consumption_accesses(layer['dims'], all_factors, num_pes, debug_data)
                
                for level_name, tensors in intra_level_consumption_accesses.items():
                    # 根据层级名称获取对应的EPA
                    if level_name == 'L0_Registers':
                        epa = self.config.L0_REG_BASE_EPA_PJ
                    elif level_name == 'L1_Accumulator':
                        size_kb = hw_params.get_buffer_size_kb(level_name)
                        num_pes_sqrt = torch.sqrt(num_pes)
                        epa = self.config.L1_ACCUM_BASE_EPA_PJ + self.config.L1_ACCUM_CAPACITY_COEFF_PJ_PER_KB * (size_kb / num_pes_sqrt)
                    elif level_name == 'L2_Scratchpad':
                        size_kb = hw_params.get_buffer_size_kb(level_name)
                        epa = self.config.L2_SPM_BASE_EPA_PJ + self.config.L2_SPM_CAPACITY_COEFF_PJ_PER_KB * size_kb
                    elif level_name == 'L3_DRAM':
                        epa = self.config.L3_DRAM_EPA_PJ
                    else:
                        epa = torch.tensor(0.0, device=self.config.DEVICE)
                    
                    # 累加该层级所有张量的所有操作的能耗
                    for tensor_type, operations in tensors.items():
                        for op_type, count in operations.items():
                            energy_intra_level += count * epa
                
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