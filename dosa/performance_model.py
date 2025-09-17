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

def debug_grad(target_var, mapping, path="L2_Scratchpad.R.spatial", var_name="target_var", fusion_params=None):
    try:
        # Recursively get parameter
        target = mapping.factors
        for p in path.split("."):
            target = getattr(target, p)
    except AttributeError:
        print(f"[GRADIENT DEBUG] Path not found: {path}")
        return
    
    if not getattr(target, "requires_grad", False):
        print(f"[GRADIENT DEBUG] {path} does not require gradient (requires_grad=False)")
        return

    grad = torch.autograd.grad(target_var, target,
                               retain_graph=True, allow_unused=True)[0]
    if grad is None:
        print(f"[GRADIENT DEBUG] {path} gradient is None - may not be involved in computation")
    else:
        print(f"[GRADIENT DEBUG] ∂({var_name})/∂({path}) = {grad.item()}")
        print(f"[GRADIENT DEBUG] {path}.value = {target.item()}")
        print(f"[GRADIENT DEBUG] {var_name} = {target_var.item()}")
    
    # Debug gradients for fusion parameters if provided
    if fusion_params is not None:
        for name, param in fusion_params.items():
            if param.requires_grad:
                grad = torch.autograd.grad(target_var, param,
                                       retain_graph=True, allow_unused=True)[0]
                if grad is not None:
                    print(f"[GRADIENT DEBUG] ∂({var_name})/∂(fusion.{name}) = {grad.item()}")
                    print(f"[GRADIENT DEBUG] fusion.{name}.value = {param.item()}")


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

def format_mapping_as_all_factors(mapping):
    """
    将 mapping.factors 转换为 all_factors 的嵌套结构，
    但保持原始数值（不做乘积/累积转换）。
    """
    formatted = {}

    # 遍历每一层级
    for level_name, dims in mapping.factors.items():
        for dim_name, factor_dict in dims.items():
            if dim_name not in formatted:
                formatted[dim_name] = {}

            # 拿到 spatial/temporal，没有的话默认 1.0
            spatial_val = factor_dict.get("spatial", 1.0)
            temporal_val = factor_dict.get("temporal", 1.0)

            # 转成 torch.Tensor，保持和 get_all_factors 输出一致的风格
            if not isinstance(spatial_val, torch.Tensor):
                spatial_val = torch.tensor(float(spatial_val), device="cuda:0")
            if not isinstance(temporal_val, torch.Tensor):
                temporal_val = torch.tensor(float(temporal_val), device="cuda:0")

            formatted[dim_name][level_name] = {
                "spatial": spatial_val,
                "temporal": temporal_val,
            }

    return formatted


class HighFidelityPerformanceModel(nn.Module):
    """
    NEW: 高精度性能模型，能够处理多级存储和细粒度映射。
    """
    def __init__(self, config: Config, debug_latency: bool = False, fusion_aware: bool = True):
        super().__init__()
        self.config = config
        self.debug_latency = debug_latency
        self.fusion_aware = fusion_aware
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
        启用/关闭"融合组全权重常驻"模式。
        若 enabled=True，则本 perf_model 实例生命周期内仅第一次对指定 residency_level 的权重填充计入流量，
        后续层的权重填充记为 0（视作已驻留）。
        """
        self._group_w_persist_enabled = bool(enabled)
        self._group_w_residency_level = residency_level
        self._group_w_first_load_done = False

    def compute_invalid_penalty(self, mapping: nn.Module) -> torch.Tensor:
        """
        Compute penalty to ensure tiling factors >= 1.
        This keeps optimization smooth (no hard clamp).
        """
        invalid_mapping_loss = torch.tensor(0.0, device=self.config.DEVICE)
        for name, p in mapping.named_parameters():
            real_val = torch.exp(p)  # log-param → real factor
            invalid_mapping_loss += torch.square(torch.clamp(1 - real_val, min=0)).sum()
        return invalid_mapping_loss


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
            for tensor_type in ['W', 'I','O']:  # 只处理需要填充的张量
                
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
                    tiles = self._tiles_above_for_W(i, layer_dims, mapping_table, persist_W=False)

                elif tensor_type == 'I':
                    tiles = self._tiles_above_for_I(i, layer_dims, mapping_table)

                elif tensor_type == 'O':
                    tiles = self._tiles_above_for_O(i, layer_dims, mapping_table)

                else:
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

                # debug
                # debug_grad(tensor_fill_bytes, mapping, "L2_Scratchpad.S.temporal")
                
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
        
        result = {}
        for interface_name, interface_data in detailed_fill_traffic.items():
            result[interface_name] = {
                'total_bytes': interface_data['total_bytes'],      # 保持 Tensor
                'breakdown': interface_data['breakdown'],          # 保持 Tensor
                'drivers': interface_data['drivers']               # drivers 可以是 float
            }

        # 如果需要序列化
        if debug_data is not None:
            debug_data["writeback_serialized"] = {
                name: {
                    'total_bytes': float(data['total_bytes'].detach().cpu().item()),
                    'breakdown': {k: float(v.detach().cpu().item()) for k, v in data['breakdown'].items()},
                    'drivers': data['drivers']
                }
                for name, data in result.items()
            }

        return result

    def calculate_inter_level_writeback_traffic(self, layer_dims: dict, mapping_table: dict, num_pes: torch.Tensor, debug_data: dict = None) -> dict:
        """
        在本口径下：updates = 从 PEs (L0) 写入到 L1_Accumulator 的 O 流量。
        公式：Updates_O(L1) = MACs / F_{S,O}(L1)，其中 F_{S,O}(L1) = ∏_{d∈{C,R,S}} f_S(L1, d)
        字节 = Updates_O(L1) * bytes_per_O
        """
        dev = self.config.DEVICE

        from dosa.utils import calculate_macs
            
        # 计算总MAC运算次数
        macs = calculate_macs(layer_dims)

        # 2) 计算 F_{S,O}(L1) —— 只看与 O 无关的归约维 {C,R,S} 在 L1 的空间因子
        L1 = 'L1_Accumulator'
        red_dims = ('C')
        # Initialize spatial reuse factor for output tensor at L1 level
        # Get spatial factor for C dimension at L0 level
        F_S_O_L0 = mapping_table.get('C', {}).get('L0_Registers', {}).get('spatial', 1.0)
        if not isinstance(F_S_O_L0, torch.Tensor):
            F_S_O_L0 = torch.tensor(float(F_S_O_L0), device=dev)
        F_S_O_L0 = torch.clamp(F_S_O_L0, min=1.0)

        # 3) Updates_O(L1) 与写入字节
        updates_L1 = macs / torch.clamp(F_S_O_L0, min=1.0)
        bytes_per_O = self.config.BYTES_PER_ELEMENT  # 若有 per-tensor 精度，替成 per-O
        write_bytes = updates_L1 * bytes_per_O

        # 4) 组织结果（接口命名：PE 到 L1）
        interface_name = 'PE_to_L1_Accumulator'
        result = {
            interface_name: {
                'total_bytes': write_bytes,
                'breakdown': {
                    'Weight': torch.tensor(0.0, device=dev),
                    'Input':  torch.tensor(0.0, device=dev),
                    'Output': write_bytes,
                },
                'drivers': {
                    'Output': {
                        'MACs': float(macs),
                        'F_S_O(L1)': float(F_S_O_L0),
                        'Updates_O(L1)': float(updates_L1),
                        'note': 'updates = PE→L1 partial-sum writes; discount only from spatial on {C,R,S} at L1'
                    }
                }
            }
        }

        # 可选：调试信息
        if debug_data is not None:
            debug_data['updates_pe_to_l1'] = {
                'MACs': float(macs),
                'F_S_O_L1': float(F_S_O_L0),
                'Updates_O_L1': float(updates_L1),
                'bytes': float(write_bytes)
            }

        return result

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
        Tiles_above(i, W) —— DOSA 口径（基线）：
            只乘“i 之外各层”的【时间分块因子】，
            只乘 W 自身的索引维 D_W = {K, C, R, S}，
            不乘空间因子；不把 {N, P, Q} 乘进来；
            不用 ceil(total/coverage)。

        说明：
        - 为保持兼容性保留 persist_W 参数，但此处不使用（忽略）。
        - 不依赖 loop order；仅依赖 mapping_table 中各层的 temporal 因子。
        """
        dev = self.config.DEVICE
        D_W = ('K', 'C', 'R', 'S')

        # Fixed memory hierarchy order from inner to outer levels
        levels_order = ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad', 'L3_DRAM']

        tiles = torch.tensor(1.0, device=dev)
        # 只乘“i 的外层(k>i)”的时间因子
        for k in range(i + 1, len(levels_order)):
            lvl = levels_order[k]
            for d in D_W:
                tf = mapping_table.get(d, {}).get(lvl, {}).get('temporal', 1.0)
                if not isinstance(tf, torch.Tensor):
                    tf = torch.tensor(float(tf), device=dev)
                tiles = tiles * torch.clamp(tf, min=1.0)

        return tiles

    def _tiles_above_for_I(self, i: int, layer_dims: dict, mapping_table: dict) -> torch.Tensor:
        """
        Tiles_above(i, I) —— DOSA 基线：
        只乘“i 之外各层”的【时间因子】；
        只乘 I 自身索引维 D_I = {N, C, P, Q}；
        不乘空间因子；不乘 R/S；不用 ceil(total/coverage)。
        """
        dev = self.config.DEVICE
        D_I = ('N', 'C', 'P', 'Q')

        levels_order = ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad', 'L3_DRAM']  #['L0_Registers','L1_Accumulator','L2_Scratchpad','L3_DRAM']

        tiles = torch.tensor(1.0, device=dev)
        for lvl in levels_order[i+1:]:  # 只看外层
            for d in D_I:
                tf = mapping_table.get(d, {}).get(lvl, {}).get('temporal', 1.0)
                if not isinstance(tf, torch.Tensor):
                    tf = torch.tensor(float(tf), device=dev)
                tiles = tiles * torch.clamp(tf, min=1.0)
        return tiles

    def _tiles_above_for_O(self, i: int, layer_dims: dict, mapping_table: dict) -> torch.Tensor:
        """
        Tiles_above(i, O) —— 基线（fill==write 的口径下）：
        只乘“i 之外各层”的【时间因子】；
        只乘 O 的索引维 D_O = {N, K, P, Q}；
        不乘空间因子；不做广播/累加折扣；不使用 ceil(total/coverage)。
        """
        dev = self.config.DEVICE
        D_O = ('N', 'K', 'P', 'Q')
        levels_order = ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad', 'L3_DRAM']  #['L0_Registers','L1_Accumulator','L2_Scratchpad','L3_DRAM']

        tiles = torch.tensor(1.0, device=dev)
        for lvl in levels_order[i+1:]:  # 只看 i 的外层
            for d in D_O:
                tf = mapping_table.get(d, {}).get(lvl, {}).get('temporal', 1.0)
                if not isinstance(tf, torch.Tensor):
                    tf = torch.tensor(float(tf), device=dev)
                tiles = tiles * torch.clamp(tf, min=1.0)
        return tiles

    def calculate_inter_level_read_traffic(
        self,
        layer_dims: dict,
        mapping_table: dict,
        num_pes: torch.Tensor,
        hw_params: "HardwareParameters",
        debug_data: dict = None,
    ) -> dict:
        """
        计算跨层级读取（read）流量。路径按你的口径固定：
        - Input:  L3->L2, L2->PE    （L2->PE 有广播，独立维 K）
        - Weight: L3->L2, L2->L0, L0->PE （L2->L0 与 L0->PE 有广播，独立维 C,K）
        - Output: L1->PE            （L1->PE 有广播，独立维 C；基数按 MACs）

        返回结构与 fill 一致：
        {
            'src_to_dst': {
            'total_bytes': Tensor,
            'breakdown': {'Weight': Tensor, 'Input': Tensor, 'Output': Tensor},
            'drivers': {'Weight': {...}, 'Input': {...}, 'Output': {...}}
            }, ...
        }
        """
        dev = self.config.DEVICE
        memory_levels = ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad', 'L3_DRAM']
        # 便于复用的 BYTES（如有 per-tensor 精度，这里可改成 per-tensor）
        bytes_per_elem = self.config.BYTES_PER_ELEMENT

        def _bcast_factor(tensor_type: str, mapping_table: dict, dev: torch.device) -> torch.Tensor:
            """
            广播折扣因子：统一从 L0_Registers 层获取无关维度的 spatial 因子。
            """
            if tensor_type == 'I':
                dims = ('K',)
            elif tensor_type == 'W':
                dims = ('C', 'K')
            elif tensor_type == 'O':
                dims = ('C',)
            else:
                dims = ()

            f = torch.tensor(1.0, device=dev)
            for d in dims:
                s = mapping_table.get(d, {}).get('L0_Registers', {}).get('spatial', 1.0)
                if not isinstance(s, torch.Tensor):
                    s = torch.tensor(float(s), device=dev)
                f = f * torch.clamp(s, min=1.0)

            return f

        def _add(interface_name: str, tname: str, elems: torch.Tensor, drivers: dict):
            """把某接口、某张量类型的元素量（elems）累加到结果字典里。"""
            if interface_name not in detailed_read_traffic:
                detailed_read_traffic[interface_name] = {
                    'total_bytes': torch.tensor(0.0, device=dev),
                    'breakdown': {
                        'Weight': torch.tensor(0.0, device=dev),
                        'Input':  torch.tensor(0.0, device=dev),
                        'Output': torch.tensor(0.0, device=dev),
                    },
                    'drivers': {'Weight': {}, 'Input': {}, 'Output': {}}
                }
            bytes_ = elems * bytes_per_elem
            detailed_read_traffic[interface_name]['breakdown'][tname] += bytes_
            detailed_read_traffic[interface_name]['total_bytes'] += bytes_
            detailed_read_traffic[interface_name]['drivers'][tname] = drivers

        # === 结果容器 & 调试容器 ===
        detailed_read_traffic = {}
        if debug_data is not None:
            debug_data["inter_level_read_traffic_trace"] = {}

        from dosa.utils import calculate_macs
        total_macs = calculate_macs(layer_dims)


        # ---------- Input Reads ----------
        # I: L3 -> L2 （无广播）
        i = memory_levels.index('L2_Scratchpad')
        C_i_I = self._calculate_data_block_size(i, 'I', layer_dims, mapping_table)
        tiles_i_I = self._tiles_above_for_I(i, layer_dims, mapping_table)
        base_I_L3_to_L2 = C_i_I * tiles_i_I  # 元素计
        _add(
            'L3_DRAM_to_L2_Scratchpad',
            'Input',
            base_I_L3_to_L2,
            {
                'C_i,t': float(C_i_I.item()) if isinstance(C_i_I, torch.Tensor) else C_i_I,
                'Tiles_above': float(tiles_i_I.item()) if isinstance(tiles_i_I, torch.Tensor) else tiles_i_I,
                'broadcast_factor': 1.0,
                'note': 'Input read L3->L2 (no broadcast)'
            }
        )

        # I: L2 -> PE  （有广播，独立维 K）
        bcast_I_L2 = _bcast_factor('I', mapping_table, dev)
        read_I_L2_to_PE = total_macs  / torch.clamp(bcast_I_L2, min=1.0)
        _add(
            'L2_Scratchpad_to_PE',
            'Input',
            read_I_L2_to_PE,
            {
                'base_elements(L2)': float(base_I_L3_to_L2.item()) if isinstance(base_I_L3_to_L2, torch.Tensor) else base_I_L3_to_L2,
                'broadcast_factor(K@L2)': float(bcast_I_L2.item()),
                'note': 'Input read L2->PE with broadcast over K'
            }
        )

        # ---------- Weight Reads ----------
        # W: L3 -> L2 （无广播）
        i = memory_levels.index('L2_Scratchpad')
        C_i_W = self._calculate_data_block_size(i, 'W', layer_dims, mapping_table)
        tiles_i_W = self._tiles_above_for_W(i, layer_dims, mapping_table, persist_W=False)
        base_W_L3_to_L2 = C_i_W * tiles_i_W
        _add(
            'L3_DRAM_to_L2_Scratchpad',
            'Weight',
            base_W_L3_to_L2,
            {
                'C_i,t': float(C_i_W.item()) if isinstance(C_i_W, torch.Tensor) else C_i_W,
                'Tiles_above': float(tiles_i_W.item()) if isinstance(tiles_i_W, torch.Tensor) else tiles_i_W,
                'broadcast_factor': 1.0,
                'note': 'Weight read L3->L2 (no broadcast)'
            }
        )

        # W: L2 -> L0 （有广播，独立维 C,K）
        i_L0 = memory_levels.index('L0_Registers')
        C_L0_W = self._calculate_data_block_size(i_L0, 'W', layer_dims, mapping_table)
        tiles_L0_W = self._tiles_above_for_W(i_L0, layer_dims, mapping_table, persist_W=False)
        base_W_L2_to_L0 = C_L0_W * tiles_L0_W
        bcast_W_L2 = _bcast_factor('W', mapping_table, dev)
        read_W_L2_to_L0 = base_W_L2_to_L0 / torch.clamp(bcast_W_L2, min=1.0)
        _add(
            'L2_Scratchpad_to_L0_Registers',
            'Weight',
            read_W_L2_to_L0,
            {
                'base_elements(L0)': float(base_W_L2_to_L0.item()) if isinstance(base_W_L2_to_L0, torch.Tensor) else base_W_L2_to_L0,
                'broadcast_factor(C,K@L2)': float(bcast_W_L2.item()),
                'note': 'Weight read L2->L0 with broadcast over C,K'
            }
        )

        # W: L0 -> PE （**无广播**）
        read_W_L0_to_PE = total_macs
        _add(
            'L0_Registers_to_PE',
            'Weight',
            read_W_L0_to_PE,
            {
                'base_elements(L0)': float(base_W_L2_to_L0.item()) if isinstance(base_W_L2_to_L0, torch.Tensor) else base_W_L2_to_L0,
                'broadcast_factor': 1.0,
                'note': 'Weight read L0->PE (no broadcast)'
            }
        )


        # ---------- Output Reads ----------
        # O: L1 -> PE  （有广播，独立维 C；基数按 MACs）
        # 你偏好用统一接口：total_macs = calculate_macs(layer['dims'])；这里直接用 layer_dims
        total_macs = calculate_macs(layer_dims)
        bcast_O_L1 = _bcast_factor('O', mapping_table, dev)
        read_O_L1_to_PE = total_macs / torch.clamp(bcast_O_L1, min=1.0)
        _add(
            'L1_Accumulator_to_PE',
            'Output',
            read_O_L1_to_PE,
            {
                'MACs': float(total_macs.item()) if isinstance(total_macs, torch.Tensor) else total_macs,
                'broadcast_factor(C@L1)': float(bcast_O_L1.item()),
                'note': 'Output read L1->PE; base=MACs, broadcast over C'
            }
        )

        # ---- 调试轨迹（可选） ----
        if debug_data is not None:
            debug_data["inter_level_read_traffic_trace"] = {
                **debug_data.get("inter_level_read_traffic_trace", {})
            }

        # ---- 收尾：组装返回 ----
        result = {}
        for iface, data in detailed_read_traffic.items():
            result[iface] = {
                'total_bytes': data['total_bytes'],
                'breakdown': data['breakdown'],
                'drivers': data['drivers']
            }
        return result

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

        WARNING: This implementation has serious issues:
        - Does not properly handle gradient propagation for mapping parameters
        - Oversimplified memory access patterns
        - Inaccurate modeling of partial sums
        - Memory bandwidth calculations need revision
        
        This model assumes:
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

        

        # debug_grad(dram_bytes, mapping, "L2_Scratchpad.S.spatial")
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

    def _evaluate_single_layer(self, layer_name: str, graph, hw_params: HardwareParameters,
                               mapping: FineGrainedMapping, all_factors: dict,
                               debug_data: dict = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate a single layer using the high fidelity model."""
        from dosa.utils import calculate_macs

        layer = graph.layers[layer_name]
        total_macs = calculate_macs(layer['dims'])
        num_pes = hw_params.get_projected_num_pes()

        # 计算实际利用的PE数量
        utilized_pes = torch.tensor(1.0, device=self.config.DEVICE)

        # print("[DEBUG] all_factors =", all_factors)
        
        for dim_name, dim_mapping in all_factors.items():
            for level_name, level_factors in dim_mapping.items():
                if 'spatial' in level_factors:
                    spatial_factor = level_factors['spatial']

                    # 打印原始值和类型
                    print(f"[DEBUG] dim={dim_name}, level={level_name}, "
                        f"raw spatial_factor={spatial_factor} "
                        f"(type={type(spatial_factor)})")

                    # 类型转换
                    if not isinstance(spatial_factor, torch.Tensor):
                        spatial_factor = torch.tensor(
                            float(spatial_factor),
                            device=self.config.DEVICE
                        )
                        print(f"[DEBUG] converted spatial_factor={spatial_factor}")

                    # 累乘之前
                    prev_val = utilized_pes.clone()
                    utilized_pes *= spatial_factor

                    # 打印累乘结果
                    print(f"[DEBUG] utilized_pes update: {prev_val.item()} * "
                        f"{spatial_factor.item()} = {utilized_pes.item()}")


        effective_pes = torch.max(utilized_pes, torch.tensor(1.0, device=self.config.DEVICE))

        # Print total MACs and utilized PEs for debugging
        print(f"[DEBUG] Total MACs: {total_macs}")
        print(f"[DEBUG] Utilized PEs: {utilized_pes}")

        compute_cycles = total_macs / effective_pes

        detailed_fill_traffic_info = self.calculate_inter_level_fill_traffic(
            layer['dims'], all_factors, num_pes, hw_params, debug_data)
        detailed_read_traffic_info = self.calculate_inter_level_read_traffic(
            layer['dims'], all_factors, num_pes, debug_data)
        detailed_writeback_traffic_info = self.calculate_inter_level_writeback_traffic(
            layer['dims'], all_factors, num_pes, debug_data)

        memory_cycles_list = []

        # ===== 聚合：按层级累加 Reads / Writes / Updates =====
        device = self.config.DEVICE
        levels = ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad', 'L3_DRAM']

        zeros = {lvl: torch.tensor(0.0, device=device) for lvl in levels}
        per_level_reads    = {lvl: torch.tensor(0.0, device=device) for lvl in levels}
        per_level_writes   = {lvl: torch.tensor(0.0, device=device) for lvl in levels}
        per_level_updates  = {lvl: torch.tensor(0.0, device=device) for lvl in levels}

        def _iface_split(name: str):
            # 形如 "SRC_to_DST"
            parts = name.split('_to_')
            if len(parts) != 2:
                return None, None
            return parts[0], parts[1]

        # 1) fill → 写入到目的层（Writes(dst)）
        for iface, info in detailed_fill_traffic_info.items():
            src, dst = _iface_split(iface)
            if src is None: 
                continue
            bytes_total = info['total_bytes']
            per_level_writes[dst] = per_level_writes[dst] + bytes_total

        # 2) read → 从源层被读出（Reads(src)）
        #   （确保你已经有 detailed_read_traffic_info；若没有，请先调用你写的 read 统计函数）
        for iface, info in detailed_read_traffic_info.items():
            src, dst = _iface_split(iface)
            if src is None: 
                continue
            bytes_total = info['total_bytes']
            per_level_reads[src] = per_level_reads[src] + bytes_total

        # 3) writeback/update → 写入到目的层（Updates(dst)）
        for iface, info in detailed_writeback_traffic_info.items():
            src, dst = _iface_split(iface)
            if src is None: 
                continue
            bytes_total = info['total_bytes']
            per_level_updates[dst] = per_level_updates[dst] + bytes_total

        # ===== 计算每层的内存侧延迟：Accesses(i)/BW(i) =====
        eps = torch.tensor(1e-9, device=device)
        memory_cycles_list = []
        per_level_cycles_debug = {}

        for lvl in levels:
            accesses_bytes = per_level_reads[lvl] + per_level_writes[lvl] + per_level_updates[lvl]
            bw_bytes_per_cycle = calculate_bandwidth_bytes_per_cycle(lvl, num_pes, self.config)
            mem_cycles = accesses_bytes / (bw_bytes_per_cycle + eps)
            memory_cycles_list.append(mem_cycles)
            # 可选：调试
            per_level_cycles_debug[lvl] = {
                'Reads_bytes':   float(per_level_reads[lvl].detach().cpu().item()),
                'Writes_bytes':  float(per_level_writes[lvl].detach().cpu().item()),
                'Updates_bytes': float(per_level_updates[lvl].detach().cpu().item()),
                'Accesses_bytes_total': float(accesses_bytes.detach().cpu().item()),
                'BW_bytes_per_cycle': float(bw_bytes_per_cycle.detach().cpu().item() if isinstance(bw_bytes_per_cycle, torch.Tensor) else bw_bytes_per_cycle),
                'Mem_cycles': float(mem_cycles.detach().cpu().item()),
            }

        if memory_cycles_list:
            bottleneck_memory_cycles = torch.max(torch.stack(memory_cycles_list))
        else:
            bottleneck_memory_cycles = torch.tensor(0.0, device=device)

        # ===== Roofline 组合：与 compute 比较 =====
        stall_cycles = torch.relu(bottleneck_memory_cycles - compute_cycles)
        total_cycles = compute_cycles + stall_cycles
        latency = total_cycles / (self.config.CLOCK_FREQUENCY_MHZ * 1e6)



        # 能耗计算
        energy = torch.tensor(0.0, device=self.config.DEVICE)
        energy_compute = total_macs * self.config.PE_MAC_EPA_PJ
        energy += energy_compute

        energy_fill = torch.tensor(0.0, device=self.config.DEVICE)
        for interface, interface_info in detailed_fill_traffic_info.items():
            lower_level_name = interface.split('_to_')[1]
            fill_bytes = interface_info['total_bytes']
            fill_words = fill_bytes / self.config.BYTES_PER_ELEMENT
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

        energy_writeback = torch.tensor(0.0, device=self.config.DEVICE)
        for interface, interface_info in detailed_writeback_traffic_info.items():
            upper_level_name = interface.split('_to_')[1]
            writeback_bytes = interface_info['total_bytes']
            writeback_words = writeback_bytes / self.config.BYTES_PER_ELEMENT
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


        # ===== Read Energy =====
        energy_read = torch.tensor(0.0, device=self.config.DEVICE)
        for interface, interface_info in detailed_read_traffic_info.items():
            source_level_name = interface.split('_to_')[0]   # read 从 source 发出
            read_bytes = interface_info['total_bytes']
            read_words = read_bytes / self.config.BYTES_PER_ELEMENT
            if source_level_name == 'L0_Registers':
                epa = self.config.L0_REG_BASE_EPA_PJ
            elif source_level_name == 'L1_Accumulator':
                size_kb = hw_params.get_buffer_size_kb(source_level_name)
                num_pes_sqrt = torch.sqrt(num_pes)
                epa = (self.config.L1_ACCUM_BASE_EPA_PJ +
                       self.config.L1_ACCUM_CAPACITY_COEFF_PJ_PER_KB * (size_kb / num_pes_sqrt))
            elif source_level_name == 'L2_Scratchpad':
                size_kb = hw_params.get_buffer_size_kb(source_level_name)
                epa = (self.config.L2_SPM_BASE_EPA_PJ +
                       self.config.L2_SPM_CAPACITY_COEFF_PJ_PER_KB * size_kb)
            elif source_level_name == 'L3_DRAM':
                epa = self.config.L3_DRAM_EPA_PJ
            else:
                epa = torch.tensor(0.0, device=self.config.DEVICE)
            energy_read += read_words * epa

        # ===== Total Energy =====
        energy = energy_compute + energy_fill + energy_writeback + energy_read




        buffer_mismatch_loss = torch.tensor(0.0, device=self.config.DEVICE)
        for i, level in enumerate(self.config.MEMORY_HIERARCHY):
            if level['type'] == 'buffer':
                required_kb = self.calculate_buffer_req_kb(layer['dims'], all_factors, i)
                available_kb = hw_params.get_buffer_size_kb(level['name'])
                buffer_deficit = torch.relu(required_kb - available_kb)
                relative_deficit = buffer_deficit / (required_kb + 1e-9)
                buffer_mismatch_loss += torch.pow(relative_deficit, 2)

        # debug
        # debug_grad(energy, mapping, "L2_Scratchpad.S.temporal")

        return latency, energy, buffer_mismatch_loss

    def forward(self, graph, hw_params: HardwareParameters, mapping: FineGrainedMapping,
                fusion_params: nn.Module = None, direct_mapping_table: dict = None,
                debug_output_path: str = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        # if direct_mapping_table:
        #     all_factors = direct_mapping_table
        # else:



        # 格式转换版本（数值不变，只是结构化）
        all_factors = format_mapping_as_all_factors(mapping)


            # print("=== Gradient Debug Information 1 ===")
            
            # Print type information
            # print(f"Type of temporal factor: {type(all_factors['S']['L2_Scratchpad']['temporal'])}")
            
            # debug_grad(all_factors["S"]["L2_Scratchpad"]["temporal"],
            #mapping, "L2_Scratchpad.S.temporal")

        # num_pes = hw_params.get_projected_num_pes()

        
        if self.fusion_aware and fusion_params is not None:
            # 避免squeeze导致0维张量的问题
            fusion_logits = fusion_params.fusion_logits
            if fusion_logits.dim() == 1 and fusion_logits.size(0) == 1:
                fusion_probs = torch.sigmoid(fusion_logits)  # 保持1维
            else:
                fusion_probs = torch.sigmoid(fusion_logits.squeeze())
            fusion_groups = graph.fusion_groups
            if debug_data is not None:
                decisions = (fusion_probs > 0.5).detach().cpu().tolist()
                debug_data["fusion_decisions"] = decisions
        else:
            # 如果 fusion_groups 为空，就退化为单层模式
            if len(graph.fusion_groups) == 0:
                fusion_groups = [[layer_name] for layer_name in graph.layers.keys()]
                fusion_probs = torch.ones(len(fusion_groups), device=self.config.DEVICE)
            else:
                fusion_probs = torch.ones(len(graph.fusion_groups), device=self.config.DEVICE)
                fusion_groups = (
                    [[layer_name] for group in graph.fusion_groups for layer_name in group]
                    if not self.fusion_aware else graph.fusion_groups
                )

            # Debug 打印
            # print("\n[DEBUG] Fusion Info ----------------------------")
            # print("[DEBUG] fusion_aware =", self.fusion_aware)
            # print("[DEBUG] graph.fusion_groups =", graph.fusion_groups)
            # print("[DEBUG] effective fusion_groups =", fusion_groups)
            # print("[DEBUG] fusion_probs =", fusion_probs)
            # print("------------------------------------------------\n")


        # Print fusion groups with index
        # print(f"[DEBUG] INFO: Fusion Groups:")
        # for i, group in enumerate(fusion_groups):
            # print(f"Group {i}: {group}")
            
        for idx, group in enumerate(fusion_groups):
            

            weight = fusion_probs[idx] if self.fusion_aware and fusion_params is not None else torch.tensor(1.0, device=self.config.DEVICE)
            fused_latency = torch.tensor(0.0, device=self.config.DEVICE)
            fused_energy = torch.tensor(0.0, device=self.config.DEVICE)
            fused_mismatch = torch.tensor(0.0, device=self.config.DEVICE)
            fused_comp = torch.tensor(0.0, device=self.config.DEVICE)

            split_latency = torch.tensor(0.0, device=self.config.DEVICE)
            split_energy = torch.tensor(0.0, device=self.config.DEVICE)
            split_mismatch = torch.tensor(0.0, device=self.config.DEVICE)
            split_comp = torch.tensor(0.0, device=self.config.DEVICE)

            current_pattern = tuple(graph.layers[layer_name]['type'] for layer_name in group)
            dmt_model = self.dmt_registry.get(current_pattern) if (self.fusion_aware and fusion_params is not None) else None
            
            # debug
            # print(f"DMT Model for pattern {current_pattern}: {dmt_model}")

            if dmt_model is not None:
                # print(f"[DEBUG] INFO: {dmt_model}")
                fused_latency, fused_energy, fused_mismatch, fused_comp, _ = dmt_model(group, graph, hw_params, mapping, self.config)
            else:
                for layer_name in group:
                    
                    print(f"[DEBUG] Evaluating layer {layer_name} with mapping:")
                    for name, param in mapping.named_parameters():
                        print(f"  {name}: {param.data}")
                        
                    print(f"[DEBUG] all_factors for {layer_name}:", all_factors)
                    lat, en, mismatch = self._evaluate_single_layer(layer_name, graph, hw_params, mapping, all_factors, debug_data)
                    fused_latency += lat
                    fused_energy += en
                    fused_mismatch += mismatch

            for layer_name in group:
                lat, en, mismatch = self._evaluate_single_layer(layer_name, graph, hw_params, mapping, all_factors, debug_data)
                split_latency += lat
                split_energy += en
                
                split_mismatch += mismatch

            # bug
            latency = weight * fused_latency + (1 - weight) * split_latency
            energy = weight * fused_energy + (1 - weight) * split_energy            

            group_buffer_mismatch_loss = weight * fused_mismatch + (1 - weight) * split_mismatch
            compatibility_penalty = weight * fused_comp + (1 - weight) * split_comp

            total_latency += latency
            total_energy += energy
            total_buffer_mismatch_loss += group_buffer_mismatch_loss
            total_compatibility_penalty += compatibility_penalty

        area_cost = hw_params.get_area_cost()

        # Compute invalid mapping penalty
        mapping_invalid_penalty = self.compute_invalid_penalty(mapping)

        if debug_data is not None and debug_output_path is not None:
            import json
            
            def tensor_to_serializable(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.detach().cpu().numpy().tolist()
                elif isinstance(obj, dict):
                    return {k: tensor_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [tensor_to_serializable(item) for item in obj]
                else:
                    return obj
            
            with torch.no_grad():
                serializable_debug_data = tensor_to_serializable(debug_data)
                with open(debug_output_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable_debug_data, f, indent=4, ensure_ascii=False)

        total_latency = total_latency.squeeze() if total_latency.dim() > 0 else total_latency
        total_energy = total_energy.squeeze() if total_energy.dim() > 0 else total_energy
        area_cost = area_cost.squeeze() if area_cost.dim() > 0 else area_cost
        total_buffer_mismatch_loss = total_buffer_mismatch_loss.squeeze() if total_buffer_mismatch_loss.dim() > 0 else total_buffer_mismatch_loss
        total_compatibility_penalty = total_compatibility_penalty.squeeze() if total_compatibility_penalty.dim() > 0 else total_compatibility_penalty
        penalty = mapping_invalid_penalty + total_buffer_mismatch_loss + total_compatibility_penalty
        
        

        return total_latency, total_energy, area_cost, total_buffer_mismatch_loss, total_compatibility_penalty, mapping_invalid_penalty, penalty

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