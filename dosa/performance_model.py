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
    formatted = {}
    for level_name, dims in mapping.factors.items():
        for dim_name, factor_dict in dims.items():
            if dim_name not in formatted:
                formatted[dim_name] = {}
            spatial_val = factor_dict.get("spatial", 1.0)
            temporal_val = factor_dict.get("temporal", 1.0)
            if not isinstance(spatial_val, torch.Tensor):
                spatial_val = torch.tensor(float(spatial_val), device="cuda:0")
            if not isinstance(temporal_val, torch.Tensor):
                temporal_val = torch.tensor(float(temporal_val), device="cuda:0")
            formatted[dim_name][level_name] = {
                "spatial": spatial_val,
                "temporal": temporal_val,
            }

    # --- Complete L3_DRAM ---
    for d in ['N','K','C','P','Q','R','S']:
        if d in mapping.dims:  # Only process existing dimensions
            # Ensure dimension exists in formatted
            if d not in formatted:
                formatted[d] = {}
            
            # Check if L3_DRAM mapping already exists
            if 'L3_DRAM' not in formatted[d]:
                # Calculate product of temporal factors for all on-chip levels
                on_chip_temporal_product = torch.tensor(1.0, device="cuda:0")
                print(f"\nProcessing dimension {d}:")
                print(f"Initial on-chip temporal product = {on_chip_temporal_product}")
                
                for level_name in formatted[d]:
                    temporal_factor = formatted[d][level_name].get('temporal', torch.tensor(1.0, device="cuda:0"))
                    on_chip_temporal_product *= temporal_factor
                    print(f"  {level_name} temporal factor = {temporal_factor}")
                    print(f"  Updated product = {on_chip_temporal_product}")

                on_chip_product = torch.tensor(1.0, device="cuda:0")
                for level_name in formatted[d]:
                    spatial_factor = formatted[d][level_name].get('spatial', torch.tensor(1.0, device="cuda:0"))
                    temporal_factor = formatted[d][level_name].get('temporal', torch.tensor(1.0, device="cuda:0"))
                    on_chip_product *= spatial_factor * temporal_factor

                                
                # DRAM temporal factor is the remaining portion
                problem_dim_size = torch.tensor(float(mapping.dims[d]), device="cuda:0")
                print(f"Problem dimension size = {problem_dim_size}")
                
                dram_temporal = problem_dim_size / torch.clamp(on_chip_product, min=1.0)
                print(f"Calculated DRAM temporal factor = {dram_temporal}")
                
                # Clamp to ensure minimum of 1
                clamped_dram_temporal = torch.clamp(dram_temporal, min=1.0)
                print(f"Final clamped DRAM temporal factor = {clamped_dram_temporal}")
                
                formatted[d]['L3_DRAM'] = {
                    "spatial": torch.tensor(1.0, device="cuda:0"),
                    "temporal": clamped_dram_temporal
                }

    return formatted

class HighFidelityPerformanceModel(nn.Module):
    """
    NEW: 高精度性能模型，能够处理多级存储和细粒度映射。
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        # 用于存放“层间边界融合决策”的可训练参数字典。
        # key: "l0->l1" 表示从 layer0 到 layer1 的边界
        # val: nn.Parameter，存的是 logit（需要经过 sigmoid 转换为概率）
        #     - sigmoid(logit) ≈ 1 表示倾向于把输出留在 L2（融合）
        #     - sigmoid(logit) ≈ 0 表示倾向于写回 DRAM（不融合）
        self.fusion_boundary_logits = nn.ParameterDict()

    def init_fusion_boundaries(self, group_layers):
        """为一条融合链的每个边界(l->l+1)创建一个可训练logit。"""
        for i in range(len(group_layers)-1):
            key = f"{group_layers[i]}->{group_layers[i+1]}"
            if key not in self.fusion_boundary_logits:
                # 折中初始化：sigmoid(0.7) ≈ 0.668
                self.fusion_boundary_logits[key] = nn.Parameter(
                    torch.tensor(0.7, device=self.config.DEVICE)
                )


        # === 2) 在类内新增一个方法（放在类中任意位置即可） ===
    
    def backprop_lb_from_output_tile(self, group_layers, graph, out_tile_last):
        """
        给定最后一层的输出tile (P_o, Q_o, K_o, N_o)，
        沿融合链从后往前反推每层的 I/O tile 的几何下界（LB），并返回每层的 I/O 字节下界。
        out_tile_last: dict，如 {'P':Po_L, 'Q':Qo_L, 'K':Ko_L, 'N':No_L}
        返回:
        lb = {
            layer_name: {
            'O': {'P':..., 'Q':..., 'K':..., 'N':...,'bytes': Tensor},
            'I': {'P':..., 'Q':..., 'C':..., 'N':...,'bytes': Tensor},
            }
        }
        """
        dev = self.config.DEVICE
        BYTES = self.config.BYTES_PER_ELEMENT

        # 结果容器（从后往前填，再返回正向顺序字典）
        lb_rev = {}
        # 当前层的输出tile（初始化为最后一层的输出tile）
        cur_Po = torch.tensor(float(out_tile_last.get('P', 1.0)), device=dev)
        cur_Qo = torch.tensor(float(out_tile_last.get('Q', 1.0)), device=dev)
        cur_Ko = torch.tensor(float(out_tile_last.get('K', 1.0)), device=dev)
        cur_No = torch.tensor(float(out_tile_last.get('N', 1.0)), device=dev)

        for name in reversed(group_layers):
            layer = graph.layers[name]
            # 提取完整卷积层的几何参数
            R = torch.tensor(float(layer['dims'].get('R', 1)), device=dev)
            S = torch.tensor(float(layer['dims'].get('S', 1)), device=dev)
            tP = torch.tensor(float(layer.get('stride_P', 1)), device=dev)
            tQ = torch.tensor(float(layer.get('stride_Q', 1)), device=dev)

            # 本层输出 LB（就是当前 cur_*）
            O_P = torch.clamp(cur_Po, min=1.0)
            O_Q = torch.clamp(cur_Qo, min=1.0)
            O_K = torch.clamp(cur_Ko, min=1.0)
            O_N = torch.clamp(cur_No, min=1.0)

            # 由 O 反推 I 的几何覆盖 LB: (P_o-1)*stride + R
            I_P = (O_P - 1.0) * tP + R
            I_Q = (O_Q - 1.0) * tQ + S
            I_C = torch.tensor(float(layer['dims'].get('C', 1)), device=dev)  # 与上一层K对齐由链条保证
            I_N = O_N  # batch不变

            # 字节数下界（不考虑复用，即 fully-recompute 的必需体量）
            I_bytes = I_N * I_C * I_P * I_Q * BYTES
            O_bytes = O_N * O_K * O_P * O_Q * BYTES

            lb_rev[name] = {
                'O': {'P': O_P, 'Q': O_Q, 'K': O_K, 'N': O_N, 'bytes': O_bytes},
                'I': {'P': I_P, 'Q': I_Q, 'C': I_C, 'N': I_N, 'bytes': I_bytes}
            }

            # 把本层 I 的 (C) 映射到上一层的 O 的 (K)，把几何 I(P,Q) 作为上一层 O(P,Q)
            # K/通道数沿网络定义走，这里只做几何链，C/K 对齐由 layer dims 管理
            prev_Ko = torch.tensor(float(layer['dims'].get('C', 1)), device=dev)
            cur_Po, cur_Qo, cur_Ko, cur_No = I_P, I_Q, prev_Ko, I_N

        # 翻转成正序
        lb = {k: lb_rev[k] for k in group_layers}
        return lb

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

                # === DEBUG: 打印 REG 的 fill 次数 ===
                if destination_level_name == "L0_Registers":
                    print(f"[DEBUG] REG Fill: tensor={tensor_type}, "
                        f"destination={destination_level_name}, "
                        f"fill_accesses={tensor_fill_accesses.detach().cpu().item()}")
                
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


        # === DEBUG 打印 ===
        print("[DEBUG][L1 Updates]")
        print(f"  MACs           = {macs.item() if isinstance(macs, torch.Tensor) else macs}")
        print(f"  F_S_O(L1)      = {F_S_O_L0.item() if isinstance(F_S_O_L0, torch.Tensor) else F_S_O_L0}")
        print(f"  Updates_O(L1)  = {updates_L1.item() if isinstance(updates_L1, torch.Tensor) else updates_L1}")
        print(f"  Write bytes    = {write_bytes.item() if isinstance(write_bytes, torch.Tensor) else write_bytes}")

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
        Tiles_above(i, W) —— L0 之外所有层级的 temporal 因子乘积：
        - 包含 W 相关维 D_W = {K, C, R, S}
        - 以及 W 无关维 {N, P, Q}   # Timeloop 的口径需要
        - 只乘 temporal；不乘 spatial；不做广播折扣（广播在 reads 阶段做）
        """
        dev = self.config.DEVICE
        D_W = ('K', 'C', 'R', 'S') # 相关维度
        D_indep = ('N', 'P', 'Q')  # 无关维度

        levels_order = ['L0_Registers','L1_Accumulator','L2_Scratchpad','L3_DRAM']

        tiles = torch.tensor(1.0, device=dev)
        multipliers = []

        # For L0 weight calculation, consider both dependent and independent dims （L2->L2流量）
        if i == 0:
            # For L0 weight calculation, consider both dependent and independent dims
            for k in range(i + 1, len(levels_order)):   # Only look at outer levels
                lvl = levels_order[k]
                level_multiplier = torch.tensor(1.0, device=dev)

                # Related dims {K,C,R,S} 
                for d in D_W:
                    tf = mapping_table.get(d, {}).get(lvl, {}).get('temporal', 1.0)
                    tf = torch.tensor(float(tf), device=dev) if not isinstance(tf, torch.Tensor) else tf
                    level_multiplier *= torch.clamp(tf, min=1.0)

                # Independent dims {N,P,Q}
                for d in D_indep:
                    tf = mapping_table.get(d, {}).get(lvl, {}).get('temporal', 1.0)
                    tf = torch.tensor(float(tf), device=dev) if not isinstance(tf, torch.Tensor) else tf
                    level_multiplier *= torch.clamp(tf, min=1.0)

                tiles *= level_multiplier
                multipliers.append((lvl, level_multiplier))
        else:
            # For other levels, only consider dependent dims
            for k in range(i + 1, len(levels_order)):
                lvl = levels_order[k]
                level_multiplier = torch.tensor(1.0, device=dev)

                # Only related dims {K,C,R,S}
                for d in D_W:
                    tf = mapping_table.get(d, {}).get(lvl, {}).get('temporal', 1.0)
                    tf = torch.tensor(float(tf), device=dev) if not isinstance(tf, torch.Tensor) else tf
                    level_multiplier *= torch.clamp(tf, min=1.0)

                tiles *= level_multiplier
                multipliers.append((lvl, level_multiplier))

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
        for lvl in levels_order[i + 1:]:  # 只看外层
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
        for lvl in levels_order[i + 1:]:  # 只看 i 的外层
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
                dims = ('P', 'Q', 'N')
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
        
        # Print L3 Input reads
        print("[DEBUG][L3 Input Reads]")
        print(f"  C_i_I = {C_i_I.item() if isinstance(C_i_I, torch.Tensor) else C_i_I}")
        print(f"  tiles_i_I = {tiles_i_I.item() if isinstance(tiles_i_I, torch.Tensor) else tiles_i_I}")
        print(f"  base_I_L3_to_L2 = {base_I_L3_to_L2.item() if isinstance(base_I_L3_to_L2, torch.Tensor) else base_I_L3_to_L2}")
        
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

        # ---------- Weight Reads ----------
        # W: L3 -> L2 （无广播）
        i = memory_levels.index('L2_Scratchpad')
        C_i_W = self._calculate_data_block_size(i, 'W', layer_dims, mapping_table)
        tiles_i_W = self._tiles_above_for_W(i, layer_dims, mapping_table, persist_W=False)
        base_W_L3_to_L2 = C_i_W * tiles_i_W

        # Print L3 Weight reads
        print("[DEBUG][L3 Weight Reads]")
        print(f"  C_i_W = {C_i_W.item() if isinstance(C_i_W, torch.Tensor) else C_i_W}")
        print(f"  tiles_i_W = {tiles_i_W.item() if isinstance(tiles_i_W, torch.Tensor) else tiles_i_W}")
        print(f"  base_W_L3_to_L2 = {base_W_L3_to_L2.item() if isinstance(base_W_L3_to_L2, torch.Tensor) else base_W_L3_to_L2}")

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

        # W: L2 -> L0 (with broadcast over C,K dimensions)
        i_L0 = memory_levels.index('L0_Registers')
        print(f"[DEBUG][L2 Weight Reads] L0 index = {i_L0}")
        C_L0_W = self._calculate_data_block_size(i_L0, 'W', layer_dims, mapping_table) # Utilzed PE
        tiles_L0_W = self._tiles_above_for_W(i_L0, layer_dims, mapping_table, persist_W=False) # 需要算上和W的无关维度
        base_W_L2_to_L0 = C_L0_W * tiles_L0_W
        bcast_W_L2 = _bcast_factor('W', mapping_table, dev)
        fanout_L0_W = torch.tensor(1.0, device=dev) # 相当于PE数目

        for d in ('K', 'C', 'R', 'S'):
            s = mapping_table.get(d, {}).get('L0_Registers', {}).get('spatial', 1.0)
            if not isinstance(s, torch.Tensor):
                s = torch.tensor(float(s), device=dev)
            fanout_L0_W *= torch.clamp(s, min=1.0)

        read_W_L2_to_L0 = (base_W_L2_to_L0 * fanout_L0_W) / torch.clamp(bcast_W_L2, min=1.0)

        # Print calculation process
        print("\n=== Weight L2->L0 Traffic Calculation ===")
        print(f"1. Data block size at L0 (C_L0_W) = {C_L0_W.item() if isinstance(C_L0_W, torch.Tensor) else C_L0_W}")
        print(f"2. Tiles above L0 (tiles_L0_W) = {tiles_L0_W.item() if isinstance(tiles_L0_W, torch.Tensor) else tiles_L0_W}")
        print(f"3. Base traffic (C_L0_W * tiles_L0_W) = {base_W_L2_to_L0.item() if isinstance(base_W_L2_to_L0, torch.Tensor) else base_W_L2_to_L0}")
        print(f"4. Broadcast factor (C,K@L2) = {bcast_W_L2.item() if isinstance(bcast_W_L2, torch.Tensor) else bcast_W_L2}")
        print(f"5. Final traffic after broadcast = {read_W_L2_to_L0.item() if isinstance(read_W_L2_to_L0, torch.Tensor) else read_W_L2_to_L0}")
        
        _add(
            'L2_Scratchpad_to_L0_Registers',
            'Weight',
            read_W_L2_to_L0,
            {
                'base_elements(L0)': float(base_W_L2_to_L0.item()) if isinstance(base_W_L2_to_L0, torch.Tensor) else base_W_L2_to_L0,
                'broadcast_factor': float(bcast_W_L2.item()),
                'note': 'Weight read L2->L0 with broadcast over C,K'
            }
        )

        # === DEBUG: 合并打印 L2 的总读流量 ===
        total_read_L2 = read_I_L2_to_PE + read_W_L2_to_L0
        print("[DEBUG][L2 Reads]")
        print(f"  Input  (L2->PE)       = {read_I_L2_to_PE.item() if isinstance(read_I_L2_to_PE, torch.Tensor) else read_I_L2_to_PE}")
        print(f"  Weight (L2->L0)       = {read_W_L2_to_L0.item() if isinstance(read_W_L2_to_L0, torch.Tensor) else read_W_L2_to_L0}")
        print(f"  Total L2 Read Traffic = {total_read_L2.item() if isinstance(total_read_L2, torch.Tensor) else total_read_L2}")

        # W: L0 -> PE (with broadcast over P,Q,N @ L0)
        bcast_W_L0 = _bcast_factor('W', mapping_table, dev)
        read_W_L0_to_PE = total_macs / torch.clamp(bcast_W_L0, min=1.0)

        print("[DEBUG][Weight Reads L0->PE]")
        print(f"  total_macs = {total_macs.item() if isinstance(total_macs, torch.Tensor) else total_macs}")
        print(f"  bcast(P,Q,N @ L0) = {bcast_W_L0.item() if isinstance(bcast_W_L0, torch.Tensor) else bcast_W_L0}")
        print(f"  read_W_L0_to_PE = {read_W_L0_to_PE.item() if isinstance(read_W_L0_to_PE, torch.Tensor) else read_W_L0_to_PE}")

        _add(
            'L0_Registers_to_PE',
            'Weight',
            read_W_L0_to_PE,
            {
                'bcast(P,Q,N@L0)': float(bcast_W_L0.item()) if isinstance(bcast_W_L0, torch.Tensor) else bcast_W_L0,
                'note': 'Weight read L0->PE with broadcast over P,Q,N at L0 (if any)'
            }
        )


        # ---------- Output Reads ----------
        # O: L1 -> PE  （有广播，独立维 C；基数按 MACs）
        # 偏好统一接口：total_macs = calculate_macs(layer['dims'])；这里直接用 layer_dims
        total_macs = calculate_macs(layer_dims)
        bcast_O_L1 = _bcast_factor('O', mapping_table, dev)
        read_O_L1_to_PE = total_macs / torch.clamp(bcast_O_L1, min=1.0)

        # === DEBUG 打印 ===
        print("[DEBUG][Output Reads]")
        print(f"  total_macs = {total_macs.item() if isinstance(total_macs, torch.Tensor) else total_macs}")
        print(f"  bcast_O_L1 = {bcast_O_L1.item() if isinstance(bcast_O_L1, torch.Tensor) else bcast_O_L1}")
        print(f"  read_O_L1_to_PE = {read_O_L1_to_PE.item() if isinstance(read_O_L1_to_PE, torch.Tensor) else read_O_L1_to_PE}")

        # 收集到 detailed traffic
        _add(
            'L1_Accumulator_to_PE',
            'Output',
            read_O_L1_to_PE,
            {
                'MACs': float(total_macs.item()) if isinstance(total_macs, torch.Tensor) else total_macs,
                'broadcast_factor(C@L1)': float(bcast_O_L1.item()) if isinstance(bcast_O_L1, torch.Tensor) else bcast_O_L1,
                'note': 'Output read L1->PE; base=MACs, broadcast over C'
            }
        )


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
            
            print(f"\n[DEBUG] Computing C_{i},W:")
            print(f"  Relevant dimensions: {relevant_dims}")
            
            for dim_name in relevant_dims:
                if dim_name in layer_dims:
                    total_dim_size = torch.tensor(layer_dims[dim_name], device=self.config.DEVICE)
                    coverage = self._coverage_upto(i, dim_name, mapping_table, layer_dims)
                    
                    # print(f"  {dim_name}:")
                    # print(f"    Total size: {total_dim_size.item()}")
                    # print(f"    Coverage: {coverage.item()}")
                    # print(f"    Min value: {torch.min(total_dim_size, coverage).item()}")
                    
                    prev_size = size.clone()
                    size *= torch.min(total_dim_size, coverage)
                    # print(f"    Running product: {prev_size.item()} * {torch.min(total_dim_size, coverage).item()} = {size.item()}")
            
            # print(f"  Final C_{i},W = {size.item()}\n")
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
            """
            论文口径（Eq.3）：
            C_{i,I} 只由“内层覆盖（j<i）的时空因子”构造：
            C_{i,I} = (N_inner * C_inner) *
                        [P_stride*(Inner(i,P)-1) + Inner(i,R)] *
                        [Q_stride*(Inner(i,Q)-1) + Inner(i,S)]
            注意：不包含当前层 i，也不包含更外层；不在这里加入整图像或 L2 特判。
            """
            dev = self.config.DEVICE
            memory_levels = ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad', 'L3_DRAM']

            def _inner_lt_i(dim_name: str) -> torch.Tensor:
                val = torch.tensor(1.0, device=dev)
                for j in range(i):  # 关键：严格 j < i
                    lvl = memory_levels[j]
                    if dim_name in mapping_table and lvl in mapping_table[dim_name]:
                        t = mapping_table[dim_name][lvl].get('temporal', 1)
                        s = mapping_table[dim_name][lvl].get('spatial', 1)
                        val *= torch.tensor(float(t) * float(s), device=dev)
                return val

            # strides（如需：从 prob 读）——当前按 1 处理
            P_stride = 1.0
            Q_stride = 1.0

            inner_P = _inner_lt_i('P')
            inner_Q = _inner_lt_i('Q')
            inner_R = _inner_lt_i('R')
            inner_S = _inner_lt_i('S')
            inner_N = _inner_lt_i('N')
            inner_C = _inner_lt_i('C')

            H = P_stride * (inner_P - 1.0) + inner_R
            W = Q_stride * (inner_Q - 1.0) + inner_S

            size = inner_N * inner_C * H * W

            # 可选：调试打印
            print(f"[DEBUG][C_i_I paper] i={i} levels< i: {memory_levels[:i]}")
            print(f"  Inner(P)={inner_P.item()}, Inner(Q)={inner_Q.item()}, Inner(R)={inner_R.item()}, Inner(S)={inner_S.item()}")
            print(f"  Inner(N)={inner_N.item()}, Inner(C)={inner_C.item()}")
            print(f"  H={H.item()}, W={W.item()},  C_i_I={float(size.item())}")

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

    def evaluate_group_depth_first(
        self,
        group_layers: list,
        graph,
        hw_params: "HardwareParameters",
        mapping: "FineGrainedMapping",
        direct_mapping_table: dict = None,
        debug_data: dict = None,
    ):
        dev   = self.config.DEVICE
        BYTES = self.config.BYTES_PER_ELEMENT

        # =============== 辅助工具：从 all_factors 生成“夹过LB的”有效视图 =================
        def _apply_lb_to_output_tiles(all_factors_in, lb_dict):
            """
            仅对每层 O-tile 做可微下界重参数化：
                O_eff = LB_O + softplus(O_raw - LB_O)
            其它映射参数保持不变。返回一个新的 mapping 视图（不修改原 mapping/all_factors）。
            """
            # 浅拷贝到新的字典（避免改动原对象）
            eff = {k: {kk: vv for kk, vv in v.items()} for k, v in all_factors_in.items()}
            for lname in group_layers:
                if lname not in eff:  # 保险：all_factors里可能不存在该层键
                    eff[lname] = {}
                # 取 LB（都是 Tensor）
                LB_P = lb_dict[lname]["O"]["P"]
                LB_Q = lb_dict[lname]["O"]["Q"]
                LB_K = lb_dict[lname]["O"]["K"]
                LB_N = lb_dict[lname]["O"]["N"]
                # 读原始 O-tile，若没有就跳过（保持原状）
                P_key, Q_key, K_key, N_key = "O_tile_P", "O_tile_Q", "O_tile_K", "O_tile_N"
                if P_key in eff[lname]:
                    eff[lname][P_key] = LB_P + torch.nn.functional.softplus(eff[lname][P_key] - LB_P)
                if Q_key in eff[lname]:
                    eff[lname][Q_key] = LB_Q + torch.nn.functional.softplus(eff[lname][Q_key] - LB_Q)
                if K_key in eff[lname]:
                    eff[lname][K_key] = LB_K + torch.nn.functional.softplus(eff[lname][K_key] - LB_K)
                if N_key in eff[lname]:
                    eff[lname][N_key] = LB_N + torch.nn.functional.softplus(eff[lname][N_key] - LB_N)
            return eff

        # =============== 边界 logit（可训练），初始化偏向 L2（sigmoid≈0.67） =================
        self.init_fusion_boundaries(group_layers)

        # =============== 反推 LB（几何下界，保证跨层兼容） =================
        last_dims = graph.layers[group_layers[-1]]["dims"]
        out_tile_last = {
            "P": float(last_dims.get("P", 1)), "Q": float(last_dims.get("Q", 1)),
            "K": float(last_dims.get("K", 1)), "N": float(last_dims.get("N", 1)),
        }
        lb = self.backprop_lb_from_output_tile(group_layers, graph, out_tile_last)
        lb_O_bytes = [lb[name]["O"]["bytes"] for name in group_layers]
        lb_I_bytes = [lb[name]["I"]["bytes"] for name in group_layers]

        # =============== 解析 factor，并用 LB“夹住”每层 O-tile 的有效视图 =================
        all_factors_raw = format_mapping_as_all_factors(mapping)
        all_factors_eff = _apply_lb_to_output_tiles(all_factors_raw, lb)

        # =============== 逐层用单层公式计算 read / fill(write) / writeback 的基线流量 =================
        per_layer = []
        total_macs = torch.tensor(0.0, device=dev)

        # 安全访问 breakdown 小工具
        def _get_bd(d, iface, tensor):
            if d is None: 
                return torch.tensor(0.0, device=dev)
            if iface not in d: 
                return torch.tensor(0.0, device=dev)
            v = d[iface]["breakdown"].get(tensor, torch.tensor(0.0, device=dev))
            return v if isinstance(v, torch.Tensor) else torch.tensor(float(v), device=dev)

        from dosa.utils import calculate_macs

        for lname in group_layers:
            layer = graph.layers[lname]
            dims  = layer["dims"]

            # --- 单层三张表：read / fill(write) / writeback ---
            read_dict = self.calculate_inter_level_read_traffic(
                layer_dims=dims,
                mapping_table=all_factors_eff,                       # ★ 用“夹过LB”的视图
                num_pes=hw_params.get_projected_num_pes(),
                hw_params=hw_params,
                debug_data=None,
            )
            fill_dict = self.calculate_inter_level_fill_traffic(
                layer_dims=dims,
                mapping_table=all_factors_eff,                       # ★ 用“夹过LB”的视图
                num_pes=hw_params.get_projected_num_pes(),
                hw_params=hw_params,
                debug_data=None,
            )
            write_dict = self.calculate_inter_level_writeback_traffic(
                layer_dims=dims,
                mapping_table=all_factors_eff,                       # ★ 用“夹过LB”的视图
                num_pes=hw_params.get_projected_num_pes(),
                debug_data=None,
            )

            # L3 服务候选：I/W 的 L3->L2；O 的 L1->L3（基线写回）
            I_L3_to_L2_base = _get_bd(read_dict, "L3_DRAM_to_L2_Scratchpad", "Input")
            W_L3_to_L2_base = _get_bd(read_dict, "L3_DRAM_to_L2_Scratchpad", "Weight")
            O_L1_to_L3_write_base = _get_bd(fill_dict, "L1_Accumulator_to_L3_DRAM", "Output")

            # L2 服务候选：I 的 L2->PE；W 的 L2->L0（权重从 L2 到寄存器）
            I_L2_to_PE  = _get_bd(read_dict, "L2_Scratchpad_to_PE", "Input")
            W_L2_to_L0  = _get_bd(read_dict, "L2_Scratchpad_to_L0_Registers", "Weight")

            # copy-bundle(L1->L2)：不计能耗/时延（但可用于 debug）
            O_L1_to_L2_copy = _get_bd(fill_dict, "L1_Accumulator_to_L2_Scratchpad", "Output")

            # PE->L1（用于能耗参考；不计 L2/L3 服务）
            PE_to_L1 = _get_bd(write_dict, "PE_to_L1_Accumulator", "Output")

            # 记录
            per_layer.append({
                "name": lname, "dims": dims,
                # L3 服务
                "I_L3_to_L2": I_L3_to_L2_base,      # 后续乘(1-s)
                "W_L3_to_L2": W_L3_to_L2_base,
                "O_L1_to_L3_write_base": O_L1_to_L3_write_base,   # 这层自然写回（非跨层融合导致）
                "O_to_L3_extra": torch.tensor(0.0, device=dev),   # 占位：边界(1-s)*LB_O 产生的额外写回
                # L2 服务
                "I_L2_to_PE": I_L2_to_PE,
                "W_L2_to_L0": W_L2_to_L0,            # ★ 关键：权重 L2->寄存器
                # 其它
                "O_L1_to_L2_copy": O_L1_to_L2_copy,  # 仅 debug
                "PE_to_L1": PE_to_L1,                # 仅能耗参考
            })

            total_macs += calculate_macs(dims)

        # =============== 边界概率 s：倾向留 L2（融合），或回写 L3（不融合） =================
        s_list = []
        for i in range(len(group_layers) - 1):
            key = f"{group_layers[i]}->{group_layers[i+1]}"
            s_list.append(torch.sigmoid(self.fusion_boundary_logits[key]))

        # 用 s 修正跨层：减少下一层 I 的 L3->L2；增加本层 O 的“几何必需额外写回”
        for i in range(len(group_layers) - 1):
            nxt = i + 1
            s_i = s_list[i]
            per_layer[nxt]["I_L3_to_L2"] = (1.0 - s_i) * per_layer[nxt]["I_L3_to_L2"]
            per_layer[i]["O_to_L3_extra"] = (1.0 - s_i) * lb_O_bytes[i]

        # =============== 汇总服务字节，显式统计 W(L2->L0) =================
        total_L3_bytes     = torch.tensor(0.0, device=dev)
        total_L2_bytes     = torch.tensor(0.0, device=dev)
        total_W_L2_to_L0   = torch.tensor(0.0, device=dev)

        for rec in per_layer:
            # L3服务 = I_L3->L2 + W_L3->L2 + 本层自然写回 + 边界额外写回
            total_L3_bytes += rec["I_L3_to_L2"] + rec["W_L3_to_L2"] \
                            + rec["O_L1_to_L3_write_base"] + rec["O_to_L3_extra"]
            # L2服务 = I_L2->PE + W_L2->L0
            total_L2_bytes += rec["I_L2_to_PE"] + rec["W_L2_to_L0"]
            total_W_L2_to_L0 += rec["W_L2_to_L0"]

        if debug_data is not None:
            debug_data.setdefault("group_counters", {})[tuple(group_layers)] = {
                "total_W_bytes_L2_to_L0": float(total_W_L2_to_L0.detach().cpu().item()),
                "total_L3_bytes": float(total_L3_bytes.detach().cpu().item()),
                "total_L2_bytes": float(total_L2_bytes.detach().cpu().item()),
                "s_boundaries": [float(s.detach().cpu().item()) for s in s_list],
            }

        # =============== Roofline：内存 vs 计算 =================
        num_pes = hw_params.get_projected_num_pes()
        clk_hz  = self.config.CLOCK_FREQUENCY_MHZ * 1e6

        compute_cycles = total_macs / torch.clamp(
            torch.tensor(float(num_pes), device=dev), min=torch.tensor(1.0, device=dev)
        )

        bw_L3 = calculate_bandwidth_bytes_per_cycle('L3_DRAM', num_pes, self.config)
        bw_L2 = calculate_bandwidth_bytes_per_cycle('L2_Scratchpad', num_pes, self.config)

        mem_cycles = torch.max(
            total_L3_bytes / torch.clamp(bw_L3, min=torch.tensor(1e-9, device=dev)),
            total_L2_bytes / torch.clamp(bw_L2, min=torch.tensor(1e-9, device=dev)),
        )
        total_cycles = torch.max(compute_cycles, mem_cycles)
        latency = total_cycles / clk_hz

        # =============== 能耗：PE + L3 + L2（不计 copy-bundle L1->L2） =================
        e_pe = total_macs * self.config.PE_MAC_EPA_PJ
        e_l3 = (total_L3_bytes / BYTES) * hw_params.get_epa_pj('L3_DRAM')
        e_l2 = (total_L2_bytes / BYTES) * hw_params.get_epa_pj('L2_Scratchpad')
        energy = e_pe + e_l3 + e_l2

        # =============== L2 容量软惩罚（保守峰值，无 overlap） =================
        l2_cap = hw_params.get_buffer_size_kb('L2_Scratchpad') * 1024.0
        guard  = 0.05 * l2_cap
        # 近似峰值：W（整组上界） + 活跃 I/O + 为下一层预留的 O（s_i * LB_O）
        W_resident = torch.tensor(0.0, device=dev)
        for lname in group_layers:
            dims = graph.layers[lname]["dims"]
            W_elems = torch.tensor(1.0, device=dev)
            for d in ("K", "C", "R", "S"):
                W_elems *= torch.tensor(float(dims.get(d, 1)), device=dev)
            W_resident += W_elems * BYTES
        W_resident = torch.clamp(W_resident, max=torch.tensor(l2_cap, device=dev))

        U_candidates = []
        for i, lname in enumerate(group_layers):
            io_active = lb_I_bytes[i] + lb_O_bytes[i]
            stash = (s_list[i] * lb_O_bytes[i]) if i < len(group_layers) - 1 else torch.tensor(0.0, device=dev)
            U_i = W_resident + io_active + stash
            U_candidates.append(U_i)
        U_L2_peak = torch.stack(U_candidates).max()

        kappa   = 0.05 * l2_cap
        cap_pen = torch.nn.functional.softplus((U_L2_peak - (l2_cap - guard)) / kappa) ** 2

        # 兼容性罚：LB反推后相邻天然兼容，这里记0，保留接口
        comp_pen = torch.tensor(0.0, device=dev)

        # 方便画图/校核：暴露本组 W(L2->L0) 总字节
        self._last_group_W_L2_to_L0 = total_W_L2_to_L0.detach()

        return latency, energy, comp_pen, cap_pen

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
                    # print(f"[DEBUG] utilized_pes update: {prev_val.item()} * "
                    #    f"{spatial_factor.item()} = {utilized_pes.item()}")


        effective_pes = torch.max(utilized_pes, torch.tensor(1.0, device=self.config.DEVICE))

        # Print total MACs and utilized PEs for debugging
        print(f"[DEBUG] Total MACs: {total_macs}")
        print(f"[DEBUG] Utilized PEs: {utilized_pes}")

        compute_cycles = total_macs / effective_pes

        print(f"[DEBUG] compute_cycles: {compute_cycles}")

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
            print(f"[DEBUG] Fill traffic: {iface} -> {bytes_total.item():.2f} bytes")

        # 2) read → 从源层被读出（Reads(src)）
        #   （确保你已经有 detailed_read_traffic_info；若没有，请先调用你写的 read 统计函数）
        for iface, info in detailed_read_traffic_info.items():
            src, dst = _iface_split(iface)
            if src is None: 
                continue
            bytes_total = info['total_bytes']
            per_level_reads[src] = per_level_reads[src] + bytes_total
            print(f"[DEBUG] Read traffic: {iface} -> {bytes_total.item():.2f} bytes")

        # 3) writeback/update → 写入到目的层（Updates(dst)）
        for iface, info in detailed_writeback_traffic_info.items():
            src, dst = _iface_split(iface)
            if src is None: 
                continue
            bytes_total = info['total_bytes']
            per_level_updates[dst] = per_level_updates[dst] + bytes_total
            print(f"[DEBUG] Writeback traffic: {iface} -> {bytes_total.item():.2f} bytes")

        # ===== 计算每层的内存侧延迟：Accesses(i)/BW(i) =====
        eps = torch.tensor(1e-9, device=device)
        memory_cycles_list = []
        per_level_cycles_debug = {}

        for lvl in levels:
            accesses_bytes = per_level_reads[lvl] + per_level_writes[lvl] + per_level_updates[lvl]
            bw_bytes_per_cycle = calculate_bandwidth_bytes_per_cycle(lvl, num_pes, self.config)
            mem_cycles = accesses_bytes / (bw_bytes_per_cycle + eps)
            memory_cycles_list.append(mem_cycles)
            
            # Print debug info for each memory level
            print(f"[DEBUG] Memory Level: {lvl}")
            print(f"  - Reads:    {float(per_level_reads[lvl].detach().cpu().item()):.2f} bytes")
            print(f"  - Writes:   {float(per_level_writes[lvl].detach().cpu().item()):.2f} bytes") 
            print(f"  - Updates:  {float(per_level_updates[lvl].detach().cpu().item()):.2f} bytes")
            print(f"  - Total:    {float(accesses_bytes.detach().cpu().item()):.2f} bytes")
            print(f"  - BW:       {float(bw_bytes_per_cycle.detach().cpu().item() if isinstance(bw_bytes_per_cycle, torch.Tensor) else bw_bytes_per_cycle):.2f} bytes/cycle")
            print(f"  - Cycles:   {float(mem_cycles.detach().cpu().item()):.2f}")
            
            # Store debug info
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

        print(f"[DEBUG] compute_cycles: {compute_cycles}")
        print(f"[DEBUG] bottleneck_memory_cycles: {bottleneck_memory_cycles}")
        

        total_cycles = compute_cycles + stall_cycles

        print(f"[DEBUG] total_cycles: {total_cycles}")

        latency = total_cycles / (self.config.CLOCK_FREQUENCY_MHZ * 1e6)



        # 能耗计算
        energy = torch.tensor(0.0, device=self.config.DEVICE)
        energy_compute = total_macs * self.config.PE_MAC_EPA_PJ

        print(f"[DEBUG] energy_compute: {energy_compute}")

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

            # 本项能量
            energy_item = read_words * epa
            energy_read += energy_item

            # 打印调试信息
            print(f"[DEBUG][ReadEnergy] interface={interface}, "
                f"src={source_level_name}, "
                f"bytes={read_bytes}, words={read_words}, "
                f"epa={float(epa):.4f} pJ, "
                f"energy={float(energy_item):.2f} pJ")

        print(f"[DEBUG][ReadEnergy] total={float(energy_read):.2f} pJ")


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
                debug_output_path: str = None):
        """
        Args:
            graph: 计算图
            hw_params: 硬件参数
            mapping: FineGrainedMapping实例
        
        Returns:
            (total_latency, total_energy, area_cost,
            total_buffer_mismatch_loss, total_compatibility_penalty,
            mapping_invalid_penalty, penalty)
        """
        dev = self.config.DEVICE

        # 累计量初始化
        total_latency  = torch.tensor(0.0, device=dev)
        total_energy   = torch.tensor(0.0, device=dev)
        total_buffer_mismatch_loss = torch.tensor(0.0, device=dev)
        total_compatibility_penalty = torch.tensor(0.0, device=dev)
        total_capacity_penalty = torch.tensor(0.0, device=dev)

        # factor 解析
        all_factors = format_mapping_as_all_factors(mapping)

        # === 确定融合组 ===
        if len(graph.fusion_groups) == 0:
            fusion_groups = [[layer_name] for layer_name in graph.layers.keys()]
        else:
            fusion_groups = graph.fusion_groups

        # === 遍历融合组 ===
        for _, group in enumerate(fusion_groups):
            if len(group) == 1:
                # 单层
                layer_name = group[0]
                lat, en, mismatch = self._evaluate_single_layer(
                    layer_name, graph, hw_params, mapping, all_factors, debug_data=None
                )
                total_latency += lat
                total_energy  += en
                total_buffer_mismatch_loss += mismatch
            else:
                # 多层组：概率路由 + 容量约束
                lat, en, comp_pen, cap_pen = self.evaluate_group_depth_first(
                    group, graph, hw_params, mapping, all_factors, debug_data=None
                )
                total_latency += lat
                total_energy  += en
                total_compatibility_penalty += comp_pen
                total_capacity_penalty      += cap_pen

        # === 额外成本 ===
        area_cost = hw_params.get_area_cost()
        mapping_invalid_penalty = self.compute_invalid_penalty(mapping)

        # === 总 penalty ===
        penalty = (mapping_invalid_penalty
                + total_buffer_mismatch_loss
                + total_compatibility_penalty
                + total_capacity_penalty)

        return (total_latency, total_energy, area_cost,
                total_buffer_mismatch_loss, total_compatibility_penalty,
                mapping_invalid_penalty, penalty)

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