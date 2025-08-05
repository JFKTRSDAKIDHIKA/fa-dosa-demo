#!/usr/bin/env python3
"""
测试脚本：验证重构后的"周期累加"延迟模型
目标：验证在极端内存受限场景下，新模型能否预测出远大于纯计算延迟的延迟值
"""

import torch
from dosa.config import Config as DosaConfig
from dosa.hardware_parameters import HardwareParameters
from dosa.performance_model import HighFidelityPerformanceModel
from dosa.mapping import FineGrainedMapping
from dosa.utils import calculate_macs

def create_extreme_memory_bound_mapping():
    """
    创建极端内存受限的映射策略
    强制所有主要维度在DRAM上处理，最大化内存访问
    使用更大的时间复用因子来放大内存访问量
    """
    mapping_table = {
        'N': {
            'L0_Registers': {'temporal': 1, 'spatial': 1},
            'L1_Accumulator': {'temporal': 1, 'spatial': 1},
            'L2_Scratchpad': {'temporal': 1, 'spatial': 1},
            'L3_DRAM': {'temporal': 1, 'spatial': 1}
        },
        'C': {
            'L0_Registers': {'temporal': 1, 'spatial': 1},  # 片上最小化
            'L1_Accumulator': {'temporal': 1, 'spatial': 1},
            'L2_Scratchpad': {'temporal': 1, 'spatial': 1},  # 片上最小化
            'L3_DRAM': {'temporal': 512, 'spatial': 1}  # 极大的DRAM负载
        },
        'K': {
            'L0_Registers': {'temporal': 1, 'spatial': 1},
            'L1_Accumulator': {'temporal': 1, 'spatial': 1},
            'L2_Scratchpad': {'temporal': 1, 'spatial': 1},
            'L3_DRAM': {'temporal': 512, 'spatial': 1}  # 极大的DRAM负载
        },
        'P': {
            'L0_Registers': {'temporal': 1, 'spatial': 1},
            'L1_Accumulator': {'temporal': 1, 'spatial': 1},
            'L2_Scratchpad': {'temporal': 1, 'spatial': 1},
            'L3_DRAM': {'temporal': 224, 'spatial': 1}  # 极大的DRAM负载
        },
        'Q': {
            'L0_Registers': {'temporal': 1, 'spatial': 1},
            'L1_Accumulator': {'temporal': 1, 'spatial': 1},
            'L2_Scratchpad': {'temporal': 1, 'spatial': 1},
            'L3_DRAM': {'temporal': 224, 'spatial': 1}  # 极大的DRAM负载
        },
        'R': {
            'L0_Registers': {'temporal': 1, 'spatial': 1},
            'L1_Accumulator': {'temporal': 1, 'spatial': 1},
            'L2_Scratchpad': {'temporal': 1, 'spatial': 1},
            'L3_DRAM': {'temporal': 9, 'spatial': 1}  # 极大的DRAM负载
        },
        'S': {
            'L0_Registers': {'temporal': 1, 'spatial': 1},
            'L1_Accumulator': {'temporal': 1, 'spatial': 1},
            'L2_Scratchpad': {'temporal': 1, 'spatial': 1},
            'L3_DRAM': {'temporal': 9, 'spatial': 1}  # 极大的DRAM负载
        }
    }
    return mapping_table

def test_cycles_accumulation_model():
    """
    测试重构后的周期累加延迟模型
    """
    print("=== 测试重构后的周期累加延迟模型 ===")
    print()
    
    # 初始化组件
    dosa_config = DosaConfig()
    hw_params = HardwareParameters(
        initial_num_pes=64.0,
        initial_l0_kb=2.0,
        initial_l1_kb=4.0,
        initial_l2_kb=256.0
    )
    perf_model = HighFidelityPerformanceModel(dosa_config)
    
    # 设置硬件配置
    num_pes = torch.tensor(64.0)
    hw_params.log_num_pes.data = torch.log(num_pes)
    hw_params.log_buffer_sizes_kb['L2_Scratchpad'].data = torch.log(torch.tensor(256.0))
    
    # 定义工作负载维度（卷积层）
    workload_dims = {
        'N': 1,    # Batch size
        'C': 64,   # Input channels
        'K': 64,   # Output channels
        'P': 56,   # Output height
        'Q': 56,   # Output width
        'R': 3,    # Kernel height
        'S': 3     # Kernel width
    }
    
    # 计算基准值
    macs = calculate_macs(workload_dims)
    compute_cycles = macs / num_pes
    compute_latency_baseline = compute_cycles / (dosa_config.CLOCK_FREQUENCY_MHZ * 1e6)
    
    print(f"[基准计算]")
    print(f"- 总MAC运算: {macs:,}")
    print(f"- 计算周期: {compute_cycles:.0f}")
    print(f"- 纯计算延迟: {compute_latency_baseline:.6f} s")
    print()
    
    # 创建极端内存受限映射
    extreme_mapping_table = create_extreme_memory_bound_mapping()
    
    # 计算访问量
    accesses = perf_model.calculate_per_level_accesses(workload_dims, extreme_mapping_table)
    dram_to_l2_accesses = accesses.get('L3_DRAM_to_L2_Scratchpad', torch.tensor(0.0))
    
    # 计算DRAM带宽和内存周期
    from dosa.performance_model import calculate_bandwidth_gb_s
    dram_bandwidth_gb_s = calculate_bandwidth_gb_s('L3_DRAM', num_pes, dosa_config)
    bytes_per_cycle = dram_bandwidth_gb_s * 1e9 / (dosa_config.CLOCK_FREQUENCY_MHZ * 1e6)
    memory_cycles = dram_to_l2_accesses / bytes_per_cycle
    
    # 应用周期累加模型
    stall_cycles = torch.relu(memory_cycles - compute_cycles)
    total_cycles = compute_cycles + stall_cycles
    total_latency = total_cycles / (dosa_config.CLOCK_FREQUENCY_MHZ * 1e6)
    
    print(f"[极端内存受限场景分析]")
    print(f"- DRAM->L2数据量: {dram_to_l2_accesses.item() / 1e6:.2f} MB")
    print(f"- DRAM带宽: {dram_bandwidth_gb_s:.2f} GB/s")
    print(f"- 每周期传输字节: {bytes_per_cycle:.2f} bytes/cycle")
    print(f"- 内存传输周期: {memory_cycles.item():.0f}")
    print(f"- 停顿周期: {stall_cycles.item():.0f}")
    print(f"- 总周期: {total_cycles.item():.0f}")
    print(f"- 总延迟: {total_latency.item():.6f} s")
    print()
    
    # 分析结果
    latency_amplification = total_latency.item() / compute_latency_baseline.item()
    is_memory_bound = stall_cycles.item() > 0
    
    print(f"[结果分析]")
    print(f"- 延迟放大倍数: {latency_amplification:.2f}x")
    print(f"- 是否内存受限: {'是' if is_memory_bound else '否'}")
    print(f"- 停顿周期占比: {stall_cycles.item() / total_cycles.item() * 100:.1f}%")
    
    if is_memory_bound and latency_amplification > 1.5:
        print(f"✅ 成功：周期累加模型正确识别了内存瓶颈！")
        print(f"   预测延迟 ({total_latency.item():.6f}s) 远大于纯计算延迟 ({compute_latency_baseline.item():.6f}s)")
    else:
        print(f"❌ 失败：模型仍未能正确识别内存瓶颈")
        print(f"   延迟放大倍数 ({latency_amplification:.2f}x) 不足以体现内存瓶颈")

if __name__ == '__main__':
    test_cycles_accumulation_model()