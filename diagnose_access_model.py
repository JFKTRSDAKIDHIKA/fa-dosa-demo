#!/usr/bin/env python3
"""
诊断脚本：验证内存访问模型对映射策略的敏感性

本脚本通过构建两个对比鲜明的映射策略来验证核心假设：
calculate_per_level_accesses 函数是否能正确识别由糟糕映射策略导致的内存瓶颈。
"""

import torch
from dosa.config import Config as DosaConfig
from dosa.hardware_parameters import HardwareParameters
from dosa.performance_model import HighFidelityPerformanceModel, calculate_bandwidth_gb_s
from dosa.utils import calculate_macs

def create_performance_optimal_mapping():
    """
    场景A: 性能优先映射策略
    将绝大部分Tiling因子放在片上缓存，DRAM的Tiling因子尽可能为1
    """
    mapping_table = {
        'N': {
            'L0_Registers': {'temporal': 1, 'spatial': 1},
            'L1_Accumulator': {'temporal': 1, 'spatial': 1}, 
            'L2_Scratchpad': {'temporal': 1, 'spatial': 1},
            'L3_DRAM': {'temporal': 1, 'spatial': 1}
        },
        'C': {
            'L0_Registers': {'temporal': 2, 'spatial': 2},  # 片上复用
            'L1_Accumulator': {'temporal': 4, 'spatial': 1},
            'L2_Scratchpad': {'temporal': 4, 'spatial': 1}, 
            'L3_DRAM': {'temporal': 1, 'spatial': 1}  # DRAM最小化
        },
        'K': {
            'L0_Registers': {'temporal': 2, 'spatial': 2},
            'L1_Accumulator': {'temporal': 4, 'spatial': 1},
            'L2_Scratchpad': {'temporal': 4, 'spatial': 1},
            'L3_DRAM': {'temporal': 1, 'spatial': 1}
        },
        'P': {
            'L0_Registers': {'temporal': 2, 'spatial': 1},
            'L1_Accumulator': {'temporal': 7, 'spatial': 1},
            'L2_Scratchpad': {'temporal': 4, 'spatial': 1},
            'L3_DRAM': {'temporal': 1, 'spatial': 1}
        },
        'Q': {
            'L0_Registers': {'temporal': 2, 'spatial': 1},
            'L1_Accumulator': {'temporal': 7, 'spatial': 1},
            'L2_Scratchpad': {'temporal': 4, 'spatial': 1},
            'L3_DRAM': {'temporal': 1, 'spatial': 1}
        },
        'R': {
            'L0_Registers': {'temporal': 1, 'spatial': 1},
            'L1_Accumulator': {'temporal': 1, 'spatial': 1},
            'L2_Scratchpad': {'temporal': 3, 'spatial': 1},  # 片上复用
            'L3_DRAM': {'temporal': 1, 'spatial': 1}  # DRAM最小化
        },
        'S': {
            'L0_Registers': {'temporal': 1, 'spatial': 1},
            'L1_Accumulator': {'temporal': 1, 'spatial': 1},
            'L2_Scratchpad': {'temporal': 3, 'spatial': 1},  # 片上复用
            'L3_DRAM': {'temporal': 1, 'spatial': 1}  # DRAM最小化
        }
    }
    return mapping_table

def create_dram_heavy_mapping():
    """
    场景B: DRAM主导映射策略
    使用更极端的内存受限映射来验证重构后的周期累加模型
    将大部分计算负载强制放在DRAM层级
    """
    mapping_table = {
        'N': {
            'L0_Registers': {'temporal': 1, 'spatial': 1},
            'L1_Accumulator': {'temporal': 1, 'spatial': 1},
            'L2_Scratchpad': {'temporal': 1, 'spatial': 1},
            'L3_DRAM': {'temporal': 1, 'spatial': 1}
        },
        'C': {
            'L0_Registers': {'temporal': 1, 'spatial': 1},
            'L1_Accumulator': {'temporal': 1, 'spatial': 1},
            'L2_Scratchpad': {'temporal': 1, 'spatial': 1},
            'L3_DRAM': {'temporal': 64, 'spatial': 1}  # 使用最大合法值64
        },
        'K': {
            'L0_Registers': {'temporal': 1, 'spatial': 1},
            'L1_Accumulator': {'temporal': 1, 'spatial': 1},
            'L2_Scratchpad': {'temporal': 1, 'spatial': 1},
            'L3_DRAM': {'temporal': 64, 'spatial': 1}  # 使用最大合法值64
        },
        'P': {
            'L0_Registers': {'temporal': 1, 'spatial': 1},
            'L1_Accumulator': {'temporal': 1, 'spatial': 1},
            'L2_Scratchpad': {'temporal': 1, 'spatial': 1},
            'L3_DRAM': {'temporal': 56, 'spatial': 1}  # 使用最大合法值56
        },
        'Q': {
            'L0_Registers': {'temporal': 1, 'spatial': 1},
            'L1_Accumulator': {'temporal': 1, 'spatial': 1},
            'L2_Scratchpad': {'temporal': 1, 'spatial': 1},
            'L3_DRAM': {'temporal': 56, 'spatial': 1}  # 使用最大合法值56
        },
        'R': {
            'L0_Registers': {'temporal': 1, 'spatial': 1},
            'L1_Accumulator': {'temporal': 1, 'spatial': 1},
            'L2_Scratchpad': {'temporal': 1, 'spatial': 1},
            'L3_DRAM': {'temporal': 3, 'spatial': 1}  # DRAM承担主要负载
        },
        'S': {
            'L0_Registers': {'temporal': 1, 'spatial': 1},
            'L1_Accumulator': {'temporal': 1, 'spatial': 1},
            'L2_Scratchpad': {'temporal': 1, 'spatial': 1},
            'L3_DRAM': {'temporal': 3, 'spatial': 1}  # DRAM承担主要负载
        }
    }
    return mapping_table

def verify_mapping_consistency(mapping_table, workload_dims):
    """
    验证映射策略的一致性：确保每个维度的所有因子乘积等于工作负载维度
    """
    for dim_name, dim_size in workload_dims.items():
        if dim_name in mapping_table:
            total_product = 1
            for level_name in ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad', 'L3_DRAM']:
                if level_name in mapping_table[dim_name]:
                    temporal = mapping_table[dim_name][level_name]['temporal']
                    spatial = mapping_table[dim_name][level_name]['spatial']
                    total_product *= temporal * spatial
            
            if total_product != dim_size:
                print(f"警告：维度 {dim_name} 的因子乘积 ({total_product}) 不等于工作负载大小 ({dim_size})")
                return False
    return True

def run_diagnosis():
    """
    执行诊断测试的主函数
    """
    print("--- 诊断测试: 内存访问模型对映射策略的敏感性 ---")
    print()
    
    # 初始化FA-DOSA组件
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
    
    # 计算基准计算延迟
    macs = calculate_macs(workload_dims)
    compute_latency = macs / (num_pes * dosa_config.CLOCK_FREQUENCY_MHZ * 1e6 + 1e-9)
    
    print("[实验设置]")
    print(f"- 硬件配置: num_pes = {int(num_pes.item())}, L2 Cache = 256 KB")
    print(f"- 基准计算延迟 (Compute Latency Benchmark): {compute_latency:.6f} s")
    print()
    
    # 场景A: 性能优先映射
    print("-" * 50)
    print("[场景 A: Performance-Optimal Mapping]")
    print("- 策略描述: 数据尽可能在片上复用，DRAM访问最小化。")
    
    performance_mapping = create_performance_optimal_mapping()
    
    # 验证映射一致性
    if not verify_mapping_consistency(performance_mapping, workload_dims):
        print("错误：性能优先映射策略不一致！")
        return
    
    # 计算访问量
    perf_accesses = perf_model.calculate_per_level_accesses(workload_dims, performance_mapping)
    dram_to_l2_accesses_perf = perf_accesses.get('L3_DRAM_to_L2_Scratchpad', torch.tensor(0.0))
    
    # 计算DRAM带宽和延迟
    dram_bandwidth = calculate_bandwidth_gb_s('L3_DRAM', num_pes, dosa_config)
    if hasattr(dram_bandwidth, 'item'):
        dram_bw_val = dram_bandwidth.item()
    else:
        dram_bw_val = float(dram_bandwidth)
    
    if hasattr(dram_to_l2_accesses_perf, 'item'):
        dram_accesses_perf_val = dram_to_l2_accesses_perf.item()
    else:
        dram_accesses_perf_val = float(dram_to_l2_accesses_perf)
    
    # 使用重构后的周期累加模型进行瓶颈判断
    compute_cycles = macs / num_pes
    bytes_per_cycle = dram_bw_val * 1e9 / (dosa_config.CLOCK_FREQUENCY_MHZ * 1e6)
    memory_cycles_perf = dram_accesses_perf_val / bytes_per_cycle
    stall_cycles_perf = max(0, memory_cycles_perf - compute_cycles)
    total_cycles_perf = compute_cycles + stall_cycles_perf
    total_latency_perf = total_cycles_perf / (dosa_config.CLOCK_FREQUENCY_MHZ * 1e6)
    
    print(f"- 计算出的DRAM->L2数据量: {dram_accesses_perf_val / 1e6:.2f} MB")
    print(f"- 预测的总延迟: {total_latency_perf:.6f} s")
    
    if stall_cycles_perf > 0:
        bottleneck_perf = "内存受限"
    else:
        bottleneck_perf = "计算受限"
    print(f"- [诊断结果]: (停顿周期 = {stall_cycles_perf:.0f}) -> 模型判定为 [{bottleneck_perf}]")
    print()
    
    # 场景B: DRAM主导映射
    print("-" * 50)
    print("[场景 B: DRAM-Heavy (Pathological) Mapping]")
    print("- 策略描述: 强制部分和在DRAM和L2之间交换，模拟最差情况。")
    
    dram_heavy_mapping = create_dram_heavy_mapping()
    
    # 验证映射一致性
    if not verify_mapping_consistency(dram_heavy_mapping, workload_dims):
        print("错误：DRAM主导映射策略不一致！")
        return
    
    # 计算访问量
    dram_accesses = perf_model.calculate_per_level_accesses(workload_dims, dram_heavy_mapping)
    dram_to_l2_accesses_heavy = dram_accesses.get('L3_DRAM_to_L2_Scratchpad', torch.tensor(0.0))
    
    if hasattr(dram_to_l2_accesses_heavy, 'item'):
        dram_accesses_heavy_val = dram_to_l2_accesses_heavy.item()
    else:
        dram_accesses_heavy_val = float(dram_to_l2_accesses_heavy)
    
    # 使用重构后的周期累加模型进行瓶颈判断
    memory_cycles_heavy = dram_accesses_heavy_val / bytes_per_cycle
    stall_cycles_heavy = max(0, memory_cycles_heavy - compute_cycles)
    total_cycles_heavy = compute_cycles + stall_cycles_heavy
    total_latency_heavy = total_cycles_heavy / (dosa_config.CLOCK_FREQUENCY_MHZ * 1e6)
    
    print(f"- 计算出的DRAM->L2数据量: {dram_accesses_heavy_val / 1e6:.2f} MB")
    print(f"- 预测的总延迟: {total_latency_heavy:.6f} s")
    
    if stall_cycles_heavy > 0:
        bottleneck_heavy = "内存受限"
    else:
        bottleneck_heavy = "计算受限"
    print(f"- [诊断结果]: (停顿周期 = {stall_cycles_heavy:.0f}) -> 模型判定为 [{bottleneck_heavy}]")
    print()
    
    # 最终诊断结论
    print("=" * 50)
    print("[最终诊断结论]")
    
    # 计算访问量放大效应
    if dram_accesses_perf_val > 0:
        amplification_factor = dram_accesses_heavy_val / dram_accesses_perf_val
    else:
        amplification_factor = float('inf')
    
    print(f"- 访问量放大效应: {amplification_factor:.2f} 倍")
    
    # 判断瓶颈误判
    expected_bottleneck_change = (bottleneck_perf == "计算受限" and bottleneck_heavy == "内存受限")
    if expected_bottleneck_change:
        bottleneck_detection = "成功"
    else:
        bottleneck_detection = "失败"
    
    print(f"- 瓶颈误判: {bottleneck_detection}")
    
    # 结论
    print("- 结论: ", end="")
    if amplification_factor > 2.0 and expected_bottleneck_change:
        print("calculate_per_level_accesses 函数对映射策略敏感，能够正确识别内存瓶颈。")
    elif amplification_factor > 2.0 and not expected_bottleneck_change:
        print("calculate_per_level_accesses 函数能够检测到访问量差异，但由于计算延迟过大，仍无法正确识别内存瓶颈。")
    elif amplification_factor <= 2.0:
        print("calculate_per_level_accesses 函数对映射策略不敏感，未能正确计算数据访问放大效应，这是导致瓶颈误判的根源。")
    else:
        print("诊断结果不明确，需要进一步分析。")

if __name__ == '__main__':
    run_diagnosis()