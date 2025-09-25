import json
import subprocess
import os
import itertools
import pandas as pd
import random
import argparse
import math
import yaml
import multiprocessing
import contextlib
from pathlib import Path
import torch
import math
import torch.nn as nn

# FA-DOSA core modules
from dosa.config import Config as DosaConfig
from dosa.hardware_parameters import HardwareParameters
from dosa.mapping import FineGrainedMapping
from dosa.utils import ComputationGraph, get_divisors
from dosa.performance_model import HighFidelityPerformanceModel

# Timeloop integration
import pytimeloop.timeloopfe.v4 as tl

# Timeloop YAML node classes
class Component(dict):
    """Represents a Timeloop Component node (leaf node)."""
    pass

class Container(dict):
    """Represents a Timeloop Container node (leaf node)."""
    pass

class Hierarchical(dict):
    """Represents a Timeloop Hierarchical node (branch node)."""
    pass

# YAML representers
def represent_component(dumper, data):
    """YAML representer for Component class."""
    return dumper.represent_mapping('!Component', data.items())

def represent_container(dumper, data):
    """YAML representer for Container class."""
    return dumper.represent_mapping('!Container', data.items())

def represent_hierarchical(dumper, data):
    """YAML representer for Hierarchical class."""
    return dumper.represent_mapping('!Hierarchical', data.items())

# Register the representers globally
yaml.add_representer(Component, represent_component)
yaml.add_representer(Container, represent_container)
yaml.add_representer(Hierarchical, represent_hierarchical)

# Set up environment variables for Timeloop
os.environ['PATH'] = os.environ.get('PATH', '') + ':/root/accelergy-timeloop-infrastructure/src/timeloop/bin:/root/accelergy-timeloop-infrastructure/src/timeloop/build'
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':/root/accelergy-timeloop-infrastructure/src/timeloop/lib:/root/accelergy-timeloop-infrastructure/src/timeloop/build'

# --- Define Search Spaces ---

# Global configuration
MAX_VALIDATION_RUNS = None  # Set to None for full sweep, or specify a number (e.g., 1, 2, 4) for quick testing
VALIDATION_TIMEOUT_SECONDS = 600  # 10 minutes timeout for each validation point

# 1. Single Convolution Layer to be tested
CONV_LAYER_CONFIG = {
    "layer_name": "conv1",
    "layer_type": "Conv",
    "dims": {"N": 1, "C": 64, "K": 64, "P": 56, "Q": 56, "R": 3, "S": 3}
}

def make_hw_config(num_pes: int = 64, l0_kb: float = 128.0, l1_kb: float = 128.0, l2_kb: float = 256.0) -> dict:
    """创建标准化的硬件配置字典，避免硬编码和重复定义。
    
    Args:
        num_pes: PE数量
        l0_kb: L0寄存器大小（KB）
        l1_kb: L1累加器大小（KB）  
        l2_kb: L2暂存器大小（KB）
        
    Returns:
        dict: 标准化的硬件配置字典
    """
    return {
        "num_pes": num_pes,
        "l0_registers_size_kb": l0_kb,
        "l1_accumulator_size_kb": l1_kb,
        "l2_scratchpad_size_kb": l2_kb
    }

# 3. Mapping Space (simplified, needs to be dimension-dependent)
MAPPING_SPACE = {
    "K": [1, 2, 4, 8, 16, 32],
    "C": [1, 2, 4, 8, 16, 32]
}

def generate_factors_by_strategy(dim_size: int, num_pes_sqrt: int, strategy: str) -> dict:
    """根据指定的策略，为单个维度生成跨存储层级的tiling因子。

    Args:
        dim_size (int): 需要分解的维度大小。
        num_pes_sqrt (int): PE数量的平方根，用于约束空间因子。
        strategy (str): 生成策略，可选值为 "performance", "dram_heavy", "random"。

    Returns:
        dict: 包含 'spatial', 'L0', 'L1', 'L2', 'DRAM' 因子的字典。
    """
    if dim_size == 1:
        return {'spatial': 1, 'L0': 1, 'L1': 1, 'L2': 1, 'DRAM': 1}

    # 1. 首先，确定空间因子 (spatial factor)
    divisors = get_divisors(dim_size).tolist()
    spatial_candidates = [d for d in divisors if d <= num_pes_sqrt]
    spatial_factor = random.choice(spatial_candidates) if spatial_candidates else 1
    
    remaining_size = dim_size // spatial_factor
    
    # 2. 根据策略分配剩余的时间因子 (temporal factors)
    factors = {'spatial': int(spatial_factor), 'L0': 1, 'L1': 1, 'L2': 1, 'DRAM': 1}
    
    if remaining_size > 1:
        if strategy == "performance":
            # 性能优先：从L0开始，从剩余约数中随机选一个，然后迭代
            temp_size = remaining_size
            for level in ['L1', 'L2']:
                if temp_size == 1: break
                level_divs = [d for d in get_divisors(temp_size).tolist() if d > 1]
                if not level_divs: continue
                factor_for_level = random.choice(level_divs)
                factors[level] = int(factor_for_level)
                temp_size //= factor_for_level
            factors['DRAM'] = int(temp_size)

        elif strategy == "dram_heavy":
            # DRAM主导：将所有剩余因子全部分配给DRAM
            factors['DRAM'] = int(remaining_size)

        elif strategy == "random":
            # 均衡随机：将剩余尺寸随机分解给 L0, L1, L2, DRAM
            temp_size = remaining_size
            levels = ['L1', 'L2', 'DRAM']
            random.shuffle(levels)
            for level in levels:
                if temp_size == 1: break
                level_divs = [d for d in get_divisors(temp_size).tolist() if d > 1]
                if not level_divs: continue
                factor_for_level = random.choice(level_divs)
                factors[level] = int(factor_for_level)
                temp_size //= factor_for_level
            # 确保乘积正确，将余数给最后一个分配的level
            if temp_size > 1:
                 factors[levels[-1]] *= int(temp_size)

    # 3. 验证所有因子的乘积是否等于原始维度大小
    product = 1
    for val in factors.values(): product *= val
    assert product == dim_size, f"Factor product check failed for strategy {strategy}!"

    return factors

def _assert_constraints(mapping_config, dims, pe_count):
    m = mapping_config["conv1"]
    # (1) spatial 只在 L0，且乘积 ≤ PE
    assert "spatial" in m["L0_Registers"] and all(
        lvl not in m or "spatial" not in m[lvl] for lvl in ["L1_Accumulator", "L2_Scratchpad", "DRAM"]
    ), "Spatial tiling must exist only at L0."
    spatial_prod = 1
    for d in m["L0_Registers"]["spatial"].values():
        spatial_prod *= d
    assert spatial_prod <= pe_count, f"Spatial product {spatial_prod} exceeds PE count {pe_count}."

    # (2) L0 temporal 除 PQ 外均为 1
    l0t = m["L0_Registers"]["temporal"]
    for k, v in l0t.items():
        if k not in ("P", "Q"):
            assert v == 1, f"L0 temporal on {k} must be 1 (got {v})."

    # (3) PQ 仅在 L0/L3
    for lvl in ["L1_Accumulator", "L2_Scratchpad"]:
        assert m[lvl]["temporal"]["P"] == 1 and m[lvl]["temporal"]["Q"] == 1, f"PQ must not be temporally tiled at {lvl}."
    assert m["DRAM"]["temporal"]["P"] >= 1 and m["DRAM"]["temporal"]["Q"] >= 1

    # (4) R/S/C 的 temporal 仅在 L1
    for lvl in ["L0_Registers", "L2_Scratchpad", "DRAM"]:
        for red in ["R", "S", "C"]:
            tv = m[lvl]["temporal"].get(red, 1)
            assert tv == 1, f"Reduction dim {red} must not be temporally tiled at {lvl} (got {tv})."

    # (5) permutation 一致
    perms = {m[lvl]["permutation"] for lvl in ["L0_Registers","L1_Accumulator","L2_Scratchpad","DRAM"]}
    assert len(perms) == 1, f"Permutations must be identical across levels (got {perms})."


def generate_constrained_mapping_configs(num_configs: int = 10):
    """
    生成符合约束条件的mapping配置
    """
    fixed_permutation = "P Q K C R S N"
    dims = CONV_LAYER_CONFIG["dims"]
    pe_count = 64
    pe_sqrt = int(pe_count ** 0.5)

    for config_id in range(num_configs):
        # ---- L0 spatial (K/C only) ----
        k_spatial_candidates = [d for d in get_divisors(dims["K"]).tolist() if d <= pe_sqrt]
        c_spatial_candidates = [d for d in get_divisors(dims["C"]).tolist() if d <= pe_sqrt]
        k_spatial = random.choice(k_spatial_candidates)
        max_c_spatial = min(pe_count // k_spatial, max(c_spatial_candidates))
        c_spatial_valid = [d for d in c_spatial_candidates if d <= max_c_spatial]
        c_spatial = random.choice(c_spatial_valid) if c_spatial_valid else 1

        # ---- L1 R/S/C temporal ----
        # 改成：吃满 R/S
        r_l1 = dims["R"]
        s_l1 = dims["S"]

        # C 仍然吃满 after L0 spatial
        c_after_l0 = dims["C"] // c_spatial
        c_l1 = c_after_l0

        # ---- L0 + DRAM PQ temporal ----
        p_l0_candidates = get_divisors(dims["P"]).tolist()
        q_l0_candidates = get_divisors(dims["Q"]).tolist()
        p_l0 = random.choice(p_l0_candidates)
        q_l0 = random.choice(q_l0_candidates)
        p_dram = dims["P"] // p_l0
        q_dram = dims["Q"] // q_l0

        # ---- K 分解：L2/L1 ----
        k_remaining_after_l0 = dims["K"] // k_spatial
        k_l2_candidates = get_divisors(k_remaining_after_l0).tolist()
        k_l2 = random.choice(k_l2_candidates)
        k_l1 = k_remaining_after_l0 // k_l2

        # ---- DRAM 禁止 R/S/C temporal ----
        c_dram, r_dram, s_dram = 1, 1, 1

        # ---- build mapping ----
        mapping_config = {
            "conv1": {
                "DRAM": {
                    "temporal": {"N": 1, "C": c_dram, "K": 1, "P": p_dram, "Q": q_dram, "R": r_dram, "S": s_dram},
                    "permutation": fixed_permutation
                },
                "L2_Scratchpad": {
                    "temporal": {"N": 1, "C": 1, "K": k_l2, "P": 1, "Q": 1, "R": 1, "S": 1},
                    "permutation": fixed_permutation
                },
                "L1_Accumulator": {
                    "temporal": {"N": 1, "C": c_l1, "K": k_l1, "P": 1, "Q": 1, "R": r_l1, "S": s_l1},
                    "permutation": fixed_permutation
                },
                "L0_Registers": {
                    "temporal": {"N": 1, "C": 1, "K": 1, "P": p_l0, "Q": q_l0, "R": 1, "S": 1},
                    "spatial": {"C": c_spatial, "K": k_spatial},
                    "permutation": fixed_permutation
                }
            }
        }

        # ---- 验证维度乘积 ----
        for dim_name, total_size in dims.items():
            product = 1
            for level in ["DRAM", "L2_Scratchpad", "L1_Accumulator", "L0_Registers"]:
                if level == "L0_Registers":
                    t = mapping_config["conv1"][level]["temporal"].get(dim_name, 1)
                    s = mapping_config["conv1"][level].get("spatial", {}).get(dim_name, 1)
                    product *= t * s
                else:
                    t = mapping_config["conv1"][level]["temporal"].get(dim_name, 1)
                    product *= t
            if product != total_size:
                print(f"Warning: {dim_name} product mismatch {product} != {total_size}")

        # ---- 构建 config ----
        config = {
            "hardware_config": make_hw_config(num_pes=64, l0_kb=128.0, l1_kb=4.0, l2_kb=256.0),
            "mapping_config": mapping_config,
            "layer_info": {
                "layer_name": "conv1",
                "layer_type": "Conv",
                "dims": dims
            }
        }

        # ---- 硬断言约束 ----
        _assert_constraints(mapping_config, dims, pe_count)
        yield config


def get_fixed_validation_config():
    """返回 validation_point_id=1 的固定配置字典"""
    config_vp1 = {
        "hardware_config": make_hw_config(num_pes=64, l0_kb=128.0, l1_kb=4.0, l2_kb=256.0),
        "mapping_config": {
            "conv1": {
                "DRAM": {
                    "temporal": {"N":1,"K":1,"C":1,"P":28,"Q":2,"R":1,"S":1},
                    "permutation": "P Q K C R S N"
                },
                "L2_Scratchpad": {
                    "temporal": {"N":1,"K":2,"C":1,"P":1,"Q":1,"R":1,"S":1},
                    "permutation": "P Q K C R S N"
                },
                "L1_Accumulator": {
                    "temporal": {"N":1,"K":32,"C":8,"P":1,"Q":1,"R":3,"S":3},
                    "permutation": "P Q K C R S N"
                },
                "L0_Registers": {
                    "temporal": {"N":1,"K":1,"C":1,"P":2,"Q":28,"R":1,"S":1},
                    "spatial": {"K":1,"C":8},
                    "permutation": "P Q K C R S N"
                }
            }
        },
        "layer_info": {
            "layer_name": "conv1",
            "layer_type": "Conv",
            "dims": {"N":1,"C":64,"K":64,"P":56,"Q":56,"R":3,"S":3}
        }
    }
    return config_vp1

def generate_configurations(num_configs: int):
    """Generates a stream of unique configurations for single convolution layer validation.
    
    Args:
        num_configs (int): Number of configurations to generate.
        
    Yields:
        dict: Configuration dictionary containing layer_info, hardware_config, mapping_config.
    """
    # Define some typical permutation patterns
    permutation_patterns = [
        'K', 'C', 'R', 'S', 'N', 'P', 'Q',
    ]
    
    # Define available strategies
    strategies = ["performance"]
    
    # Main loop for generating configurations
    for i in range(num_configs):
        layer_name = CONV_LAYER_CONFIG["layer_name"]
        dims = CONV_LAYER_CONFIG["dims"]
        
        # Generate hardware configuration using make_hw_config with default parameters
        hardware_config = make_hw_config()
        
        # Calculate PE mesh size for spatial constraints
        pe_mesh_size = int(hardware_config['num_pes'] ** 0.5)
        
        # Randomly select a strategy for this configuration
        chosen_strategy = random.choice(strategies)
        
        # Generate complete factorization for each dimension using the chosen strategy
        dim_factors = {}
        for dim_name, dim_size in dims.items():
            if dim_name in ['N', 'K', 'C', 'P', 'Q', 'R', 'S']:
                dim_factors[dim_name] = generate_factors_by_strategy(dim_size, pe_mesh_size, chosen_strategy)
        
        # Build mapping config with complete factors
        mapping_config = {
            layer_name: {
                'DRAM': {
                    'temporal': {dim: factors['DRAM'] for dim, factors in dim_factors.items()},
                    'permutation': random.choice(permutation_patterns)
                },
                'L2_Scratchpad': {
                    'temporal': {dim: factors['L2'] for dim, factors in dim_factors.items()},
                    'permutation': random.choice(permutation_patterns)
                },
                'L1_Accumulator': {
                    'temporal': {dim: factors['L1'] for dim, factors in dim_factors.items()},
                    'permutation': random.choice(permutation_patterns)
                },
                'L0_Registers': {
                    'temporal': {dim: factors['L0'] for dim, factors in dim_factors.items()},
                    'spatial': {
                        dim: factors['spatial']
                        for dim, factors in dim_factors.items()
                        if dim in ['K', 'C']
                    },
                    'permutation': random.choice(permutation_patterns)
                }
            }
        }
        
        yield {
             "layer_info": CONV_LAYER_CONFIG,
             "hardware_config": hardware_config,
             "mapping_config": mapping_config
         }

def run_dosa_prediction(config: dict, validation_point_id: int = None, output_dir: Path = None) -> dict:
    print("[INFO] Running FA-DOSA analytical model for single convolution layer...")
    # 1. 解包配置字典
    hw_config = config['hardware_config']
    mapping_config = config['mapping_config']
    layer_info = config['layer_info']

    # 2. 设置调试输出路径
    debug_filename = f"debug_performance_model_point_{validation_point_id}.json" if validation_point_id is not None else "debug_performance_model.json"
    if output_dir is not None:
        debug_filename = str(output_dir / debug_filename)
    
    # 3. 初始化核心对象
    dosa_config = DosaConfig()
    perf_model = HighFidelityPerformanceModel(dosa_config)

    hw_params = HardwareParameters(
        initial_num_pes=hw_config['num_pes'],
        initial_l0_kb=hw_config.get('l0_registers_size_kb', 128.0),
        initial_l1_kb=hw_config.get('l1_accumulator_size_kb', 4.0),
        initial_l2_kb=hw_config['l2_scratchpad_size_kb']
    )
    hw_params.to(dosa_config.DEVICE)
            
    layer_name = layer_info['layer_name']
    problem_dims = layer_info['dims']
    hierarchy = dosa_config.MEMORY_HIERARCHY
    
    mapping = FineGrainedMapping(problem_dims, hierarchy)

    print(f"[INFO] Loading mapping config for layer: {layer_name}")
    layer_mapping = mapping_config.get(layer_name, {})

    for level_name, level_mapping in layer_mapping.items():
        if level_name in mapping.factors:
            temporal_factors = level_mapping.get('temporal', {})
            spatial_factors  = level_mapping.get('spatial', {})

            for dim_name in mapping.factors[level_name]:
                t_real = float(temporal_factors.get(dim_name, 1.0))
                s_real = float(spatial_factors.get(dim_name, 1.0))

                # 防御：因子至少为 1（避免 log(0) 或 <1 导致负 log）
                t_real = max(t_real, 1.0)
                s_real = max(s_real, 1.0)

                t_log = math.log(t_real)
                s_log = math.log(s_real)

                mapping.factors[level_name][dim_name]['temporal'] = nn.Parameter(
                    torch.tensor(t_log, device=dosa_config.DEVICE)
                )
                mapping.factors[level_name][dim_name]['spatial'] = nn.Parameter(
                    torch.tensor(s_log, device=dosa_config.DEVICE)
                )
    mapping.to(dosa_config.DEVICE)
    
    # Print mapping information
    print("\n[INFO] Mapping Configuration:")
    for level_name, level_factors in mapping.factors.items():
        print(f"\nLevel: {level_name}")
        print("Temporal factors:", {dim: factors['temporal'] for dim, factors in level_factors.items()})
        if any('spatial' in factors for factors in level_factors.values()):
            print("Spatial factors:", {dim: factors['spatial'] for dim, factors in level_factors.items() if 'spatial' in factors})
    
    # 5. 创建计算图
    graph = ComputationGraph()
    graph.add_layer(layer_name, problem_dims, layer_info['layer_type'])
    
    # Print graph information
    # print("\n[INFO] Computation Graph:")
    # print(f"Layer Name: {layer_name}")
    # print(f"Layer Type: {layer_info['layer_type']}")
    # print(f"Problem Dimensions: {problem_dims}")
    
    

    with torch.no_grad():
        # Create layer2mapping dictionary with layer name as key
        layer2mapping = {layer_name: mapping}
        
        total_latency, total_energy, area_cost, mismatch_loss, comp_penalty, _, _ = perf_model(
            graph=graph,
            hw_params=hw_params,
            layer2mapping=layer2mapping,  # <--- 传递字典而不是单个mapping对象
            fusion_params=None,
            debug_output_path=debug_filename
        )

    predicted_latency_s = float(total_latency)
    predicted_energy_pj = float(total_energy)

    print(f"[INFO] FA-DOSA prediction complete: latency={predicted_latency_s} s, energy={predicted_energy_pj} pJ")
    return {
        "predicted_latency_s": predicted_latency_s,
        "predicted_energy_pj": predicted_energy_pj
    }

def run_timeloop_simulation(config: dict, work_dir: Path) -> dict:
    """
    为给定的单个卷积层配置运行Timeloop/Accelergy仿真并返回结果。

    Args:
        config (dict): 与 run_dosa_prediction 使用的完全相同的配置字典。
        work_dir (Path): 用于存放临时YAML文件的目录路径对象。

    Returns:
        dict: 包含仿真结果的字典，例如 {'simulated_latency_s': 0.0012, 'simulated_energy_pj': 5200.0}
    """
    print("[INFO] Running Timeloop/Accelergy simulation for single convolution layer...")
    try:
        # 确保工作目录存在
        work_dir.mkdir(exist_ok=True)
        
        # 生成Timeloop配置文件
        generate_timeloop_files(config, work_dir)
        
        # 定义输入文件列表
        input_files = [
            str(work_dir / "arch.yaml"),
            str(work_dir / "problem.yaml"),
            str(work_dir / "constraints.yaml"),
            str(work_dir / "env.yaml")
        ]
        
        # 从YAML文件加载规范
        spec = tl.Specification.from_yaml_files(input_files)
        
        # 调用Timeloop映射器
        stats = tl.call_mapper(spec, output_dir=str(work_dir))
        
        # 从stats对象中提取性能指标
        if stats and hasattr(stats, 'cycles') and hasattr(stats, 'energy'):
            # 获取原始值并明确单位
            simulated_cycles = float(stats.cycles)
            simulated_energy_uj = float(stats.energy)
            simulated_energy_pj = simulated_energy_uj * 1e12
            
            # 获取DosaConfig实例以访问时钟频率
            dosa_config = DosaConfig()
            
            # 计算时钟周期时长（秒）
            cycle_time_s = 1.0 / (dosa_config.CLOCK_FREQUENCY_MHZ * 1e6)
            
            # 将时钟周期转换为秒
            simulated_latency_s = simulated_cycles * cycle_time_s
            
            print(f"[INFO] Timeloop raw output: cycles={simulated_cycles}, energy={simulated_energy_pj} pJ")
            print(f"[INFO] Converted to: latency={simulated_latency_s} s")
            
            return {
                "simulated_latency_s": simulated_latency_s,
                "simulated_energy_pj": simulated_energy_pj
            }
        else:
            print("[ERROR] Timeloop simulation failed to return valid stats")
            return {
                "simulated_latency_s": -1.0,
                "simulated_energy_pj": -1.0
            }
            
    except Exception as e:
        print(f"[ERROR] Timeloop simulation failed: {e}")
        return {
            "simulated_latency_s": -1.0,
            "simulated_energy_pj": -1.0
        }

def generate_timeloop_files(config: dict, work_dir: Path):
    """
    为【单个卷积层】生成 Timeloop v0.4 的 arch.yaml / problem.yaml / constraints.yaml / env.yaml
    - 完全对齐你“融合链”版本的节点/字段风格
    - 从 layer_info['dims'] 取实例大小
    - 从 mapping_config[layer_name] 取 per-level tiling & permutation
    """
    hw_config       = config['hardware_config']
    mapping_config  = config['mapping_config']
    layer_info      = config['layer_info']
    dims            = layer_info['dims']
    layer_name      = layer_info['layer_name']

    # ---------------------------
    # 1) 生成 arch.yaml (v0.4)
    # ---------------------------
    meshX     = int(math.sqrt(hw_config['num_pes']))
    datawidth = 16  # 与融合链版本一致：按 16-bit 数据宽度
    def get_depth(size_kb: float) -> int:
        return int(size_kb * 1024 * 8 / datawidth)

    arch_dict = {
        'architecture': {
            'version': '0.4',
            'nodes': [
                Hierarchical({
                    'nodes': [
                        Component({
                            'name': 'DRAM',
                            'class': 'DRAM',
                            'attributes': {
                                # 注意：v0.4 的 DRAM 一般给 depth/width/datawidth
                                'depth': 1048576,
                                'width': 256,
                                'datawidth': datawidth
                            }
                        }),
                        Hierarchical({
                            'nodes': [
                                Component({
                                    'name': 'L2_Scratchpad',
                                    'class': 'SRAM',
                                    'attributes': {
                                        'depth': get_depth(hw_config['l2_scratchpad_size_kb']),
                                        'width': datawidth,
                                        'datawidth': datawidth
                                    }
                                }),
                                Hierarchical({
                                    'nodes': [
                                        Component({
                                            'name': 'L1_Accumulator',
                                            'class': 'SRAM',
                                            'attributes': {
                                                'depth': get_depth(hw_config.get('l1_accumulator_size_kb', 4.0)),
                                                'width': datawidth,
                                                'datawidth': datawidth
                                            }
                                        }),
                                        Hierarchical({
                                            'nodes': [
                                                Container({
                                                    'name': 'PE_array_container',
                                                    'spatial': {'meshX': meshX, 'meshY': meshX}
                                                }),
                                                Component({
                                                    'name': 'L0_Registers',
                                                    'class': 'regfile',
                                                    'attributes': {
                                                        'depth': get_depth(hw_config.get('l0_registers_size_kb', 128.0)),
                                                        'width': datawidth,
                                                        'datawidth': datawidth
                                                    }
                                                }),
                                                Component({
                                                    'name': 'MAC',
                                                    'class': 'intmac',
                                                    'attributes': {'datawidth': datawidth, 'width': 8}
                                                })
                                            ]
                                        })
                                    ]
                                })
                            ]
                        })
                    ]
                })
            ]
        }
    }

    with open(work_dir / 'arch.yaml', 'w') as f:
        yaml.dump(arch_dict, f, sort_keys=False)

    # ---------------------------
    # 2) 生成 problem.yaml
    # ---------------------------
    # v0.4 形状用“维度符号列表”，实例大小放 instance。
    # 投影按你融合链那段的规范来（Outputs 要 read_write=True）
    problem_config = {
        'problem': {
            'version': '0.4',
            'shape': {
                'name': 'CNN_Layer',
                'dimensions': ['K', 'C', 'R', 'S', 'P', 'Q', 'N'],
                'data_spaces': [
                    {
                        'name': 'Weights',
                        'projection': [[['K']], [['C']], [['R']], [['S']]]
                    },
                    {
                        'name': 'Inputs',
                        'projection': [[['N']], [['C']], [['R'], ['P']], [['S'], ['Q']]]
                    },
                    {
                        'name': 'Outputs',
                        'projection': [[['N']], [['K']], [['P']], [['Q']]],
                        'read_write': True
                    }
                ]
            },
            'instance': {
                'N': int(dims['N']),
                'C': int(dims['C']),
                'K': int(dims['K']),
                'P': int(dims['P']),
                'Q': int(dims['Q']),
                'R': int(dims['R']),
                'S': int(dims['S']),
            }
        }
    }

    with open(work_dir / 'problem.yaml', 'w') as f:
        yaml.dump(problem_config, f, sort_keys=False)

    # ---------------------------
    # 3) 生成 constraints.yaml
    # ---------------------------
    layer_mapping = mapping_config[layer_name]
    all_dims = ['N','K','C','P','Q','R','S']

    # 收集 per-dim 的空间/时间因子
    dim_factors = {d: {} for d in all_dims}

    # spatial（通常只在 PE 层做）
    if 'spatial' in layer_mapping.get('L0_Registers', {}):
        # 兼容：有些你的 mapping 把 spatial 因子挂在 L0_Registers 下存储
        for d, v in layer_mapping['L0_Registers']['spatial'].items():
            if d in dim_factors:
                dim_factors[d]['spatial'] = int(v)

    # temporal（各存储层与 DRAM）
    for level in ['DRAM', 'L2_Scratchpad', 'L1_Accumulator', 'L0_Registers']:
        if level in layer_mapping and 'temporal' in layer_mapping[level]:
            for d, v in layer_mapping[level]['temporal'].items():
                if d in dim_factors:
                    dim_factors[d][level] = int(v)

    # 严格校验：按 v0.4 规范，要求 product(spatial, L0, L1, L2, DRAM) == 实例大小
    def fix_factorization_for_dim(dim_name, dim_size, entries):
        s  = entries.get('spatial', 1)
        l0 = entries.get('L0_Registers', 1)
        l1 = entries.get('L1_Accumulator', 1)
        l2 = entries.get('L2_Scratchpad', 1)
        known = s * l0 * l1 * l2
        if dim_size % known != 0:
            raise ValueError(
                f"[{dim_name}] 非整除: dim={dim_size}, known={known}. "
                f"请修正各层因子，别用近似/截断。"
            )
        entries['DRAM'] = dim_size // known
        return entries

    for d in all_dims:
        if d in dims and int(dims[d]) > 0:
            dim_factors[d] = fix_factorization_for_dim(d, int(dims[d]), dim_factors[d])

    # 格式化 factors 为 Timeloop 期望的 "K=xx" 字符串列表
    def format_factors(level_type, level_name):
        items = []
        for d in all_dims:
            if d not in dims: 
                continue
            if level_type == 'spatial':
                # 仅对有空间因子的维度输出
                if dim_factors[d].get('spatial', 1) > 1 and d in ['K','C']:
                    items.append(f"{d}={dim_factors[d]['spatial']}")
            else:
                items.append(f"{d}={dim_factors[d].get(level_name, 1)}")
        return items

    # 生成一个合理的空间排列（优先 K/C，避免把 N 放到 mesh 前两维）
    def generate_spatial_permutation():
        priority = ['K','C','N','P','Q','S','R']
        active = [d for d in priority if dim_factors.get(d, {}).get('spatial', 1) > 1]
        first_two = active[:2] if len(active) >= 2 else active + [d for d in priority if d not in active][:2-len(active)]
        rest = [d for d in priority if d not in first_two]
        perm = first_two + rest

        # 如果 K/C 有空间分解，不要让 N 占前两位
        if any(dim_factors.get(x, {}).get('spatial', 1) > 1 for x in ['K','C']) and 'N' in perm[:2]:
            for i in range(2):
                if perm[i] == 'N':
                    for j in range(2, len(perm)):
                        if perm[j] in ['K','C'] and dim_factors.get(perm[j], {}).get('spatial', 1) > 1:
                            perm[i], perm[j] = perm[j], perm[i]
                            break
        return perm

    spatial_perm = generate_spatial_permutation()

    # 调试打印（可保留可去掉）
    # print("[DEBUG] Final dimension factors:")
    for d in ['N','C','K','P','Q','R','S']:
        s  = dim_factors[d].get('spatial', 1)
        l0 = dim_factors[d].get('L0_Registers', 1)
        l1 = dim_factors[d].get('L1_Accumulator', 1)
        l2 = dim_factors[d].get('L2_Scratchpad', 1)
        dr = dim_factors[d].get('DRAM', 1)
        product = s*l0*l1*l2*dr
        expected = int(dims[d])
        # print(f"  {d}: spatial={s}, L0={l0}, L1={l1}, L2={l2}, DRAM={dr}, product={product}, expected={expected}")
        assert product == expected, f"Dimension {d}: product {product} != expected {expected}"

    # print(f"[DEBUG] Generated spatial permutation: {spatial_perm}")

    # 目标约束
    targets = []
    # 空间目标：绑在 PE_array_container
    targets.append({
        'target': 'PE_array_container',
        'type': 'spatial',
        'factors': format_factors('spatial', 'L0_Registers'),
        'permutation': "P Q K C R S N"
    })

    # 时间目标：DRAM/L2/L1/L0（permutation 使用映射里给的）
    for level in ['DRAM','L2_Scratchpad','L1_Accumulator','L0_Registers']:
        targets.append({
            'target': level,
            'type': 'temporal',
            'factors': format_factors('temporal', level),
            'permutation': layer_mapping[level]['permutation']
        })

    # dataspace 语义（与融合链版本一致）
    targets.append({
        'target': 'L1_Accumulator',
        'type': 'dataspace',
        'keep':   ['Outputs'],
        'bypass': ['Inputs','Weights']
    })
    targets.append({
        'target': 'L0_Registers',
        'type': 'dataspace',
        'keep':   ['Weights'],
        'bypass': ['Inputs','Outputs']
    })
    targets.append({
        'target': 'L2_Scratchpad',
        'type': 'dataspace',
        'keep': ['Inputs','Weights'],
        'bypass': ['Outputs']
    })
    # 如需让 W/I 旁路 L2，可改成仅 keep Outputs

    with open(work_dir / 'constraints.yaml', 'w') as f:
        yaml.dump({'constraints': {'targets': targets}}, f, sort_keys=False)

    # ---------------------------
    # 4) 生成 env.yaml
    # ---------------------------
    env_config = {
        'globals': {'environment_variables': {
            'ACCELERGY_COMPONENT_LIBRARIES': '/root/accelergy-timeloop-infrastructure/src/accelergy-library-plug-in/library/'
        }},
        'variables': {'global_cycle_seconds': 1e-9, 'technology': "40nm"}
    }
    with open(work_dir / 'env.yaml', 'w') as f:
        yaml.dump(env_config, f, sort_keys=False, default_style="'")

def generate_validation_configs(max_runs: int = None, use_fixed_config: bool = True, use_constrained_mapping: bool = False):
    """Generate validation configurations for single convolution layer testing.
    
    Args:
        max_runs: Maximum number of validation runs
        use_fixed_config: If True, always use fixed configuration for debugging
        use_constrained_mapping: If True, use constrained mapping configurations
    """
    configs = []
    
    if use_fixed_config:
        # 调试模式：使用固定配置
        print("[INFO] Using fixed configuration for debugging")
        if max_runs is None or max_runs >= 1:
            configs.append((1, get_fixed_validation_config()))
        # 如果需要多个配置点但仍使用固定配置，可以复制相同配置
        if max_runs is not None and max_runs > 1:
            for i in range(2, max_runs + 1):
                configs.append((i, get_fixed_validation_config()))
    elif use_constrained_mapping:
        # 约束mapping扫描模式：使用符合约束条件的mapping配置
        print("[INFO] Generating constrained mapping configurations")
        num_configs = max_runs if max_runs is not None else 10
        for i, config in enumerate(generate_constrained_mapping_configs(num_configs)):
            configs.append((i + 1, config))
            if max_runs is not None and len(configs) >= max_runs:
                break
    else:
        # 生产模式：生成随机配置
        print("[INFO] Generating random configurations")
        num_configs = max_runs if max_runs is not None else 10
        for i, config in enumerate(generate_configurations(num_configs)):
            configs.append((i + 1, config))
            if max_runs is not None and len(configs) >= max_runs:
                break
    
    return configs

def run_traffic_comparison(work_dir, validation_point_id):
    """
    运行traffic comparison分析，调用diff_traffic.py模块
    """
    try:
        # 导入diff_traffic模块
        import diff_traffic
        
        # 构建文件路径 - 现在都从work_dir读取
        model_csv_path = work_dir / "traffic_summary_tensor.csv"  # 模型生成的文件，已移动到work_dir
        timeloop_csv_path = work_dir / "scalar_access_summary.csv"  # extract_scalar_access.py在work_dir中生成的文件
        output_csv_path = work_dir / "traffic_diff.csv"
        
        print(f"[DEBUG] Traffic comparison for validation point {validation_point_id}:")
        print(f"[DEBUG]   Model CSV: {model_csv_path} (exists: {model_csv_path.exists()})")
        print(f"[DEBUG]   Timeloop CSV: {timeloop_csv_path} (exists: {timeloop_csv_path.exists()})")
        print(f"[DEBUG]   Output CSV: {output_csv_path}")
        
        # 检查必需文件是否存在
        if not model_csv_path.exists():
            print(f"[WARNING] Model traffic CSV not found: {model_csv_path}")
            return {"traffic_comparison_status": "model_csv_missing", 
                   "traffic_accuracy_percent": -1.0, 
                   "traffic_total_rows": -1, 
                   "traffic_correct_rows": -1}
        
        if not timeloop_csv_path.exists():
            print(f"[WARNING] Timeloop traffic CSV not found: {timeloop_csv_path}")
            return {"traffic_comparison_status": "timeloop_csv_missing", 
                   "traffic_accuracy_percent": -1.0, 
                   "traffic_total_rows": -1, 
                   "traffic_correct_rows": -1}
        
        # 检查文件大小
        model_size = model_csv_path.stat().st_size
        timeloop_size = timeloop_csv_path.stat().st_size
        print(f"[DEBUG] File sizes - Model CSV: {model_size} bytes, Timeloop CSV: {timeloop_size} bytes")
        
        if model_size == 0:
            print(f"[WARNING] Model CSV file is empty: {model_csv_path}")
            return {"traffic_comparison_status": "model_csv_empty", 
                   "traffic_accuracy_percent": -1.0, 
                   "traffic_total_rows": -1, 
                   "traffic_correct_rows": -1}
        
        if timeloop_size == 0:
            print(f"[WARNING] Timeloop CSV file is empty: {timeloop_csv_path}")
            return {"traffic_comparison_status": "timeloop_csv_empty", 
                   "traffic_accuracy_percent": -1.0, 
                   "traffic_total_rows": -1, 
                   "traffic_correct_rows": -1}
        
        # 调用diff_traffic模块的分析函数
        print(f"[INFO] Running traffic comparison analysis for validation point {validation_point_id}")
        
        # 构建stats文件路径
        stats_file_path = work_dir / "timeloop-mapper.stats.txt"
        if not stats_file_path.exists():
            stats_file_path = work_dir / "timeloop-output" / "stats.txt"
        
        print(f"[DEBUG] Stats file: {stats_file_path} (exists: {stats_file_path.exists()})")
        
        stats = diff_traffic.run_traffic_diff_analysis(
            str(model_csv_path), 
            str(timeloop_csv_path), 
            str(output_csv_path),
            str(stats_file_path) if stats_file_path.exists() else None
        )
        
        print(f"[INFO] Traffic comparison completed for validation point {validation_point_id}: {stats.get('correct_count', 0)}/{stats.get('total_count', 0)} ({stats.get('accuracy_percent', 0.0):.1f}%)")
        
        return {
            "traffic_comparison_status": "success",
            "traffic_accuracy_percent": stats.get("accuracy_percent", 0.0),
            "traffic_total_rows": stats.get("total_count", 0),
            "traffic_correct_rows": stats.get("correct_count", 0)
        }
        
    except ImportError as e:
        print(f"[ERROR] Failed to import diff_traffic module for validation point {validation_point_id}: {e}")
        return {"traffic_comparison_status": "import_error", 
               "traffic_accuracy_percent": -1.0, 
               "traffic_total_rows": -1, 
               "traffic_correct_rows": -1}
    
    except Exception as e:
        print(f"[ERROR] Traffic comparison analysis failed for validation point {validation_point_id}: {e}")
        import traceback
        traceback.print_exc()
        return {"traffic_comparison_status": "analysis_error", 
               "traffic_accuracy_percent": -1.0, 
               "traffic_total_rows": -1, 
               "traffic_correct_rows": -1}

def main():
    """Main control script to run single convolution layer validation experiments."""
    parser = argparse.ArgumentParser(description="Run single convolution layer validation experiments")
    parser.add_argument('--max-runs', type=int, default=MAX_VALIDATION_RUNS,
                       help='Maximum number of validation runs (default: None for full sweep)')
    parser.add_argument('--output-dir', type=str, default="output",
                       help='Output directory for all generated files (default: output)')
    parser.add_argument('--timeout', type=int, default=VALIDATION_TIMEOUT_SECONDS,
                       help=f'Timeout in seconds for each validation point (default: {VALIDATION_TIMEOUT_SECONDS})')
    parser.add_argument('--use-random-config', action='store_true', default=False,
                       help='Use random configurations instead of fixed configuration (default: False, use fixed config for debugging)')
    parser.add_argument('--use-constrained-mapping', action='store_true', default=False,
                       help='Use constrained mapping configurations for systematic scanning (default: False)')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    all_results = []
    all_diff_data = []  # 存储所有配置点的diff数据
    master_configs = {}  # Dictionary to store all validation point configurations
    max_runs = args.max_runs
    timeout_seconds = args.timeout
    use_fixed_config = not args.use_random_config and not args.use_constrained_mapping  # 默认使用固定配置
    use_constrained_mapping = args.use_constrained_mapping
    
    # Statistics counters
    success_count = 0
    error_count = 0
    timeout_count = 0

    print("Starting single convolution layer validation run...")
    print(f"[INFO] Output directory: {output_dir.resolve()}")
    print(f"[INFO] Timeout per validation point: {timeout_seconds} seconds ({timeout_seconds/60:.1f} minutes)")
    
    if use_constrained_mapping:
        print("[INFO] Configuration mode: Constrained mapping scanning")
    elif use_fixed_config:
        print("[INFO] Configuration mode: Fixed (debugging)")
    else:
        print("[INFO] Configuration mode: Random (production)")
        
    if max_runs is not None:
        print(f"[INFO] Limited to {max_runs} validation runs for quick testing")
    else:
        print("[INFO] Running full validation sweep")

    try:
        # Generate validation configurations
        validation_configs = generate_validation_configs(max_runs, use_fixed_config, use_constrained_mapping)
        print(f"[INFO] Generated {len(validation_configs)} validation configurations")
        
        for validation_point_id, config in validation_configs:
            print(f"--- Running Validation Point {validation_point_id} ---")
            
            # Store configuration in master dictionary
            master_configs[str(validation_point_id)] = config
            
            # Create workspace directory in output folder
            work_dir = output_dir / f'validation_workspace_{validation_point_id}'
            work_dir.mkdir(exist_ok=True)
            
            # Clean up any residual CSV files from previous runs
            for csv_file in work_dir.glob("*.csv"):
                csv_file.unlink()
                print(f"[INFO] Cleaned up residual CSV file: {csv_file.name}")
            
            try:
                # Run dual-track evaluation
                dosa_results = run_dosa_prediction(config, validation_point_id, output_dir)
                
                # Move model-side traffic_summary_tensor.csv to work_dir
                model_csv_path = Path("traffic_summary_tensor.csv")
                if model_csv_path.exists():
                    target_csv_path = work_dir / "traffic_summary_tensor.csv"
                    model_csv_path.rename(target_csv_path)
                    print(f"[INFO] Moved traffic_summary_tensor.csv to {target_csv_path}")
                else:
                    print(f"[WARNING] Model CSV file not found: {model_csv_path}")
                
                timeloop_results = run_timeloop_simulation(config, work_dir)
                
                # Check if Timeloop simulation failed (more comprehensive check)
                timeloop_failed = (timeloop_results.get("simulated_latency_s", -1) <= 0 or 
                                 timeloop_results.get("simulated_energy_pj", -1) <= 0 or
                                 timeloop_results.get("simulated_latency_s", -1) == -1.0 or 
                                 timeloop_results.get("simulated_energy_pj", -1) == -1.0)
                
                # 检查stats文件是否存在（这是运行diff分析的前提）
                stats_file = work_dir / "timeloop-mapper.stats.txt"
                if not stats_file.exists():
                    stats_file = work_dir / "timeloop-output" / "stats.txt"
                
                stats_file_exists = stats_file.exists()
                
                if timeloop_failed:
                    print(f"[WARNING] Timeloop simulation failed for validation point {validation_point_id} (latency={timeloop_results.get('simulated_latency_s', 'N/A')}, energy={timeloop_results.get('simulated_energy_pj', 'N/A')})")
                    traffic_results = {"traffic_comparison_status": "timeloop_simulation_failed", 
                                     "traffic_accuracy_percent": -1.0, 
                                     "traffic_total_rows": -1, 
                                     "traffic_correct_rows": -1}
                elif not stats_file_exists:
                    print(f"[WARNING] Timeloop stats file not found for validation point {validation_point_id}, cannot run traffic analysis")
                    traffic_results = {"traffic_comparison_status": "stats_file_missing", 
                                     "traffic_accuracy_percent": -1.0, 
                                     "traffic_total_rows": -1, 
                                     "traffic_correct_rows": -1}
                else:
                    # Timeloop成功且stats文件存在，运行traffic分析
                    print(f"[INFO] Running extract_scalar_access.py on {stats_file}")
                    try:
                        # 使用相对于工作目录的文件名
                        stats_filename = stats_file.name  # 只取文件名，如 "timeloop-mapper.stats.txt"
                        subprocess.run(
                            ["python", str(Path.cwd() / "extract_scalar_access.py"), stats_filename],
                            cwd=str(work_dir),
                            check=True
                        )
                        
                        # 运行traffic comparison分析
                        traffic_results = run_traffic_comparison(work_dir, validation_point_id)
                        
                        # 收集diff数据用于汇总
                        diff_csv_path = work_dir / "traffic_diff.csv"
                        if diff_csv_path.exists():
                            try:
                                diff_df = pd.read_csv(diff_csv_path)
                                # 添加配置点ID列
                                diff_df['validation_point_id'] = validation_point_id
                                all_diff_data.append(diff_df)
                            except Exception as e:
                                print(f"[WARNING] Failed to read diff CSV for validation point {validation_point_id}: {e}")
                        
                    except subprocess.CalledProcessError as e:
                        print(f"[ERROR] extract_scalar_access.py failed for validation point {validation_point_id}: {e}")
                        traffic_results = {"traffic_comparison_status": "extract_failed", 
                                         "traffic_accuracy_percent": -1.0, 
                                         "traffic_total_rows": -1, 
                                         "traffic_correct_rows": -1}
                    except Exception as e:
                        print(f"[ERROR] Unexpected error during traffic analysis for validation point {validation_point_id}: {e}")
                        traffic_results = {"traffic_comparison_status": "analysis_error", 
                                         "traffic_accuracy_percent": -1.0, 
                                         "traffic_total_rows": -1, 
                                         "traffic_correct_rows": -1}

                # Combine results (including traffic comparison results)
                flat_config = {
                    "validation_point_id": validation_point_id,
                    "layer_name": config['layer_info']['layer_name'],
                    **config['hardware_config']
                }
                
                # Determine overall status
                overall_status = "timeloop_failed" if timeloop_failed else "success"
                combined_result = {**flat_config, **dosa_results, **timeloop_results, **traffic_results, "status": overall_status}
                all_results.append(combined_result)
                
                if timeloop_failed:
                    error_count += 1
                    print(f"[PARTIAL SUCCESS] Validation point {validation_point_id} completed with Timeloop simulation failure")
                else:
                    success_count += 1
                    print(f"[SUCCESS] Validation point {validation_point_id} completed successfully")
                
            except Exception as e:
                print(f"[ERROR] Validation point {validation_point_id} failed: {e}")
                import traceback
                traceback.print_exc()
                
                # Add error result
                flat_config = {
                    "validation_point_id": validation_point_id,
                    "layer_name": config['layer_info']['layer_name'],
                    **config['hardware_config']
                }
                error_result = {**flat_config, "predicted_latency_s": -1.0, "predicted_energy_pj": -1.0, 
                               "simulated_latency_s": -1.0, "simulated_energy_pj": -1.0, 
                               "status": "error", "error_message": str(e)}
                all_results.append(error_result)
                error_count += 1

        # Save results to CSV file
        if all_results:
            csv_file = output_dir / "single_conv_validation_results.csv"
            df = pd.DataFrame(all_results)
            # Ensure all required columns are present
            required_columns = ['validation_point_id', 'layer_name', 'num_pes', 'l2_scratchpad_size_kb', 
                               'predicted_latency_s', 'predicted_energy_pj', 'simulated_latency_s', 'simulated_energy_pj', 
                               'traffic_comparison_status', 'traffic_accuracy_percent', 'status']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = -1.0 if 'latency' in col or 'energy' in col else 'unknown'
            
            df.to_csv(csv_file, index=False)
            print(f"[INFO] Results saved to {csv_file}")
        
        # Save aggregated diff data to a single CSV file
        if all_diff_data:
            aggregated_diff_csv = output_dir / "aggregated_traffic_diff.csv"
            combined_diff_df = pd.concat(all_diff_data, ignore_index=True)
            combined_diff_df.to_csv(aggregated_diff_csv, index=False)
            print(f"[INFO] Aggregated diff data saved to {aggregated_diff_csv}")
            print(f"[INFO] Total diff records: {len(combined_diff_df)}")
        
        # Final statistics
        total_points = len(all_results)
        if total_points > 0:
            print(f"\nValidation complete. Total: {total_points}, Succeeded: {success_count}, Failed: {error_count}, Timed out: {timeout_count}")
        else:
            print("\nValidation run finished, but no results were collected.")
            
        # Save configuration information to JSON file in output directory
        try:
            config_file = output_dir / "validation_configs.json"
            with open(config_file, 'w') as f:
                json.dump(master_configs, f, indent=4)
            print(f"[INFO] Configuration information saved to {config_file}")
            print(f"[INFO] Total configurations saved: {len(master_configs)}")
        except Exception as json_error:
            print(f"[ERROR] Failed to save configuration JSON: {json_error}")
            
    except Exception as e:
        print(f"[ERROR] Main validation loop failed: {e}")
        import traceback
        traceback.print_exc()
            
    finally:
        # All files are now organized in the output directory
        print(f"[INFO] All validation results and workspaces are preserved in: {output_dir.resolve()}")

if __name__ == "__main__":
    main()