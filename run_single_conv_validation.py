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

# 2. Hardware Configuration Space
HW_CONFIG_SPACE = {
    "num_pes": [16, 32, 64, 128, 256],
    "l2_scratchpad_size_kb": [128, 256, 512]
}

# 3. Mapping Space (simplified, needs to be dimension-dependent)
MAPPING_SPACE = {
    "K": [1, 4, 8, 16, 32],
    "C": [1, 4, 8, 16, 32]
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
            for level in ['L0', 'L1', 'L2']:
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
            levels = ['L0', 'L1', 'L2', 'DRAM']
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

def get_fixed_validation_config():
    """返回 validation_point_id=1 的固定配置字典"""
    config_vp1 = {
        "hardware_config": {
            "num_pes": 64,
            "l2_scratchpad_size_kb": 256
        },
        "mapping_config": {
            "conv1": {
                "DRAM": {"temporal": {"N":1,"C":1,"K":1,"P":8,"Q":1,"R":1,"S":1}, "permutation": "K N C P Q S R"},
                "L2_Scratchpad": {"temporal": {"N":1,"C":1,"K":2,"P":1,"Q":7,"R":1,"S":1}, "permutation": "K N C P Q S R"},
                "L1_Accumulator": {"temporal": {"N":1,"C":1,"K":2,"P":1,"Q":2,"R":1,"S":1}, "permutation": "K N C P Q S R"},
                "L0_Registers": {"temporal": {"N":1,"C":8,"K":2,"P":7,"Q":2,"R":3,"S":3}, "spatial": {"C":8,"K":8}, "permutation": "K N C P Q S R"}
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
        'K N C P Q S R'
    ]
    
    # Define available strategies
    strategies = ["performance"]
    
    # Main loop for generating configurations
    for i in range(num_configs):
        layer_name = CONV_LAYER_CONFIG["layer_name"]
        dims = CONV_LAYER_CONFIG["dims"]
        
        # Randomly sample hardware configuration
        hardware_config = {}
        for hw_key, hw_value_list in HW_CONFIG_SPACE.items():
            hardware_config[hw_key] = random.choice(hw_value_list)
        
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
    perf_model = HighFidelityPerformanceModel(dosa_config, debug_latency=False, fusion_aware=False)

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
            spatial_factors = level_mapping.get('spatial', {})
            
            for dim_name in mapping.factors[level_name]:
                t_factor = temporal_factors.get(dim_name, 1.0)
                s_factor = spatial_factors.get(dim_name, 1.0)
                
                import torch.nn as nn
                mapping.factors[level_name][dim_name]['temporal'] = nn.Parameter(torch.tensor(float(t_factor)))
                mapping.factors[level_name][dim_name]['spatial'] = nn.Parameter(torch.tensor(float(s_factor)))

    mapping.to(dosa_config.DEVICE)
    
    # Print mapping information
    # print("\n[INFO] Mapping Configuration:")
    # for level_name, level_factors in mapping.factors.items():
    #     print(f"\nLevel: {level_name}")
    #     print("Temporal factors:", {dim: factors['temporal'] for dim, factors in level_factors.items()})
    #     if any('spatial' in factors for factors in level_factors.values()):
    #         print("Spatial factors:", {dim: factors['spatial'] for dim, factors in level_factors.items() if 'spatial' in factors})
    
    # 5. 创建计算图
    graph = ComputationGraph()
    graph.add_layer(layer_name, problem_dims, layer_info['layer_type'])
    
    # Print graph information
    # print("\n[INFO] Computation Graph:")
    # print(f"Layer Name: {layer_name}")
    # print(f"Layer Type: {layer_info['layer_type']}")
    # print(f"Problem Dimensions: {problem_dims}")
    
    

    with torch.no_grad():
        total_latency, total_energy, area_cost, mismatch_loss, comp_penalty, _, _ = perf_model(
            graph=graph,
            hw_params=hw_params,
            mapping=mapping,  # <--- 使用 mapping 对象
            fusion_params=None,
            direct_mapping_table=None, # <--- 明确地不再使用它
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
                                                'depth': get_depth(hw_config.get('l1_buffer_size_kb', 4.0)),
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
                                                        'depth': get_depth(hw_config.get('l0_registers_size_kb', 2.0)),
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
        'permutation': ' '.join(spatial_perm)
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

def generate_validation_configs(max_runs: int = None, use_fixed_config: bool = True):
    """Generate validation configurations for single convolution layer testing.
    
    Args:
        max_runs: Maximum number of validation runs
        use_fixed_config: If True, always use fixed configuration for debugging
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
    else:
        # 生产模式：生成随机配置
        print("[INFO] Generating random configurations")
        num_configs = max_runs if max_runs is not None else 10
        for i, config in enumerate(generate_configurations(num_configs)):
            configs.append((i + 1, config))
            if max_runs is not None and len(configs) >= max_runs:
                break
    
    return configs

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
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    all_results = []
    master_configs = {}  # Dictionary to store all validation point configurations
    max_runs = args.max_runs
    timeout_seconds = args.timeout
    use_fixed_config = not args.use_random_config  # 默认使用固定配置
    
    # Statistics counters
    success_count = 0
    error_count = 0
    timeout_count = 0

    print("Starting single convolution layer validation run...")
    print(f"[INFO] Output directory: {output_dir.resolve()}")
    print(f"[INFO] Timeout per validation point: {timeout_seconds} seconds ({timeout_seconds/60:.1f} minutes)")
    print(f"[INFO] Configuration mode: {'Fixed (debugging)' if use_fixed_config else 'Random (production)'}")
    if max_runs is not None:
        print(f"[INFO] Limited to {max_runs} validation runs for quick testing")
    else:
        print("[INFO] Running full validation sweep")

    try:
        # Generate validation configurations
        validation_configs = generate_validation_configs(max_runs, use_fixed_config)
        print(f"[INFO] Generated {len(validation_configs)} validation configurations")
        
        for validation_point_id, config in validation_configs:
            print(f"--- Running Validation Point {validation_point_id} ---")
            
            # Store configuration in master dictionary
            master_configs[str(validation_point_id)] = config
            
            # Create workspace directory in output folder
            work_dir = output_dir / f'validation_workspace_{validation_point_id}'
            work_dir.mkdir(exist_ok=True)
            
            try:
                # Run dual-track evaluation
                dosa_results = run_dosa_prediction(config, validation_point_id, output_dir)
                timeloop_results = run_timeloop_simulation(config, work_dir)
                
                # Combine results
                flat_config = {
                    "validation_point_id": validation_point_id,
                    "layer_name": config['layer_info']['layer_name'],
                    **config['hardware_config']
                }
                combined_result = {**flat_config, **dosa_results, **timeloop_results, "status": "success"}
                all_results.append(combined_result)
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
                               'predicted_latency_s', 'predicted_energy_pj', 'simulated_latency_s', 'simulated_energy_pj', 'status']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = -1.0 if 'latency' in col or 'energy' in col else 'unknown'
            
            df.to_csv(csv_file, index=False)
            print(f"[INFO] Results saved to {csv_file}")
        
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