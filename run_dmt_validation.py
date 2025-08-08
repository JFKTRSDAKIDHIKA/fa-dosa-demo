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
from pathlib import Path

# FA-DOSA core modules
import torch
from dosa.config import Config as DosaConfig
from dosa.hardware_parameters import HardwareParameters
from dosa.mapping import FineGrainedMapping
from dosa.utils import ComputationGraph, get_divisors
from dosa.dmt import InPlaceFusionDMT, convert_tensors_to_native

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

# Configuration for number of validation runs# Global configuration
MAX_VALIDATION_RUNS = None  # Set to None for full sweep, or specify a number (e.g., 1, 2, 4) for quick testing
VALIDATION_TIMEOUT_SECONDS = 600  # 10 minutes timeout for each validation point

# 1. Fused Groups to be tested (example from ResNet-18)
FUSION_GROUPS_TO_TEST = [
    {
        "group_name": "layer1.0.conv1_relu",
        "layers": ["layer1.0.conv1", "layer1.0.relu"],
        "pattern": ["Conv", "ReLU"],
        "producer_layer": "layer1.0.conv1",
        "consumer_layer": "layer1.0.relu"
    },
    # Add more groups as needed
]

# 2. Hardware Configuration Space
HW_CONFIG_SPACE = {
    "num_pes": [64, 256],
    "l2_scratchpad_size_kb": [256, 512]
}

# 3. Mapping Space (simplified, needs to be dimension-dependent)
# A more robust implementation would dynamically get divisors
MAPPING_SPACE = {
    "K": [16, 32, 64],
    "C": [16, 32, 64]
}

# 4. Workload Dimensions (example for a specific layer)
# In a real scenario, this would be loaded dynamically per layer
WORKLOAD_DIMS = {
    "layer1.0.conv1": {"N": 1, "C": 64, "K": 64, "P": 56, "Q": 56, "R": 3, "S": 3},
    "layer1.0.relu": {"N": 1, "C": 64, "K": 64, "P": 56, "Q": 56, "R": 1, "S": 1} # ReLU output dims match Conv output dims
}


def generate_dynamic_mapping_space(workload_dims, producer_layer):
    """Dynamically generate valid mapping factors based on workload dimensions."""
    dims = workload_dims[producer_layer]
    mapping_space = {}
    
    # For each dimension, get its divisors
    for dim_name, dim_size in dims.items():
        if dim_name in ['N', 'K', 'C', 'P', 'Q', 'R', 'S']:
            divisors = get_divisors(dim_size).tolist()
            # Sample a few divisors for efficiency
            if len(divisors) > 4:
                divisors = random.sample(divisors, 4)
            mapping_space[dim_name] = divisors
    
    return mapping_space

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
    # 逻辑保持不变：从所有约数中，找到一个小于等于 num_pes_sqrt 的最大可能值
    divisors = get_divisors(dim_size).tolist()
    spatial_candidates = [d for d in divisors if d <= num_pes_sqrt]
    spatial_factor = random.choice(spatial_candidates) if spatial_candidates else 1
    
    remaining_size = dim_size // spatial_factor
    
    # 2. 根据策略分配剩余的时间因子 (temporal factors)
    factors = {'spatial': int(spatial_factor), 'L0': 1, 'L1': 1, 'L2': 1, 'DRAM': 1}
    
    if remaining_size > 1:
        remaining_divisors = get_divisors(remaining_size).tolist()
        
        if strategy == "performance":
            # 性能优先：从L0开始，从剩余约数中随机选一个最大的，然后迭代
            temp_size = remaining_size
            for level in ['L0', 'L1', 'L2']:
                if temp_size == 1: break
                level_divs = [d for d in get_divisors(temp_size).tolist() if d > 1]
                if not level_divs: continue
                factor_for_level = random.choice(level_divs) # 引入随机性
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
            random.shuffle(levels) # 随机化分配顺序
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

def generate_complete_dimension_factors(dim_size, num_pes_sqrt):
    """Generate complete factor decomposition for a dimension across all levels.
    保留此函数以保持向后兼容性。
    
    Args:
        dim_size (int): The problem dimension size to factorize.
        num_pes_sqrt (int): Square root of the number of PEs, used to constrain spatial factors.
        
    Returns:
        dict: A dictionary with factors for each level ['spatial', 'L0', 'L1', 'L2', 'DRAM'].
    """
    return generate_factors_by_strategy(dim_size, num_pes_sqrt, "random")

def get_fixed_validation_config():
    """返回 validation_point_id=1 的固定配置字典"""
    config_vp1 = {
        "hardware_config": {
            "num_pes": 64,
            "l2_scratchpad_size_kb": 256
        },
        "mapping_config": {
            "layer1.0.conv1": {
                "DRAM": {"temporal": {"N":1,"C":1,"K":1,"P":1,"Q":1,"R":1,"S":1}, "permutation": "K N C P Q S R"},
                "L2_Scratchpad": {"temporal": {"N":1,"C":1,"K":2,"P":1,"Q":7,"R":1,"S":1}, "permutation": "K N C P Q S R"},
                "L1_Accumulator": {"temporal": {"N":1,"C":1,"K":2,"P":1,"Q":2,"R":1,"S":1}, "permutation": "K N C P Q S R"},
                "L0_Registers": {"temporal": {"N":1,"C":32,"K":2,"P":7,"Q":2,"R":3,"S":3}, "spatial": {"C":2,"K":8}, "permutation": "K N C P Q S R"}
            }
        },
        "fusion_group_info": {
            "group_name": "layer1.0.conv1_relu",
            "layers": ["layer1.0.conv1", "layer1.0.relu"],
            "pattern": ["Conv", "ReLU"],
            "producer_layer": "layer1.0.conv1",
            "consumer_layer": "layer1.0.relu"
        },
        "workload_dims": {
            "layer1.0.conv1": {"N":1,"C":64,"K":64,"P":56,"Q":56,"R":3,"S":3},
            "layer1.0.relu": {"N":1,"C":64,"K":64,"P":56,"Q":56,"R":1,"S":1}
        }
    }
    return config_vp1

def generate_configurations(num_configs: int):
    """Generates a stream of unique configurations with complete dimension factorization using strategy-driven approach.
    
    Args:
        num_configs (int): Number of configurations to generate.
        
    Yields:
        dict: Configuration dictionary containing fusion_group_info, hardware_config, mapping_config, workload_dims.
    """
    hw_keys, hw_values = zip(*HW_CONFIG_SPACE.items())
    
    # Define some typical permutation patterns
    permutation_patterns = [
        'K N C P Q S R'
    ]
    
    # Define available strategies
    strategies = ["performance"]
    
    # Main loop for generating configurations
    for i in range(num_configs):
        # Randomly select a fusion group
        group = random.choice(FUSION_GROUPS_TO_TEST)
        producer_layer = group["producer_layer"]
        dims = WORKLOAD_DIMS[producer_layer]
        
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
            producer_layer: {
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
             "fusion_group_info": group,
             "hardware_config": hardware_config,
             "mapping_config": mapping_config,
             "workload_dims": WORKLOAD_DIMS
         }

def run_dosa_prediction(config: dict, validation_point_id: int = None, output_dir: Path = None) -> dict:
    """
    使用FA-DOSA的内部分析模型，为给定的配置计算PPA预测值。

    Args:
        config (dict): 包含'hardware_config', 'mapping_config', 等信息的完整配置字典。
        validation_point_id (int): 验证点ID，用于生成唯一的调试文件名
        output_dir (Path): 输出目录路径，用于保存调试文件

    Returns:
        dict: 包含预测结果的字典，例如 {'predicted_latency_s': 0.001, 'predicted_energy_pj': 5000.0}
    """
    print("[INFO] Running FA-DOSA analytical model...")
    try:
        hw_config = config['hardware_config']
        mapping_config = config['mapping_config']
        fusion_group_info = config['fusion_group_info']
        workload_dims = config['workload_dims']

        if fusion_group_info['pattern'] != ['Conv', 'ReLU']:
            raise ValueError(f"Unsupported DMT pattern for validation: {fusion_group_info['pattern']}")
        
        # 为每个验证点生成唯一的调试文件名，保存到output目录
        debug_filename = f"debug_performance_model_point_{validation_point_id}.json" if validation_point_id is not None else "debug_performance_model.json"
        if output_dir is not None:
            debug_filename = str(output_dir / debug_filename)
        dmt_model = InPlaceFusionDMT(debug_latency=True, debug_output_path=debug_filename)

        hw_params = HardwareParameters(
            initial_num_pes=hw_config['num_pes'],
            initial_l0_kb=hw_config.get('l0_registers_size_kb', 2.0),
            initial_l1_kb=hw_config.get('l1_accumulator_size_kb', 4.0),
            initial_l2_kb=hw_config['l2_scratchpad_size_kb']
        )
        
        dosa_config = DosaConfig()
        
        producer_layer = fusion_group_info['producer_layer']
        problem_dims = workload_dims[producer_layer]
        hierarchy = dosa_config.MEMORY_HIERARCHY
        
        mapping = FineGrainedMapping(problem_dims, hierarchy)

        with torch.no_grad():
            producer_mapping = mapping_config.get(producer_layer, {})
            for level_name, level_mapping in producer_mapping.items():
                if level_name in mapping.factors:
                    temporal_factors = level_mapping.get('temporal', {})
                    spatial_factors = level_mapping.get('spatial', {})
                    for dim_name in mapping.factors[level_name]:
                        t_factor = temporal_factors.get(dim_name, 1.0)
                        s_factor = spatial_factors.get(dim_name, 1.0)
                        mapping.factors[level_name][dim_name]['temporal'].data = torch.log(torch.tensor(float(t_factor), device=dosa_config.DEVICE))
                        mapping.factors[level_name][dim_name]['spatial'].data = torch.log(torch.tensor(float(s_factor), device=dosa_config.DEVICE))

        hw_params.to(dosa_config.DEVICE)
        mapping.to(dosa_config.DEVICE)
        
        graph = ComputationGraph()
        graph.add_layer(producer_layer, problem_dims, fusion_group_info['pattern'][0])
        consumer_layer = fusion_group_info['consumer_layer']
        graph.add_layer(consumer_layer, workload_dims[consumer_layer], fusion_group_info['pattern'][1])
        
        group = (producer_layer, consumer_layer)
        
        # 准备验证快车道的直接映射表
        producer_mapping = mapping_config.get(producer_layer, {})
        direct_mapping_table = {}
        
        if producer_mapping:
            # 1. 收集所有出现过的维度和层级
            all_dims = set()
            all_levels = producer_mapping.keys()
            for level_data in producer_mapping.values():
                all_dims.update(level_data.get('temporal', {}).keys())
                all_dims.update(level_data.get('spatial', {}).keys())
            
            # 2. 进行数据结构的行列转置
            for dim_name in all_dims:
                direct_mapping_table[dim_name] = {}
                for level_name in all_levels:
                    level_data = producer_mapping.get(level_name, {})
                    # 获取temporal和spatial因子，如果不存在则默认为1.0
                    temporal_val = level_data.get('temporal', {}).get(dim_name, 1.0)
                    spatial_val = level_data.get('spatial', {}).get(dim_name, 1.0)
                    
                    direct_mapping_table[dim_name][level_name] = {
                        'temporal': torch.tensor(float(temporal_val), device=dosa_config.DEVICE),
                        'spatial': torch.tensor(float(spatial_val), device=dosa_config.DEVICE)
                    }
        
        with torch.no_grad():
            result = dmt_model(
                group, graph, hw_params, mapping, dosa_config, 
                direct_mapping_table=direct_mapping_table
            )
            # Handle different return value formats
            if len(result) == 5:
                predicted_latency, predicted_energy, _, _, detailed_metrics = result
            elif len(result) == 4:
                predicted_latency, predicted_energy, _, detailed_metrics = result
            elif len(result) == 3:
                predicted_latency, predicted_energy, detailed_metrics = result
            elif len(result) == 2:
                predicted_latency, predicted_energy = result
                detailed_metrics = {}
            else:
                raise ValueError(f"Unexpected DMT model return format: {len(result)} values")

        # 明确变量命名和单位
        predicted_latency_s = predicted_latency.item()
        predicted_energy_pj = predicted_energy.item()
        
        # 将detailed_metrics中的Tensor转换为Python原生数字类型
        native_detailed_metrics = convert_tensors_to_native(detailed_metrics)
        
        print(f"[INFO] FA-DOSA prediction complete: latency={predicted_latency_s} s, energy={predicted_energy_pj} pJ")
        return {
            "predicted_latency_s": predicted_latency_s,
            "predicted_energy_pj": predicted_energy_pj
        }
    except Exception as e:
        print(f"[ERROR] FA-DOSA prediction failed: {e}")
        return {
            "predicted_latency_s": -1.0,
            "predicted_energy_pj": -1.0
        }


def run_timeloop_simulation(config: dict, work_dir: Path) -> dict:
    """
    为给定配置运行Timeloop/Accelergy仿真并返回结果。

    Args:
        config (dict): 与 run_dosa_prediction 使用的完全相同的配置字典。
        work_dir (Path): 用于存放临时YAML文件的目录路径对象。

    Returns:
        dict: 包含仿真结果的字典，例如 {'simulated_latency_s': 0.0012, 'simulated_energy_pj': 5200.0}
    """
    print("[INFO] Running Timeloop/Accelergy simulation...")
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
            simulated_energy_pj = simulated_energy_uj * 1e6
            
            # 获取DosaConfig实例以访问时钟频率
            dosa_config = DosaConfig()
            
            # 计算时钟周期时长（秒）
            cycle_time_s = 1.0 / (dosa_config.CLOCK_FREQUENCY_MHZ * 1e6)
            
            # 将时钟周期转换为秒
            simulated_latency_s = simulated_cycles * cycle_time_s
            
            # 增强日志输出
            print(f"[INFO] Timeloop raw output: cycles={simulated_cycles}, energy={simulated_energy_pj} pJ")
            print(f"[INFO] Converted to: latency={simulated_latency_s} s")
            
            return {
                "simulated_latency_s": simulated_latency_s,
                "simulated_energy_pj": simulated_energy_pj
            }
        else:
            print("[ERROR] Failed to extract performance metrics from Timeloop results.")
            print(f"[DEBUG] stats object: {stats}")
            if stats:
                print(f"[DEBUG] stats attributes: {dir(stats)}")
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


def validate_one_point(config: dict, result_queue: multiprocessing.Queue, validation_point_id: int):
    """
    Worker function to validate a single design point with complete error handling.
    
    This function encapsulates the entire validation process for one configuration,
    including both FA-DOSA prediction and Timeloop simulation. It runs in a separate
    process to enable timeout functionality.
    
    Args:
        config (dict): Complete configuration dictionary
        result_queue (multiprocessing.Queue): Queue to return results to main process
        validation_point_id (int): ID of the validation point for logging
    """
    try:
        print(f"[WORKER] Starting validation for point {validation_point_id}")
        
        # Create temporary workspace for this worker
        work_dir = Path(f'./validation_workspace_{validation_point_id}')
        work_dir.mkdir(exist_ok=True)
        
        # Run dual-track evaluation
        dosa_results = run_dosa_prediction(config, validation_point_id)
        timeloop_results = run_timeloop_simulation(config, work_dir)
        
        # Combine configuration info and results
        flat_config = {
            "validation_point_id": validation_point_id,
            "group_name": config['fusion_group_info']['group_name'],
            **config['hardware_config'],
        }
        
        combined_result = {
            **flat_config,
            **dosa_results,
            **timeloop_results,
            "status": "success"
        }
        
        # Clean up worker workspace (DISABLED to preserve Timeloop reports)
        # import shutil
        # if work_dir.exists():
        #     try:
        #         shutil.rmtree(work_dir)
        #     except Exception as cleanup_error:
        #         print(f"[WARNING] Worker cleanup failed: {cleanup_error}")
        print(f"[INFO] Timeloop reports preserved in: {work_dir.resolve()}")
        
        # Send result back to main process
        result_queue.put(combined_result)
        print(f"[WORKER] Validation point {validation_point_id} completed successfully")
        
    except Exception as e:
        print(f"[ERROR] Worker validation failed for point {validation_point_id}: {e}")
        import traceback
        traceback.print_exc()
        
        # Send error result back to main process
        error_result = {
            "validation_point_id": validation_point_id,
            "group_name": config.get('fusion_group_info', {}).get('group_name', 'unknown'),
            **config.get('hardware_config', {}),
            "predicted_latency_s": -1.0,
            "predicted_energy_pj": -1.0,
            "simulated_latency_s": -1.0,
            "simulated_energy_pj": -1.0,
            "status": "error",
            "error_message": str(e)
        }
        result_queue.put(error_result)


def generate_timeloop_files(config, work_dir):
    """Generates arch.yaml, problem.yaml, constraints.yaml, and env.yaml for Timeloop v0.4."""
    hw_config = config['hardware_config']
    mapping_config = config['mapping_config']
    producer_layer = config['fusion_group_info']['producer_layer']
    workload_dims = config['workload_dims'][producer_layer]

    # --- 1. Generate arch.yaml (Hierarchical v0.4 format) ---
    meshX = int(math.sqrt(hw_config['num_pes']))
    datawidth = 16 # Assuming 16-bit data for this validation

    # Helper to calculate depth from size_kb
    def get_depth(size_kb):
        return int(size_kb * 1024 * 8 / datawidth)

    # Step 3: Reconstruct arch_dict using correct node types
    arch_dict = {
        'architecture': {
            'version': '0.4',
            'nodes': [
                Hierarchical({
                    'nodes': [
                        Component({
                            'name': 'DRAM',
                            'class': 'DRAM',
                            'attributes': {'depth': 1048576, 'width': 256, 'datawidth': datawidth} 
                        }),
                        Hierarchical({
                            'nodes': [
                                Component({
                                    'name': 'L2_Scratchpad',
                                    'class': 'SRAM',
                                    'attributes': {'depth': get_depth(hw_config['l2_scratchpad_size_kb']), 'width': datawidth, 'datawidth': datawidth}
                                }),
                                Hierarchical({
                                    'nodes': [
                                        Container({
                                            'name': 'PE_array_container',
                                            'spatial': {'meshX': meshX, 'meshY': meshX}
                                        }),
                                        Component({
                                            'name': 'L1_Accumulator',
                                            'class': 'regfile',
                                            'attributes': {'depth': get_depth(hw_config.get('l1_accumulator_size_kb', 4.0)), 'width': datawidth, 'datawidth': datawidth}
                                        }),
                                        Component({
                                            'name': 'L0_Registers',
                                            'class': 'regfile',
                                            'attributes': {'depth': get_depth(hw_config.get('l0_registers_size_kb', 2.0)), 'width': datawidth, 'datawidth': datawidth}
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
        }
    }
    # Step 4: Use standard yaml.dump without custom Dumper
    with open(work_dir / 'arch.yaml', 'w') as f:
        yaml.dump(arch_dict, f, sort_keys=False)

    # --- 2. Generate problem.yaml ---
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
            'instance': workload_dims
        }
    }

    # 写入文件
    with open(work_dir / 'problem.yaml', 'w') as f:
        # 使用标准的 dump 即可，无需任何特殊格式化
        yaml.dump(problem_config, f, sort_keys=False)

    # --- 3. Generate constraints.yaml ---
    layer_mapping = mapping_config[producer_layer]
    all_dims = ['N', 'K', 'C', 'P', 'Q', 'R', 'S']
    
    # Verify and adjust factors to ensure product matches problem dimensions
    def verify_and_adjust_factors(factors_dict, dim_name, dim_size):
        if dim_name not in factors_dict:
            return {dim_name: dim_size}
        
        # Calculate current product
        product = 1
        for val in factors_dict.values():
            product *= int(val)
        
        # If product doesn't match, adjust DRAM factor
        if product != dim_size and dim_size > 0:
            # Calculate adjustment factor
            adjustment = dim_size / product
            # Apply adjustment to DRAM factor
            if 'DRAM' in factors_dict:
                factors_dict['DRAM'] = int(factors_dict['DRAM'] * adjustment)
            else:
                # If no DRAM factor, add it
                factors_dict['DRAM'] = int(adjustment)
        
        return factors_dict
    
    # Collect all factors for each dimension
    dim_factors = {dim: {} for dim in all_dims}
    
    # Collect spatial factors
    for dim, val in layer_mapping['L0_Registers']['spatial'].items():
        if dim in dim_factors:
            dim_factors[dim]['spatial'] = int(val)
    
    # Collect temporal factors
    for target_name in ['DRAM', 'L2_Scratchpad', 'L1_Accumulator', 'L0_Registers']:
        for dim, val in layer_mapping[target_name]['temporal'].items():
            if dim in dim_factors:
                dim_factors[dim][target_name] = int(val)
    
    # Verify and adjust factors for each dimension
    for dim in all_dims:
        if dim in workload_dims:
            dim_size = workload_dims[dim]
            # Calculate current product
            product = 1
            for level, val in dim_factors[dim].items():
                product *= val
            
            # If product doesn't match, adjust DRAM factor
            if product != dim_size and dim_size > 0:
                # Calculate adjustment factor
                adjustment = dim_size / product
                # Apply adjustment to DRAM factor
                if 'DRAM' in dim_factors[dim]:
                    dim_factors[dim]['DRAM'] = int(dim_factors[dim]['DRAM'] * adjustment)
                else:
                    # If no DRAM factor, add it
                    dim_factors[dim]['DRAM'] = int(adjustment)
    
    # Format factors for constraints.yaml
    def format_factors(level_type, level_name):
        factors = []
        for dim in all_dims:
            if dim in workload_dims:
                if level_type == 'spatial' and 'spatial' in dim_factors[dim]:
                    if dim in ['K', 'C']:  # Only K and C have spatial factors
                        factors.append(f"{dim}={dim_factors[dim]['spatial']}")
                elif level_type == 'temporal' and level_name in dim_factors[dim]:
                    factors.append(f"{dim}={dim_factors[dim][level_name]}")
                elif level_type == 'temporal' and level_name not in dim_factors[dim]:
                    # Default to 1 if not specified
                    factors.append(f"{dim}=1")
        return factors
    
    targets = []
    # Spatial Target
    targets.append({
        'target': 'PE_array_container', 'type': 'spatial',
        'factors': format_factors('spatial', 'L0_Registers'),
        'permutation': layer_mapping['L0_Registers']['permutation']
    })
    # Temporal Targets
    for target_name in ['DRAM', 'L2_Scratchpad', 'L1_Accumulator', 'L0_Registers']:
        targets.append({
            'target': target_name, 'type': 'temporal',
            'factors': format_factors('temporal', target_name),
            'permutation': layer_mapping[target_name]['permutation']
        })
    # Dataspace Target for Fusion
    targets.append({'target': 'L2_Scratchpad', 'type': 'dataspace', 'keep': ['Outputs']})

    with open(work_dir / 'constraints.yaml', 'w') as f:
        yaml.dump({'constraints': {'targets': targets}}, f, sort_keys=False)

    # --- 4. Generate env.yaml ---
    env_config = {
        'globals': {'environment_variables': {
            'ACCELERGY_COMPONENT_LIBRARIES': '/root/accelergy-timeloop-infrastructure/src/accelergy-library-plug-in/library/'
        }},
        'variables': {'global_cycle_seconds': 1e-9, 'technology': "40nm"}
    }
    with open(work_dir / 'env.yaml', 'w') as f:
        yaml.dump(env_config, f, sort_keys=False, default_style="'")

def generate_validation_configs(max_runs=None):
    """Generate validation configurations for multiple data points."""
    configs = []
    validation_point_id = 1
    
    # Generate configurations based on configuration space
    for fusion_group in FUSION_GROUPS_TO_TEST:
        for num_pes in HW_CONFIG_SPACE["num_pes"]:
            for l2_size in HW_CONFIG_SPACE["l2_scratchpad_size_kb"]:
                # Generate dynamic mapping space for this workload
                producer_layer = fusion_group["producer_layer"]
                mapping_space = generate_dynamic_mapping_space(WORKLOAD_DIMS, producer_layer)
                
                # Generate a few mapping configurations
                for k_factor in mapping_space.get("K", [1])[:2]:  # Limit to 2 for efficiency
                    for c_factor in mapping_space.get("C", [1])[:2]:
                        # Generate complete factor decomposition
                        num_pes_sqrt = int(math.sqrt(num_pes))
                        k_factors = generate_factors_by_strategy(WORKLOAD_DIMS[producer_layer]["K"], num_pes_sqrt, "performance")
                        c_factors = generate_factors_by_strategy(WORKLOAD_DIMS[producer_layer]["C"], num_pes_sqrt, "random")
                        
                        config = {
                            "hardware_config": {
                                "num_pes": num_pes,
                                "l2_scratchpad_size_kb": l2_size
                            },
                            "mapping_config": {
                                producer_layer: {
                                    "DRAM": {"temporal": {"N":1,"C":c_factors["DRAM"],"K":k_factors["DRAM"],"P":1,"Q":1,"R":1,"S":1}, "permutation": "K N C P Q S R"},
                                    "L2_Scratchpad": {"temporal": {"N":1,"C":c_factors["L2"],"K":k_factors["L2"],"P":1,"Q":7,"R":1,"S":1}, "permutation": "K N C P Q S R"},
                                    "L1_Accumulator": {"temporal": {"N":1,"C":c_factors["L1"],"K":k_factors["L1"],"P":1,"Q":2,"R":1,"S":1}, "permutation": "K N C P Q S R"},
                                    "L0_Registers": {"temporal": {"N":1,"C":c_factors["L0"],"K":k_factors["L0"],"P":7,"Q":2,"R":3,"S":3}, "spatial": {"C":c_factors["spatial"],"K":k_factors["spatial"]}, "permutation": "K N C P Q S R"}
                                }
                            },
                            "fusion_group_info": fusion_group,
                            "workload_dims": WORKLOAD_DIMS
                        }
                        
                        configs.append((validation_point_id, config))
                        validation_point_id += 1
                        
                        # Limit total configurations if max_runs is specified
                        if max_runs is not None and len(configs) >= max_runs:
                            return configs
    
    return configs

def main():
    """Main control script to run DMT validation experiments with timeout support."""
    parser = argparse.ArgumentParser(description="Run DMT validation experiments")
    parser.add_argument('--max-runs', type=int, default=MAX_VALIDATION_RUNS,
                       help='Maximum number of validation runs (default: None for full sweep)')
    parser.add_argument('--output-dir', type=str, default="output",
                       help='Output directory for all generated files (default: output)')
    parser.add_argument('--timeout', type=int, default=VALIDATION_TIMEOUT_SECONDS,
                       help=f'Timeout in seconds for each validation point (default: {VALIDATION_TIMEOUT_SECONDS})')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    all_results = []
    master_configs = {}  # Dictionary to store all validation point configurations
    max_runs = args.max_runs
    timeout_seconds = args.timeout
    
    # Statistics counters
    success_count = 0
    error_count = 0
    timeout_count = 0

    print("Starting DMT validation run with timeout support...")
    print(f"[INFO] Output directory: {output_dir.resolve()}")
    print(f"[INFO] Timeout per validation point: {timeout_seconds} seconds ({timeout_seconds/60:.1f} minutes)")
    if max_runs is not None:
        print(f"[INFO] Limited to {max_runs} validation runs for quick testing")
    else:
        print("[INFO] Running full validation sweep")

    try:
        # Generate validation configurations
        validation_configs = generate_validation_configs(max_runs)
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
                    "group_name": config['fusion_group_info']['group_name'],
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
                    "group_name": config['fusion_group_info']['group_name'],
                    **config['hardware_config']
                }
                error_result = {**flat_config, "predicted_latency_s": -1.0, "predicted_energy_pj": -1.0, 
                               "simulated_latency_s": -1.0, "simulated_energy_pj": -1.0, 
                               "status": "error", "error_message": str(e)}
                all_results.append(error_result)
                error_count += 1

        # Save results to CSV file
        if all_results:
            csv_file = output_dir / "dmt_validation_results.csv"
            df = pd.DataFrame(all_results)
            # Ensure all required columns are present
            required_columns = ['validation_point_id', 'group_name', 'num_pes', 'l2_scratchpad_size_kb', 
                               'predicted_latency_s', 'predicted_energy_pj', 'simulated_latency_s', 'simulated_energy_pj', 'status']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = -1.0 if 'latency' in col or 'energy' in col else 'unknown'
            
            df.to_csv(csv_file, index=False)
            print(f"[INFO] Results saved to {csv_file}")
        
        # Debug JSON files are already saved directly to output directory
        
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