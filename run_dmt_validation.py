import json
import subprocess
import os
import itertools
import pandas as pd
import random
import argparse
import math
import yaml
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

# Configuration for number of validation runs
MAX_VALIDATION_RUNS = None  # Set to None for full sweep, or specify a number (e.g., 1, 2, 4) for quick testing

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

def generate_complete_dimension_factors(dim_size, num_pes_sqrt):
    """Generate complete factor decomposition for a dimension across all levels.
    
    Args:
        dim_size (int): The problem dimension size to factorize.
        num_pes_sqrt (int): Square root of the number of PEs, used to constrain spatial factors.
        
    Returns:
        dict: A dictionary with factors for each level ['spatial', 'L0', 'L1', 'L2', 'DRAM'].
    """
    if dim_size == 1:
        return {'spatial': 1, 'L0': 1, 'L1': 1, 'L2': 1, 'DRAM': 1}
    
    # Initialize factors with default values
    factors = {'spatial': 1, 'L0': 1, 'L1': 1, 'L2': 1, 'DRAM': 1}
    
    # Get all prime factors of the dimension size
    def get_prime_factors(n):
        i = 2
        factors = []
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                factors.append(i)
        if n > 1:
            factors.append(n)
        return factors
    
    prime_factors = get_prime_factors(dim_size)
    
    # If no prime factors (dim_size is 1), return default factors
    if not prime_factors:
        return factors
    
    # Sort prime factors in descending order for better distribution
    prime_factors.sort(reverse=True)
    
    # First, assign spatial factors (limited by num_pes_sqrt)
    spatial_product = 1
    remaining_factors = []
    
    for factor in prime_factors:
        if spatial_product * factor <= num_pes_sqrt:
            spatial_product *= factor
        else:
            remaining_factors.append(factor)
    
    factors['spatial'] = spatial_product
    
    # If there are no remaining factors, assign 1 to all memory levels
    if not remaining_factors:
        return factors
    
    # Distribute remaining factors among memory levels
    # We'll use a more deterministic approach to ensure all factors are used
    memory_levels = ['L0', 'L1', 'L2', 'DRAM']
    level_index = 0
    
    for factor in remaining_factors:
        factors[memory_levels[level_index]] *= factor
        level_index = (level_index + 1) % len(memory_levels)
    
    # Verification
    product = 1
    for val in factors.values():
        product *= val
    
    # If verification fails, use a direct approach to ensure correctness
    if product != dim_size:
        # Reset factors
        factors = {'spatial': 1, 'L0': 1, 'L1': 1, 'L2': 1, 'DRAM': dim_size}
        
        # Try to assign a valid spatial factor
        divisors = get_divisors(dim_size).tolist()
        spatial_candidates = [d for d in divisors if d <= num_pes_sqrt]
        
        if spatial_candidates:
            factors['spatial'] = spatial_candidates[-1]  # Take the largest valid divisor
            factors['DRAM'] = dim_size // factors['spatial']
    
    # Final verification
    product = 1
    for val in factors.values():
        product *= val
    assert product == dim_size, f"Factor product {product} does not match dim_size {dim_size}"
    
    return factors

def generate_configurations(num_configs: int):
    """Generates a stream of unique configurations with complete dimension factorization.
    
    Args:
        num_configs (int): Number of configurations to generate.
        
    Yields:
        dict: Configuration dictionary containing fusion_group_info, hardware_config, mapping_config, workload_dims.
    """
    hw_keys, hw_values = zip(*HW_CONFIG_SPACE.items())
    
    # Define some typical permutation patterns
    permutation_patterns = [
        'K C N P Q S R',
        'N K C P Q S R', 
        'C K N P Q S R',
        'K N C P Q S R'
    ]
    
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
        
        # Generate complete factorization for each dimension
        dim_factors = {}
        for dim_name, dim_size in dims.items():
            if dim_name in ['N', 'K', 'C', 'P', 'Q', 'R', 'S']:
                dim_factors[dim_name] = generate_complete_dimension_factors(dim_size, pe_mesh_size)
        
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
                    'permutation': random.choice(permutation_patterns)
                },
                'PE_array': {
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

def run_dosa_prediction(config: dict) -> dict:
    """
    使用FA-DOSA的内部分析模型，为给定的配置计算PPA预测值。

    Args:
        config (dict): 包含'hardware_config', 'mapping_config', 等信息的完整配置字典。

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
        
        dmt_model = InPlaceFusionDMT()

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
        
        with torch.no_grad():
            result = dmt_model(
                group, graph, hw_params, mapping, dosa_config
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
    for dim, val in layer_mapping['PE_array']['spatial'].items():
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
        'factors': format_factors('spatial', 'PE_array'),
        'permutation': layer_mapping['PE_array']['permutation']
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

def main():
    """Main control script to run DMT validation experiments."""
    parser = argparse.ArgumentParser(description="Run DMT validation experiments")
    parser.add_argument('--max-runs', type=int, default=MAX_VALIDATION_RUNS,
                       help='Maximum number of validation runs (default: None for full sweep)')
    parser.add_argument('--output', type=str, default="dmt_validation_results.csv",
                       help='Output CSV file path (default: dmt_validation_results.csv)')
    args = parser.parse_args()
    
    output_csv_path = args.output
    all_results = []
    max_runs = args.max_runs

    print("Starting DMT validation run...")
    if max_runs is not None:
        print(f"[INFO] Limited to {max_runs} validation runs for quick testing")
    else:
        print("[INFO] Running full validation sweep")

    # Create temporary workspace
    work_dir = Path('./validation_workspace')
    work_dir.mkdir(exist_ok=True)

    try:
        for i, config in enumerate(generate_configurations(max_runs if max_runs is not None else 1000)):
            # Check if we've reached the maximum number of runs
            if max_runs is not None and i >= max_runs:
                print(f"[INFO] Reached maximum validation runs limit ({max_runs}). Stopping.")
                break
                
            print(f"--- Running Validation Point {i+1} ---")
            
            # 双轨评估
            dosa_results = run_dosa_prediction(config)
            timeloop_results = run_timeloop_simulation(config, work_dir)

            # 数据合并
            # 将配置信息和两边的结果合并到一个字典中
            flat_config = {
                "validation_point_id": i + 1,
                "group_name": config['fusion_group_info']['group_name'],
                **config['hardware_config'],
                # 可以有选择性地扁平化一些关键的mapping参数以方便分析
            }
            
            combined_result = {
                **flat_config,
                **dosa_results,
                **timeloop_results
            }
            
            # 数据追加
            all_results.append(combined_result)

            # 增量写入CSV文件
            df = pd.DataFrame([combined_result])
            if i == 0:
                # 第一次写入，包含表头
                df.to_csv(args.output, index=False, mode='w')
            else:
                # 后续写入，追加模式，不包含表头
                df.to_csv(args.output, index=False, mode='a', header=False)
            
            print(f"[SUCCESS] Validation point {i+1} completed and saved to {args.output}")

        # 结果持久化
        if all_results:
            print(f"\nValidation complete. {len(all_results)} data points saved to {args.output}")
        else:
            print("\nValidation run finished, but no results were collected.")
            
    except Exception as e:
        print(f"[ERROR] Main validation loop failed: {e}")
        import traceback
        traceback.print_exc()
            
    finally:
        # 可选：清理工作目录
        import shutil
        if work_dir.exists():
            try:
                shutil.rmtree(work_dir)
                print("[INFO] Cleaned up temporary workspace")
            except Exception as e:
                print(f"[WARNING] Failed to clean up workspace: {e}")

if __name__ == "__main__":
    main()