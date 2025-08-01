import json
import argparse
import subprocess
import os
from pathlib import Path
import math
import yaml

import torch
import pytimeloop.timeloopfe.v4 as tl
from dosa.dmt import InPlaceFusionDMT
from dosa.utils import ComputationGraph
from dosa.hardware_parameters import HardwareParameters
from dosa.mapping import FineGrainedMapping
from dosa.config import Config as DosaConfig

# Step 1: Define node classes that inherit from dict
class Component(dict):
    """Represents a Timeloop Component node (leaf node)."""
    pass

class Container(dict):
    """Represents a Timeloop Container node (leaf node)."""
    pass

class Hierarchical(dict):
    """Represents a Timeloop Hierarchical node (branch node)."""
    pass

# Step 2: Implement and register YAML representer functions
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


def convert_tensors_to_native(obj):
    """递归将PyTorch Tensor转换为Python原生数字类型。"""
    if isinstance(obj, torch.Tensor):
        return float(obj.item())
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors_to_native(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_tensors_to_native(item) for item in obj)
    else:
        return obj


def get_dmt_prediction(config):
    """Calculates performance prediction using the DMT model.
    Returns prediction results with latency in seconds and energy in picojoules."""
    print("[INFO] Calculating DMT prediction...")

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
        predicted_latency, predicted_energy, _, detailed_metrics = dmt_model(
            group, graph, hw_params, mapping, dosa_config
        )

    # 明确变量命名和单位
    predicted_latency_s = predicted_latency.item()
    predicted_energy_pj = predicted_energy.item()
    
    # 将detailed_metrics中的Tensor转换为Python原生数字类型
    native_detailed_metrics = convert_tensors_to_native(detailed_metrics)
    
    print(f"[INFO] DMT prediction complete: latency={predicted_latency_s} s, energy={predicted_energy_pj} pJ")
    return predicted_latency_s, predicted_energy_pj, native_detailed_metrics


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


def parse_timeloop_output(stats_file):
    """Parses the timeloop-model.stats.txt file to get key performance metrics."""
    if not stats_file.exists(): return -1.0, -1.0
    with open(stats_file, 'r') as f: content = f.read()
    try:
        summary_section = content.split('Summary Stats')[1]
        cycles = float(summary_section.split('Cycles: ')[1].split('\n')[0])
        energy = float(summary_section.split('Energy: ')[1].split(' ')[0])
        return cycles, energy
    except (IndexError, ValueError) as e:
        print(f"[ERROR] Could not parse Timeloop stats file: {e}")
        return -1.0, -1.0

def get_timeloop_simulation(config):
    """Generates Timeloop files, runs simulation, and parses results.
    Returns simulation results with latency in seconds and energy in picojoules."""
    print("[INFO] Running Timeloop simulation...")
    work_dir = Path('./timeloop_workspace')
    work_dir.mkdir(exist_ok=True)

    try:
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
        else:
            print("[ERROR] Failed to extract performance metrics from Timeloop results.")
            print(f"[DEBUG] stats object: {stats}")
            if stats:
                print(f"[DEBUG] stats attributes: {dir(stats)}")
            return -1.0, -1.0
        
        # Clean up only on success
        # for f in work_dir.glob('*'): f.unlink()
        # work_dir.rmdir()

    except Exception as e:
        print(f"[ERROR] An exception occurred during Timeloop simulation: {e}")
        print(f"[DEBUG] Timeloop files preserved in: {work_dir.resolve()}")
        return -1.0, -1.0

    print("[INFO] Timeloop simulation complete.")
    return simulated_latency_s, simulated_energy_pj


def main():
    """Main function to run the DMT validation.
    Compares DMT predictions with Timeloop simulation results using unified units:
    - Latency: seconds (s)
    - Energy: picojoules (pJ)
    """
    parser = argparse.ArgumentParser(description="Validate DMT performance prediction against Timeloop.")
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file for the validation point.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    # 获取预测值和仿真值，使用明确的变量名称
    predicted_latency_s, predicted_energy_pj, detailed_metrics = get_dmt_prediction(config)
    simulated_latency_s, simulated_energy_pj = get_timeloop_simulation(config)

    # 更新结果字典结构，使键名能够清晰地反映物理单位
    results = {
        "config": config,
        "prediction": {
            "latency_s": predicted_latency_s, 
            "energy_pj": predicted_energy_pj
        },
        "simulation": {
            "latency_s": simulated_latency_s, 
            "energy_pj": simulated_energy_pj
        },
        "detailed_prediction_metrics": detailed_metrics
    }

    print("\n---DMT_VALIDATION_RESULT_START---")
    print(json.dumps(results, indent=4))
    print("---DMT_VALIDATION_RESULT_END---")

if __name__ == "__main__":
    main()