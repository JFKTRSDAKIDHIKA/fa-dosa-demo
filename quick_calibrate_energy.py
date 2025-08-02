#!/usr/bin/env python3
"""
Quick Energy Model Calibration Script

This script provides a fast, lightweight way to compare DOSA custom energy model
predictions against Timeloop/Accelergy simulation results for a single hardware
and mapping configuration point. Designed for rapid iteration and debugging of
energy model parameters.

Usage:
    python quick_calibrate_energy.py --config <path_to_config.json>
"""

import json
import argparse
import os
import math
from pathlib import Path
import yaml
import torch

# Import DOSA components
from dosa.config import Config as DosaConfig
from dosa.hardware_parameters import HardwareParameters
from dosa.mapping import FineGrainedMapping
from dosa.utils import ComputationGraph
from dosa.performance_model import HighFidelityPerformanceModel

# Import Timeloop components
import pytimeloop.timeloopfe.v4 as tl


# Timeloop YAML node classes (reused from validate_dmt.py)
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
    return dumper.represent_mapping('!Component', data.items())

def represent_container(dumper, data):
    return dumper.represent_mapping('!Container', data.items())

def represent_hierarchical(dumper, data):
    return dumper.represent_mapping('!Hierarchical', data.items())

# Register representers
yaml.add_representer(Component, represent_component)
yaml.add_representer(Container, represent_container)
yaml.add_representer(Hierarchical, represent_hierarchical)


def run_dosa_energy_model(config: dict) -> float:
    """
    Run DOSA custom energy model and return predicted energy in picojoules (pJ).
    
    This function focuses exclusively on energy calculation, skipping latency
    computations for maximum speed.
    
    Args:
        config: Complete configuration dictionary
        
    Returns:
        float: Predicted energy in picojoules (pJ)
    """
    print("[INFO] Running DOSA energy model...")
    
    # Extract configuration components
    hw_config = config['hardware_config']
    mapping_config = config['mapping_config']
    fusion_group_info = config['fusion_group_info']
    workload_dims = config['workload_dims']
    
    # Validate fusion pattern
    if fusion_group_info['pattern'] != ['Conv', 'ReLU']:
        raise ValueError(f"Unsupported pattern: {fusion_group_info['pattern']}")
    
    # Initialize DOSA components
    dosa_config = DosaConfig()
    
    hw_params = HardwareParameters(
        initial_num_pes=hw_config['num_pes'],
        initial_l0_kb=hw_config.get('l0_registers_size_kb', 2.0),
        initial_l1_kb=hw_config.get('l1_accumulator_size_kb', 4.0),
        initial_l2_kb=hw_config['l2_scratchpad_size_kb']
    )
    
    # Setup problem dimensions and mapping
    producer_layer = fusion_group_info['producer_layer']
    problem_dims = workload_dims[producer_layer]
    hierarchy = dosa_config.MEMORY_HIERARCHY
    
    mapping = FineGrainedMapping(problem_dims, hierarchy)
    
    # Configure mapping factors from config
    with torch.no_grad():
        producer_mapping = mapping_config.get(producer_layer, {})
        for level_name, level_mapping in producer_mapping.items():
            if level_name in mapping.factors:
                temporal_factors = level_mapping.get('temporal', {})
                spatial_factors = level_mapping.get('spatial', {})
                for dim_name in mapping.factors[level_name]:
                    t_factor = temporal_factors.get(dim_name, 1.0)
                    s_factor = spatial_factors.get(dim_name, 1.0)
                    mapping.factors[level_name][dim_name]['temporal'].data = torch.log(
                        torch.tensor(float(t_factor), device=dosa_config.DEVICE)
                    )
                    mapping.factors[level_name][dim_name]['spatial'].data = torch.log(
                        torch.tensor(float(s_factor), device=dosa_config.DEVICE)
                    )
    
    # Move to device
    hw_params.to(dosa_config.DEVICE)
    mapping.to(dosa_config.DEVICE)
    
    # Create computation graph
    graph = ComputationGraph()
    graph.add_layer(producer_layer, problem_dims, fusion_group_info['pattern'][0])
    consumer_layer = fusion_group_info['consumer_layer']
    graph.add_layer(consumer_layer, workload_dims[consumer_layer], fusion_group_info['pattern'][1])
    
    # Initialize performance model
    perf_model = HighFidelityPerformanceModel(dosa_config)
    
    # Calculate energy only (skip latency for speed)
    with torch.no_grad():
        # Get layer information from graph.layers
        layer_info = graph.layers[producer_layer]
        
        # Get all mapping factors
        all_factors = mapping.get_all_factors()
        
        # Calculate per-level accesses
        per_level_accesses = perf_model.calculate_per_level_accesses(layer_info['dims'], all_factors)
        
        # Calculate energy from memory accesses
        energy = torch.tensor(0.0, device=dosa_config.DEVICE)
        
        # Add energy from each memory interface
        for interface, accesses in per_level_accesses.items():
            upper_level_name = interface.split('_to_')[0]
            level_info_hw = next((level for level in dosa_config.MEMORY_HIERARCHY if level['name'] == upper_level_name), None)
            
            if level_info_hw:
                # Calculate energy per access (pJ per byte)
                energy_per_byte = level_info_hw.get('energy_per_access_pj', 1.0) / dosa_config.BYTES_PER_ELEMENT
                energy += accesses * energy_per_byte
        
        # Add intra-level energy if available
        if hasattr(perf_model, 'calculate_intra_level_accesses'):
            num_pes = hw_params.get_projected_num_pes()
            intra_level_accesses = perf_model.calculate_intra_level_accesses(
                layer_info['dims'], all_factors, num_pes
            )
            
            # Convert intra-level accesses to energy
            for level_name, tensors in intra_level_accesses.items():
                level_info_hw = next((level for level in dosa_config.MEMORY_HIERARCHY if level['name'] == level_name), None)
                if level_info_hw:
                    energy_per_access_pj = level_info_hw.get('energy_per_access_pj', 1.0)
                    for tensor_type, operations in tensors.items():
                        for op_type, count in operations.items():
                            energy += count * energy_per_access_pj * dosa_config.BYTES_PER_ELEMENT
    
    predicted_energy_pj = energy.item()
    print(f"[INFO] DOSA energy prediction: {predicted_energy_pj:.2f} pJ")
    
    return predicted_energy_pj


def generate_timeloop_files_quick(config: dict, work_dir: Path):
    """
    Generate Timeloop configuration files for energy simulation.
    Simplified version focused on energy calculation.
    """
    hw_config = config['hardware_config']
    mapping_config = config['mapping_config']
    producer_layer = config['fusion_group_info']['producer_layer']
    workload_dims = config['workload_dims'][producer_layer]
    
    # Calculate mesh dimensions
    meshX = int(math.sqrt(hw_config['num_pes']))
    datawidth = 16
    
    def get_depth(size_kb):
        return int(size_kb * 1024 * 8 / datawidth)
    
    # Generate arch.yaml
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
                                    'attributes': {
                                        'depth': get_depth(hw_config['l2_scratchpad_size_kb']),
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
                                            'name': 'L1_Accumulator',
                                            'class': 'regfile',
                                            'attributes': {
                                                'depth': get_depth(hw_config.get('l1_accumulator_size_kb', 4.0)),
                                                'width': datawidth,
                                                'datawidth': datawidth
                                            }
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
        }
    }
    
    with open(work_dir / 'arch.yaml', 'w') as f:
        yaml.dump(arch_dict, f, sort_keys=False)
    
    # Generate problem.yaml
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
    
    with open(work_dir / 'problem.yaml', 'w') as f:
        yaml.dump(problem_config, f, sort_keys=False)
    
    # Generate constraints.yaml (simplified)
    layer_mapping = mapping_config[producer_layer]
    all_dims = ['N', 'K', 'C', 'P', 'Q', 'R', 'S']
    
    def format_factors(level_type, level_name):
        factors = []
        for dim in all_dims:
            if dim in workload_dims:
                if level_type == 'spatial' and level_name == 'PE_array':
                    spatial_factors = layer_mapping['PE_array']['spatial']
                    if dim in spatial_factors:
                        factors.append(f"{dim}={spatial_factors[dim]}")
                elif level_type == 'temporal' and level_name in layer_mapping:
                    temporal_factors = layer_mapping[level_name]['temporal']
                    if dim in temporal_factors:
                        factors.append(f"{dim}={temporal_factors[dim]}")
                    else:
                        factors.append(f"{dim}=1")
        return factors
    
    targets = [
        {
            'target': 'PE_array_container',
            'type': 'spatial',
            'factors': format_factors('spatial', 'PE_array'),
            'permutation': layer_mapping['PE_array']['permutation']
        }
    ]
    
    for target_name in ['DRAM', 'L2_Scratchpad', 'L1_Accumulator', 'L0_Registers']:
        targets.append({
            'target': target_name,
            'type': 'temporal',
            'factors': format_factors('temporal', target_name),
            'permutation': layer_mapping[target_name]['permutation']
        })
    
    targets.append({'target': 'L2_Scratchpad', 'type': 'dataspace', 'keep': ['Outputs']})
    
    with open(work_dir / 'constraints.yaml', 'w') as f:
        yaml.dump({'constraints': {'targets': targets}}, f, sort_keys=False)
    
    # Generate env.yaml
    env_config = {
        'globals': {
            'environment_variables': {
                'ACCELERGY_COMPONENT_LIBRARIES': '/root/accelergy-timeloop-infrastructure/src/accelergy-library-plug-in/library/'
            }
        },
        'variables': {
            'global_cycle_seconds': 1e-9,
            'technology': "40nm"
        }
    }
    
    with open(work_dir / 'env.yaml', 'w') as f:
        yaml.dump(env_config, f, sort_keys=False, default_style="'")


def run_timeloop_energy_simulation(config: dict) -> float:
    """
    Run Timeloop/Accelergy simulation and return energy in picojoules (pJ).
    
    Args:
        config: Complete configuration dictionary
        
    Returns:
        float: Simulated energy in picojoules (pJ)
    """
    print("[INFO] Running Timeloop energy simulation...")
    
    work_dir = Path('./quick_cal_workspace')
    work_dir.mkdir(exist_ok=True)
    
    try:
        # Generate Timeloop configuration files
        generate_timeloop_files_quick(config, work_dir)
        
        # Define input files
        input_files = [
            str(work_dir / "arch.yaml"),
            str(work_dir / "problem.yaml"),
            str(work_dir / "constraints.yaml"),
            str(work_dir / "env.yaml")
        ]
        
        # Load specification and run mapper
        spec = tl.Specification.from_yaml_files(input_files)
        stats = tl.call_mapper(spec, output_dir=str(work_dir))
        
        # Extract energy from results
        if stats and hasattr(stats, 'energy'):
            simulated_energy_pj = float(stats.energy)
            print(f"[INFO] Timeloop energy simulation: {simulated_energy_pj:.2f} pJ")
            return simulated_energy_pj
        else:
            print("[ERROR] Failed to extract energy from Timeloop results.")
            return -1.0
            
    except Exception as e:
        print(f"[ERROR] Timeloop simulation failed: {e}")
        print(f"[DEBUG] Workspace preserved in: {work_dir.resolve()}")
        return -1.0


def main():
    """
    Main function for quick energy model calibration.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Quick Energy Model Calibration - Fast comparison of DOSA vs Timeloop energy predictions"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the JSON configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load configuration file: {e}")
        return 1
    
    # Run energy predictions
    try:
        predicted_energy_pj = run_dosa_energy_model(config)
        simulated_energy_pj = run_timeloop_energy_simulation(config)
        
        # Calculate errors
        if simulated_energy_pj > 0:
            absolute_error = predicted_energy_pj - simulated_energy_pj
            percentage_error = (absolute_error / simulated_energy_pj) * 100
        else:
            absolute_error = float('nan')
            percentage_error = float('nan')
        
        # Print calibration report
        print("\n" + "=" * 60)
        print("        ⚡️ Quick Energy Model Calibration Report ⚡️")
        print("=" * 60)
        print(f"Configuration File:  {args.config}")
        print("\n--- Comparison ---")
        print(f"[DOSA Model] Predicted Energy : {predicted_energy_pj:.2f} pJ")
        print(f"[Timeloop]   Simulated Energy : {simulated_energy_pj:.2f} pJ")
        print("\n--- Analysis ---")
        if not (abs(absolute_error) == float('inf') or abs(percentage_error) == float('inf')):
            print(f"Absolute Error : {absolute_error:.2f} pJ")
            print(f"Percentage Error : {percentage_error:.2f} %")
        else:
            print("Error calculation failed (Timeloop simulation error)")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] Calibration failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())