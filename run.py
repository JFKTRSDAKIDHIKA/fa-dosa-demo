import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import json
import onnx
from typing import Dict, Tuple, List, Any
from functools import reduce
from operator import mul

from dosa.config import Config
from dosa.hardware_parameters import HardwareParameters
from dosa.mapping import FineGrainedMapping, ProjectToNearestDivisor
from dosa.performance_model import HighFidelityPerformanceModel, TENSOR_DIM_MAP
from dosa.utils import (
    ComputationGraph, FusionParameters, 
    calculate_macs, save_configuration_to_json, get_divisors,
    OptimizationLogger
)

def parse_onnx_to_graph(model_name: str) -> ComputationGraph:
    """
    Enhanced ONNX parser that dynamically loads and parses ONNX models.
    
    Args:
        model_name: Name of the model (e.g., 'resnet18', 'bert_base')
    
    Returns:
        ComputationGraph: Parsed computation graph with layers and fusion groups
    """
    # Construct ONNX file path
    onnx_path = f"onnx_models/{model_name}.onnx"
    
    try:
        # Load ONNX model
        if not os.path.exists(onnx_path):
            print(f"Warning: ONNX file {onnx_path} not found. Using fallback graph.")
            return _create_fallback_graph()
        
        model = onnx.load(onnx_path)
        graph = ComputationGraph()
        
        # Extract tensor shape information from model inputs/outputs
        tensor_shapes = _extract_tensor_shapes(model)
        
        # Parse ONNX nodes and convert to layers
        layer_sequence = []
        for i, node in enumerate(model.graph.node):
            layer_name = f"{node.op_type.lower()}_{i}"
            layer_dims = _convert_onnx_node_to_dims(node, tensor_shapes)
            
            if layer_dims:  # Only add if we could extract valid dimensions
                graph.add_layer(layer_name, layer_dims, node.op_type)
                layer_sequence.append((layer_name, node.op_type))
        
        # Identify and add fusion groups based on common patterns
        _add_fusion_groups(graph, layer_sequence)
        
        print(f"Successfully parsed {model_name}: {len(graph.layers)} layers, {len(graph.fusion_groups)} fusion groups")
        return graph
        
    except Exception as e:
        print(f"Error parsing ONNX model {model_name}: {e}")
        print("Using fallback graph.")
        return _create_fallback_graph()

def _extract_tensor_shapes(model) -> Dict[str, List[int]]:
    """Extract tensor shapes from ONNX model."""
    tensor_shapes = {}
    
    # Extract from model inputs
    for input_info in model.graph.input:
        name = input_info.name
        if input_info.type.tensor_type.shape.dim:
            shape = []
            for dim in input_info.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                else:
                    shape.append(1)  # Default for dynamic dimensions
            tensor_shapes[name] = shape
    
    # Extract from value_info (intermediate tensors)
    for value_info in model.graph.value_info:
        name = value_info.name
        if value_info.type.tensor_type.shape.dim:
            shape = []
            for dim in value_info.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                else:
                    shape.append(1)
            tensor_shapes[name] = shape
    
    return tensor_shapes

def _convert_onnx_node_to_dims(node, tensor_shapes: Dict[str, List[int]]) -> Dict[str, int]:
    """Convert ONNX node to DOSA layer dimensions."""
    op_type = node.op_type
    
    # Get input tensor shape if available
    input_shape = None
    if node.input and node.input[0] in tensor_shapes:
        input_shape = tensor_shapes[node.input[0]]
    
    # Default dimensions
    dims = {'N': 1, 'C': 1, 'K': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1}
    
    if op_type in ['Conv', 'ConvTranspose']:
        # Extract convolution parameters
        if input_shape and len(input_shape) >= 4:
            dims['N'] = input_shape[0] if input_shape[0] > 0 else 1
            dims['C'] = input_shape[1] if input_shape[1] > 0 else 64
            dims['P'] = input_shape[2] if input_shape[2] > 0 else 56
            dims['Q'] = input_shape[3] if input_shape[3] > 0 else 56
        
        # Extract kernel size and output channels from attributes
        for attr in node.attribute:
            if attr.name == 'kernel_shape' and len(attr.ints) >= 2:
                dims['R'] = attr.ints[0]
                dims['S'] = attr.ints[1]
        
        # Try to infer output channels from weight tensor
        if len(node.input) > 1 and node.input[1] in tensor_shapes:
            weight_shape = tensor_shapes[node.input[1]]
            if len(weight_shape) >= 1:
                dims['K'] = weight_shape[0]  # Output channels
        else:
            dims['K'] = dims['C']  # Default assumption
    
    elif op_type in ['Relu', 'ReLU', 'Sigmoid', 'Tanh', 'BatchNormalization']:
        # Activation and normalization layers preserve input dimensions
        if input_shape and len(input_shape) >= 4:
            dims['N'] = input_shape[0] if input_shape[0] > 0 else 1
            dims['C'] = input_shape[1] if input_shape[1] > 0 else 64
            dims['P'] = input_shape[2] if input_shape[2] > 0 else 56
            dims['Q'] = input_shape[3] if input_shape[3] > 0 else 56
            dims['K'] = dims['C']
    
    elif op_type in ['MatMul', 'Gemm']:
        # Matrix multiplication operations
        if input_shape and len(input_shape) >= 2:
            dims['N'] = input_shape[0] if input_shape[0] > 0 else 1
            dims['C'] = input_shape[-1] if input_shape[-1] > 0 else 512
        
        # Try to infer output size from weight tensor
        if len(node.input) > 1 and node.input[1] in tensor_shapes:
            weight_shape = tensor_shapes[node.input[1]]
            if len(weight_shape) >= 2:
                dims['K'] = weight_shape[-1]  # Output features
        else:
            dims['K'] = dims['C']
        
        # For linear layers, set spatial dimensions to 1
        dims['P'] = dims['Q'] = dims['R'] = dims['S'] = 1
    
    elif op_type in ['Add', 'Mul', 'Sub', 'Div']:
        # Element-wise operations preserve dimensions
        if input_shape:
            if len(input_shape) >= 4:
                dims['N'] = input_shape[0] if input_shape[0] > 0 else 1
                dims['C'] = input_shape[1] if input_shape[1] > 0 else 64
                dims['P'] = input_shape[2] if input_shape[2] > 0 else 56
                dims['Q'] = input_shape[3] if input_shape[3] > 0 else 56
                dims['K'] = dims['C']
    
    return dims

def _add_fusion_groups(graph: ComputationGraph, layer_sequence: List[Tuple[str, str]]):
    """Identify and add fusion groups based on common patterns."""
    # Add individual layers as single-layer fusion groups
    for layer_name, _ in layer_sequence:
        graph.add_fusion_group([layer_name])
    
    # Identify common fusion patterns
    i = 0
    while i < len(layer_sequence) - 1:
        current_layer, current_op = layer_sequence[i]
        next_layer, next_op = layer_sequence[i + 1]
        
        # Conv -> ReLU fusion
        if current_op == 'Conv' and next_op in ['Relu', 'ReLU']:
            graph.add_fusion_group([current_layer, next_layer])
            i += 2
        # Conv -> BatchNormalization -> ReLU fusion
        elif (current_op == 'Conv' and next_op == 'BatchNormalization' and 
              i + 2 < len(layer_sequence) and layer_sequence[i + 2][1] in ['Relu', 'ReLU']):
            third_layer = layer_sequence[i + 2][0]
            graph.add_fusion_group([current_layer, next_layer, third_layer])
            i += 3
        # MatMul -> Add fusion
        elif current_op in ['MatMul', 'Gemm'] and next_op == 'Add':
            graph.add_fusion_group([current_layer, next_layer])
            i += 2
        # MatMul -> Add -> Activation fusion
        elif (current_op in ['MatMul', 'Gemm'] and next_op == 'Add' and 
              i + 2 < len(layer_sequence) and layer_sequence[i + 2][1] in ['Relu', 'ReLU', 'Gelu']):
            third_layer = layer_sequence[i + 2][0]
            graph.add_fusion_group([current_layer, next_layer, third_layer])
            i += 3
        else:
            i += 1

def _create_fallback_graph() -> ComputationGraph:
    """Create a fallback graph when ONNX parsing fails."""
    graph = ComputationGraph()
    
    # Create a simple 2-layer Conv+ReLU network
    for i in range(2):
        dims_conv = {
            'N': 1, 
            'C': 64 * (2 ** (i // 2)), 
            'K': 64 * (2 ** (i // 2)), 
            'P': 56 // (2 ** (i // 2)), 
            'Q': 56 // (2 ** (i // 2)), 
            'R': 3, 
            'S': 3
        }
        dims_relu = dims_conv.copy()
        
        graph.add_layer(f'conv_{i}', dims_conv, 'Conv')
        graph.add_layer(f'relu_{i}', dims_relu, 'ReLU')
        
        # Add fusion groups
        graph.add_fusion_group([f'conv_{i}', f'relu_{i}'])
        graph.add_fusion_group([f'conv_{i}'])
        graph.add_fusion_group([f'relu_{i}'])
    
    return graph


# --- 主实验流程 ---

def run_experiment(model_name: str = "resnet18", num_outer_steps=10, num_mapping_steps=200, num_hardware_steps=50, lr_mapping=1e-2, lr_hardware=1e-2):
    # Initialize the optimization logger
    logger = OptimizationLogger("optimization_log.jsonl")
    print(f"--- Running High-Fidelity FA-DOSA Experiment for {model_name} ---")
    config = Config()
    graph = parse_onnx_to_graph(model_name)
    
    # Loss function strategy selection
    loss_strategy = 'strategy_A'  # Options: 'strategy_A', 'strategy_B', or 'original'
    
    # Unified loss weight configuration
    loss_weights = {
        'area_weight': 0.01,           # Weight for area penalty
        'edp_weight': 10.0,           # For Strategy B: Amplified EDP Benefit
        'pe_penalty_weight_phase_a': 0.01,  # PE penalty weight for mapping phase
        'pe_penalty_weight_phase_b': 0.01,  # PE penalty weight for hardware phase
        'mismatch_penalty_weight': config.MISMATCH_PENALTY_WEIGHT  # From config
    }

    hw_params = HardwareParameters()
    # 使用一个共享的mapping对象
    mapping = FineGrainedMapping(graph.problem_dims, config.MEMORY_HIERARCHY)
    fusion_params = FusionParameters(graph)
    perf_model = HighFidelityPerformanceModel(config)
    
    # Alternating optimization scheme
    for outer_step in range(num_outer_steps):
        print(f"\n--- Outer Step {outer_step + 1}/{num_outer_steps} ---")
        
        # Phase A: Optimize Mapping & Fusion (freeze hardware)
        print("--- Phase A: Optimizing Mapping & Fusion ---")
        
        # Freeze hardware parameters
        for p in hw_params.parameters():
            p.requires_grad = False
        # Unfreeze mapping and fusion parameters
        for p in list(mapping.parameters()) + list(fusion_params.parameters()):
            p.requires_grad = True
            
        # Create optimizer for mapping and fusion parameters
        map_fus_params = list(mapping.parameters()) + list(fusion_params.parameters())
        optimizer_map = optim.Adam(map_fus_params, lr=lr_mapping)
        
        for i in range(num_mapping_steps):
            optimizer_map.zero_grad()
            latency, energy, area, mismatch_loss = perf_model(graph, hw_params, mapping)
            
            # Calculate loss components
            continuous_pes = hw_params.get_num_pes()
            sqrt_pes = torch.sqrt(continuous_pes)
            pe_square_penalty = torch.pow(sqrt_pes - torch.round(sqrt_pes), 2)

            # Strategy-based loss calculation
            if loss_strategy == 'strategy_A':
                # Strategy A: Log-space Penalty ("Soft Wall")
                edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
                area_loss = loss_weights['area_weight'] * area
                mismatch_penalty = torch.log(1.0 + mismatch_loss * loss_weights['mismatch_penalty_weight'])
                loss = edp_loss + area_loss + mismatch_penalty + pe_square_penalty * loss_weights['pe_penalty_weight_phase_a']
            elif loss_strategy == 'strategy_B':
                # Strategy B: Amplified EDP Benefit
                edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
                area_loss = loss_weights['area_weight'] * area
                loss = loss_weights['edp_weight'] * edp_loss + area_loss + mismatch_loss * loss_weights['mismatch_penalty_weight'] + pe_square_penalty * loss_weights['pe_penalty_weight_phase_a']
            else:  # Fallback to original
                edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
                area_loss = loss_weights['area_weight'] * area
                loss = edp_loss + area_loss + mismatch_loss * loss_weights['mismatch_penalty_weight'] + pe_square_penalty * loss_weights['pe_penalty_weight_phase_a']
            loss.backward()
            optimizer_map.step()
            # Anneal temperature for Gumbel-Softmax after each step
            mapping.anneal_tau()
            
            if i % 10 == 0:
                if loss_strategy == 'strategy_A':
                    print(f"[Map] Iter {i}: Loss={loss.item():.4f}, EDP={edp_loss.item():.4f}, Area={area_loss.item():.4f}, Mismatch_Penalty={mismatch_penalty.item():.4f}, PE_Penalty={pe_square_penalty.item():.4f}")
                else:
                    print(f"[Map] Iter {i}: Loss={loss.item():.4f}, EDP={edp_loss.item():.4f}, Area={area_loss.item():.4f}, Mismatch={mismatch_loss.item():.4f}, PE_Penalty={pe_square_penalty.item():.4f}")
                print(f"         Latency={latency.item():.2e}s, Energy={energy.item():.2e}pJ, Area={area.item():.2f}mm²")
                
                # Log optimization data
                log_data = {
                    'phase': 'A: Mapping',
                    'outer_step': outer_step,
                    'inner_step': i,
                    'loss_total': loss,
                    'loss_components': {
                        'edp': edp_loss,
                        'area': area_loss,
                        'mismatch': mismatch_loss if loss_strategy != 'strategy_A' else mismatch_penalty,
                        'pe_penalty': pe_square_penalty
                    },
                    'performance_metrics': {
                        'latency_sec': latency,
                        'energy_pj': energy,
                        'area_mm2': area
                    },
                    'hardware_params': {
                        'num_pes': hw_params.get_num_pes(),
                        'projected_num_pes': hw_params.get_projected_num_pes(),
                        'l0_size_kb': hw_params.get_buffer_size_kb('L0_Registers'),
                        'l1_size_kb': hw_params.get_buffer_size_kb('L1_Accumulator'),
                        'l2_size_kb': hw_params.get_buffer_size_kb('L2_Scratchpad')
                    },
                    'mapping_params_snapshot': mapping.get_all_factors(),
                    'gumbel_tau': mapping.projector.tau,
                    'fusion_decisions': fusion_params.get_fusion_decisions_serializable(graph)
                }
                logger.log_step(log_data)
        
        # Phase B: Optimize Hardware (freeze mapping and fusion)
        print("--- Phase B: Optimizing Hardware ---")
        
        # Freeze mapping and fusion parameters
        for p in list(mapping.parameters()) + list(fusion_params.parameters()):
            p.requires_grad = False
        # Unfreeze hardware parameters
        for p in hw_params.parameters():
            p.requires_grad = True
            
        # Create optimizer for hardware parameters
        optimizer_hw = optim.Adam(hw_params.parameters(), lr=lr_hardware)
        
        for i in range(num_hardware_steps):
            optimizer_hw.zero_grad()
            latency, energy, area, mismatch_loss = perf_model(graph, hw_params, mapping)
            
            # Calculate loss components
            continuous_pes = hw_params.get_num_pes()
            sqrt_pes = torch.sqrt(continuous_pes)
            pe_square_penalty = torch.pow(sqrt_pes - torch.round(sqrt_pes), 2)

            # Strategy-based loss calculation (Phase B: consistent area weight)
            if loss_strategy == 'strategy_A':
                # Strategy A: Log-space Penalty ("Soft Wall")
                edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
                area_loss = loss_weights['area_weight'] * area
                mismatch_penalty = torch.log(1.0 + mismatch_loss * loss_weights['mismatch_penalty_weight'])
                loss = edp_loss + area_loss + mismatch_penalty + pe_square_penalty * loss_weights['pe_penalty_weight_phase_b']
            elif loss_strategy == 'strategy_B':
                # Strategy B: Amplified EDP Benefit
                edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
                area_loss = loss_weights['area_weight'] * area
                loss = loss_weights['edp_weight'] * edp_loss + area_loss + mismatch_loss * loss_weights['mismatch_penalty_weight'] + pe_square_penalty * loss_weights['pe_penalty_weight_phase_b']
            else:  # Fallback to original
                edp_loss = torch.log(latency + 1e-9) + torch.log(energy + 1e-9)
                area_loss = loss_weights['area_weight'] * area
                loss = edp_loss + area_loss + mismatch_loss * loss_weights['mismatch_penalty_weight'] + pe_square_penalty * loss_weights['pe_penalty_weight_phase_b']
            loss.backward()
            optimizer_hw.step()
            
            if i % 10 == 0:
                if loss_strategy == 'strategy_A':
                    print(f"[HW] Iter {i}: Loss={loss.item():.4f}, EDP={edp_loss.item():.4f}, Area={area_loss.item():.4f}, Mismatch_Penalty={mismatch_penalty.item():.4f}, PE_Penalty={pe_square_penalty.item():.4f}")
                else:
                    print(f"[HW] Iter {i}: Loss={loss.item():.4f}, EDP={edp_loss.item():.4f}, Area={area_loss.item():.4f}, Mismatch={mismatch_loss.item():.4f}, PE_Penalty={pe_square_penalty.item():.4f}")
                print(f"         Latency={latency.item():.2e}s, Energy={energy.item():.2e}pJ, Area={area.item():.2f}mm²")
                
                # Log optimization data
                log_data = {
                    'phase': 'B: Hardware',
                    'outer_step': outer_step,
                    'inner_step': i,
                    'loss_total': loss,
                    'loss_components': {
                        'edp': edp_loss,
                        'area': area_loss,
                        'mismatch': mismatch_loss if loss_strategy != 'strategy_A' else mismatch_penalty,
                        'pe_penalty': pe_square_penalty
                    },
                    'performance_metrics': {
                        'latency_sec': latency,
                        'energy_pj': energy,
                        'area_mm2': area
                    },
                    'hardware_params': {
                        'num_pes': hw_params.get_num_pes(),
                        'projected_num_pes': hw_params.get_projected_num_pes(),
                        'l0_size_kb': hw_params.get_buffer_size_kb('L0_Registers'),
                        'l1_size_kb': hw_params.get_buffer_size_kb('L1_Accumulator'),
                        'l2_size_kb': hw_params.get_buffer_size_kb('L2_Scratchpad')
                    },
                    'mapping_params_snapshot': mapping.get_all_factors(),
                    'gumbel_tau': mapping.projector.tau,
                    'fusion_decisions': fusion_params.get_fusion_decisions_serializable(graph)
                }
                logger.log_step(log_data)
    
    print("\n--- Final Configuration ---")
    print(f"PEs: {hw_params.get_projected_num_pes().item():.0f}")
    for level in config.MEMORY_HIERARCHY:
        if level['type'] == 'buffer':
            print(f"{level['name']} Size: {hw_params.get_buffer_size_kb(level['name']).item():.2f} KB")

    # Get the final projected mapping
    final_mapping = mapping.get_all_factors()

    # Convert tensors to floats for JSON serialization
    for dim_name, dim_factors in final_mapping.items():
        for level_name, level_factors in dim_factors.items():
            final_mapping[dim_name][level_name]['temporal'] = level_factors['temporal'].item()
            final_mapping[dim_name][level_name]['spatial'] = level_factors['spatial'].item()

    # Get final fusion decisions
    final_fusion_decisions = fusion_params.get_fusion_decisions_serializable(graph)
    
    # Save the final configuration to JSON
    save_configuration_to_json(hw_params, final_mapping, final_fusion_decisions, "final_configuration.json")
    
    # Close the logger
    logger.close()

if __name__ == "__main__":
    # Example: Run experiment with ResNet-18
    run_experiment(
        model_name="resnet18",
        num_outer_steps=5, 
        num_mapping_steps=50, 
        num_hardware_steps=50
    )
    
    # Additional examples for other models:
    # run_experiment(model_name="bert_base", num_outer_steps=3)
    # run_experiment(model_name="unet", num_outer_steps=3)