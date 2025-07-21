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

def run_experiment(
    model_name: str = "resnet18", 
    searcher_type: str = "fa-dosa",
    num_trials: int = 500,
    **kwargs
) -> Dict[str, Any]:
    """
    重构后的实验启动器，支持多种搜索算法
    
    Args:
        model_name: 模型名称 (e.g., 'resnet18', 'bert_base')
        searcher_type: 搜索器类型 ('fa-dosa', 'random_search', 'bayesian_opt', 'genetic_algo')
        num_trials: 试验次数或评估次数
        **kwargs: 其他搜索器特定参数
        
    Returns:
        搜索结果字典
    """
    print(f"--- Running DSE Experiment: {searcher_type.upper()} on {model_name} ---")
    
    # 初始化核心组件
    config = Config()
    graph = parse_onnx_to_graph(model_name)
    hw_params = HardwareParameters()
    mapping = FineGrainedMapping(graph.problem_dims, config.MEMORY_HIERARCHY)
    fusion_params = FusionParameters(graph)
    perf_model = HighFidelityPerformanceModel(config)
    
    # 确保output目录存在
    import os
    os.makedirs('output', exist_ok=True)
    
    # 创建日志器
    log_filename = f"output/optimization_log_{searcher_type.replace('-', '_')}.jsonl"
    logger = OptimizationLogger(log_filename)
    
    # 根据searcher_type实例化对应的搜索器
    searcher = create_searcher(
        searcher_type, graph, hw_params, mapping, 
        fusion_params, perf_model, config, logger, **kwargs
    )
    
    # 执行搜索
    start_time = time.time()
    results = searcher.search(num_trials)
    end_time = time.time()
    
    # 输出结果
    print(f"\n--- Search Completed in {end_time - start_time:.2f}s ---")
    print(f"Best Loss: {results['best_loss']:.4f}")
    
    # 安全地访问best_metrics
    best_metrics = results.get('best_metrics', {})
    if best_metrics and 'edp' in best_metrics:
        print(f"Best EDP: {best_metrics['edp']:.2e}")
        print(f"Best Area: {best_metrics['area_mm2']:.2f}mm²")
    else:
        print("No valid solutions found.")
    
    print(f"Total Trials: {results['total_trials']}")
    
    # 保存最终配置（仅当找到有效解时）
    if results['best_params'] is not None:
        final_config_filename = f"output/final_configuration_{searcher_type.replace('-', '_')}.json"
        
        # 从最佳参数中提取映射和融合决策
        best_mapping = results['best_params'].get('mapping', {})
        best_fusion_decisions = results['best_params'].get('fusion_decisions', [])
        
        # 设置硬件参数到最佳配置
        if 'num_pes' in results['best_params']:
            hw_params.log_num_pes.data = torch.log(torch.tensor(float(results['best_params']['num_pes'])))
        for level in ['l0_registers', 'l1_accumulator', 'l2_scratchpad']:
            key = f'{level}_size_kb'
            if key in results['best_params']:
                level_name = level.replace('_', ' ').title().replace(' ', '_')
                if level == 'l0_registers':
                    level_name = 'L0_Registers'
                elif level == 'l1_accumulator':
                    level_name = 'L1_Accumulator'
                elif level == 'l2_scratchpad':
                    level_name = 'L2_Scratchpad'
                if level_name in hw_params.log_buffer_sizes_kb:
                    hw_params.log_buffer_sizes_kb[level_name].data = torch.log(torch.tensor(float(results['best_params'][key])))
        
        save_configuration_to_json(
            hw_params, best_mapping, best_fusion_decisions, final_config_filename
        )
        print(f"Configuration saved to {final_config_filename}")
    else:
        print("No valid configuration to save.")
    
    # 关闭日志器
    logger.close()
    
    return results


def create_searcher(
    searcher_type: str, graph, hw_params, mapping, fusion_params, 
    perf_model, config, logger, **kwargs
):
    """
    工厂函数：根据类型创建对应的搜索器
    
    Args:
        searcher_type: 搜索器类型
        其他参数: 搜索器初始化所需的组件
        **kwargs: 搜索器特定参数
        
    Returns:
        对应的搜索器实例
    """
    from dosa.searcher import (
        FADOSASearcher, RandomSearcher, 
        BayesianOptimizationSearcher, GeneticAlgorithmSearcher
    )
    
    # 为FA-DOSA设置默认参数
    if searcher_type == 'fa-dosa':
        config.NUM_OUTER_STEPS = kwargs.get('num_outer_steps', 5)
        config.NUM_MAPPING_STEPS = kwargs.get('num_mapping_steps', 50)
        config.NUM_HARDWARE_STEPS = kwargs.get('num_hardware_steps', 50)
        config.LR_MAPPING = kwargs.get('lr_mapping', 0.01)
        config.LR_HARDWARE = kwargs.get('lr_hardware', 0.01)
        
        return FADOSASearcher(
            graph, hw_params, mapping, fusion_params, perf_model, config, logger
        )
    
    elif searcher_type == 'random_search':
        return RandomSearcher(
            graph, hw_params, mapping, fusion_params, perf_model, config, logger
        )
    
    elif searcher_type == 'bayesian_opt':
        return BayesianOptimizationSearcher(
            graph, hw_params, mapping, fusion_params, perf_model, config, logger
        )
    
    elif searcher_type == 'genetic_algo':
        # 设置遗传算法参数
        config.GA_POPULATION_SIZE = kwargs.get('population_size', 50)
        config.GA_MUTATION_RATE = kwargs.get('mutation_rate', 0.1)
        config.GA_CROSSOVER_RATE = kwargs.get('crossover_rate', 0.8)
        
        return GeneticAlgorithmSearcher(
            graph, hw_params, mapping, fusion_params, perf_model, config, logger
        )
    
    else:
        raise ValueError(f"Unknown searcher type: {searcher_type}. "
                        f"Supported types: 'fa-dosa', 'random_search', 'bayesian_opt', 'genetic_algo'")


def run_comparison_experiment(
    model_name: str = "resnet18",
    searcher_types: List[str] = ["fa-dosa", "random_search"],
    num_trials: int = 500
) -> Dict[str, Dict[str, Any]]:
    """
    运行多个搜索器的对比实验
    
    Args:
        model_name: 模型名称
        searcher_types: 要对比的搜索器类型列表
        num_trials: 每个搜索器的试验次数
        
    Returns:
        所有搜索器的结果字典
    """
    print(f"\n=== Running Comparison Experiment on {model_name} ===")
    print(f"Searchers: {searcher_types}")
    print(f"Trials per searcher: {num_trials}\n")
    
    all_results = {}
    
    for searcher_type in searcher_types:
        print(f"\n{'='*60}")
        print(f"Running {searcher_type.upper()}")
        print(f"{'='*60}")
        
        try:
            results = run_experiment(
                model_name=model_name,
                searcher_type=searcher_type,
                num_trials=num_trials
            )
            all_results[searcher_type] = results
            
        except Exception as e:
            print(f"Error running {searcher_type}: {e}")
            all_results[searcher_type] = {'error': str(e)}
    
    # 输出对比总结
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for searcher_type, results in all_results.items():
        if 'error' not in results:
            best_metrics = results.get('best_metrics', {})
            if best_metrics and 'edp' in best_metrics:
                print(f"{searcher_type.upper():15s}: "
                      f"Best EDP = {best_metrics['edp']:.2e}, "
                      f"Best Loss = {results['best_loss']:.4f}")
            else:
                print(f"{searcher_type.upper():15s}: "
                      f"No valid solutions found, Best Loss = {results['best_loss']:.4f}")
        else:
            print(f"{searcher_type.upper():15s}: ERROR - {results['error']}")
    
    return all_results

if __name__ == "__main__":
    # 示例1: 运行单个搜索器实验
    print("=== Single Searcher Experiment ===")
    
    # FA-DOSA实验 - 轻量级测试配置
    fa_dosa_results = run_experiment(
        model_name="resnet18",
        searcher_type="fa-dosa",
        num_trials=50,  # 减少总评估次数以加快测试
        num_outer_steps=2,  # 减少外层步数
        num_mapping_steps=5,  # 减少映射优化步数
        num_hardware_steps=5  # 减少硬件优化步数
    )
    
    # 随机搜索实验 - 轻量级测试配置
    random_results = run_experiment(
        model_name="resnet18",
        searcher_type="random_search",
        num_trials=50  # 减少试验次数以加快测试
    )
    
    # 贝叶斯优化实验 - 轻量级测试配置
    bayesian_results = run_experiment(
        model_name="resnet18",
        searcher_type="bayesian_opt",
        num_trials=50  # 减少试验次数以加快测试
    )
    
    # 遗传算法实验 - 轻量级测试配置
    genetic_results = run_experiment(
        model_name="resnet18",
        searcher_type="genetic_algo",
        num_trials=50,  # 减少试验次数以加快测试
        population_size=20,  # 减少种群大小以加快测试
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    
    # 示例2: 运行对比实验 - 四种算法完整对比
    # print("\n=== Comprehensive Comparison Experiment ===")
    # comparison_results = run_comparison_experiment(
    #     model_name="resnet18",
    #     searcher_types=["fa-dosa", "random_search", "bayesian_opt", "genetic_algo"],
    #     num_trials=50  # 统一试验次数以确保公平对比
    # )
    
    # 示例3: 其他模型的实验（注释掉，可根据需要启用）
    # run_experiment(model_name="bert_base", searcher_type="fa-dosa", num_trials=200)
    # run_experiment(model_name="unet", searcher_type="random_search", num_trials=300)
    
    print("\n=== All Experiments Completed ===")