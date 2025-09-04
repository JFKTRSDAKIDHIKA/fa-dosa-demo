import os
import time
import math
import numpy as np
import sys
from pathlib import Path
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
from dosa.structured_logger import StructuredLogger

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
            # 使用fallback graph，警告将通过logger输出
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
                # 提取输入和输出张量名称
                inputs = list(node.input) if node.input else []
                outputs = list(node.output) if node.output else []
                
                graph.add_layer(layer_name, layer_dims, node.op_type, inputs, outputs)
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
    """Identify and add fusion groups based on graph structure patterns."""
    # 首先识别ResNet残差块模式
    skip_patterns = graph.find_skip_connection_patterns()
    
    # 为每个识别出的残差块创建融合组
    for pattern in skip_patterns:
        main_path = pattern['main_path']
        add_node = pattern['add_node']
        
        # 创建包含主路径和Add操作的融合组
        fusion_group = main_path + [add_node]
        graph.add_fusion_group(fusion_group)
        print(f"Added ResNet skip connection fusion group: {fusion_group}")
    
    # 识别其他常见的融合模式
    processed_layers = set()
    
    # 收集已经在残差块中处理的层
    for pattern in skip_patterns:
        processed_layers.update(pattern['main_path'])
        processed_layers.add(pattern['add_node'])
    
    # 为未处理的层识别其他融合模式
    for layer_name, layer_type in layer_sequence:
        if layer_name in processed_layers:
            continue
            
        # 尝试识别以当前层开始的融合模式
        fusion_group = _identify_fusion_pattern_from_layer(graph, layer_name, processed_layers)
        
        if len(fusion_group) > 1:
            graph.add_fusion_group(fusion_group)
            processed_layers.update(fusion_group)
        else:
            # 单层融合组
            graph.add_fusion_group([layer_name])
            processed_layers.add(layer_name)

def _identify_fusion_pattern_from_layer(graph: ComputationGraph, start_layer: str, processed_layers: set) -> List[str]:
    """从指定层开始识别融合模式。"""
    fusion_group = [start_layer]
    current_layer = start_layer
    
    # 获取当前层的输出层
    output_layers = graph.get_layer_outputs(current_layer)
    
    while output_layers:
        # 如果有多个输出，停止融合
        if len(output_layers) > 1:
            break
            
        next_layer = output_layers[0]
        
        # 如果下一层已经被处理，停止融合
        if next_layer in processed_layers:
            break
            
        next_layer_info = graph.layers[next_layer]
        next_layer_type = next_layer_info['type']
        current_layer_type = graph.layers[current_layer]['type']
        
        # 检查是否可以融合
        can_fuse = False
        
        # Conv -> BatchNormalization
        if current_layer_type == 'Conv' and next_layer_type == 'BatchNormalization':
            can_fuse = True
        # BatchNormalization -> ReLU
        elif current_layer_type == 'BatchNormalization' and next_layer_type in ['Relu', 'ReLU']:
            can_fuse = True
        # Conv -> ReLU (直接连接)
        elif current_layer_type == 'Conv' and next_layer_type in ['Relu', 'ReLU']:
            can_fuse = True
        # MatMul/Gemm -> Add
        elif current_layer_type in ['MatMul', 'Gemm'] and next_layer_type == 'Add':
            can_fuse = True
        # Add -> Activation
        elif current_layer_type == 'Add' and next_layer_type in ['Relu', 'ReLU', 'Gelu']:
            can_fuse = True
            
        if can_fuse:
            fusion_group.append(next_layer)
            current_layer = next_layer
            output_layers = graph.get_layer_outputs(current_layer)
        else:
            break
    
    return fusion_group

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
    # 初始化核心组件
    config = Config()
    
    # 创建结构化日志器
    logger = StructuredLogger(
        log_dir=config.LOG_DIR,
        minimal_console=config.MINIMAL_CONSOLE,
        log_intermediate=config.LOG_INTERMEDIATE
    )
    
    # 记录实验开始
    logger.event("experiment_start", 
                model_name=model_name, 
                searcher_type=searcher_type, 
                num_trials=num_trials,
                **kwargs)
    logger.console(f"--- Running DSE Experiment: {searcher_type.upper()} on {model_name} ---")
    
    graph = parse_onnx_to_graph(model_name)
    hw_params = HardwareParameters()
    mapping = FineGrainedMapping(graph.problem_dims, config.MEMORY_HIERARCHY)
    fusion_params = FusionParameters(graph)
    perf_model = HighFidelityPerformanceModel(config)
    
    # 记录ONNX解析结果
    if not os.path.exists(f"onnx_models/{model_name}.onnx"):
        logger.event("onnx_fallback", model_name=model_name, 
                    message=f"ONNX file not found, using fallback graph")
    
    logger.event("components_initialized", 
                graph_layers=len(graph.layers),
                fusion_groups=len(graph.fusion_groups))
    
    # 根据searcher_type实例化对应的搜索器
    searcher = create_searcher(
        searcher_type, graph, hw_params, mapping, 
        fusion_params, perf_model, config, logger, **kwargs
    )
    
    # 执行搜索
    start_time = time.time()
    results = searcher.search(num_trials)
    end_time = time.time()
    
    # 记录搜索完成
    duration = end_time - start_time
    logger.event("search_completed", duration=duration)
    
    # 输出结果摘要
    best_metrics = results.get('best_metrics', {})
    if results['best_loss'] != float('inf') and best_metrics:
        logger.console(f"\n--- Search Completed in {duration:.2f}s ---")
        logger.console(f"Best Loss: {results['best_loss']:.4f}")
        logger.console(f"Best EDP: {best_metrics.get('edp', 0):.2e}")
        logger.console(f"Best Area: {best_metrics.get('area_mm2', 0):.2f}mm²")
    else:
        logger.console(f"\n--- Search Completed in {duration:.2f}s ---")
        logger.console("No valid solutions found.")
    
    logger.console(f"Total Trials: {results['total_trials']}")
    
    # 保存最终配置（仅当找到有效解时）
    if results['best_params'] is not None:
        final_config_filename = logger.get_run_dir() / f"final_configuration_{searcher_type.replace('-', '_')}.json"
        
        # 重构映射参数
        best_mapping = {}
        best_params = results['best_params']
        
        # 从扁平化参数重构mapping结构
        for key, value in best_params.items():
            if '_temporal' in key or '_spatial' in key:
                # 解析键名：例如 'N_L0_Registers_temporal'
                parts = key.split('_')
                if len(parts) >= 3:
                    dim_name = parts[0]
                    level_name = '_'.join(parts[1:-1])  # 处理多词level名称
                    factor_type = parts[-1]
                    
                    if dim_name not in best_mapping:
                        best_mapping[dim_name] = {}
                    if level_name not in best_mapping[dim_name]:
                        best_mapping[dim_name][level_name] = {}
                    
                    best_mapping[dim_name][level_name][factor_type] = value
        
        # 重构融合决策
        best_fusion_decisions = []
        if 'fusion_logits' in best_params:
            fusion_logits = best_params['fusion_logits']
            if isinstance(fusion_logits, list):
                for i, logit in enumerate(fusion_logits):
                    best_fusion_decisions.append({
                        "group": f"group_{i}",
                        "fused": logit > 0.0  # 简单的阈值判断
                    })
        
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
            hw_params, best_mapping, best_fusion_decisions, str(final_config_filename)
        )
        logger.artifact(str(final_config_filename), {
            "type": "final_configuration",
            "searcher_type": searcher_type,
            "model_name": model_name
        })
        logger.console(f"Configuration saved to {final_config_filename}")
    else:
        logger.console("No valid configuration to save.")
    
    # 完成日志记录
    logger.finalize(results)
    
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
    import argparse

    parser = argparse.ArgumentParser(description="FA-DOSA runner with optional Act I experiment")
    parser.add_argument("--exp", default="legacy", choices=["legacy", "act1"], help="Experiment to run")
    parser.add_argument("--config", default=str(Path("configs/act1.yaml").resolve()), help="Config file for act1 experiment")
    args, unknown = parser.parse_known_args()

    if args.exp == "act1":
        from experiments.act1 import run_act1  # lazy import
        run_act1(args.config)
        sys.exit(0)

    # ------------------------------------------------------------------
    # Legacy demo flow (original behaviour)
    # ------------------------------------------------------------------
    # 示例1: 运行单个搜索器实验
    print("\n" + "="*60)
    print("SINGLE SEARCHER EXPERIMENT")
    print("="*60)
    
    # FA-DOSA实验 - 轻量级测试配置
    fa_dosa_results = run_experiment(
        model_name="resnet18",
        searcher_type="fa-dosa",
        num_trials=100,  # 减少总评估次数以加快测试
        num_outer_steps=5,  # 减少外层步数
        num_mapping_steps=10,  # 减少映射优化步数
        num_hardware_steps=10  # 减少硬件优化步数
    )
    
    # # Random search experiment - lightweight test configuration
    # random_results = run_experiment(
    #     model_name="resnet18",
    #     searcher_type="random_search",
    #     num_trials=50  # Reduce trials to speed up testing
    # )
    
    # # Bayesian optimization experiment - lightweight test configuration
    # bayesian_results = run_experiment(
    #     model_name="resnet18",
    #     searcher_type="bayesian_opt",
    #     num_trials=50  # Reduce trials to speed up testing
    # )
    
    # # Genetic algorithm experiment - lightweight test configuration
    # genetic_results = run_experiment(
    #     model_name="resnet18",
    #     searcher_type="genetic_algo",
    #     num_trials=50,  # Reduce trials to speed up testing
    #     population_size=20,  # Reduce population size to speed up testing
    #     mutation_rate=0.1,
    #     crossover_rate=0.8
    # )
    
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
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*60)