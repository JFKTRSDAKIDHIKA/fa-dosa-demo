#!/usr/bin/env python3
"""
面积预算功能演示脚本

展示如何使用DOSA框架的面积预算功能来区分edge和cloud场景，
通过Loss惩罚项限定硬件在特定面积区域内优化。

使用方法:
    python examples/area_budget_demo.py --scenario edge
    python examples/area_budget_demo.py --scenario cloud
    python examples/area_budget_demo.py --scenario mobile
"""

import sys
import os
import argparse

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dosa.config import Config
from dosa.searcher import FADOSASearcher
from dosa.utils import ComputationGraph, FusionParameters
from dosa.hardware_parameters import HardwareParameters
from dosa.mapping import FineGrainedMapping
from dosa.performance_model import HighFidelityPerformanceModel
from dosa.structured_logger import StructuredLogger

def create_demo_workload():
    """
    创建一个简单的演示工作负载
    """
    graph = ComputationGraph()
    
    # Create a simple 2-layer Conv+ReLU network (similar to run.py fallback)
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
    
    return graph

def run_area_budget_demo(scenario: str, num_trials: int = 100):
    """
    运行面积预算演示
    
    Args:
        scenario: 场景类型 ('edge', 'cloud', 'mobile')
        num_trials: 优化试验次数
    """
    print(f"\n=== 面积预算功能演示 - {scenario.upper()}场景 ===")
    
    # 初始化配置
    config = Config()
    
    # 应用场景预设配置
    print(f"应用{scenario}场景预设配置...")
    config.apply_scenario_preset(scenario)
    
    # 显示当前面积预算配置
    print(f"\n当前面积预算配置:")
    print(f"  - 启用状态: {config.ENABLE_AREA_BUDGET}")
    print(f"  - 面积预算: {config.AREA_BUDGET_MM2} mm²")
    print(f"  - 容忍区间: ±{config.AREA_BUDGET_TOLERANCE*100:.1f}%")
    print(f"  - 惩罚策略: {config.AREA_BUDGET_PENALTY_STRATEGY}")
    print(f"  - 惩罚权重: {config.AREA_BUDGET_PENALTY_WEIGHT}")
    
    # 显示损失权重配置
    print(f"\n损失权重配置:")
    for key, value in config.LOSS_WEIGHTS.items():
        print(f"  - {key}: {value}")
    
    # 创建演示工作负载
    graph = create_demo_workload()
    
    # 初始化硬件参数
    hw_params = HardwareParameters()
    
    # 初始化映射参数
    mapping = FineGrainedMapping(graph.problem_dims, config.MEMORY_HIERARCHY)
    
    # 初始化融合参数
    fusion_params = FusionParameters(graph)
    
    # 初始化性能模型
    perf_model = HighFidelityPerformanceModel(config)
    
    # 移动到设备并设置梯度
    device = config.DEVICE
    hw_params.to(device)
    mapping.to(device)
    fusion_params.to(device)
    
    # 初始化日志记录器
    logger = StructuredLogger(log_dir=f"output/area_budget_demo_{scenario}")
    
    # 创建搜索器
    searcher = FADOSASearcher(
        graph=graph,
        hw_params=hw_params,
        mapping=mapping,
        fusion_params=fusion_params,
        perf_model=perf_model,
        config=config,
        logger=logger
    )
    
    print(f"\n开始优化 ({num_trials} 次试验)...")
    
    # 执行搜索
    try:
        results = searcher.search(num_trials=num_trials)
        
        print(f"\n=== 优化结果 ===")
        print(f"最佳损失: {results['best_loss']:.6f}")
        print(f"最佳EDP: {results['best_metrics']['edp']:.2e}")
        print(f"最佳面积: {results['best_metrics']['area_mm2']:.2f} mm²")
        print(f"最佳延迟: {results['best_metrics']['latency_sec']:.2e} s")
        print(f"最佳能耗: {results['best_metrics']['energy_pj']:.2e} pJ")
        
        # 检查面积是否在预算范围内
        area = results['best_metrics']['area_mm2']
        budget = config.AREA_BUDGET_MM2
        tolerance = config.AREA_BUDGET_TOLERANCE
        lower_bound = budget * (1 - tolerance)
        upper_bound = budget * (1 + tolerance)
        
        print(f"\n=== 面积预算分析 ===")
        print(f"面积预算: {budget} mm²")
        print(f"容忍区间: [{lower_bound:.2f}, {upper_bound:.2f}] mm²")
        print(f"实际面积: {area:.2f} mm²")
        
        if lower_bound <= area <= upper_bound:
            print("✅ 面积在预算范围内")
        elif area < lower_bound:
            print(f"⚠️  面积低于预算下限 {(lower_bound - area):.2f} mm²")
        else:
            print(f"❌ 面积超出预算上限 {(area - upper_bound):.2f} mm²")
        
        # 显示损失组成分析
        if 'loss_breakdown' in results:
            breakdown = results['loss_breakdown']
            print(f"\n=== 损失组成分析 ===")
            for key, value in breakdown.items():
                if isinstance(value, (int, float)):
                    print(f"  - {key}: {value:.6f}")
                else:
                    print(f"  - {key}: {value}")
    
    except Exception as e:
        print(f"优化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n=== 演示完成 ===")

def compare_scenarios():
    """
    比较不同场景的配置差异
    """
    print("\n=== 场景配置比较 ===")
    
    scenarios = ['edge', 'cloud', 'mobile']
    configs = {}
    
    for scenario in scenarios:
        config = Config()
        config.apply_scenario_preset(scenario)
        configs[scenario] = {
            'area_budget': config.AREA_BUDGET_MM2,
            'tolerance': config.AREA_BUDGET_TOLERANCE,
            'penalty_weight': config.AREA_BUDGET_PENALTY_WEIGHT,
            'penalty_strategy': config.AREA_BUDGET_PENALTY_STRATEGY
        }
    
    print(f"{'场景':<10} {'面积预算(mm²)':<15} {'容忍度':<10} {'惩罚权重':<12} {'惩罚策略':<12}")
    print("-" * 65)
    
    for scenario, cfg in configs.items():
        print(f"{scenario:<10} {cfg['area_budget']:<15} {cfg['tolerance']:<10.1%} "
              f"{cfg['penalty_weight']:<12} {cfg['penalty_strategy']:<12}")

def main():
    parser = argparse.ArgumentParser(description='面积预算功能演示')
    parser.add_argument('--scenario', choices=['edge', 'cloud', 'mobile'], 
                       default='edge', help='选择场景类型')
    parser.add_argument('--trials', type=int, default=50, 
                       help='优化试验次数')
    parser.add_argument('--compare', action='store_true', 
                       help='比较不同场景的配置')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_scenarios()
    else:
        run_area_budget_demo(args.scenario, args.trials)

if __name__ == '__main__':
    main()