#!/usr/bin/env python3
"""
plot_results.py - 结果可视化脚本

该脚本用于读取不同搜索器生成的日志文件，并绘制EDP收敛曲线对比图，
复现类似DOSA论文中Figure 7的效果。
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from typing import Dict, List, Tuple


def load_jsonl_log(log_file: str) -> List[Dict]:
    """
    加载JSONL格式的日志文件
    
    Args:
        log_file: 日志文件路径
        
    Returns:
        日志条目列表
    """
    if not os.path.exists(log_file):
        print(f"Warning: Log file {log_file} not found.")
        return []
    
    log_entries = []
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line.strip())
                    log_entries.append(entry)
    except Exception as e:
        print(f"Error loading {log_file}: {e}")
        return []
    
    return log_entries


def extract_convergence_data(log_entries: List[Dict]) -> Tuple[List[int], List[float]]:
    """
    从日志条目中提取收敛数据
    
    Args:
        log_entries: 日志条目列表
        
    Returns:
        (trial_numbers, best_edp_so_far) 元组
    """
    trial_numbers = []
    best_edp_so_far = []
    current_best_edp = float('inf')
    
    for i, entry in enumerate(log_entries):
        # 计算当前EDP
        if 'performance_metrics' in entry:
            latency = float(entry['performance_metrics']['latency_sec'])
            energy = float(entry['performance_metrics']['energy_pj'])
            current_edp = latency * energy
        elif 'edp' in entry:
            current_edp = float(entry['edp'])
        else:
            # 尝试从loss_components中获取
            if 'loss_components' in entry and 'edp' in entry['loss_components']:
                # 这里的edp是log(latency) + log(energy)，需要转换
                log_edp = float(entry['loss_components']['edp'])
                current_edp = np.exp(log_edp)
            else:
                continue
        
        # 更新最佳EDP
        if current_edp < current_best_edp:
            current_best_edp = current_edp
        
        trial_numbers.append(i + 1)
        best_edp_so_far.append(current_best_edp)
    
    return trial_numbers, best_edp_so_far


def plot_convergence_comparison(
    log_files: Dict[str, str], 
    output_file: str = "convergence_comparison.png",
    title: str = "DSE Algorithm Convergence Comparison",
    figsize: Tuple[int, int] = (12, 8)
):
    """
    绘制多个算法的收敛曲线对比图
    
    Args:
        log_files: {算法名称: 日志文件路径} 字典
        output_file: 输出图片文件名
        title: 图表标题
        figsize: 图表尺寸
    """
    plt.figure(figsize=figsize)
    
    # 定义颜色和线型
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    algorithm_data = {}
    
    for i, (algorithm_name, log_file) in enumerate(log_files.items()):
        print(f"Processing {algorithm_name} from {log_file}...")
        
        # 加载日志数据
        log_entries = load_jsonl_log(log_file)
        if not log_entries:
            print(f"Skipping {algorithm_name} due to empty log.")
            continue
        
        # 提取收敛数据
        trial_numbers, best_edp_so_far = extract_convergence_data(log_entries)
        if not trial_numbers:
            print(f"Skipping {algorithm_name} due to no valid data.")
            continue
        
        algorithm_data[algorithm_name] = {
            'trials': trial_numbers,
            'edp': best_edp_so_far,
            'final_edp': best_edp_so_far[-1] if best_edp_so_far else float('inf')
        }
        
        # 绘制收敛曲线
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]
        
        # 使用对数坐标
        plt.semilogy(
            trial_numbers, best_edp_so_far,
            color=color, linestyle=linestyle, marker=marker,
            label=f"{algorithm_name.upper()} (Final: {best_edp_so_far[-1]:.2e})",
            linewidth=2, markersize=4, markevery=max(1, len(trial_numbers)//20)
        )
        
        print(f"  {algorithm_name}: {len(trial_numbers)} trials, "
              f"Final EDP: {best_edp_so_far[-1]:.2e}")
    
    # 设置图表属性
    plt.xlabel('Number of Evaluations', fontsize=14, fontweight='bold')
    plt.ylabel('Best EDP So Far (Energy × Delay)', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # 设置坐标轴
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nConvergence plot saved to: {output_file}")
    
    # 显示图片（如果在交互环境中）
    try:
        plt.show()
    except:
        pass
    
    return algorithm_data


def print_summary_table(algorithm_data: Dict[str, Dict]):
    """
    打印算法性能对比表格
    
    Args:
        algorithm_data: 算法数据字典
    """
    print("\n" + "="*80)
    print("ALGORITHM PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Algorithm':<20} {'Final EDP':<15} {'Trials':<10} {'Improvement':<15}")
    print("-"*80)
    
    # 找到最佳EDP作为基准
    best_edp = min(data['final_edp'] for data in algorithm_data.values())
    
    for algorithm_name, data in algorithm_data.items():
        final_edp = data['final_edp']
        num_trials = len(data['trials'])
        improvement = f"{(final_edp / best_edp - 1) * 100:+.1f}%" if final_edp != best_edp else "BEST"
        
        print(f"{algorithm_name.upper():<20} {final_edp:<15.2e} {num_trials:<10} {improvement:<15}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Plot convergence comparison for DSE algorithms"
    )
    parser.add_argument(
        '--logs', '-l', nargs='+', 
        default=[
            'optimization_log_fa_dosa.jsonl',
            'optimization_log_random_search.jsonl'
        ],
        help='List of log files to compare'
    )
    parser.add_argument(
        '--output', '-o', default='convergence_comparison.png',
        help='Output plot filename'
    )
    parser.add_argument(
        '--title', '-t', default='DSE Algorithm Convergence Comparison',
        help='Plot title'
    )
    parser.add_argument(
        '--model', '-m', default='resnet18',
        help='Model name for title'
    )
    
    args = parser.parse_args()
    
    # 构建日志文件字典
    log_files = {}
    for log_file in args.logs:
        if os.path.exists(log_file):
            # 从文件名推断算法名称
            algorithm_name = log_file.replace('optimization_log_', '').replace('.jsonl', '')
            algorithm_name = algorithm_name.replace('_', '-')
            log_files[algorithm_name] = log_file
        else:
            print(f"Warning: Log file {log_file} not found.")
    
    if not log_files:
        print("Error: No valid log files found.")
        return
    
    # 更新标题以包含模型名称
    full_title = f"{args.title} ({args.model.upper()})"
    
    # 绘制对比图
    algorithm_data = plot_convergence_comparison(
        log_files, args.output, full_title
    )
    
    # 打印性能总结
    if algorithm_data:
        print_summary_table(algorithm_data)
    
    print(f"\nVisualization completed. Check {args.output} for the plot.")


if __name__ == "__main__":
    main()