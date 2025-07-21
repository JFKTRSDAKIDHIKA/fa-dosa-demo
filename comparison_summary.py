#!/usr/bin/env python3
"""
算法对比结果总结脚本

该脚本分析和可视化四种优化算法的性能对比结果：
- FA-DOSA (Fusion-Aware Design Space Optimization for Systolic Arrays)
- Random Search
- Bayesian Optimization  
- Genetic Algorithm
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_optimization_logs():
    """
    加载所有算法的优化日志
    
    Returns:
        dict: 包含所有算法结果的字典
    """
    results = {}
    
    # 定义日志文件映射
    log_files = {
        'FA-DOSA': 'output/optimization_log_fa_dosa.jsonl',
        'Random Search': 'output/optimization_log_random_search.jsonl', 
        'Bayesian Optimization': 'output/optimization_log_bayesian_opt.jsonl',
        'Genetic Algorithm': 'output/optimization_log_genetic_algo.jsonl'
    }
    
    for algo_name, log_file in log_files.items():
        if Path(log_file).exists():
            trials = []
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        trial = json.loads(line.strip())
                        trials.append(trial)
                    except json.JSONDecodeError:
                        continue
            results[algo_name] = trials
        else:
            print(f"Warning: {log_file} not found")
            results[algo_name] = []
    
    return results

def extract_convergence_data(results):
    """
    提取收敛数据用于绘图
    
    Args:
        results: 算法结果字典
        
    Returns:
        dict: 包含收敛数据的字典
    """
    convergence_data = {}
    
    for algo_name, trials in results.items():
        if not trials:
            convergence_data[algo_name] = {'trials': [], 'best_edp': [], 'best_loss': []}
            continue
            
        trial_nums = []
        best_edp_so_far = []
        best_loss_so_far = []
        
        current_best_edp = float('inf')
        current_best_loss = float('inf')
        
        for trial in trials:
            trial_num = trial.get('trial_number', 0)
            edp = trial.get('current_edp', float('inf'))
            loss = trial.get('loss_total', float('inf'))
            
            # 跳过无效值
            if edp == float('inf') or loss == float('inf') or np.isnan(edp) or np.isnan(loss):
                continue
                
            trial_nums.append(trial_num)
            
            # 更新最佳值
            if edp < current_best_edp:
                current_best_edp = edp
            if loss < current_best_loss:
                current_best_loss = loss
                
            best_edp_so_far.append(current_best_edp)
            best_loss_so_far.append(current_best_loss)
        
        convergence_data[algo_name] = {
            'trials': trial_nums,
            'best_edp': best_edp_so_far,
            'best_loss': best_loss_so_far
        }
    
    return convergence_data

def create_comparison_plots(convergence_data):
    """
    创建对比图表
    
    Args:
        convergence_data: 收敛数据字典
    """
    # 设置图表样式
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 颜色映射
    colors = {
        'FA-DOSA': '#1f77b4',
        'Random Search': '#ff7f0e', 
        'Bayesian Optimization': '#2ca02c',
        'Genetic Algorithm': '#d62728'
    }
    
    # 绘制EDP收敛图
    ax1.set_title('Energy-Delay Product (EDP) Convergence', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Best EDP (log scale)')
    ax1.set_yscale('log')
    
    for algo_name, data in convergence_data.items():
        if data['trials'] and data['best_edp']:
            ax1.plot(data['trials'], data['best_edp'], 
                    label=algo_name, color=colors.get(algo_name, 'gray'),
                    linewidth=2, marker='o', markersize=3)
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制损失收敛图
    ax2.set_title('Loss Function Convergence', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Trial Number')
    ax2.set_ylabel('Best Loss (log scale)')
    ax2.set_yscale('log')
    
    for algo_name, data in convergence_data.items():
        if data['trials'] and data['best_loss']:
            ax2.plot(data['trials'], data['best_loss'],
                    label=algo_name, color=colors.get(algo_name, 'gray'),
                    linewidth=2, marker='o', markersize=3)
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comparison plot saved as 'output/algorithm_comparison.png'")

def print_summary_table(convergence_data):
    """
    打印结果总结表格
    
    Args:
        convergence_data: 收敛数据字典
    """
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON SUMMARY")
    print("="*80)
    
    print(f"{'Algorithm':<20} {'Best EDP':<15} {'Best Loss':<15} {'Trials':<10} {'Status':<15}")
    print("-"*80)
    
    for algo_name, data in convergence_data.items():
        if data['trials'] and data['best_edp']:
            best_edp = min(data['best_edp'])
            best_loss = min(data['best_loss'])
            num_trials = len(data['trials'])
            status = "Success"
        else:
            best_edp = float('inf')
            best_loss = float('inf')
            num_trials = 0
            status = "No valid solutions"
        
        edp_str = f"{best_edp:.2e}" if best_edp != float('inf') else "N/A"
        loss_str = f"{best_loss:.2e}" if best_loss != float('inf') else "N/A"
        
        print(f"{algo_name:<20} {edp_str:<15} {loss_str:<15} {num_trials:<10} {status:<15}")
    
    print("="*80)
    
    # 找出最佳算法
    valid_results = {name: data for name, data in convergence_data.items() 
                    if data['trials'] and data['best_edp']}
    
    if valid_results:
        best_algo = min(valid_results.keys(), 
                       key=lambda x: min(valid_results[x]['best_edp']))
        best_edp = min(valid_results[best_algo]['best_edp'])
        
        print(f"\n🏆 WINNER: {best_algo} with EDP = {best_edp:.2e}")
        
        # 计算改进比例
        print("\n📊 PERFORMANCE COMPARISON (relative to best):")
        for algo_name, data in valid_results.items():
            if data['best_edp']:
                algo_best_edp = min(data['best_edp'])
                improvement = algo_best_edp / best_edp
                print(f"   {algo_name}: {improvement:.2f}x")
    else:
        print("\n❌ No algorithm found valid solutions.")

def main():
    """
    主函数
    """
    print("Loading optimization results...")
    results = load_optimization_logs()
    
    print("Extracting convergence data...")
    convergence_data = extract_convergence_data(results)
    
    print("Creating comparison plots...")
    create_comparison_plots(convergence_data)
    
    print_summary_table(convergence_data)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()