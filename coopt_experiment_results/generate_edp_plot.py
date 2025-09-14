#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from pathlib import Path
import glob

def find_result_files(directory, pattern="*.json"):
    """递归查找结果文件"""
    result_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json') and ('result' in file.lower() or 'summary' in file.lower() or 'meta' in file.lower() or 'best' in file.lower()):
                result_files.append(os.path.join(root, file))
    return result_files

def extract_edp_from_file(filepath):
    """从结果文件中提取EDP数据"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        edp = None
        
        # 特殊处理COOPT的run_meta.json文件
        if 'final_results' in data and isinstance(data['final_results'], dict):
            final_results = data['final_results']
            if 'best_edp' in final_results:
                edp = float(final_results['best_edp'])
            elif 'best_edp_metrics' in final_results and isinstance(final_results['best_edp_metrics'], dict):
                if 'edp' in final_results['best_edp_metrics']:
                    edp = float(final_results['best_edp_metrics']['edp'])
        
        # 直接查找EDP
        if edp is None and 'edp' in data:
            edp = float(data['edp'])
        
        return edp
    except Exception as e:
        print(f"解析文件 {filepath} 时出错: {e}")
        return None

def main():
    results_dir = Path(__file__).parent
    baseline_dir = results_dir / "baseline_results"
    coopt_dir = results_dir / "coopt_results"
    
    # 收集baseline结果
    baseline_results = {}
    if baseline_dir.exists():
        for baseline in ['baselineA_A1', 'baselineA_A2', 'baselineB']:
            baseline_path = baseline_dir / baseline
            if baseline_path.exists():
                files = find_result_files(str(baseline_path))
                edps = []
                for file in files:
                    edp = extract_edp_from_file(file)
                    if edp is not None:
                        edps.append(edp)
                
                if edps:
                    baseline_results[baseline] = {
                        'edp_mean': np.mean(edps),
                        'edp_std': np.std(edps) if len(edps) > 1 else 0,
                        'edp_values': edps
                    }
    
    # 收集核心优化结果
    coopt_results = []
    if coopt_dir.exists():
        files = find_result_files(str(coopt_dir))
        for file in files:
            edp = extract_edp_from_file(file)
            if edp is not None:
                coopt_results.append(edp)
    
    # 生成对比图
    plt.figure(figsize=(12, 8))
    
    # 准备数据
    methods = []
    edp_means = []
    edp_stds = []
    
    # 添加baseline结果
    for baseline, data in baseline_results.items():
        methods.append(baseline)
        edp_means.append(data['edp_mean'])
        edp_stds.append(data['edp_std'])
    
    # 添加核心优化结果
    if coopt_results:
        methods.append('Co-optimization')
        edp_means.append(np.mean(coopt_results))
        edp_stds.append(np.std(coopt_results) if len(coopt_results) > 1 else 0)
    
    if not methods:
        print("未找到有效的EDP数据")
        return
    
    # 绘制柱状图
    x = np.arange(len(methods))
    bars = plt.bar(x, edp_means, yerr=edp_stds, capsize=5, 
                   color=['lightcoral', 'lightblue', 'lightgreen', 'gold'][:len(methods)],
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    # 设置图表属性
    plt.xlabel('优化方法', fontsize=12, fontweight='bold')
    plt.ylabel('EDP (Energy-Delay Product)', fontsize=12, fontweight='bold')
    plt.title('硬件-软件协同优化 vs Baseline方法\nEDP性能对比', fontsize=14, fontweight='bold')
    plt.xticks(x, methods, rotation=45, ha='right')
    
    # 添加数值标签
    for i, (bar, mean, std) in enumerate(zip(bars, edp_means, edp_stds)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std + height*0.01,
                f'{mean:.2e}', ha='center', va='bottom', fontsize=10)
    
    # 添加网格
    plt.grid(True, alpha=0.3, axis='y')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plot_path = results_dir / "edp_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"EDP对比图已保存到: {plot_path}")
    
    # 生成实验摘要
    summary = {
        'baseline_results': baseline_results,
        'coopt_results': {
            'edp_mean': np.mean(coopt_results) if coopt_results else None,
            'edp_std': np.std(coopt_results) if len(coopt_results) > 1 else 0,
            'edp_values': coopt_results
        }
    }
    
    summary_path = results_dir / "experiment_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"实验摘要已保存到: {summary_path}")
    
    # 打印结果摘要
    print("\n=== 实验结果摘要 ===")
    for method, mean in zip(methods, edp_means):
        print(f"{method}: EDP = {mean:.2e}")
    
    if len(methods) > 1 and coopt_results:
        baseline_best = min(edp_means[:-1]) if len(edp_means) > 1 else edp_means[0]
        coopt_edp = edp_means[-1]
        improvement = (baseline_best - coopt_edp) / baseline_best * 100
        print(f"\n协同优化相比最佳baseline改进: {improvement:.1f}%")

if __name__ == "__main__":
    main()
