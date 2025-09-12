#!/usr/bin/env python3
"""
简单的帕累托前沿分析脚本
用于分析现有的trials.csv和best.json文件
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_pareto_results(results_dir):
    """分析帕累托前沿结果"""
    results_path = Path(results_dir)
    
    # 读取trials.csv文件
    trials_file = results_path / "trials.csv"
    best_file = results_path / "best.json"
    
    if not trials_file.exists():
        print(f"未找到trials.csv文件: {trials_file}")
        return
        
    if not best_file.exists():
        print(f"未找到best.json文件: {best_file}")
        return
    
    # 读取数据
    print("读取实验数据...")
    trials_df = pd.read_csv(trials_file)
    
    with open(best_file, 'r') as f:
        best_result = json.load(f)
    
    print(f"\n=== 帕累托前沿扫描结果分析 ===")
    print(f"总试验次数: {len(trials_df)}")
    print(f"\n最佳结果:")
    for key, value in best_result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # 分析trials数据
    print(f"\n=== 所有试验统计 ===")
    print(f"面积范围: {trials_df['area_mm2'].min():.2f} - {trials_df['area_mm2'].max():.2f} mm²")
    print(f"延迟范围: {trials_df['latency_sec'].min():.6f} - {trials_df['latency_sec'].max():.6f} 秒")
    print(f"能耗范围: {trials_df['energy_pj'].min():.0f} - {trials_df['energy_pj'].max():.0f} pJ")
    print(f"EDP范围: {trials_df['edp'].min():.0f} - {trials_df['edp'].max():.0f}")
    
    # 创建可视化
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 面积 vs EDP散点图
    ax1.scatter(trials_df['area_mm2'], trials_df['edp'], alpha=0.6, s=50)
    ax1.scatter(best_result['area_mm2'], best_result['edp'], color='red', s=100, label='最佳结果')
    ax1.set_xlabel('面积 (mm²)')
    ax1.set_ylabel('EDP')
    ax1.set_title('面积 vs EDP')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 延迟 vs 能耗散点图
    ax2.scatter(trials_df['latency_sec'], trials_df['energy_pj'], alpha=0.6, s=50)
    ax2.scatter(best_result['latency_sec'], best_result['energy_pj'], color='red', s=100, label='最佳结果')
    ax2.set_xlabel('延迟 (秒)')
    ax2.set_ylabel('能耗 (pJ)')
    ax2.set_title('延迟 vs 能耗')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # EDP分布直方图
    ax3.hist(trials_df['edp'], bins=20, alpha=0.7, edgecolor='black')
    ax3.axvline(best_result['edp'], color='red', linestyle='--', linewidth=2, label='最佳EDP')
    ax3.set_xlabel('EDP')
    ax3.set_ylabel('频次')
    ax3.set_title('EDP分布')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 面积分布直方图
    ax4.hist(trials_df['area_mm2'], bins=20, alpha=0.7, edgecolor='black')
    ax4.axvline(best_result['area_mm2'], color='red', linestyle='--', linewidth=2, label='最佳面积')
    ax4.set_xlabel('面积 (mm²)')
    ax4.set_ylabel('频次')
    ax4.set_title('面积分布')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_plot = results_path / "pareto_analysis.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"\n可视化图表已保存到: {output_plot}")
    
    # 不显示图表，只保存
    # plt.show()
    
    # 找到帕累托前沿点（简化版本）
    print(f"\n=== 帕累托前沿分析 ===")
    
    # 按面积排序，找到每个面积水平下的最小EDP
    sorted_trials = trials_df.sort_values('area_mm2')
    pareto_points = []
    min_edp = float('inf')
    
    for _, row in sorted_trials.iterrows():
        if row['edp'] < min_edp:
            min_edp = row['edp']
            pareto_points.append(row)
    
    print(f"帕累托前沿包含 {len(pareto_points)} 个点:")
    for i, point in enumerate(pareto_points):
        print(f"  点{i+1}: 面积={point['area_mm2']:.2f}mm², EDP={point['edp']:.0f}, 延迟={point['latency_sec']:.6f}s")
    
    return trials_df, best_result, pareto_points

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "output/pareto_example"
    
    analyze_pareto_results(results_dir)