#!/bin/bash

# 硬件-软件协同优化必要性实验脚本
# 运行baseline和核心优化实验，并生成EDP对比图

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="/root/fa-dosa-demo"
EXPERIMENTS_DIR="$PROJECT_ROOT/experiments"
BASELINE_OUTPUT_DIR="$PROJECT_ROOT/experiments/output"
COOPT_OUTPUT_DIR="$PROJECT_ROOT/output"
RESULTS_DIR="$PROJECT_ROOT/coopt_experiment_results"

# 创建结果目录
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  硬件-软件协同优化必要性实验${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# 函数：打印带时间戳的日志
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# 函数：打印错误信息
error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# 函数：打印警告信息
warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 检查必要的文件是否存在
check_prerequisites() {
    log "检查实验环境..."
    
    if [ ! -f "$EXPERIMENTS_DIR/act1.py" ]; then
        error "未找到 act1.py 文件"
        exit 1
    fi
    
    if [ ! -f "$PROJECT_ROOT/run.py" ]; then
        error "未找到 run.py 文件"
        exit 1
    fi
    
    log "环境检查完成"
}

# 运行baseline实验
run_baseline_experiments() {
    log "开始运行baseline实验..."
    
    cd "$EXPERIMENTS_DIR"
    
    # 设置Python路径
    export PYTHONPATH="$PROJECT_ROOT"
    
    # 运行baseline实验（act1.py会自动运行所有配置的baseline）
    log "运行所有baseline实验（baselineA_A1, baselineA_A2, baselineB）..."
    
    if python act1.py --scenario edge; then
        log "所有baseline实验完成"
    else
        error "baseline实验失败"
        exit 1
    fi
    
    echo
}

# 运行核心优化实验
run_coopt_experiment() {
    log "开始运行核心优化实验..."
    
    cd "$PROJECT_ROOT"
    
    # 运行核心优化
    if python3 run.py --scenario edge; then
        log "核心优化实验完成"
    else
        error "核心优化实验失败"
        exit 1
    fi
    
    echo
}

# 收集实验结果
collect_results() {
    log "收集实验结果..."
    
    # 创建结果汇总目录
    mkdir -p "$RESULTS_DIR/baseline_results"
    mkdir -p "$RESULTS_DIR/coopt_results"
    
    # 复制baseline结果（从experiments/output/act1/最新时间戳目录）
    latest_baseline_dir=$(find "$BASELINE_OUTPUT_DIR/act1" -maxdepth 1 -type d -name "*" | sort | tail -1 2>/dev/null)
    if [ -n "$latest_baseline_dir" ] && [ -d "$latest_baseline_dir" ]; then
        log "复制baseline结果从: $latest_baseline_dir"
        cp -r "$latest_baseline_dir"/* "$RESULTS_DIR/baseline_results/" 2>/dev/null || warn "部分baseline结果复制失败"
    else
        warn "未找到baseline结果目录"
    fi
    
    # 复制核心优化结果（从output/最新run目录）
    latest_coopt_dir=$(find "$COOPT_OUTPUT_DIR" -maxdepth 1 -type d -name "run_*" | sort | tail -1 2>/dev/null)
    if [ -n "$latest_coopt_dir" ] && [ -d "$latest_coopt_dir" ]; then
        log "复制核心优化结果从: $latest_coopt_dir"
        cp -r "$latest_coopt_dir"/* "$RESULTS_DIR/coopt_results/" 2>/dev/null || warn "部分核心优化结果复制失败"
    else
        warn "未找到核心优化结果目录"
    fi
    
    log "结果收集完成"
}

# 生成EDP对比图
generate_edp_comparison() {
    log "生成EDP对比图..."
    
    cd "$PROJECT_ROOT"
    
    # 创建Python脚本来生成对比图
    cat > "$RESULTS_DIR/generate_edp_plot.py" << 'EOF'
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
            if file.endswith('.json') and ('result' in file.lower() or 'summary' in file.lower()):
                result_files.append(os.path.join(root, file))
    return result_files

def extract_edp_from_file(filepath):
    """从结果文件中提取EDP数据"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 尝试不同的键名来提取EDP数据
        edp_keys = ['edp', 'EDP', 'energy_delay_product', 'latency_energy_product']
        latency_keys = ['latency', 'delay', 'execution_time']
        energy_keys = ['energy', 'power', 'energy_consumption']
        
        edp = None
        latency = None
        energy = None
        
        # 直接查找EDP
        for key in edp_keys:
            if key in data:
                edp = float(data[key])
                break
        
        # 如果没有直接的EDP，尝试从latency和energy计算
        if edp is None:
            for lat_key in latency_keys:
                if lat_key in data:
                    latency = float(data[lat_key])
                    break
            
            for eng_key in energy_keys:
                if eng_key in data:
                    energy = float(data[eng_key])
                    break
            
            if latency is not None and energy is not None:
                edp = latency * energy
        
        return edp, latency, energy
    
    except Exception as e:
        print(f"解析文件 {filepath} 时出错: {e}")
        return None, None, None

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
                    edp, lat, eng = extract_edp_from_file(file)
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
            edp, lat, eng = extract_edp_from_file(file)
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
    
    # 保存图片
    output_path = results_dir / "edp_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"EDP对比图已保存到: {output_path}")
    
    # 保存数据摘要
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
    
    if len(methods) > 1:
        baseline_best = min(edp_means[:-1]) if len(edp_means) > 1 else edp_means[0]
        coopt_edp = edp_means[-1]
        improvement = (baseline_best - coopt_edp) / baseline_best * 100
        print(f"\n协同优化相比最佳baseline改进: {improvement:.1f}%")

if __name__ == "__main__":
    main()
EOF

    # 运行Python脚本生成图表
    if python3 "$RESULTS_DIR/generate_edp_plot.py"; then
        log "EDP对比图生成完成"
    else
        error "EDP对比图生成失败"
    fi
}

# 主函数
main() {
    log "开始硬件-软件协同优化必要性实验"
    
    # 检查环境
    check_prerequisites
    
    # 运行实验
    run_baseline_experiments
    run_coopt_experiment
    
    # 收集结果并生成图表
    collect_results
    generate_edp_comparison
    
    log "实验完成！结果保存在: $RESULTS_DIR"
    echo -e "${GREEN}主要输出文件:${NC}"
    echo -e "  - EDP对比图: ${BLUE}$RESULTS_DIR/edp_comparison.png${NC}"
    echo -e "  - 实验摘要: ${BLUE}$RESULTS_DIR/experiment_summary.json${NC}"
    echo -e "  - Baseline结果: ${BLUE}$RESULTS_DIR/baseline_results/${NC}"
    echo -e "  - 协同优化结果: ${BLUE}$RESULTS_DIR/coopt_results/${NC}"
}

# 脚本入口点
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi