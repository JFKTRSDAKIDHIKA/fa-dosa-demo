#!/bin/bash

# 测试硬件-软件协同优化实验脚本
# 这是一个简化版本，用于快速验证实验流程

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 项目根目录
PROJECT_ROOT="/root/fa-dosa-demo"
EXPERIMENTS_DIR="$PROJECT_ROOT/experiments"
RESULTS_DIR="$PROJECT_ROOT/test_experiment_results"

# 创建结果目录
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  测试硬件-软件协同优化实验${NC}"
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

# 测试环境检查
test_environment() {
    log "检查测试环境..."
    
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

# 测试baseline实验（快速版本）
test_baseline() {
    log "测试baseline实验..."
    
    cd "$EXPERIMENTS_DIR"
    export PYTHONPATH="$PROJECT_ROOT"
    
    # 检查act1.py是否能正常导入
    if python -c "import act1; print('act1.py导入成功')"; then
        log "act1.py模块导入测试通过"
    else
        error "act1.py模块导入失败"
        return 1
    fi
    
    # 检查配置文件
    if [ -f "$PROJECT_ROOT/configs/act1.yaml" ]; then
        log "找到配置文件: act1.yaml"
    else
        error "未找到配置文件: act1.yaml"
        return 1
    fi
    
    log "baseline测试完成"
}

# 测试核心优化
test_coopt() {
    log "测试核心优化..."
    
    cd "$PROJECT_ROOT"
    
    # 检查run.py是否能正常导入
    if python3 -c "import run; print('run.py导入成功')"; then
        log "run.py模块导入测试通过"
    else
        error "run.py模块导入失败"
        return 1
    fi
    
    log "核心优化测试完成"
}

# 创建示例EDP数据和图表
create_sample_plot() {
    log "创建示例EDP对比图..."
    
    cat > "$RESULTS_DIR/create_sample_plot.py" << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import json

def create_sample_edp_plot():
    """创建示例EDP对比图"""
    
    # 示例数据（模拟实验结果）
    methods = ['baselineA_A1', 'baselineA_A2', 'baselineB', 'Co-optimization']
    edp_values = [1.2e6, 1.1e6, 0.95e6, 0.75e6]  # 示例EDP值
    edp_stds = [0.1e6, 0.08e6, 0.07e6, 0.05e6]   # 示例标准差
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(methods))
    colors = ['lightcoral', 'lightblue', 'lightgreen', 'gold']
    
    bars = plt.bar(x, edp_values, yerr=edp_stds, capsize=5, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # 设置图表属性
    plt.xlabel('优化方法', fontsize=12, fontweight='bold')
    plt.ylabel('EDP (Energy-Delay Product)', fontsize=12, fontweight='bold')
    plt.title('硬件-软件协同优化 vs Baseline方法\nEDP性能对比（示例数据）', fontsize=14, fontweight='bold')
    plt.xticks(x, methods, rotation=45, ha='right')
    
    # 添加数值标签
    for i, (bar, value, std) in enumerate(zip(bars, edp_values, edp_stds)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std + height*0.01,
                f'{value:.2e}', ha='center', va='bottom', fontsize=10)
    
    # 添加网格
    plt.grid(True, alpha=0.3, axis='y')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    results_dir = Path(__file__).parent
    output_path = results_dir / "sample_edp_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"示例EDP对比图已保存到: {output_path}")
    
    # 保存示例数据
    sample_data = {
        'methods': methods,
        'edp_values': edp_values,
        'edp_stds': edp_stds,
        'note': '这是示例数据，用于测试图表生成功能'
    }
    
    data_path = results_dir / "sample_data.json"
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"示例数据已保存到: {data_path}")
    
    # 计算改进百分比
    baseline_best = min(edp_values[:-1])
    coopt_edp = edp_values[-1]
    improvement = (baseline_best - coopt_edp) / baseline_best * 100
    
    print(f"\n=== 示例结果摘要 ===")
    for method, value in zip(methods, edp_values):
        print(f"{method}: EDP = {value:.2e}")
    print(f"\n协同优化相比最佳baseline改进: {improvement:.1f}%")

if __name__ == "__main__":
    create_sample_edp_plot()
EOF

    # 运行示例图表生成
    if python3 "$RESULTS_DIR/create_sample_plot.py"; then
        log "示例EDP对比图生成完成"
    else
        error "示例图表生成失败"
    fi
}

# 主测试函数
main() {
    log "开始测试硬件-软件协同优化实验环境"
    
    # 运行测试
    test_environment
    test_baseline
    test_coopt
    create_sample_plot
    
    log "测试完成！"
    echo -e "${GREEN}测试结果:${NC}"
    echo -e "  - 环境检查: ${GREEN}通过${NC}"
    echo -e "  - Baseline测试: ${GREEN}通过${NC}"
    echo -e "  - 核心优化测试: ${GREEN}通过${NC}"
    echo -e "  - 示例图表: ${BLUE}$RESULTS_DIR/sample_edp_comparison.png${NC}"
    echo -e "  - 示例数据: ${BLUE}$RESULTS_DIR/sample_data.json${NC}"
    echo
    echo -e "${YELLOW}提示: 如果所有测试都通过，可以运行完整实验:${NC}"
    echo -e "  ${BLUE}./run_coopt_experiment.sh${NC}"
}

# 脚本入口点
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi