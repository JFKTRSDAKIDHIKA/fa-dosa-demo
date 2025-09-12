#!/usr/bin/env python3
"""
帕累托前沿扫描实验运行脚本

该脚本用于运行ParetoFrontierRunner，扫描面积-性能权衡空间，
并自动生成分析报告和可视化图表。
"""

import os
import sys
import argparse
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.act1 import run_baseline_experiment
from experiments.pareto_analysis import ParetoAnalyzer


def create_pareto_config(base_config: dict = None, num_trials: int = 30) -> dict:
    """创建帕累托前沿实验配置
    
    Args:
        base_config: 基础配置字典
        num_trials: 每个权重点的试验次数
        
    Returns:
        实验配置字典
    """
    if base_config is None:
        base_config = {
            "workload": "resnet18",
            "device": "cuda:0",
            "output_dir": "output"
        }
    
    config = {
        "shared": {
            "workload": base_config.get("workload", "resnet18"),
            "device": base_config.get("device", "cuda:0"),
            "num_trials": num_trials,
            "output_dir": base_config.get("output_dir", "output")
        },
        "baselines": ["pareto_frontier"],
        "seeds": [42]  # 使用单个种子以保持一致性
    }
    
    return config


def run_pareto_experiment(workload: str = "resnet18", 
                         num_trials: int = 30,
                         output_dir: str = "output",
                         device: str = "cuda:0") -> str:
    """运行帕累托前沿扫描实验
    
    Args:
        workload: 工作负载名称
        num_trials: 每个权重点的试验次数
        output_dir: 输出目录
        device: 计算设备
        
    Returns:
        实验输出目录路径
    """
    print("=" * 60)
    print("帕累托前沿扫描实验")
    print("=" * 60)
    print(f"工作负载: {workload}")
    print(f"每个权重点试验次数: {num_trials}")
    print(f"输出目录: {output_dir}")
    print(f"计算设备: {device}")
    print()
    
    # 创建配置
    config = create_pareto_config({
        "workload": workload,
        "device": device,
        "output_dir": output_dir
    }, num_trials)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config_file = os.path.join(output_dir, "pareto_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"配置已保存到: {config_file}")
    
    # 运行实验
    print("\n开始运行帕累托前沿扫描...")
    try:
        run_baseline_experiment(config)
        print("\n实验完成！")
        return output_dir
    except Exception as e:
        print(f"\n实验运行出错: {e}")
        raise


def analyze_results(output_dir: str, 
                   experiment_name: str = "pareto_frontier",
                   generate_plots: bool = True,
                   generate_report: bool = True) -> None:
    """分析帕累托前沿实验结果
    
    Args:
        output_dir: 结果目录
        experiment_name: 实验名称前缀
        generate_plots: 是否生成图表
        generate_report: 是否生成报告
    """
    print("\n" + "=" * 60)
    print("分析帕累托前沿结果")
    print("=" * 60)
    
    # 创建分析器
    analyzer = ParetoAnalyzer(output_dir)
    
    # 加载结果
    analyzer.load_results(experiment_name)
    
    if not analyzer.pareto_data:
        print("未找到实验结果数据")
        return
        
    print(f"加载了 {len(analyzer.pareto_data)} 个数据点")
    
    # 生成图表
    if generate_plots:
        plot_file = os.path.join(output_dir, "pareto_frontier.png")
        analyzer.plot_pareto_frontier(save_path=plot_file)
        print(f"帕累托前沿图已保存到: {plot_file}")
    
    # 生成报告
    if generate_report:
        report_file = os.path.join(output_dir, "pareto_report.txt")
        analyzer.generate_report(report_file)
        print(f"分析报告已保存到: {report_file}")
    
    # 导出数据
    data_file = os.path.join(output_dir, "pareto_data.json")
    analyzer.export_data(data_file)
    print(f"数据已导出到: {data_file}")
    
    # 显示简要统计
    pareto_points = analyzer.find_pareto_frontier()
    print(f"\n帕累托前沿包含 {len(pareto_points)} 个点")
    
    if pareto_points:
        best_edp = min(pareto_points, key=lambda x: x['edp'])
        min_area = min(pareto_points, key=lambda x: x['area_mm2'])
        
        print(f"最佳性能点: 权重={best_edp['area_weight']}, EDP={best_edp['edp']:.2e}, 面积={best_edp['area_mm2']:.2f}mm²")
        print(f"最小面积点: 权重={min_area['area_weight']}, EDP={min_area['edp']:.2e}, 面积={min_area['area_mm2']:.2f}mm²")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Run Pareto Frontier Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行完整的帕累托前沿实验
  python run_pareto_frontier.py --workload resnet18 --num-trials 50
  
  # 仅分析已有结果
  python run_pareto_frontier.py --analyze-only --output-dir output
  
  # 运行实验但不生成图表
  python run_pareto_frontier.py --no-plots
        """
    )
    
    parser.add_argument('--workload', default='resnet18', 
                       help='工作负载名称 (默认: resnet18)')
    parser.add_argument('--num-trials', type=int, default=30,
                       help='每个权重点的试验次数 (默认: 30)')
    parser.add_argument('--output-dir', default='output',
                       help='输出目录 (默认: output)')
    parser.add_argument('--device', default='cuda:0',
                       help='计算设备 (默认: cuda:0)')
    parser.add_argument('--experiment-name', default='pareto_frontier',
                       help='实验名称前缀 (默认: pareto_frontier)')
    
    # 控制选项
    parser.add_argument('--analyze-only', action='store_true',
                       help='仅分析结果，不运行实验')
    parser.add_argument('--no-plots', action='store_true',
                       help='不生成图表')
    parser.add_argument('--no-report', action='store_true',
                       help='不生成文本报告')
    
    args = parser.parse_args()
    
    try:
        # 运行实验（如果需要）
        if not args.analyze_only:
            output_dir = run_pareto_experiment(
                workload=args.workload,
                num_trials=args.num_trials,
                output_dir=args.output_dir,
                device=args.device
            )
        else:
            output_dir = args.output_dir
            
        # 分析结果
        analyze_results(
            output_dir=output_dir,
            experiment_name=args.experiment_name,
            generate_plots=not args.no_plots,
            generate_report=not args.no_report
        )
        
        print("\n" + "=" * 60)
        print("帕累托前沿实验完成！")
        print(f"结果保存在: {output_dir}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n实验被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n实验失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()