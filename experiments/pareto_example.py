#!/usr/bin/env python3
"""
帕累托前沿扫描示例脚本

这是一个简化的示例，展示如何使用ParetoFrontierRunner进行
面积-性能权衡分析。
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.baselines import ParetoFrontierRunner
from logging_utils.recorder import Recorder


def run_simple_pareto_example():
    """运行简单的帕累托前沿示例"""
    print("=" * 50)
    print("帕累托前沿扫描示例")
    print("=" * 50)
    
    # 创建输出目录
    output_dir = Path("output/pareto_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建记录器
    recorder = Recorder(output_dir)
    
    # 创建帕累托前沿运行器
    runner = ParetoFrontierRunner("pareto_example")
    
    # 配置实验参数
    config = {
        "shared": {
            "workload": "resnet18",
            "num_trials": 10,  # 较少的试验次数用于快速演示
            "device": "cuda:0" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"
        }
    }
    
    seed = 42
    
    print(f"配置: {config}")
    print(f"种子: {seed}")
    print(f"面积权重范围: {runner.area_weights}")
    print()
    
    try:
        # 运行帕累托前沿扫描
        print("开始帕累托前沿扫描...")
        runner.run(config, seed, recorder)
        
        print("\n扫描完成！")
        print(f"结果保存在: {output_dir}")
        
        # 显示结果摘要
        print(f"\n扫描了 {len(runner.area_weights)} 个面积权重点:")
        for i, weight in enumerate(runner.area_weights):
            print(f"  {i+1}. 权重 = {weight}")
            
    except Exception as e:
        print(f"运行出错: {e}")
        raise


def analyze_example_results():
    """分析示例结果"""
    print("\n" + "=" * 50)
    print("分析示例结果")
    print("=" * 50)
    
    try:
        from experiments.pareto_analysis import ParetoAnalyzer
        
        # 创建分析器
        analyzer = ParetoAnalyzer("output")
        
        # 加载结果
        analyzer.load_results("pareto_example")
        
        if analyzer.pareto_data:
            print(f"找到 {len(analyzer.pareto_data)} 个数据点")
            
            # 生成简单的文本报告
            pareto_points = analyzer.find_pareto_frontier()
            print(f"帕累托前沿包含 {len(pareto_points)} 个点")
            
            if pareto_points:
                print("\n帕累托前沿点:")
                print(f"{'权重':<8} {'面积(mm²)':<12} {'EDP':<15}")
                print("-" * 35)
                for point in pareto_points:
                    print(f"{point['area_weight']:<8.2f} {point['area_mm2']:<12.2f} {point['edp']:<15.2e}")
                    
            # 尝试生成图表（如果matplotlib可用）
            try:
                analyzer.plot_pareto_frontier(save_path="output/example_pareto.png")
                print("\n图表已保存到: output/example_pareto.png")
            except ImportError:
                print("\n注意: matplotlib未安装，跳过图表生成")
                
        else:
            print("未找到结果数据")
            
    except ImportError as e:
        print(f"分析模块导入失败: {e}")
    except Exception as e:
        print(f"分析出错: {e}")


def main():
    """主函数"""
    print("帕累托前沿扫描示例")
    print("这个示例将演示如何使用ParetoFrontierRunner进行面积-性能权衡分析")
    print()
    
    try:
        # 运行示例
        run_simple_pareto_example()
        
        # 分析结果
        analyze_example_results()
        
        print("\n" + "=" * 50)
        print("示例完成！")
        print("=" * 50)
        print("\n使用说明:")
        print("1. 查看 output/ 目录中的结果文件")
        print("2. 运行 python experiments/pareto_analysis.py 进行详细分析")
        print("3. 运行 python experiments/run_pareto_frontier.py 进行完整实验")
        
    except KeyboardInterrupt:
        print("\n示例被用户中断")
    except Exception as e:
        print(f"\n示例运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()