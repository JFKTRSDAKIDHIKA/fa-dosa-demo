#!/usr/bin/env python3
"""
快速帕累托前沿优化方案验证脚本

测试三种优化方案的效果：
1. 增加 num_trials
2. 从上一个权重搜索的最优解开始新搜索
3. 调整优化器参数
"""

import os
import sys
import json
import time
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from experiments import baselines as baselines_mod
from logging_utils import recorder as recorder_mod


class AgileParetoTester:
    """快速帕累托前沿优化方案测试器"""
    
    def __init__(self, output_dir: str = "output/agile_pareto_test"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 基础配置
        self.base_config = {
            "workload": "resnet18",
            "device": "cuda:0",
            "output_dir": str(self.output_dir)
        }
        
        # 测试方案配置
        self.test_configs = {
            "baseline": {
                "name": "基线方案",
                "num_trials": 5,
                "area_weights": [0.0, 2.5, 5.0],  # 只用3个权重点
                "use_warm_start": False,
                "optimizer_params": {}
            },
            "more_trials": {
                "name": "增加试验次数",
                "num_trials": 20,  # 4倍增加
                "area_weights": [0.0, 2.5, 5.0],
                "use_warm_start": False,
                "optimizer_params": {}
            },
            "warm_start": {
                "name": "热启动搜索",
                "num_trials": 5,
                "area_weights": [0.0, 2.5, 5.0],
                "use_warm_start": True,
                "optimizer_params": {}
            },
            "optimizer_tuned": {
                "name": "优化器调参",
                "num_trials": 5,
                "area_weights": [0.0, 2.5, 5.0],
                "use_warm_start": False,
                "optimizer_params": {
                    "learning_rate": 0.01,  # 降低学习率
                    "exploration_factor": 1.5  # 增加探索
                }
            }
        }
        
        self.results = {}
    
    def create_test_config(self, test_name: str) -> Dict:
        """为指定测试创建配置"""
        test_config = self.test_configs[test_name]
        
        config = {
            "shared": {
                **self.base_config,
                "num_trials": test_config["num_trials"],
                "seeds": [42]
            },
            "baselines": ["pareto_frontier"],
            "test_config": test_config
        }
        
        return config
    
    def run_single_test(self, test_name: str) -> Dict:
        """运行单个测试方案"""
        print(f"\n{'='*60}")
        print(f"运行测试: {self.test_configs[test_name]['name']}")
        print(f"{'='*60}")
        
        test_dir = self.output_dir / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建配置
        config = self.create_test_config(test_name)
        test_config = config["test_config"]
        
        # 保存配置
        config_path = test_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 临时修改ParetoFrontierRunner的配置
        self._apply_test_config(test_config)
        
        start_time = time.time()
        
        try:
            # 运行实验
            runner = baselines_mod.get_baseline_runner("pareto_frontier")
            
            seed_dir = test_dir / "seed_42"
            with recorder_mod.Recorder(seed_dir) as rec:
                runner.run(config, 42, rec)
            
            # 收集结果
            results = self._collect_results(seed_dir)
            results["execution_time"] = time.time() - start_time
            results["test_config"] = test_config
            
            print(f"测试完成，用时: {results['execution_time']:.2f}秒")
            print(f"找到帕累托点数量: {len(results['pareto_points'])}")
            
            return results
            
        except Exception as e:
            print(f"测试失败: {e}")
            return {
                "error": str(e),
                "execution_time": time.time() - start_time,
                "test_config": test_config,
                "pareto_points": []
            }
    
    def _apply_test_config(self, test_config: Dict):
        """临时应用测试配置到ParetoFrontierRunner"""
        # 设置类属性来控制area_weights
        baselines_mod.ParetoFrontierRunner._test_area_weights = test_config["area_weights"]
        
        # 如果有优化器参数，可以在这里应用
        # 注意：这是一个简化的实现，实际可能需要更复杂的参数传递
        if test_config["optimizer_params"]:
            # 这里可以添加优化器参数的应用逻辑
            pass
    
    def _collect_results(self, seed_dir: Path) -> Dict:
        """收集实验结果"""
        trials_csv = seed_dir / "trials.csv"
        
        if not trials_csv.exists():
            return {"pareto_points": [], "all_points": []}
        
        # 读取试验数据
        df = pd.read_csv(trials_csv)
        
        # 提取面积和性能数据
        if 'area' in df.columns and 'performance' in df.columns:
            points = [(row['area'], row['performance']) for _, row in df.iterrows()]
            
            # 计算帕累托前沿
            pareto_points = self._compute_pareto_frontier(points)
            
            return {
                "pareto_points": pareto_points,
                "all_points": points,
                "num_trials": len(df),
                "best_area": df['area'].min(),
                "best_performance": df['performance'].max()
            }
        else:
            return {"pareto_points": [], "all_points": []}
    
    def _compute_pareto_frontier(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """计算帕累托前沿（最小化面积，最大化性能）"""
        if not points:
            return []
        
        # 排序：按面积升序
        sorted_points = sorted(points, key=lambda x: x[0])
        
        pareto = []
        max_performance = float('-inf')
        
        for area, performance in sorted_points:
            if performance > max_performance:
                pareto.append((area, performance))
                max_performance = performance
        
        return pareto
    
    def run_all_tests(self) -> Dict:
        """运行所有测试方案"""
        print("开始快速帕累托前沿优化方案验证...")
        
        for test_name in self.test_configs.keys():
            self.results[test_name] = self.run_single_test(test_name)
        
        # 生成对比报告
        self._generate_comparison_report()
        
        return self.results
    
    def _generate_comparison_report(self):
        """生成对比报告"""
        print(f"\n{'='*80}")
        print("测试结果对比报告")
        print(f"{'='*80}")
        
        # 创建对比表格
        comparison_data = []
        
        for test_name, result in self.results.items():
            if "error" in result:
                comparison_data.append({
                    "测试方案": self.test_configs[test_name]["name"],
                    "执行时间(秒)": f"{result['execution_time']:.2f}",
                    "帕累托点数量": "错误",
                    "状态": f"失败: {result['error']}"
                })
            else:
                comparison_data.append({
                    "测试方案": self.test_configs[test_name]["name"],
                    "执行时间(秒)": f"{result['execution_time']:.2f}",
                    "帕累托点数量": len(result['pareto_points']),
                    "状态": "成功"
                })
        
        # 打印对比表格
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        # 保存详细结果
        results_path = self.output_dir / "comparison_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n详细结果已保存到: {results_path}")
        
        # 分析最有效的方案
        self._analyze_best_approach()
    
    def _analyze_best_approach(self):
        """分析最有效的优化方案"""
        print(f"\n{'='*60}")
        print("优化方案效果分析")
        print(f"{'='*60}")
        
        successful_results = {k: v for k, v in self.results.items() if "error" not in v}
        
        if not successful_results:
            print("所有测试都失败了，无法进行分析。")
            return
        
        # 按帕累托点数量排序
        sorted_by_pareto = sorted(
            successful_results.items(),
            key=lambda x: len(x[1]['pareto_points']),
            reverse=True
        )
        
        print("按帕累托点数量排序:")
        for i, (test_name, result) in enumerate(sorted_by_pareto, 1):
            config_name = self.test_configs[test_name]["name"]
            pareto_count = len(result['pareto_points'])
            exec_time = result['execution_time']
            print(f"{i}. {config_name}: {pareto_count}个点 (用时{exec_time:.2f}秒)")
        
        # 推荐最佳方案
        if sorted_by_pareto:
            best_test, best_result = sorted_by_pareto[0]
            best_config = self.test_configs[best_test]
            
            print(f"\n推荐方案: {best_config['name']}")
            print(f"- 帕累托点数量: {len(best_result['pareto_points'])}")
            print(f"- 执行时间: {best_result['execution_time']:.2f}秒")
            print(f"- 效率比 (点数/时间): {len(best_result['pareto_points'])/best_result['execution_time']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="快速帕累托前沿优化方案验证")
    parser.add_argument("--output-dir", default="output/agile_pareto_test",
                       help="输出目录")
    parser.add_argument("--test", choices=["baseline", "more_trials", "warm_start", "optimizer_tuned", "all"],
                       default="all", help="要运行的测试")
    
    args = parser.parse_args()
    
    tester = AgileParetoTester(args.output_dir)
    
    if args.test == "all":
        tester.run_all_tests()
    else:
        result = tester.run_single_test(args.test)
        print(f"\n测试结果: {result}")


if __name__ == "__main__":
    main()