#!/usr/bin/env python3
"""
帕累托前沿分析和可视化工具

该脚本用于分析ParetoFrontierRunner的实验结果，
生成面积-性能权衡的帕累托前沿图。
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse


class ParetoAnalyzer:
    """帕累托前沿分析器"""
    
    def __init__(self, results_dir: str = "output"):
        self.results_dir = Path(results_dir)
        self.pareto_data = []
        
    def load_results(self, experiment_name: str = "pareto_frontier") -> None:
        """加载帕累托前沿实验结果
        
        Args:
            experiment_name: 实验名称前缀
        """
        print(f"Loading results from {self.results_dir}...")
        
        # 查找所有相关的结果文件
        pattern = f"*{experiment_name}*best*.json"
        result_files = list(self.results_dir.glob(pattern))
        
        if not result_files:
            print(f"No result files found matching pattern: {pattern}")
            return
            
        print(f"Found {len(result_files)} result files")
        
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # 提取面积权重信息（从文件名或数据中）
                area_weight = self._extract_area_weight(file_path.name, data)
                
                if 'best_metrics' in data and data['best_metrics']:
                    metrics = data['best_metrics']
                    point = {
                        'area_weight': area_weight,
                        'area_mm2': metrics.get('area_mm2', 0),
                        'edp': metrics.get('edp', float('inf')),
                        'latency_sec': metrics.get('latency_sec', 0),
                        'energy_pj': metrics.get('energy_pj', 0),
                        'log_edp': metrics.get('log_edp', 0),
                        'file': file_path.name
                    }
                    self.pareto_data.append(point)
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        # 按面积权重排序
        self.pareto_data.sort(key=lambda x: x['area_weight'])
        print(f"Loaded {len(self.pareto_data)} data points")
        
    def _extract_area_weight(self, filename: str, data: Dict[str, Any]) -> float:
        """从文件名或数据中提取面积权重"""
        # 尝试从文件名中提取
        import re
        match = re.search(r'weight_([0-9.]+)', filename)
        if match:
            return float(match.group(1))
            
        # 尝试从数据中提取
        if 'area_weight' in data:
            return data['area_weight']
            
        # 默认值
        return 0.0
        
    def find_pareto_frontier(self) -> List[Dict[str, Any]]:
        """找到帕累托前沿点
        
        Returns:
            帕累托前沿上的点列表
        """
        if not self.pareto_data:
            return []
            
        # 按面积排序
        sorted_points = sorted(self.pareto_data, key=lambda x: x['area_mm2'])
        
        pareto_points = []
        min_edp = float('inf')
        
        for point in sorted_points:
            if point['edp'] < min_edp:
                min_edp = point['edp']
                pareto_points.append(point)
                
        return pareto_points
        
    def plot_pareto_frontier(self, save_path: str = None, show_all_points: bool = True) -> None:
        """绘制帕累托前沿图
        
        Args:
            save_path: 保存路径，如果为None则显示图像
            show_all_points: 是否显示所有点
        """
        if not self.pareto_data:
            print("No data to plot")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 提取数据
        areas = [p['area_mm2'] for p in self.pareto_data]
        edps = [p['edp'] for p in self.pareto_data]
        weights = [p['area_weight'] for p in self.pareto_data]
        
        # 找到帕累托前沿
        pareto_points = self.find_pareto_frontier()
        pareto_areas = [p['area_mm2'] for p in pareto_points]
        pareto_edps = [p['edp'] for p in pareto_points]
        
        # 图1: 面积 vs EDP
        if show_all_points:
            scatter = ax1.scatter(areas, edps, c=weights, cmap='viridis', alpha=0.6, s=50)
            plt.colorbar(scatter, ax=ax1, label='Area Weight')
            
        # 绘制帕累托前沿
        ax1.plot(pareto_areas, pareto_edps, 'r-o', linewidth=2, markersize=8, 
                label=f'Pareto Frontier ({len(pareto_points)} points)')
        
        ax1.set_xlabel('Area (mm²)')
        ax1.set_ylabel('EDP (Energy-Delay Product)')
        ax1.set_title('Pareto Frontier: Area vs EDP')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 图2: 面积权重 vs 性能指标
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(weights, areas, 'b-o', label='Area (mm²)', linewidth=2)
        line2 = ax2_twin.plot(weights, edps, 'r-s', label='EDP', linewidth=2)
        
        ax2.set_xlabel('Area Weight')
        ax2.set_ylabel('Area (mm²)', color='b')
        ax2_twin.set_ylabel('EDP (Energy-Delay Product)', color='r')
        ax2_twin.set_yscale('log')
        ax2.set_title('Trade-off: Area Weight vs Performance Metrics')
        
        # 合并图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
            
    def generate_report(self, output_file: str = "pareto_report.txt") -> None:
        """生成帕累托前沿分析报告
        
        Args:
            output_file: 输出文件路径
        """
        if not self.pareto_data:
            print("No data for report generation")
            return
            
        pareto_points = self.find_pareto_frontier()
        
        with open(output_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("帕累托前沿分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"总数据点数: {len(self.pareto_data)}\n")
            f.write(f"帕累托前沿点数: {len(pareto_points)}\n\n")
            
            # 统计信息
            areas = [p['area_mm2'] for p in self.pareto_data]
            edps = [p['edp'] for p in self.pareto_data]
            
            f.write("数据统计:\n")
            f.write(f"  面积范围: {min(areas):.2f} - {max(areas):.2f} mm²\n")
            f.write(f"  EDP范围: {min(edps):.2e} - {max(edps):.2e}\n\n")
            
            # 帕累托前沿点详情
            f.write("帕累托前沿点详情:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'权重':<8} {'面积(mm²)':<12} {'EDP':<15} {'延迟(s)':<12} {'能耗(pJ)':<12}\n")
            f.write("-" * 80 + "\n")
            
            for point in pareto_points:
                f.write(f"{point['area_weight']:<8.2f} {point['area_mm2']:<12.2f} "
                       f"{point['edp']:<15.2e} {point['latency_sec']:<12.2e} "
                       f"{point['energy_pj']:<12.2e}\n")
                       
            f.write("\n")
            
            # 推荐配置
            if pareto_points:
                # 找到最佳EDP点
                best_edp_point = min(pareto_points, key=lambda x: x['edp'])
                # 找到最小面积点
                min_area_point = min(pareto_points, key=lambda x: x['area_mm2'])
                
                f.write("推荐配置:\n")
                f.write(f"  最佳性能 (最低EDP): 权重={best_edp_point['area_weight']}, "
                       f"面积={best_edp_point['area_mm2']:.2f}mm², EDP={best_edp_point['edp']:.2e}\n")
                f.write(f"  最小面积: 权重={min_area_point['area_weight']}, "
                       f"面积={min_area_point['area_mm2']:.2f}mm², EDP={min_area_point['edp']:.2e}\n")
                       
        print(f"Report saved to {output_file}")
        
    def export_data(self, output_file: str = "pareto_data.json") -> None:
        """导出帕累托数据为JSON格式
        
        Args:
            output_file: 输出文件路径
        """
        pareto_points = self.find_pareto_frontier()
        
        export_data = {
            'all_points': self.pareto_data,
            'pareto_frontier': pareto_points,
            'summary': {
                'total_points': len(self.pareto_data),
                'pareto_points': len(pareto_points),
                'area_range': [min(p['area_mm2'] for p in self.pareto_data),
                              max(p['area_mm2'] for p in self.pareto_data)],
                'edp_range': [min(p['edp'] for p in self.pareto_data),
                             max(p['edp'] for p in self.pareto_data)]
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        print(f"Data exported to {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Pareto Frontier Analysis Tool')
    parser.add_argument('--results-dir', default='output', help='Results directory')
    parser.add_argument('--experiment-name', default='pareto_frontier', help='Experiment name prefix')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--report', action='store_true', help='Generate text report')
    parser.add_argument('--export', action='store_true', help='Export data to JSON')
    parser.add_argument('--save-plot', help='Save plot to file instead of showing')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = ParetoAnalyzer(args.results_dir)
    
    # 加载结果
    analyzer.load_results(args.experiment_name)
    
    if not analyzer.pareto_data:
        print("No data loaded. Exiting.")
        return
        
    # 生成图表
    if args.plot or args.save_plot:
        analyzer.plot_pareto_frontier(save_path=args.save_plot)
        
    # 生成报告
    if args.report:
        analyzer.generate_report()
        
    # 导出数据
    if args.export:
        analyzer.export_data()
        
    # 如果没有指定任何操作，默认生成所有输出
    if not any([args.plot, args.report, args.export, args.save_plot]):
        print("Generating all outputs...")
        analyzer.plot_pareto_frontier(save_path="pareto_frontier.png")
        analyzer.generate_report()
        analyzer.export_data()


if __name__ == '__main__':
    main()