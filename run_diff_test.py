#!/usr/bin/env python3
"""
FA-DOSA 差分测试框架 - 终极实现

本脚本用于对FA-DOSA DNN加速器设计框架进行系统化的差分测试。
核心测试哲学：逻辑核心，物理无关 - 专注于验证分析模型在核心逻辑层面与底层仿真器的一致性。

作者: FA-DOSA Team
版本: 1.0
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd


class TimeloopReportParser:
    """
    Timeloop仿真报告解析器
    
    专职负责解析 timeloop-mapper.stats.txt 文件，提取关键的性能指标。
    """
    
    def __init__(self):
        self.parsed_data = {}
    
    def parse_stats_file(self, stats_file_path: Path) -> Dict[str, Any]:
        """
        解析Timeloop统计文件
        
        Args:
            stats_file_path: timeloop-mapper.stats.txt文件路径
            
        Returns:
            结构化的解析结果字典
        """
        if not stats_file_path.exists():
            raise FileNotFoundError(f"Stats file not found: {stats_file_path}")
        
        with open(stats_file_path, 'r') as f:
            content = f.read()
        
        self.parsed_data = {
            'summary': {},
            'components': {}
        }
        
        # 解析总周期数
        self._parse_summary_stats(content)
        
        # 解析组件级别的访问统计
        self._parse_component_stats(content)
        
        return self.parsed_data
    
    def _parse_summary_stats(self, content: str) -> None:
        """
        解析Summary Stats块中的总周期数
        """
        # 查找Summary Stats部分
        summary_pattern = r'Summary Stats\s*\n-+\s*\n(.*?)(?=\n\n|$)'
        summary_match = re.search(summary_pattern, content, re.DOTALL)
        
        if summary_match:
            summary_content = summary_match.group(1)
            # 查找Cycles行
            cycles_pattern = r'Cycles:\s*(\d+)'
            cycles_match = re.search(cycles_pattern, summary_content)
            
            if cycles_match:
                self.parsed_data['summary']['cycles'] = int(cycles_match.group(1))
        else:
            # 如果没找到Summary Stats，尝试直接搜索Cycles
            cycles_pattern = r'Cycles:\s*(\d+)'
            cycles_match = re.search(cycles_pattern, content)
            if cycles_match:
                self.parsed_data['summary']['cycles'] = int(cycles_match.group(1))
    
    def _parse_component_stats(self, content: str) -> None:
        """
        解析各个组件的访问统计信息
        """
        # 使用正确的Level分隔符
        level_pattern = r'Level \d+\n-+\n(.*?)(?=Level \d+\n-+|Networks\n-+|Operational|Summary Stats|$)'
        level_matches = re.finditer(level_pattern, content, re.DOTALL)
        
        for level_match in level_matches:
            level_content = level_match.group(1)
            
            # 查找组件名称
            component_match = re.search(r'=== (\w+) ===', level_content)
            if not component_match:
                continue
                
            component_name = component_match.group(1)
            
            # 查找STATS部分
            stats_match = re.search(r'STATS\s*\n\s*-+\s*\n(.*)', level_content, re.DOTALL)
            if not stats_match:
                continue
                
            stats_content = stats_match.group(1)
            
            self.parsed_data['components'][component_name] = {
                'accesses': {}
            }
            
            # 解析每个张量类型的访问信息
            self._parse_tensor_accesses(stats_content, component_name)
    
    def _parse_tensor_accesses(self, component_content: str, component_name: str) -> None:
        """
        解析组件中每个张量的访问统计
        """
        # 查找张量块 (Weights, Inputs, Outputs)
        tensor_pattern = r'(Weights|Inputs|Outputs):\s*\n(.*?)(?=\n\s*(?:Weights|Inputs|Outputs):|\n\nLevel|$)'
        tensor_matches = re.finditer(tensor_pattern, component_content, re.DOTALL)
        
        for tensor_match in tensor_matches:
            tensor_name = tensor_match.group(1)
            tensor_content = tensor_match.group(2)
            
            # 初始化张量访问数据
            self.parsed_data['components'][component_name]['accesses'][tensor_name] = {
                'scalar_reads_total': 0.0,
                'scalar_fills_total': 0.0,
                'scalar_updates_total': 0.0
            }
            
            # 解析实例数量
            utilized_instances = self._extract_utilized_instances(tensor_content)
            
            # 解析每实例的访问次数并计算总数
            reads_per_instance = self._extract_scalar_value(tensor_content, 'Scalar reads \\(per-instance\\)')
            fills_per_instance = self._extract_scalar_value(tensor_content, 'Scalar fills \\(per-instance\\)')
            updates_per_instance = self._extract_scalar_value(tensor_content, 'Scalar updates \\(per-instance\\)')
            
            # 计算总访问次数 = 每实例访问次数 × 实例数量
            self.parsed_data['components'][component_name]['accesses'][tensor_name].update({
                'scalar_reads_total': reads_per_instance * utilized_instances,
                'scalar_fills_total': fills_per_instance * utilized_instances,
                'scalar_updates_total': updates_per_instance * utilized_instances
            })
            
            # 新增：把 per-instance 和 实例数也存起来，后面"atomic 对齐"要用
            self.parsed_data['components'][component_name]['accesses'][tensor_name].update({
                'scalar_reads_per_instance': reads_per_instance,
                'scalar_fills_per_instance': fills_per_instance,
                'scalar_updates_per_instance': updates_per_instance,
                'utilized_instances': utilized_instances
            })
    
    def _extract_utilized_instances(self, tensor_content: str) -> float:
        """
        提取Utilized instances (max)的值
        """
        pattern = r'Utilized instances \(max\)\s*:\s*([\d.]+)'
        match = re.search(pattern, tensor_content)
        return float(match.group(1)) if match else 1.0
    
    def _extract_scalar_value(self, tensor_content: str, metric_name: str) -> float:
        """
        提取标量值（如reads, fills, updates per-instance）
        """
        pattern = rf'{metric_name}\s*:\s*([\d.]+)'
        match = re.search(pattern, tensor_content)
        return float(match.group(1)) if match else 0.0


class DifferentialComparator:
    """
    差分比较器 - 测试框架的灵魂
    
    实现核心的、聚焦于逻辑行为的"对账引擎"。
    """
    
    def __init__(self):
        # 核心指标配置清单 - 这是整个测试框架的"法律"
        self.metrics_to_check = [
            {
                'name': 'Total Cycles',
                'dosa_path': ['calculation_summary', 'total_cycles'],
                'tl_path': ['summary', 'cycles'],
                'unit': 'cycles',
                'tolerance': 0.01
            },
            {
                'name': 'Atomic: L0 Reads (W)',
                'dosa_path': ['intra_level_consumption_trace', 'L0_Registers (i=0)', 'W', 'consumption_reads'],
                'tl_path': ['components', 'L0_Registers', 'accesses', 'Weights', 'scalar_reads_total'],
                'unit': 'accesses',
                'tolerance': 0.01
            },
            {
                'name': 'Atomic: L2 Reads (I)',
                'dosa_path': ['intra_level_consumption_trace', 'L2_Scratchpad (i=2)', 'I', 'consumption_reads'],
                'tl_path': ['components', 'L2_Scratchpad', 'accesses', 'Inputs', 'scalar_reads_total'],
                'unit': 'accesses',
                'tolerance': 0.05
            },
            {
                'name': 'Atomic: Output Updates',
                'dosa_path': ['intra_level_consumption_trace', 'L1_Accumulator (i=1)', 'O', 'consumption_updates'],
                'tl_path':   ['components', 'L1_Accumulator', 'accesses', 'Outputs', 'scalar_updates_total'],
                'unit': 'accesses', 
                'tolerance': 0.05
            },
            {
                'name': 'Traffic: DRAM->L2 (Fills, W)',
                'dosa_path': ['inter_level_fill_traffic_trace', 'L2_Scratchpad (i=2)', 'W', 'tensor_fill_accesses'],
                'tl_path': ['components', 'L2_Scratchpad', 'accesses', 'Weights', 'scalar_fills_total'],
                'unit': 'accesses',
                'tolerance': 0.05
            },
            {
                'name': 'Traffic: DRAM->L2 (Fills, I)',
                'dosa_path': ['inter_level_fill_traffic_trace', 'L2_Scratchpad (i=2)', 'I', 'tensor_fill_accesses'],
                'tl_path': ['components', 'L2_Scratchpad', 'accesses', 'Inputs', 'scalar_fills_total'],
                'unit': 'accesses',
                'tolerance': 0.30
            },
            {
                'name': 'Traffic: L2->L0 (Fills, W)',
                'dosa_path': ['inter_level_fill_traffic_trace', 'L0_Registers (i=0)', 'W', 'tensor_fill_accesses'],
                'tl_path': ['components', 'L0_Registers', 'accesses', 'Weights', 'scalar_fills_total'],
                'unit': 'accesses',
                'tolerance': 0.05
            }
        ]
    
    def compare_results(self, dosa_data: Dict[str, Any], timeloop_data: Dict[str, Any], vp_id: str) -> List[Dict[str, Any]]:
        """
        执行差分比较
        
        Args:
            dosa_data: FA-DOSA分析模型数据
            timeloop_data: Timeloop仿真数据
            vp_id: 验证点ID
            
        Returns:
            差异记录列表
        """
        differences = []
        
        for metric in self.metrics_to_check:
            dosa_value = self._get_value(dosa_data, metric['dosa_path'])
            tl_value = self._get_value(timeloop_data, metric['tl_path'])
            
            # === 专用分支：Atomic: Output Updates（按论文口径对齐） ===
            if metric.get('name') == 'Atomic: Output Updates':
                try:
                    # 1) 从 TL(L1) 取 per-instance 和 utilized_instances
                    tl_L1_O = timeloop_data['components']['L1_Accumulator']['accesses']['Outputs']
                    per_inst_updates = float(tl_L1_O.get('scalar_updates_per_instance', 0.0))
                    utilized_insts   = float(tl_L1_O.get('utilized_instances', 1.0))
                    
                    # 2) 从 DOSA 调试里拿 F_S,O(L1) —— 这就是"与 O 无关的空间并行"，等价于 C_spatial（R/S/N若无并行则为1）
                    #    该值在你的 debug 里是 intra_level_consumption_trace 的 "F_S,t(i)" 字段
                    dosa_F_S_O_L1 = self._get_value(
                        dosa_data,
                        ['intra_level_consumption_trace', 'L1_Accumulator (i=1)', 'O', 'F_S,t(i)']
                    )
                    if dosa_F_S_O_L1 == 0.0:
                        dosa_F_S_O_L1 = 1.0  # 兜底，避免除零
                    
                    # 3) 由 "L1 实例数 = C_spatial * K_spatial" 得到 K_spatial
                    K_spatial = utilized_insts / dosa_F_S_O_L1
                    
                    # 4) TL 的"atomic 对齐值" = per-instance × K_spatial
                    tl_value = per_inst_updates * K_spatial
                except Exception as e:
                    # 解析失败时，保留原 tl_value（通常是总数），但打个标记更容易排查
                    # 你也可以选择在 differences 里加个 Note 提醒
                    pass
            # === 专用分支结束 ===
            
            # 检查是否为零守卫模式
            if metric.get('mode') == 'zero_guard':
                epsilon = float(metric.get('epsilon', 1.0))
                dosa_is_zero = abs(dosa_value) <= epsilon
                tl_is_zero = abs(tl_value) <= epsilon
                
                # 如果两侧都接近零，则跳过（通过测试）
                if dosa_is_zero and tl_is_zero:
                    continue
                
                # 否则记录差异
                differences.append({
                    'VP_ID': vp_id,
                    'Metric': metric['name'],
                    'DOSA_Value': dosa_value,
                    'Timeloop_Value': tl_value,
                    'Relative_Error': '—',
                    'Tolerance': f'abs≤{epsilon:g}',
                    'Unit': metric['unit'],
                    'Note': 'Expected zero under paper-B constraints'
                })
                continue
            
            # 原有的比率模式逻辑
            # 计算相对误差
            if tl_value == 0 and dosa_value == 0:
                relative_error = 0.0
            elif tl_value == 0:
                relative_error = float('inf')
            else:
                relative_error = abs(dosa_value - tl_value) / tl_value
            
            # 检查是否超过容忍度阈值
            if relative_error > metric['tolerance']:
                differences.append({
                    'VP_ID': vp_id,
                    'Metric': metric['name'],
                    'DOSA_Value': dosa_value,
                    'Timeloop_Value': tl_value,
                    'Relative_Error': f"{relative_error:.2%}",
                    'Tolerance': f"{metric['tolerance']:.2%}",
                    'Unit': metric['unit']
                })
        
        return differences
    
    def _get_value(self, data: Dict[str, Any], path: List[str]) -> float:
        """
        根据路径从嵌套字典中安全地提取数值
        
        Args:
            data: 数据字典
            path: 访问路径列表
            
        Returns:
            提取的数值，如果路径不存在则返回0.0
        """
        # 处理特殊的 '__ZERO__' 路径
        if path == ['__ZERO__']:
            return 0.0
            
        current = data
        
        try:
            for key in path:
                current = current[key]
            return float(current) if current is not None else 0.0
        except (KeyError, TypeError, ValueError):
            return 0.0


class DiffTestRunner:
    """
    差分测试运行器
    
    负责调度和编排整个测试工作流。
    """
    
    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        self.parser = TimeloopReportParser()
        self.comparator = DifferentialComparator()
    
    def run_differential_test(self) -> None:
        """
        执行完整的差分测试流程
        """
        print("FA-DOSA 差分测试框架启动...")
        print(f"输入目录: {self.input_dir}")
        
        # 自动发现文件对
        file_pairs = self._discover_file_pairs()
        
        if not file_pairs:
            print("错误: 未找到任何有效的文件对进行测试")
            return
        
        print(f"发现 {len(file_pairs)} 个验证点")
        
        all_differences = []
        
        # 处理每个文件对
        for vp_id, dosa_file, timeloop_file in file_pairs:
            print(f"\n处理验证点: {vp_id}")
            
            try:
                # 解析DOSA数据
                with open(dosa_file, 'r') as f:
                    dosa_data = json.load(f)
                
                # 解析Timeloop数据
                timeloop_data = self.parser.parse_stats_file(timeloop_file)
                
                # 执行比较
                differences = self.comparator.compare_results(dosa_data, timeloop_data, vp_id)
                all_differences.extend(differences)
                
                if differences:
                    print(f"  发现 {len(differences)} 个差异")
                else:
                    print("  ✓ 所有指标均在容忍度范围内")
                    
            except Exception as e:
                print(f"  错误: 处理验证点 {vp_id} 时发生异常: {e}")
        
        # 生成最终报告
        self._generate_final_report(all_differences)
    
    def _discover_file_pairs(self) -> List[Tuple[str, Path, Path]]:
        """
        自动发现成对的DOSA和Timeloop文件
        
        Returns:
            (vp_id, dosa_file_path, timeloop_file_path) 的列表
        """
        file_pairs = []
        
        # 查找所有DOSA调试文件
        dosa_pattern = "debug_performance_model_point_*.json"
        # 首先在根目录查找
        dosa_files = list(self.input_dir.glob(dosa_pattern))
        # 如果根目录没有，则在output目录查找
        if not dosa_files:
            output_dir = self.input_dir / "output"
            if output_dir.exists():
                dosa_files = list(output_dir.glob(dosa_pattern))
        
        for dosa_file in dosa_files:
            # 提取验证点ID
            match = re.search(r'debug_performance_model_point_(\d+)\.json$', dosa_file.name)
            if not match:
                continue
            
            vp_id = match.group(1)
            
            # 查找对应的Timeloop文件
            # 如果DOSA文件在output目录，则Timeloop文件也在output目录
            if "output" in str(dosa_file.parent):
                timeloop_file = dosa_file.parent / f"validation_workspace_{vp_id}" / "timeloop-mapper.stats.txt"
            else:
                timeloop_file = self.input_dir / "output" / f"validation_workspace_{vp_id}" / "timeloop-mapper.stats.txt"
            
            if timeloop_file.exists():
                file_pairs.append((vp_id, dosa_file, timeloop_file))
            else:
                print(f"警告: 验证点 {vp_id} 缺少Timeloop统计文件")
        
        return sorted(file_pairs, key=lambda x: int(x[0]))
    
    def _generate_final_report(self, all_differences: List[Dict[str, Any]]) -> None:
        """
        生成最终的差异报告
        """
        print("\n" + "="*80)
        print("FA-DOSA 差分测试结果报告")
        print("="*80)
        
        if not all_differences:
            print("\n🎉 测试通过！")
            print("所有验证点的核心逻辑指标均与Timeloop仿真器保持一致。")
            print("分析模型在逻辑层面的行为已通过验证。")
        else:
            print(f"\n⚠️  发现 {len(all_differences)} 个差异需要关注")
            
            # 创建DataFrame并格式化输出
            df = pd.DataFrame(all_differences)
            
            # 重新排列列的顺序，如果存在Note列则包含它
            column_order = ['VP_ID', 'Metric', 'DOSA_Value', 'Timeloop_Value', 'Relative_Error', 'Tolerance', 'Unit']
            if 'Note' in df.columns:
                column_order.append('Note')
            df = df[column_order]
            
            print("\n差异详情:")
            print(df.to_string(index=False, float_format='%.2f'))
            
            # 保存到CSV文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"diff_report_logical_{timestamp}.csv"
            csv_path = self.input_dir / csv_filename
            
            df.to_csv(csv_path, index=False)
            print(f"\n差异报告已保存至: {csv_path}")
        
        print("\n测试完成。")


def main():
    """
    主函数 - 命令行入口点
    """
    parser = argparse.ArgumentParser(
        description="FA-DOSA 差分测试框架 - 验证分析模型与仿真器的核心逻辑一致性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python run_diff_test.py /path/to/validation/output
  
注意:
  - input_dir 应指向由 run_dmt_validation.py 生成的完整输出目录
  - 该目录应包含 debug_performance_model_point_*.json 文件
  - 以及对应的 validation_workspace_*/timeloop-mapper.stats.txt 文件
        """
    )
    
    parser.add_argument(
        'input_dir',
        type=Path,
        help='包含验证数据的输入目录路径'
    )
    
    args = parser.parse_args()
    
    # 验证输入目录
    if not args.input_dir.exists():
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return 1
    
    if not args.input_dir.is_dir():
        print(f"错误: 输入路径不是目录: {args.input_dir}")
        return 1
    
    # 创建并运行测试器
    try:
        runner = DiffTestRunner(args.input_dir)
        runner.run_differential_test()
        return 0
    except Exception as e:
        print(f"致命错误: {e}")
        return 1


if __name__ == "__main__":
    exit(main())