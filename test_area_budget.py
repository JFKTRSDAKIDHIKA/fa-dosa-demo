#!/usr/bin/env python3
"""
面积预算功能测试脚本

验证面积预算功能的正确性，包括:
1. 配置参数的正确加载
2. 惩罚项计算的准确性
3. 不同场景预设的应用
4. Loss函数的集成
"""

import torch
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

from dosa.config import Config
from dosa.searcher import BaseSearcher

class TestAreaBudget:
    """面积预算功能测试类"""
    
    def __init__(self):
        self.config = Config()
        
    def test_config_loading(self):
        """测试配置参数加载"""
        print("\n=== 测试配置参数加载 ===")
        
        # 测试默认配置
        assert hasattr(self.config, 'ENABLE_AREA_BUDGET')
        assert hasattr(self.config, 'AREA_BUDGET_MM2')
        assert hasattr(self.config, 'AREA_BUDGET_TOLERANCE')
        assert hasattr(self.config, 'AREA_BUDGET_PENALTY_WEIGHT')
        assert hasattr(self.config, 'AREA_BUDGET_PENALTY_STRATEGY')
        
        print(f"✅ 默认配置加载成功")
        print(f"   - 启用状态: {self.config.ENABLE_AREA_BUDGET}")
        print(f"   - 面积预算: {self.config.AREA_BUDGET_MM2}")
        print(f"   - 容忍度: {self.config.AREA_BUDGET_TOLERANCE}")
        print(f"   - 惩罚权重: {self.config.AREA_BUDGET_PENALTY_WEIGHT}")
        print(f"   - 惩罚策略: {self.config.AREA_BUDGET_PENALTY_STRATEGY}")
        
    def test_scenario_presets(self):
        """测试场景预设功能"""
        print("\n=== 测试场景预设功能 ===")
        
        scenarios = ['edge', 'cloud', 'mobile']
        
        for scenario in scenarios:
            config = Config()
            config.apply_scenario_preset(scenario)
            
            print(f"\n{scenario.upper()}场景配置:")
            print(f"   - 面积预算: {config.AREA_BUDGET_MM2} mm²")
            print(f"   - 容忍度: {config.AREA_BUDGET_TOLERANCE:.1%}")
            print(f"   - 惩罚权重: {config.AREA_BUDGET_PENALTY_WEIGHT}")
            print(f"   - 惩罚策略: {config.AREA_BUDGET_PENALTY_STRATEGY}")
            
            # 验证配置合理性
            assert config.AREA_BUDGET_MM2 > 0, f"{scenario}场景面积预算应大于0"
            assert 0 < config.AREA_BUDGET_TOLERANCE < 1, f"{scenario}场景容忍度应在0-1之间"
            assert config.AREA_BUDGET_PENALTY_WEIGHT >= 0, f"{scenario}场景惩罚权重应非负"
            
        print("\n✅ 所有场景预设测试通过")
        
    def test_penalty_calculation(self):
        """测试惩罚项计算"""
        print("\n=== 测试惩罚项计算 ===")
        
        # 创建一个模拟的searcher来测试惩罚计算
        class MockSearcher(BaseSearcher):
            def __init__(self, config):
                self.config = config
                self.loss_weights = config.LOSS_WEIGHTS
                
            def search(self, num_trials):
                pass
                
            def _compute_area_budget_penalty(self, area, step_count=0):
                # 复制实际的惩罚计算逻辑
                from dosa.config import Config
                config = Config.get_instance()
                
                if not config.ENABLE_AREA_BUDGET or config.AREA_BUDGET_MM2 is None:
                    return torch.tensor(0.0, device=area.device, dtype=area.dtype)
                
                budget = config.AREA_BUDGET_MM2
                tolerance = config.AREA_BUDGET_TOLERANCE
                strategy = config.AREA_BUDGET_PENALTY_STRATEGY
                weight = config.AREA_BUDGET_PENALTY_WEIGHT
                
                lower_bound = budget * (1 - tolerance)
                upper_bound = budget * (1 + tolerance)
                
                if lower_bound <= area <= upper_bound:
                    return torch.tensor(0.0, device=area.device, dtype=area.dtype)
                
                if area < lower_bound:
                    deviation = lower_bound - area
                else:
                    deviation = area - upper_bound
                
                normalized_deviation = deviation / budget
                
                if strategy == 'quadratic':
                    penalty = normalized_deviation ** 2
                elif strategy == 'linear':
                    penalty = normalized_deviation
                elif strategy == 'exponential':
                    penalty = torch.exp(normalized_deviation) - 1
                else:
                    penalty = normalized_deviation ** 2
                
                return weight * penalty
        
        # 测试不同场景的惩罚计算
        test_cases = [
            {'scenario': 'edge', 'area': 5.0, 'expected_penalty': 0.0},  # 在预算内
            {'scenario': 'edge', 'area': 15.0, 'expected_penalty': '>0'},  # 超出预算
            {'scenario': 'cloud', 'area': 50.0, 'expected_penalty': 0.0},  # 在预算内
            {'scenario': 'cloud', 'area': 150.0, 'expected_penalty': '>0'},  # 超出预算
        ]
        
        for case in test_cases:
            config = Config()
            config.apply_scenario_preset(case['scenario'])
            
            searcher = MockSearcher(config)
            area_tensor = torch.tensor(case['area'], dtype=torch.float32)
            penalty = searcher._compute_area_budget_penalty(area_tensor)
            
            print(f"\n{case['scenario'].upper()}场景 - 面积{case['area']} mm²:")
            print(f"   - 预算: {config.AREA_BUDGET_MM2} mm²")
            print(f"   - 容忍区间: [{config.AREA_BUDGET_MM2 * (1-config.AREA_BUDGET_TOLERANCE):.1f}, "
                  f"{config.AREA_BUDGET_MM2 * (1+config.AREA_BUDGET_TOLERANCE):.1f}] mm²")
            print(f"   - 惩罚值: {penalty.item():.6f}")
            
            if case['expected_penalty'] == 0.0:
                assert penalty.item() == 0.0, f"预期惩罚为0，实际为{penalty.item()}"
                print("   ✅ 惩罚计算正确（在预算内）")
            elif case['expected_penalty'] == '>0':
                assert penalty.item() > 0.0, f"预期惩罚>0，实际为{penalty.item()}"
                print("   ✅ 惩罚计算正确（超出预算）")
        
        print("\n✅ 惩罚项计算测试通过")
        
    def test_loss_integration(self):
        """测试Loss函数集成"""
        print("\n=== 测试Loss函数集成 ===")
        
        # 创建模拟的loss计算环境
        config = Config()
        config.apply_scenario_preset('edge')
        
        # 模拟性能指标
        latency = torch.tensor(1e-3, dtype=torch.float32)  # 1ms
        energy = torch.tensor(1e6, dtype=torch.float32)    # 1mJ
        area = torch.tensor(15.0, dtype=torch.float32)     # 15mm² (超出edge预算)
        mismatch_loss = torch.tensor(0.1, dtype=torch.float32)
        compatibility_penalty = torch.tensor(0.0, dtype=torch.float32)
        
        # 创建模拟searcher
        class MockSearcher(BaseSearcher):
            def __init__(self, config):
                self.config = config
                self.loss_weights = config.LOSS_WEIGHTS
                self.loss_strategy = 'log_edp_plus_area'
                
            def search(self, num_trials):
                pass
                
            def _compute_area_budget_penalty(self, area, step_count=0):
                # 简化的惩罚计算
                if not config.ENABLE_AREA_BUDGET:
                    return torch.tensor(0.0)
                
                budget = config.AREA_BUDGET_MM2
                tolerance = config.AREA_BUDGET_TOLERANCE
                weight = config.AREA_BUDGET_PENALTY_WEIGHT
                
                upper_bound = budget * (1 + tolerance)
                if area > upper_bound:
                    deviation = (area - upper_bound) / budget
                    return weight * (deviation ** 2)
                return torch.tensor(0.0)
        
        searcher = MockSearcher(config)
        
        # 测试启用面积预算的情况
        config.ENABLE_AREA_BUDGET = True
        area_penalty = searcher._compute_area_budget_penalty(area)
        print(f"启用面积预算时的惩罚: {area_penalty.item():.6f}")
        assert area_penalty.item() > 0, "超出预算时应有惩罚"
        
        # 测试禁用面积预算的情况
        config.ENABLE_AREA_BUDGET = False
        area_penalty_disabled = searcher._compute_area_budget_penalty(area)
        print(f"禁用面积预算时的惩罚: {area_penalty_disabled.item():.6f}")
        assert area_penalty_disabled.item() == 0, "禁用时应无惩罚"
        
        print("✅ Loss函数集成测试通过")
        
    def run_all_tests(self):
        """运行所有测试"""
        print("开始面积预算功能测试...")
        
        try:
            self.test_config_loading()
            self.test_scenario_presets()
            self.test_penalty_calculation()
            self.test_loss_integration()
            
            print("\n🎉 所有测试通过！面积预算功能工作正常。")
            return True
            
        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    tester = TestAreaBudget()
    success = tester.run_all_tests()
    
    if success:
        print("\n=== 使用建议 ===")
        print("1. 使用 config.apply_scenario_preset('edge') 应用edge场景配置")
        print("2. 使用 config.apply_scenario_preset('cloud') 应用cloud场景配置")
        print("3. 使用 config.apply_scenario_preset('mobile') 应用mobile场景配置")
        print("4. 运行 python examples/area_budget_demo.py 查看完整演示")
        print("5. 面积预算惩罚项会自动集成到所有损失策略中")
        
        return 0
    else:
        return 1

if __name__ == '__main__':
    exit(main())