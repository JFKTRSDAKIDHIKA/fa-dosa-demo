#!/usr/bin/env python3
"""测试baseline是否正确使用ResNet-18工作负载"""

import yaml
import tempfile
import os
from pathlib import Path
from experiments.baselines import get_baseline_runner
from logging_utils.recorder import Recorder

def test_baseline_workload():
    """测试baseline是否使用正确的工作负载"""
    # 加载配置
    with open('configs/act1.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 设置最小测试参数
    cfg['shared'] = cfg.get('shared', {})
    cfg['shared']['num_trials'] = 1
    
    print(f"配置中的工作负载名称: {cfg.get('workload', {}).get('name')}")
    
    # 解析模型名称
    workload_name = cfg.get('workload', {}).get('name', 'resnet18')
    model_name = workload_name.split('_')[0] if '_' in workload_name else workload_name
    print(f"解析出的模型名称: {model_name}")
    print(f"ONNX文件是否存在: {os.path.exists(f'onnx_models/{model_name}.onnx')}")
    
    # 测试baseline运行器
    with tempfile.TemporaryDirectory() as tmpdir:
        rec = Recorder(Path(tmpdir))
        runner = get_baseline_runner('baselineA_A1')
        
        print("\n开始测试baseline实验...")
        try:
            runner.run(cfg, 42, rec)
            print("✓ 测试成功！baseline现在应该使用ResNet-18工作负载")
            return True
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            return False

if __name__ == "__main__":
    test_baseline_workload()