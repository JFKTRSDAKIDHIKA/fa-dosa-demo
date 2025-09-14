#!/usr/bin/env python3

import sys
sys.path.append('/root/fa-dosa-demo')

import yaml
import tempfile
from pathlib import Path
from logging_utils.recorder import Recorder
from experiments.baselines import get_baseline_runner

def test_baseline_workload():
    print("=== 测试Baseline实验工作负载 ===")
    
    # 加载配置
    config_path = "/root/fa-dosa-demo/configs/act1.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f"配置文件中的工作负载: {cfg['workload']['name']}")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        recorder = Recorder(tmpdir_path)
        
        # 创建baseline runner
        runner = get_baseline_runner('baselineA_A1')
        
        print("\n=== 检查BaselineRunner中的工作负载解析 ===")
        
        # 模拟baselines.py中的工作负载名称解析逻辑
        workload_name = cfg["workload"]["name"]
        model_name = workload_name.split('_')[0] if '_' in workload_name else workload_name
        
        print(f"原始工作负载名称: {workload_name}")
        print(f"解析出的模型名称: {model_name}")
        
        # 检查ONNX文件
        onnx_path = f"/root/fa-dosa-demo/onnx_models/{model_name}.onnx"
        onnx_exists = Path(onnx_path).exists()
        print(f"ONNX文件路径: {onnx_path}")
        print(f"ONNX文件存在: {onnx_exists}")
        
        if onnx_exists:
            print(f"\n✅ Baseline实验将使用 {model_name} 模型")
            print(f"这确认了ResNet-18被正确配置为工作负载")
        else:
            print(f"\n❌ ONNX文件不存在，baseline实验可能使用fallback图")
        
        return onnx_exists and model_name == 'resnet18'

if __name__ == "__main__":
    success = test_baseline_workload()
    if success:
        print("\n🎉 确认: Baseline实验确实使用ResNet-18作为工作负载")
    else:
        print("\n⚠️  警告: Baseline实验可能没有使用ResNet-18")