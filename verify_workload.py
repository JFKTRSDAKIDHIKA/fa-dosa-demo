#!/usr/bin/env python3

import sys
sys.path.append('/root/fa-dosa-demo')

import yaml
from pathlib import Path
from run import parse_onnx_to_graph

def verify_workload_loading():
    print("=== 验证工作负载加载情况 ===")
    
    # 1. 检查配置文件
    config_path = "/root/fa-dosa-demo/configs/act1.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    workload_name = config['workload']['name']
    print(f"配置文件中的工作负载名称: {workload_name}")
    
    # 2. 解析模型名称（按照baselines.py中的逻辑）
    model_name = workload_name.split('_')[0] if '_' in workload_name else workload_name
    print(f"解析出的模型名称: {model_name}")
    
    # 3. 检查ONNX文件是否存在
    onnx_path = f"/root/fa-dosa-demo/onnx_models/{model_name}.onnx"
    onnx_exists = Path(onnx_path).exists()
    print(f"ONNX文件路径: {onnx_path}")
    print(f"ONNX文件是否存在: {onnx_exists}")
    
    # 4. 尝试解析ONNX模型
    if onnx_exists:
        try:
            print("\n=== 尝试解析ONNX模型 ===")
            graph = parse_onnx_to_graph(model_name)
            print(f"成功解析模型: {model_name}")
            print(f"图中的层数: {len(graph.layers)}")
            print(f"融合组数量: {len(graph.fusion_groups)}")
            
            # 显示前几层的信息
            print("\n前5层信息:")
            for i, layer in enumerate(graph.layers[:5]):
                print(f"  层{i+1}: {layer.name} (类型: {layer.op_type})")
                
        except Exception as e:
            print(f"解析ONNX模型时出错: {e}")
    else:
        print("ONNX文件不存在，无法验证模型加载")
    
    # 5. 检查baseline实验的配置
    print("\n=== Baseline实验配置 ===")
    baselines = config.get('baselines', [])
    print(f"配置的baseline实验: {baselines}")
    
    return onnx_exists and model_name == 'resnet18'

if __name__ == "__main__":
    success = verify_workload_loading()
    if success:
        print("\n✅ 验证成功: ResNet-18工作负载配置正确")
    else:
        print("\n❌ 验证失败: ResNet-18工作负载配置有问题")