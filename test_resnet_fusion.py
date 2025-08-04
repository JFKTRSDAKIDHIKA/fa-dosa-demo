#!/usr/bin/env python3
"""
测试ResNet残差连接融合功能
"""

import torch
import torch.nn as nn
from dosa.utils import ComputationGraph
from dosa.performance_model import HighFidelityPerformanceModel
from dosa.hardware_parameters import HardwareParameters
from dosa.mapping import FineGrainedMapping
from dosa.config import Config

def create_resnet_test_graph():
    """创建一个简单的ResNet残差块测试图"""
    graph = ComputationGraph()
    
    # 添加层
    graph.add_layer('input', {'N': 1, 'C': 64, 'P': 56, 'Q': 56}, 'Input', [], ['input_out'])
    
    graph.add_layer('conv1', {'N': 1, 'C': 64, 'K': 64, 'P': 56, 'Q': 56, 'R': 3, 'S': 3}, 'Conv', ['input_out'], ['conv1_out'])
    
    graph.add_layer('bn1', {'N': 1, 'C': 64, 'P': 56, 'Q': 56}, 'BatchNormalization', ['conv1_out'], ['bn1_out'])
    
    graph.add_layer('relu1', {'N': 1, 'C': 64, 'P': 56, 'Q': 56}, 'ReLU', ['bn1_out'], ['relu1_out'])
    
    graph.add_layer('add1', {'N': 1, 'C': 64, 'P': 56, 'Q': 56}, 'Add', ['relu1_out', 'input_out'], ['add1_out'])
    
    # 连接已通过inputs/outputs参数定义
    
    return graph

def test_resnet_fusion():
    """测试ResNet残差连接融合功能"""
    print("=== 测试ResNet残差连接融合功能 ===")
    
    # 创建测试图
    graph = create_resnet_test_graph()
    
    # 识别残差连接模式
    skip_patterns = graph.find_skip_connection_patterns()
    print(f"识别到的残差连接模式: {skip_patterns}")
    
    # 添加融合组
    from run import _add_fusion_groups
    layer_sequence = [(name, layer['type']) for name, layer in graph.layers.items()]
    _add_fusion_groups(graph, layer_sequence)
    print(f"融合组: {graph.fusion_groups}")
    
    # 创建配置和硬件参数
    config = Config.get_instance()
    hw_params = HardwareParameters()
    
    # 创建简单的层次结构用于测试
    hierarchy = [
        {'name': 'L0_Registers', 'type': 'buffer'},
        {'name': 'L1_Accumulator', 'type': 'buffer'},
        {'name': 'L2_Scratchpad', 'type': 'buffer'}
    ]
    
    # 创建问题维度字典 - 使用性能模型期望的维度名称
    problem_dims = {
        'N': 1,      # batch size
        'C': 3,      # input channels
        'K': 64,     # output channels
        'P': 224,    # input height
        'Q': 224,    # input width
        'R': 7,      # kernel height
        'S': 7       # kernel width
    }
    
    mapping = FineGrainedMapping(problem_dims, hierarchy)
    
    # 创建性能模型
    perf_model = HighFidelityPerformanceModel(config)
    
    # 测试性能评估
    try:
        latency, energy, area, mismatch_loss, compatibility_penalty = perf_model(
            graph, hw_params, mapping
        )
        
        print(f"\n=== 性能评估结果 ===")
        print(f"延迟: {latency.item():.2e}")
        print(f"能耗: {energy.item():.2e}")
        print(f"面积: {area.item():.2f} mm²")
        print(f"Buffer不匹配损失: {mismatch_loss.item():.2e}")
        print(f"兼容性惩罚: {compatibility_penalty.item():.2e}")
        
        print("\n✅ ResNet残差连接融合功能测试成功！")
        return True
        
    except Exception as e:
        import traceback
        print(f"\n❌ 测试失败: {e}")
        print("\n完整错误信息:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_resnet_fusion()
    exit(0 if success else 1)