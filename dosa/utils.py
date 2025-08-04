import torch
import torch.nn as nn
import json
from typing import Dict, List, Tuple

# Memoization cache for divisors
_divisors_cache = {}

def get_divisors(n: int) -> torch.Tensor:
    """
    Get all integer divisors of n as a sorted torch.Tensor.
    Results are memoized to avoid re-computation.
    """
    if n in _divisors_cache:
        return _divisors_cache[n]
    
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort()
    divisors_tensor = torch.tensor(divisors, dtype=torch.float32)
    _divisors_cache[n] = divisors_tensor
    return divisors_tensor

class ComputationGraph:
    def __init__(self):
        self.layers = {}
        self.edges = []
        self.fusion_groups = []
        self.problem_dims = {'N':1, 'C':1, 'K':1, 'P':1, 'Q':1, 'R':1, 'S':1}
        self.tensor_to_layer = {}  # 映射张量名到产生它的层
        
    def add_layer(self, name, dims, op_type, inputs=None, outputs=None):
        """添加层到计算图中，支持非线性拓扑结构
        
        Args:
            name: 层名称
            dims: 维度字典
            op_type: 操作类型
            inputs: 输入张量名列表
            outputs: 输出张量名列表
        """
        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = []
            
        self.layers[name] = {
            'dims': dims, 
            'type': op_type,
            'inputs': inputs,
            'outputs': outputs
        }
        
        # 更新张量到层的映射
        for output_tensor in outputs:
            self.tensor_to_layer[output_tensor] = name
            
        # 更新问题维度
        for d, v in dims.items():
            self.problem_dims[d] = max(self.problem_dims[d], v)
            
    def add_fusion_group(self, group):
        self.fusion_groups.append(group)
        
    def get_layer_inputs(self, layer_name):
        """获取指定层的输入层列表"""
        if layer_name not in self.layers:
            return []
        
        input_layers = []
        for input_tensor in self.layers[layer_name]['inputs']:
            if input_tensor in self.tensor_to_layer:
                input_layers.append(self.tensor_to_layer[input_tensor])
        return input_layers
        
    def get_layer_outputs(self, layer_name):
        """获取指定层的输出层列表"""
        if layer_name not in self.layers:
            return []
            
        output_layers = []
        layer_outputs = self.layers[layer_name]['outputs']
        
        for other_layer_name, layer_info in self.layers.items():
            if other_layer_name == layer_name:
                continue
            for input_tensor in layer_info['inputs']:
                if input_tensor in layer_outputs:
                    output_layers.append(other_layer_name)
                    break
        return output_layers
        
    def find_skip_connection_patterns(self):
        """识别ResNet残差块模式"""
        patterns = []
        
        # 查找Add节点
        for layer_name, layer_info in self.layers.items():
            if layer_info['type'] == 'Add':
                input_layers = self.get_layer_inputs(layer_name)
                if len(input_layers) >= 2:
                    # 分析两个输入路径
                    main_path, skip_path = self._analyze_add_inputs(layer_name, input_layers)
                    if main_path and skip_path:
                        patterns.append({
                            'add_node': layer_name,
                            'main_path': main_path,
                            'skip_path': skip_path,
                            'pattern_type': 'skip_connection'
                        })
        return patterns
        
    def _analyze_add_inputs(self, add_layer, input_layers):
        """分析Add节点的输入，识别主路径和跳跃路径"""
        # 简化的启发式：较长的路径为主路径，较短的为跳跃路径
        paths = []
        for input_layer in input_layers:
            path = self._trace_path_backward(input_layer)
            paths.append((len(path), path, input_layer))
            
        if len(paths) >= 2:
            paths.sort(key=lambda x: x[0], reverse=True)
            main_path = paths[0][1]  # 最长路径
            skip_path = [paths[1][2]]  # 跳跃连接通常是单个层
            return main_path, skip_path
            
        return None, None
        
    def _trace_path_backward(self, layer_name, visited=None):
        """向后追踪路径"""
        if visited is None:
            visited = set()
            
        if layer_name in visited:
            return []
            
        visited.add(layer_name)
        path = [layer_name]
        
        input_layers = self.get_layer_inputs(layer_name)
        if input_layers:
            # 选择第一个输入继续追踪（简化处理）
            sub_path = self._trace_path_backward(input_layers[0], visited.copy())
            path.extend(sub_path)
            
        return path

class FusionParameters(nn.Module):
    def __init__(self, graph):
        super().__init__()
        num_groups = len(graph.fusion_groups)
        self.fusion_logits = nn.Parameter(torch.randn(num_groups, 1))

    def get_fusion_decisions(self):
        return torch.sigmoid(self.fusion_logits) > 0.5
    
    def get_fusion_decisions_serializable(self, graph):
        """Return a JSON-serializable representation of fusion decisions."""
        decisions = self.get_fusion_decisions()
        fusion_decisions = []
        
        for i, group in enumerate(graph.fusion_groups):
            fusion_decisions.append({
                "group": group,
                "fused": bool(decisions[i].item())
            })
        
        return fusion_decisions

def calculate_macs(dims):
    output_elements = dims.get('N', 1) * dims.get('K', 1) * dims.get('P', 1) * dims.get('Q', 1)
    ops_per_element = dims.get('C', 1) * dims.get('R', 1) * dims.get('S', 1)
    return output_elements * ops_per_element

def save_configuration_to_json(hw_params, projected_mapping, fusion_decisions, file_path="final_configuration.json"):
    # Helper function to convert tensors to native Python types
    def to_native_types(data):
        if isinstance(data, dict):
            return {k: to_native_types(v) for k, v in data.items()}
        if isinstance(data, list):
            return [to_native_types(i) for i in data]
        if isinstance(data, torch.Tensor):
            return data.item() if data.numel() == 1 else data.tolist()
        return data
    
    config_dict = {
        "num_pes": hw_params.get_projected_num_pes().item(),
        "l0_size_kb": hw_params.get_buffer_size_kb('L0_Registers').item(),
        "l1_size_kb": hw_params.get_buffer_size_kb('L1_Accumulator').item(),
        "l2_size_kb": hw_params.get_buffer_size_kb('L2_Scratchpad').item(),
        "mapping": to_native_types(projected_mapping),
        "fusion_strategy": fusion_decisions
    }
    with open(file_path, 'w') as f:
        json.dump(config_dict, f, indent=4)


class OptimizationLogger:
    """A simple logger to save optimization snapshots to a JSON Lines file."""

    def __init__(self, log_path="optimization_log.jsonl"):
        """Initializes the logger and opens the log file for writing."""
        self.log_path = log_path
        # Open the file in 'w' mode to clear previous logs on a new run
        self.log_file = open(self.log_path, 'w')

    def log_step(self, step_info: dict):
        """
        Logs a single dictionary of information as a JSON string on a new line.
        Converts PyTorch tensors to basic Python types for serialization.
        """
        # A helper function to recursively convert tensors to numbers
        def to_native_types(data):
            if isinstance(data, dict):
                return {k: to_native_types(v) for k, v in data.items()}
            if isinstance(data, list):
                return [to_native_types(i) for i in data]
            if isinstance(data, torch.Tensor):
                return data.item()
            return data

        serializable_info = to_native_types(step_info)
        self.log_file.write(json.dumps(serializable_info) + '\n')
        # Flush the buffer to ensure data is written immediately
        self.log_file.flush()

    def close(self):
        """Closes the log file."""
        self.log_file.close()