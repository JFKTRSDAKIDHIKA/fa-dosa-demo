import torch
import numpy as np
import random
from typing import Dict, Any, List, Tuple, Union
from .utils import get_divisors


class SearchSpace:
    """
    DSE参数空间管理类
    
    负责定义整个设计空间探索的参数空间，包括硬件参数、映射参数和融合参数。
    提供参数采样、扁平化转换等功能，为黑盒优化算法提供统一接口。
    """
    
    def __init__(self, graph):
        """
        初始化搜索空间
        
        Args:
            graph: ComputationGraph实例
        """
        self.graph = graph
        self.dimensions = []
        self._build_parameter_space()
    
    def _build_parameter_space(self):
        """
        构建完整的参数空间定义
        """
        # 1. 硬件参数空间
        self._add_hardware_dimensions()
        
        # 2. 映射参数空间
        self._add_mapping_dimensions()
        
        # 3. 融合参数空间
        self._add_fusion_dimensions()
    
    def _add_hardware_dimensions(self):
        """
        添加硬件参数维度 - 扩大搜索范围
        """
        # 大幅扩大PE数量范围
        self.dimensions.append({
            'name': 'num_pes',
            'type': 'integer_square',
            'range': (2, 64),  # sqrt范围，实际PE数量为4-4096
            'description': 'Number of PEs (will be squared)'
        })
        
        # 大幅扩大Buffer大小范围
        buffer_configs = [
            ('l0_registers_size_kb', (0.05, 50.0)),     # 0.05KB-50KB
            ('l1_accumulator_size_kb', (0.1, 200.0)),   # 0.1KB-200KB
            ('l2_scratchpad_size_kb', (0.5, 2000.0))    # 0.5KB-2MB
        ]
        
        for name, (min_val, max_val) in buffer_configs:
            self.dimensions.append({
                'name': name,
                'type': 'log_uniform',
                'range': (min_val, max_val),
                'description': f'Buffer size in KB (log-uniform)'
            })
    
    def _add_mapping_dimensions(self):
        """
        添加映射参数维度
        """
        # 只为on-chip存储层级定义映射参数
        on_chip_levels = ['L0_Registers', 'L1_Accumulator', 'L2_Scratchpad']
        
        for dim_name, dim_size in self.graph.problem_dims.items():
            # 获取该维度的所有有效约数
            divisors = get_divisors(dim_size)
            divisor_list = [int(d.item()) for d in divisors]
            
            for level_name in on_chip_levels:
                # Temporal tiling factor
                self.dimensions.append({
                    'name': f'{dim_name}_{level_name}_temporal',
                    'type': 'categorical',
                    'categories': divisor_list,
                    'description': f'Temporal tiling factor for {dim_name} at {level_name}'
                })
                
                # Spatial tiling factor
                self.dimensions.append({
                    'name': f'{dim_name}_{level_name}_spatial',
                    'type': 'categorical',
                    'categories': divisor_list,
                    'description': f'Spatial tiling factor for {dim_name} at {level_name}'
                })
    
    def _add_fusion_dimensions(self):
        """
        添加融合参数维度
        """
        for i, group in enumerate(self.graph.fusion_groups):
            self.dimensions.append({
                'name': f'fusion_group_{i}',
                'type': 'categorical',
                'categories': [0, 1],  # 0: 不融合, 1: 融合
                'description': f'Fusion decision for group {i}: {group}'
            })
    
    def sample(self) -> Dict[str, Any]:
        """
        从参数空间中随机采样一个配置
        
        Returns:
            结构化的参数字典
        """
        params = {}
        
        for dim in self.dimensions:
            name = dim['name']
            dim_type = dim['type']
            
            if dim_type == 'integer_square':
                # 特殊处理：采样sqrt值，然后平方
                min_sqrt, max_sqrt = dim['range']
                sqrt_val = random.randint(min_sqrt, max_sqrt)
                params[name] = sqrt_val * sqrt_val
                
            elif dim_type == 'log_uniform':
                # 对数均匀采样
                min_val, max_val = dim['range']
                log_min, log_max = np.log(min_val), np.log(max_val)
                params[name] = np.exp(random.uniform(log_min, log_max))
                
            elif dim_type == 'categorical':
                # 类别采样
                params[name] = random.choice(dim['categories'])
                
            else:
                raise ValueError(f"Unknown dimension type: {dim_type}")
        
        # 处理融合参数：转换为fusion_logits格式
        fusion_logits = []
        fusion_keys = [d['name'] for d in self.dimensions if d['name'].startswith('fusion_group_')]
        
        for key in sorted(fusion_keys):  # 确保顺序一致
            decision = params.pop(key)
            # 转换为logits：0 -> -2.0 (不融合), 1 -> 2.0 (融合)
            logit = 2.0 if decision == 1 else -2.0
            fusion_logits.append(logit)
        
        if fusion_logits:
            params['fusion_logits'] = fusion_logits
        
        return params
    
    def to_flat(self, params_dict: Dict[str, Any]) -> List[float]:
        """
        将结构化参数字典转换为扁平化数值列表
        
        Args:
            params_dict: 结构化参数字典
            
        Returns:
            扁平化的数值列表
        """
        flat_params = []
        
        # 处理融合参数：从fusion_logits提取
        fusion_decisions = {}
        if 'fusion_logits' in params_dict:
            fusion_logits = params_dict['fusion_logits']
            for i, logit in enumerate(fusion_logits):
                # 转换logits为决策：正值->1, 负值->0
                decision = 1 if logit > 0 else 0
                fusion_decisions[f'fusion_group_{i}'] = decision
        
        for dim in self.dimensions:
            name = dim['name']
            dim_type = dim['type']
            
            if name.startswith('fusion_group_'):
                # 融合参数：使用从fusion_logits提取的决策
                decision = fusion_decisions.get(name, 0)
                categories = dim['categories']
                index = categories.index(decision)
                flat_params.append(float(index))
                
            elif dim_type == 'categorical':
                # 其他类别参数：查找索引
                value = params_dict[name]
                categories = dim['categories']
                index = categories.index(value)
                flat_params.append(float(index))
                
            elif dim_type == 'integer_square':
                # 平方数参数：存储sqrt值
                value = params_dict[name]
                sqrt_val = int(np.sqrt(value))
                flat_params.append(float(sqrt_val))
                
            elif dim_type == 'log_uniform':
                # 连续参数：直接存储
                flat_params.append(float(params_dict[name]))
                
            else:
                raise ValueError(f"Unknown dimension type: {dim_type}")
        
        return flat_params
    
    def from_flat(self, flat_params: List[float]) -> Dict[str, Any]:
        """
        将扁平化数值列表转换为结构化参数字典
        
        Args:
            flat_params: 扁平化的数值列表
            
        Returns:
            结构化的参数字典
        """
        if len(flat_params) != len(self.dimensions):
            raise ValueError(f"Expected {len(self.dimensions)} parameters, got {len(flat_params)}")
        
        params = {}
        fusion_logits = []
        
        for i, dim in enumerate(self.dimensions):
            name = dim['name']
            dim_type = dim['type']
            value = flat_params[i]
            
            if dim_type == 'integer_square':
                # 平方数参数：平方sqrt值
                sqrt_val = int(round(value))
                params[name] = sqrt_val * sqrt_val
                
            elif dim_type == 'log_uniform':
                # 连续参数：直接使用
                params[name] = float(value)
                
            elif dim_type == 'categorical':
                # 类别参数：根据索引查找值
                index = int(round(value))
                categories = dim['categories']
                index = max(0, min(index, len(categories) - 1))  # 边界检查
                
                if name.startswith('fusion_group_'):
                    # 融合参数：转换为logits
                    decision = categories[index]
                    logit = 2.0 if decision == 1 else -2.0
                    fusion_logits.append(logit)
                else:
                    # 其他类别参数
                    params[name] = categories[index]
                    
            else:
                raise ValueError(f"Unknown dimension type: {dim_type}")
        
        # 添加融合参数
        if fusion_logits:
            params['fusion_logits'] = fusion_logits
        
        return params
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """
        获取所有参数的边界，用于黑盒优化算法
        
        Returns:
            参数边界列表
        """
        bounds = []
        
        for dim in self.dimensions:
            dim_type = dim['type']
            
            if dim_type == 'integer_square':
                min_sqrt, max_sqrt = dim['range']
                bounds.append((float(min_sqrt), float(max_sqrt)))
                
            elif dim_type == 'log_uniform':
                min_val, max_val = dim['range']
                bounds.append((min_val, max_val))
                
            elif dim_type == 'categorical':
                # 类别参数：索引范围
                num_categories = len(dim['categories'])
                bounds.append((0.0, float(num_categories - 1)))
                
            else:
                raise ValueError(f"Unknown dimension type: {dim_type}")
        
        return bounds
    
    def get_dimension_info(self) -> List[Dict[str, Any]]:
        """
        获取参数空间的详细信息
        
        Returns:
            参数维度信息列表
        """
        return self.dimensions.copy()
    
    def validate_params(self, params_dict: Dict[str, Any]) -> bool:
        """
        验证参数字典的有效性
        
        Args:
            params_dict: 参数字典
            
        Returns:
            是否有效
        """
        try:
            # 尝试转换为扁平格式再转换回来
            flat_params = self.to_flat(params_dict)
            reconstructed = self.from_flat(flat_params)
            return True
        except Exception:
            return False