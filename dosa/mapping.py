import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

def get_divisors(n):
    """Helper function to find all divisors of a given number."""
    divisors = []
    for i in range(1, int(n) + 1):
        if n % i == 0:
            divisors.append(i)
    return divisors

class GumbelSoftmaxDivisor(nn.Module):
    """
    Gumbel-Softmax based divisor selection that enforces discrete choices in forward pass
    while maintaining differentiable gradients. Forces optimizer to face true discrete costs.
    """
    def __init__(self, sharpness_alpha=1.0, initial_tau=1.0, tau_decay_rate=0.999):
        super().__init__()
        self.sharpness_alpha = sharpness_alpha
        self.tau_decay_rate = tau_decay_rate
        # Use register_buffer to store tau as non-parameter tensor
        self.register_buffer('tau', torch.tensor(initial_tau))
    
    def forward(self, continuous_factor, problem_dim):
        """
        Args:
            continuous_factor: torch.Tensor, continuous learnable parameter
            problem_dim: torch.Tensor or int, total size of problem dimension
        
        Returns:
            selected_divisor: torch.Tensor, discrete divisor selection (scalar)
        """
        # Clamp the continuous factor to a minimum of 1.0
        continuous_factor = torch.clamp(continuous_factor, min=1.0)
        
        # Ensure problem_dim is an integer
        if isinstance(problem_dim, torch.Tensor):
            problem_dim_int = int(problem_dim.item())
        else:
            problem_dim_int = int(problem_dim)
        
        # Find all valid divisors
        divisors_list = get_divisors(problem_dim_int)
        valid_divisors = torch.tensor(divisors_list, dtype=continuous_factor.dtype, device=continuous_factor.device)
        
        # Calculate logits (unnormalized log-probabilities)
        logits = -self.sharpness_alpha * (continuous_factor.squeeze().unsqueeze(-1) - valid_divisors).pow(2)
        
        if self.training:
            # Training mode: Use Gumbel-Softmax with hard=True for discrete selection
            one_hot_selection = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
        else:
            # Evaluation mode: Deterministic argmax selection
            best_idx = torch.argmax(logits, dim=-1)
            one_hot_selection = F.one_hot(best_idx, num_classes=logits.shape[-1]).float()
        
        # Calculate selected divisor using one-hot selection
        selected_divisor = torch.sum(one_hot_selection * valid_divisors, dim=-1)
        
        return selected_divisor
    
    def anneal_tau(self):
        """
        Anneal the temperature parameter towards a minimum value.
        Should be called periodically during training.
        """
        self.tau.data = torch.clamp(self.tau.data * self.tau_decay_rate, min=0.1)

class SoftmaxWeightedDivisor(nn.Module):
    """
    Differentiable softmax-weighted projection from continuous factor to valid divisor.
    Solves the dead gradient zone problem by providing smooth gradients.
    """
    def __init__(self, sharpness_alpha=1.0):
        super().__init__()
        self.sharpness_alpha = sharpness_alpha
    
    def forward(self, continuous_factor, problem_dim):
        """
        Args:
            continuous_factor: torch.Tensor, continuous learnable parameter
            problem_dim: torch.Tensor or int, total size of problem dimension
        
        Returns:
            effective_divisor: torch.Tensor, smooth differentiable approximation (scalar)
        """
        # Clamp the continuous factor to a minimum of 1.0 (like original)
        continuous_factor = torch.clamp(continuous_factor, min=1.0)
        
        # Ensure problem_dim is an integer
        if isinstance(problem_dim, torch.Tensor):
            problem_dim_int = int(problem_dim.item())
        else:
            problem_dim_int = int(problem_dim)
        
        # Find all valid divisors
        divisors_list = get_divisors(problem_dim_int)
        valid_divisors = torch.tensor(divisors_list, dtype=continuous_factor.dtype, device=continuous_factor.device)
        
        # Calculate similarity scores (negative squared distance)
        # Ensure continuous_factor is treated as scalar for broadcasting
        scores = -self.sharpness_alpha * (continuous_factor.squeeze() - valid_divisors).pow(2)
        
        # Compute softmax probabilities
        probabilities = torch.softmax(scores, dim=-1)
        
        # Calculate weighted average - this should return a scalar tensor
        effective_divisor = torch.sum(probabilities * valid_divisors)
        
        return effective_divisor

class ProjectToNearestDivisor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, continuous_factor, problem_dim):
        # Clamp the continuous factor to a minimum of 1.0
        continuous_factor = torch.clamp(continuous_factor, min=1.0)
        # Project to the nearest integer divisor
        divisors = torch.arange(1, problem_dim.item() + 1, device=continuous_factor.device)
        valid_divisors = divisors[problem_dim % divisors == 0]
        
        # Find the closest valid divisor
        abs_diff = torch.abs(valid_divisors - continuous_factor)
        nearest_divisor = valid_divisors[torch.argmin(abs_diff)]
        
        return nearest_divisor

    @staticmethod
    def backward(ctx, grad_output):
        # Pass the gradient through, ensuring correct shape
        # grad_output might be a scalar, but we need to match input shape
        if grad_output.numel() == 1 and grad_output.dim() == 0:
            grad_output = grad_output.unsqueeze(0)
        return grad_output, None

class FineGrainedMapping(nn.Module):
    """
    NEW: 细粒度的映射参数化模块，替代了旧的LearnableConvReluTemplate。
    """
    def __init__(self, problem_dims: Dict[str, int], hierarchy: List[Dict]):
        super().__init__()
        self.dims = problem_dims
        self.hierarchy = hierarchy
        
        # Initialize the Gumbel-Softmax divisor projector
        self.projector = GumbelSoftmaxDivisor(sharpness_alpha=1.0, initial_tau=1.0, tau_decay_rate=0.999)
        
        # 创建一个嵌套的参数字典来存储所有tiling因子
        # 结构: self.factors[level_name][dim_name]['temporal' or 'spatial']
        self.factors = nn.ModuleDict()

        # 只为片上存储（on-chip buffers）创建可学习的参数
        on_chip_levels = [level['name'] for level in hierarchy if level['type'] == 'buffer']
        
        for level_name in on_chip_levels:
            self.factors[level_name] = nn.ModuleDict()
            for dim_name in self.dims.keys():
                # 使用ParameterDict来正确注册参数
                # 初始化为1（在log空间中为0）
                self.factors[level_name][dim_name] = nn.ParameterDict({
                    'temporal': nn.Parameter(torch.zeros(1)), # log(1) = 0
                    'spatial': nn.Parameter(torch.zeros(1))
                })

    def get_factor(self, level_name, dim_name, factor_type):
        """获取指定level, dim, type的tiling因子。"""
        return torch.clamp(torch.exp(self.factors[level_name][dim_name][factor_type]), min=1.0)
    
    def anneal_tau(self):
        """Anneal the temperature parameter of the Gumbel-Softmax projector."""
        self.projector.anneal_tau()

    def get_all_factors(self):
        """
        NEW: Returns physically valid, integer tiling factors using differentiable projection.
        This method replaces get_all_factors() for performance evaluation.
        """
        projected_factors = {}
        on_chip_levels = [level['name'] for level in self.hierarchy if level['type'] == 'buffer']

        for dim_name, total_size in self.dims.items():
            projected_factors[dim_name] = {}
            product_of_on_chip_factors = 1.0
            
            # Project on-chip factors to valid integer divisors
            for level_name in on_chip_levels:
                continuous_temporal = self.get_factor(level_name, dim_name, 'temporal')
                continuous_spatial = self.get_factor(level_name, dim_name, 'spatial')
                
                # Apply differentiable projection using SoftmaxWeightedDivisor
                problem_dim_tensor = torch.tensor(float(total_size))
                projected_temporal = self.projector(continuous_temporal, problem_dim_tensor)
                projected_spatial = self.projector(continuous_spatial, problem_dim_tensor)
                
                projected_factors[dim_name][level_name] = {
                    'temporal': projected_temporal,
                    'spatial': projected_spatial
                }
                product_of_on_chip_factors *= projected_temporal * projected_spatial

            # Handle DRAM level factors
            dram_level = next((level for level in self.hierarchy if level['type'] == 'dram'), None)
            if dram_level:
                dram_level_name = dram_level['name']
                # The temporal factor at DRAM is what's left over from the on-chip factors
                dram_temporal_factor = total_size / product_of_on_chip_factors
                projected_dram_temporal = ProjectToNearestDivisor.apply(dram_temporal_factor, torch.tensor(float(total_size)))

                projected_factors[dim_name][dram_level_name] = {
                    'temporal': projected_dram_temporal,
                    'spatial': torch.tensor(1.0) # No spatial tiling in DRAM
                }

        return projected_factors