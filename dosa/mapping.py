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
                # 初始化为1（在log空间中为0），确保是标量张量
                self.factors[level_name][dim_name] = nn.ParameterDict({
                    'temporal': nn.Parameter(torch.tensor(0.0)), # log(1) = 0, 标量
                    'spatial': nn.Parameter(torch.tensor(0.0))   # log(1) = 0, 标量
                })

        # Flag storage for partial-sum detection
        self.partial_sum_violations = {}
        # Enforce no partial sum across DRAM (default off)
        self.enforce_no_partial_sum = False

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

        # Reset violation tracking
        self.partial_sum_violations = {}
        reduction_dims = {'C', 'R', 'S'}
        dram_level = next((level for level in self.hierarchy if level['type'] == 'dram'), None)
        dram_level_name = dram_level['name'] if dram_level else None

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
            if dram_level_name:
                # The temporal factor at DRAM is what's left over from the on-chip factors
                dram_temporal_factor = total_size / product_of_on_chip_factors
                projected_dram_temporal = ProjectToNearestDivisor.apply(
                    dram_temporal_factor, torch.tensor(float(total_size))
                )

                projected_factors[dim_name][dram_level_name] = {
                    'temporal': projected_dram_temporal,
                    'spatial': torch.tensor(1.0)  # No spatial tiling in DRAM
                }

                if dim_name in reduction_dims:
                    self.partial_sum_violations[dim_name] = projected_dram_temporal > 1
        # Convenience flags for external modules
        self.has_partial_sum = any(self.partial_sum_violations.values())
        self.partial_sum_tiles = {
            dim: projected_factors[dim][dram_level_name]['temporal']
            for dim, violated in self.partial_sum_violations.items() if violated
        }

        # === Final-Sum-Per-Tile Hard Constraint ===
        if getattr(self, 'enforce_no_partial_sum', False):
            reduction_dims = {'C', 'R', 'S'}
            on_chip_buf_levels = [lv['name'] for lv in self.hierarchy if lv['type'] == 'buffer']
            dram = next((lv for lv in self.hierarchy if lv['type'] == 'dram'), None)
            dram_name = dram['name'] if dram else None
            for dim, total in self.dims.items():
                if dim in reduction_dims:
                    self._rebalance_reduction_dim(dim, int(total), projected_factors, on_chip_buf_levels, dram_name)
            # refresh flags after elimination
            self.partial_sum_violations = {}
            self.has_partial_sum = False
            self.partial_sum_tiles = {}

        return projected_factors

    def has_partial_sums(self) -> bool:
        """Return True if any reduction dimension is temporally tiled."""
        return getattr(self, 'has_partial_sum', False)

    def get_partial_sum_tiles(self) -> Dict[str, torch.Tensor]:
        """Return the number of tiles for each violating reduction dimension."""
        return getattr(self, 'partial_sum_tiles', {})

    def set_enforce_no_partial_sum(self, enabled: bool = True):
        """Enable or disable hard constraint to disallow partial sums in DRAM."""
        self.enforce_no_partial_sum = bool(enabled)

    def _param_device(self):
        """Safely obtain a device from registered parameters."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def _rebalance_reduction_dim(self, dim_name, total_size, projected_factors, on_chip_levels, dram_level_name):
        """Rebalance reduction dimension so that DRAM.temporal == 1."""
        dev = self._param_device()
        total = torch.tensor(float(total_size), device=dev)
        onchip = torch.tensor(1.0, device=dev)
        for lvl in on_chip_levels:
            tf = projected_factors[dim_name][lvl]['temporal']
            sf = projected_factors[dim_name][lvl]['spatial']
            onchip = onchip * tf * sf
        need = torch.ceil(torch.clamp(total / torch.clamp(onchip, min=1.0), min=1.0))
        if need.item() > 1.0:
            innermost = on_chip_levels[0]
            extra = ProjectToNearestDivisor.apply(need.to(dev), torch.tensor(float(total_size), device=dev))
            projected_factors[dim_name][innermost]['temporal'] = projected_factors[dim_name][innermost]['temporal'] * extra
        if dram_level_name:
            projected_factors[dim_name][dram_level_name]['temporal'] = torch.tensor(1.0, device=dev)
            projected_factors[dim_name][dram_level_name]['spatial'] = torch.tensor(1.0, device=dev)
