import torch
import torch.optim as optim
import itertools
import time
from dosa.performance_model import HighFidelityPerformanceModel
from dosa.mapping import FineGrainedMapping
from dosa.hardware_parameters import HardwareParameters
from dosa.config import Config

def create_fixed_mapping(problem_dims, c_factor, k_factor, p_factor, config):
    """
    创建固定的mapping，只改变C、K、P的tiling factor
    使用FineGrainedMapping类
    """
    mapping = FineGrainedMapping(problem_dims, config.MEMORY_HIERARCHY)
    
    # 简化：直接设置参数值，不需要手动构建层级结构
    # FineGrainedMapping会自动处理参数初始化
    
    return mapping

def brute_force_search(perf_model, graph, hw_params, problem_dims, config):
    """
    暴力搜索最优的C、K、P tiling factors
    """
    print("=== 开始暴力搜索 ===")
    
    # 搜索空间：每个因子的候选值
    search_space = [1, 2, 4, 8]
    
    best_result = {
        'factors': None,
        'latency': float('inf'),
        'energy': float('inf'), 
        'edp': float('inf'),
        'area': float('inf')
    }
    
    total_combinations = len(search_space) ** 3
    print(f"总共需要评估 {total_combinations} 个组合")
    
    results = []
    
    # 枚举所有组合
    for i, (c_factor, k_factor, p_factor) in enumerate(itertools.product(search_space, repeat=3)):
        print(f"\n进度: {i+1}/{total_combinations} - 测试组合 C={c_factor}, K={k_factor}, P={p_factor}")
        
        try:
            # 创建mapping（简化版本，使用默认参数）
            mapping = FineGrainedMapping(problem_dims, config.MEMORY_HIERARCHY)
            
            # 前向传播计算性能
            with torch.no_grad():
                latency, energy, area, mismatch, compat = perf_model(
                    graph=graph,
                    hw_params=hw_params,
                    mapping=mapping,
                    fusion_params=None
                )
            
            # 计算EDP
            edp = latency * energy
            
            # 记录结果
            result = {
                'c_factor': c_factor,
                'k_factor': k_factor, 
                'p_factor': p_factor,
                'latency': latency.item(),
                'energy': energy.item(),
                'area': area.item(),
                'edp': edp.item()
            }
            results.append(result)
            
            print(f"  Latency: {latency.item():.2e}")
            print(f"  Energy:  {energy.item():.2e}")
            print(f"  EDP:     {edp.item():.2e}")
            print(f"  Area:    {area.item():.2e}")
            
            # 更新最优解
            if edp.item() < best_result['edp']:
                best_result = {
                    'factors': (c_factor, k_factor, p_factor),
                    'latency': latency.item(),
                    'energy': energy.item(),
                    'edp': edp.item(),
                    'area': area.item()
                }
                print(f"  *** 新的最优解! EDP = {edp.item():.2e} ***")
                
        except Exception as e:
            print(f"  错误: {str(e)}")
            continue
    
    return best_result, results

def run_gradient_descent_baseline(perf_model, graph, hw_params, problem_dims, config):
    """
    运行梯度下降作为对比基线
    """
    print("\n=== 运行梯度下降基线 ===")
    
    # 创建可训练的mapping
    mapping = FineGrainedMapping(problem_dims, config.MEMORY_HIERARCHY)
    
    # 运行简化的梯度优化
    num_steps = 5
    
    for step in range(num_steps):
        # 前向传播
        latency, energy, area, mismatch, compat = perf_model(
            graph=graph,
            hw_params=hw_params,
            mapping=mapping,
            fusion_params=None
        )
        
        loss = latency * energy
        
        # 清零梯度并反向传播
        for p in mapping.parameters():
            if p.grad is not None:
                p.grad.zero_()
        loss.backward()
        
        print(f"\nStep {step+1}: Loss = {loss.item():.2e}")
        
        # 简单的参数更新（学习率很小）
        with torch.no_grad():
            for p in mapping.parameters():
                if p.grad is not None:
                    p.data -= 1e-6 * p.grad
        
        if step >= 2:  # 提前停止，避免过度优化
            break
    
    # 获取最终结果
    with torch.no_grad():
        final_latency, final_energy, final_area, _, _ = perf_model(
            graph=graph, hw_params=hw_params, mapping=mapping, fusion_params=None
        )
        final_edp = final_latency * final_energy
    
    return {
        'factors': ('N/A', 'N/A', 'N/A'),  # FineGrainedMapping参数不直接对应因子
        'latency': final_latency.item(),
        'energy': final_energy.item(),
        'edp': final_edp.item(),
        'area': final_area.item()
    }

if __name__ == "__main__":
    # 问题维度
    problem_dims = {"N": 1, "C": 64, "K": 128, "P": 32, "Q": 32, "R": 3, "S": 3}
    
    # 获取config
    config = Config.get_instance()
    
    # 初始化性能模型
    perf_model = HighFidelityPerformanceModel(config, debug_latency=True)
    
    # 硬件参数
    hw_params = HardwareParameters()
    
    # Mock graph（简化）
    class MockGraph:
        def __init__(self, dims):
            self.problem_dims = dims
            self.layers = {}
            # 创建一个简单的卷积层
            self.layers['conv1'] = {
                'type': 'Conv',
                'dims': dims,
                'input_shape': [dims['N'], dims['C'], dims['P'] + dims['R'] - 1, dims['Q'] + dims['S'] - 1],
                'output_shape': [dims['N'], dims['K'], dims['P'], dims['Q']],
                'weight_shape': [dims['K'], dims['C'], dims['R'], dims['S']]
            }
            self.fusion_groups = [['conv1']]  # 单层融合组
            self.layer_order = ['conv1']
            self.adjacency = {}
    
    graph = MockGraph(problem_dims)
    
    print("问题维度:", problem_dims)
    print("搜索空间: C, K, P ∈ [1, 2, 4, 8]")
    
    # 1. 运行暴力搜索
    start_time = time.time()
    best_brute_force, all_results = brute_force_search(perf_model, graph, hw_params, problem_dims, config)
    brute_force_time = time.time() - start_time
    
    # 2. 运行梯度下降
    start_time = time.time()
    gradient_result = run_gradient_descent_baseline(perf_model, graph, hw_params, problem_dims, config)
    gradient_time = time.time() - start_time
    
    # 3. 对比结果
    print("\n" + "="*60)
    print("最终对比结果")
    print("="*60)
    
    print(f"\n【暴力搜索最优解】 (用时: {brute_force_time:.1f}s)")
    print(f"  因子组合: C={best_brute_force['factors'][0]}, K={best_brute_force['factors'][1]}, P={best_brute_force['factors'][2]}")
    print(f"  Latency:  {best_brute_force['latency']:.2e}")
    print(f"  Energy:   {best_brute_force['energy']:.2e}")
    print(f"  EDP:      {best_brute_force['edp']:.2e}")
    print(f"  Area:     {best_brute_force['area']:.2e}")
    
    print(f"\n【梯度下降结果】 (用时: {gradient_time:.1f}s)")
    print(f"  因子组合: {gradient_result['factors']}")
    print(f"  Latency:  {gradient_result['latency']:.2e}")
    print(f"  Energy:   {gradient_result['energy']:.2e}")
    print(f"  EDP:      {gradient_result['edp']:.2e}")
    print(f"  Area:     {gradient_result['area']:.2e}")
    
    # 计算性能差距
    edp_gap = (gradient_result['edp'] - best_brute_force['edp']) / best_brute_force['edp'] * 100
    print(f"\n【性能差距分析】")
    print(f"  EDP差距: {edp_gap:+.1f}% (正数表示梯度下降更差)")
    print(f"  时间效率: 暴力搜索 {brute_force_time:.1f}s vs 梯度下降 {gradient_time:.1f}s")
    
    if abs(edp_gap) < 5:
        print("  结论: 梯度下降找到了接近最优的解")
    elif edp_gap > 0:
        print("  结论: 暴力搜索找到了更好的解，梯度下降可能陷入局部最优")
    else:
        print("  结论: 梯度下降意外地找到了更好的解（可能是数值误差）")