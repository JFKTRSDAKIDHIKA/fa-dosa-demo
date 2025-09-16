import torch
import torch.nn as nn
import torch.optim as optim
import math

# 正确的import路径
from dosa.performance_model import HighFidelityPerformanceModel
from dosa.mapping import FineGrainedMapping
from dosa.config import Config
from dosa.hardware_parameters import HardwareParameters

# =============== 配置常量 ===============
# 优化器配置
LEARNING_RATE = 2e-8  # 学习率，控制参数更新步长
NUM_OPTIMIZATION_STEPS = 10  # 优化迭代次数
MAPPING_PENALTY_WEIGHT = 1e8  # 映射无效惩罚权重

# =============== 工具函数 ===============
def print_mapping_parameters(mapping, title="Mapping参数详情", show_projected=True):
    """
    打印mapping参数的详细信息
    
    Args:
        mapping: FineGrainedMapping对象
        title: 打印标题
        show_projected: 是否显示投影后的离散值
    """
    print(f"\n{'='*60}")
    print(f"📋 {title}")
    print(f"{'='*60}")
    
    if show_projected:
        projected_factors = mapping.get_all_factors()
    
    for i, (name, param) in enumerate(mapping.named_parameters()):
        continuous_val = param.data.item()
        real_val = math.exp(continuous_val)
        
        print(f"\n📌 参数 {i+1}: {name}")
        print(f"   📊 连续值 (log): {continuous_val:.6f}")
        print(f"   🔢 真实值: {real_val:.6f}")
        
        if show_projected:
            # Extract dimension and level from parameter name
            parts = name.split('.')
            if len(parts) >= 3:
                dim = parts[2]  # e.g. 'K'
                level = parts[1]  # e.g. 'L0_Registers'
                factor_type = parts[3]  # e.g. 'temporal' or 'spatial'
                
                # Print corresponding projected discrete factor
                if dim in projected_factors and level in projected_factors[dim]:
                    projected_val = projected_factors[dim][level][factor_type].item()
                    print(f"   🎯 投影值: {projected_val:.3f}")
    
    print(f"{'='*60}")

def print_optimization_step(step, latency, energy, penalty, loss, current_loss, best_loss):
    """
    打印优化步骤的信息
    
    Args:
        step: 当前步数
        latency, energy, penalty, loss: 性能指标
        current_loss: 当前损失
        best_loss: 历史最优损失
    """
    print(f"\n{'='*60}")
    print(f"📊 优化步骤 {step+1}/{NUM_OPTIMIZATION_STEPS}")
    print(f"{'='*60}")
    
    # 性能指标
    print(f"🔍 性能指标:")
    print(f"   ⏱️  Latency: {latency.item():.6e}")
    print(f"   ⚡ Energy: {energy.item():.6e}")
    print(f"   📈 Performance Loss: {(latency*energy).item():.6e}")
    
    # 惩罚项
    print(f"\n⚠️  惩罚项:")
    print(f"   🚫 Mapping Invalid Penalty: {penalty.item():.6e}")
    
    # 总损失
    print(f"\n🎯 总损失: {loss.item():.6e}")
    
    # 最优解状态
    if current_loss < best_loss:
        print(f"🎉 发现新的最优解！Loss = {current_loss:.6e}")
    else:
        print(f"📋 当前最优: {best_loss:.6e}")
    
    print(f"{'='*60}")

def print_fusion_gradients(fusion_params, title="Fusion参数梯度详情"):
    """
    打印fusion参数的梯度信息
    """
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")
    
    if fusion_params is None:
        print("❌ fusion_params 为 None")
        return
    
    for name, param in fusion_params.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_values = param.grad.detach().cpu().numpy()
            current_val = param.data.clone()
            update_val = current_val - LEARNING_RATE * 10 * param.grad  # Using larger learning rate for fusion
            
            print(f"\n📌 参数: {name}")
            print(f"   📊 梯度范数: {grad_norm:.8f}")
            print(f"   🔢 梯度值: {grad_values}")
            print(f"   📏 梯度形状: {param.grad.shape}")
            print(f"   📈 当前值: {current_val.detach().cpu().numpy()}")
            print(f"   🔄 更新后值: {update_val.detach().cpu().numpy()}")
            
            # Print fusion probability changes if it's logits
            if 'logits' in name:
                current_prob = torch.sigmoid(current_val)
                updated_prob = torch.sigmoid(update_val)
                print(f"   🎯 Fusion概率变化: {current_prob.item():.4f} → {updated_prob.item():.4f}")
        else:
            print(f"\n📌 参数: {name}")
            print(f"   ❌ 梯度为 None (可能未参与计算图)")
    print(f"{'='*60}")


def print_parameter_gradients(mapping, learning_rate):
    """
    打印参数梯度和更新信息
    
    Args:
        mapping: FineGrainedMapping对象
        learning_rate: 学习率
    """
    print(f"\n🔧 参数梯度和更新信息:")
    print(f"{'─'*50}")
    
    for i, (name, param) in enumerate(mapping.named_parameters()):
        if param.grad is None:
            continue
            
        old_val = param.data.clone()
        grad_val = param.grad.clone()
        update_val = old_val - learning_rate * grad_val

        real_old = math.exp(old_val.item())
        real_update = math.exp(update_val.item())

        print(f"\n📌 参数 {i+1}: {name}")
        print(f"   📊 当前值 (log): {old_val.item():.6f} → 真实值: {real_old:.6f}")
        print(f"   📉 梯度: {grad_val.item():.6f}")
        print(f"   🔄 更新后 (log): {update_val.item():.6f} → 真实值: {real_update:.6f}")

def print_fusion_parameters(fusion_params, graph, title="Fusion参数详情"):
    """
    打印fusion参数的详细信息
    
    Args:
        fusion_params: fusion参数字典
        graph: 计算图对象
        title: 打印标题
    """
    print(f"\n{'='*60}")
    print(f"🔗 {title}")
    print(f"{'='*60}")
    
    if fusion_params is None or "fusion_logits" not in fusion_params:
        print("⚠️ 未找到fusion参数")
        return
    
    fusion_logits = fusion_params["fusion_logits"]
    # 确保fusion_probs始终是1维张量，避免squeeze导致的0维问题
    if fusion_logits.dim() == 0:
        fusion_probs = torch.sigmoid(fusion_logits).unsqueeze(0)
    else:
        fusion_probs = torch.sigmoid(fusion_logits)
    
    print(f"\n📊 Fusion Groups 总数: {len(graph.fusion_groups)}")
    
    for i, group in enumerate(graph.fusion_groups):
        if i < len(fusion_probs):
            logit_val = fusion_logits[i].item() if fusion_logits.dim() > 0 else fusion_logits.item()
            prob_val = fusion_probs[i].item() if fusion_probs.dim() > 0 else fusion_probs.item()
        else:
            logit_val = 0.0
            prob_val = 0.5
        
        print(f"\n🔗 Fusion Group {i+1}: {' → '.join(group)}")
        print(f"   📊 Logit值: {logit_val:.6f}")
        print(f"   🎯 Fusion概率: {prob_val:.6f}")
        print(f"   💡 状态: {'启用' if prob_val > 0.5 else '禁用'}")
    
    print(f"{'='*60}")

def print_graph_structure(graph, title="计算图结构"):
    """
    打印计算图的结构信息
    
    Args:
        graph: 计算图对象
        title: 打印标题
    """
    print(f"\n{'='*60}")
    print(f"🏗️ {title}")
    print(f"{'='*60}")
    
    print(f"\n📋 层信息:")
    for layer_name, layer_info in graph.layers.items():
        print(f"   🔸 {layer_name}: {layer_info['type']}")
        print(f"      输入形状: {layer_info['input_shape']}")
        print(f"      输出形状: {layer_info['output_shape']}")
    
    print(f"\n🔗 融合组:")
    for i, group in enumerate(graph.fusion_groups):
        print(f"   Group {i+1}: {' → '.join(group)}")
    
    print(f"\n📝 层执行顺序: {' → '.join(graph.layer_order)}")
    print(f"{'='*60}")

def print_best_solution_summary(best_step, best_loss, best_metrics):
    """
    打印最优解的摘要信息
    
    Args:
        best_step: 最优解出现的步数
        best_loss: 最优损失值
        best_metrics: 最优解的详细指标
    """
    print(f"\n{'='*70}")
    print(f"🏆 优化完成！最优解摘要报告")
    print(f"{'='*70}")
    
    print(f"📍 最优解来源: 第 {best_step+1} 步")
    print(f"🎯 最优损失: {best_loss:.6e}")
    
    print(f"\n📊 详细性能指标:")
    print(f"   ⏱️  Latency: {best_metrics['latency']:.6e}")
    print(f"   ⚡ Energy: {best_metrics['energy']:.6e}")
    print(f"   📐 Area: {best_metrics['area']:.6e}")
    print(f"   ⚠️  Mapping Invalid Penalty: {best_metrics['mapping_invalid_penalty']:.6e}")
    print(f"   🎯 Total Penalty: {best_metrics['penalty']:.6e}")
    
    print(f"{'='*70}")

# def create_mock_graph(problem_dims):
#     """
#     创建模拟图对象
    
#     Args:
#         problem_dims: 问题维度字典
        
#     Returns:
#         MockGraph对象
#     """
#     class MockGraph:
#         def __init__(self, dims):
#             self.problem_dims = dims
#             self.layers = {}
#             # 创建一个简单的卷积层（使用字典结构）
#             self.layers['conv1'] = {
#                 'type': 'Conv',
#                 'dims': dims,
#                 'input_shape': [dims['N'], dims['C'], dims['P'] + dims['R'] - 1, dims['Q'] + dims['S'] - 1],
#                 'output_shape': [dims['N'], dims['K'], dims['P'], dims['Q']],
#                 'weight_shape': [dims['K'], dims['C'], dims['R'], dims['S']]
#             }
#             self.fusion_groups = [['conv1']]  # 单层融合组
#             self.layer_order = ['conv1']
#             self.adjacency = {}
    
#     return MockGraph(problem_dims)

def create_mock_graph(problem_dims):
    """
    Create a mock graph with simple Conv-BN-ReLU fusion
    """
    class MockGraph:
        def __init__(self, dims):
            self.problem_dims = dims
            self.layers = {}

            # Input layer
            self.layers['input'] = {
                'type': 'Input',
                'dims': dims,
                'input_shape': [dims['N'], dims['C'], dims['P'] + dims['R'] - 1, dims['Q'] + dims['S'] - 1],
                'output_shape': [dims['N'], dims['C'], dims['P'] + dims['R'] - 1, dims['Q'] + dims['S'] - 1]
            }

            # Conv-BN-ReLU fusion block
            self.layers['conv1'] = {
                'type': 'Conv',
                'dims': dims,
                'input_shape': [dims['N'], dims['C'], dims['P'] + dims['R'] - 1, dims['Q'] + dims['S'] - 1],
                'output_shape': [dims['N'], dims['K'], dims['P'], dims['Q']],
                'weight_shape': [dims['K'], dims['C'], dims['R'], dims['S']]
            }

            self.layers['bn1'] = {
                'type': 'BatchNormalization',
                'dims': dims,
                'input_shape': [dims['N'], dims['K'], dims['P'], dims['Q']],
                'output_shape': [dims['N'], dims['K'], dims['P'], dims['Q']]
            }

            self.layers['relu1'] = {
                'type': 'ReLU',
                'dims': dims,
                'input_shape': [dims['N'], dims['K'], dims['P'], dims['Q']],
                'output_shape': [dims['N'], dims['K'], dims['P'], dims['Q']]
            }

            # Output layer
            self.layers['output'] = {
                'type': 'Output',
                'dims': dims,
                'input_shape': [dims['N'], dims['K'], dims['P'], dims['Q']],
                'output_shape': [dims['N'], dims['K'], dims['P'], dims['Q']]
            }

            # Define single fusion group for Conv-BN-ReLU
            self.fusion_groups = [
                ['conv1', 'bn1', 'relu1']
            ]
            
            # Define layer execution order
            self.layer_order = [
                'input', 'conv1', 'bn1', 'relu1', 'output'
            ]
            
            self.adjacency = {}  # Empty for simple case

    return MockGraph(problem_dims)


# =============== 主程序初始化 ===============
def initialize_system():
    """
    初始化系统组件
    
    该函数负责创建和配置所有必要的系统组件，包括：
    - 配置管理器
    - 性能模型
    - 映射对象
    - 硬件参数
    - 模拟计算图
    
    Returns:
        tuple: (config, perf_model, mapping, hw_params, graph)
            - config: 配置管理器实例
            - perf_model: 高保真性能模型实例
            - mapping: 细粒度映射对象
            - hw_params: 硬件参数对象
            - graph: 模拟计算图对象
    """
    # 1. 获取config单例实例
    config = Config.get_instance()

    # 2. 创建性能模型实例，启用延迟调试
    perf_model = HighFidelityPerformanceModel(config, debug_latency=True)

    # 3. 构造简单的问题维度（模拟卷积层）
    problem_dims = {
        "N": 1,    # batch size - 批次大小
        "C": 64,   # input channels - 输入通道数
        "K": 128,  # output channels - 输出通道数
        "P": 32,   # output height - 输出高度
        "Q": 32,   # output width - 输出宽度
        "R": 3,    # kernel height - 卷积核高度
        "S": 3     # kernel width - 卷积核宽度
    }

    # 4. 创建映射对象
    mapping = FineGrainedMapping(problem_dims, config.MEMORY_HIERARCHY)

    # 使用经验良好的起始点初始化映射参数
    for i, p in enumerate(mapping.parameters()):
        if i == 0:  # 时间映射参数
            p.data.fill_(1.0)  # 偏向时间重用
        elif i == 1:  # 空间映射参数  
            p.data.fill_(2.0)  # 适度空间并行
        else:  # 内存层次参数
            p.data.fill_(1.0)  # 平衡内存利用

    # 5. 创建硬件参数
    hw_params = HardwareParameters(
        initial_num_pes=16.0,   # 初始处理单元数量
        initial_l0_kb=2.0,      # L0缓存大小(KB)
        initial_l1_kb=4.0,      # L1缓存大小(KB)
        initial_l2_kb=64.0      # L2缓存大小(KB)
    )

    # 6. 创建一个简单的模拟图对象
    graph = create_mock_graph(problem_dims)
    
    # 7. 创建fusion参数（必须在graph创建后）
    # 确保fusion_logits始终是1维张量
    num_fusion_groups = len(graph.fusion_groups)
    fusion_params = nn.ParameterDict({
        "fusion_logits": nn.Parameter(torch.zeros(max(1, num_fusion_groups)))
    })
    
    return config, perf_model, mapping, hw_params, fusion_params, graph

# =============== 优化主循环 ===============
def run_optimization(perf_model, mapping, hw_params, fusion_params,graph):
    """
    运行优化循环
    
    该函数实现了基于梯度下降的优化循环，包含以下特性：
    - Best-so-far 策略：跟踪并保存历史最优解
    - 详细的性能指标监控
    - 参数梯度和更新信息的可视化
    - 自动恢复最优解到映射对象
    
    Args:
        perf_model: 性能模型实例，用于计算延迟、能耗等指标
        mapping: 映射对象，包含可优化的映射参数
        hw_params: 硬件参数对象，定义硬件配置
        graph: 计算图对象，描述要优化的神经网络结构
        
    Note:
        优化过程中会自动保存最优解，并在优化结束后恢复到mapping对象中
    """
    print("🚀 开始性能模型优化...")

    # 初始化优化器 - 分别为mapping和fusion参数创建优化器
    mapping_optimizer = optim.SGD(mapping.parameters(), lr=LEARNING_RATE)
    fusion_optimizer = optim.SGD(fusion_params.parameters(), lr=LEARNING_RATE * 10)  # fusion参数使用更大的学习率

    # Best-so-far 策略初始化
    best_loss = float('inf')        # 历史最优损失值
    best_mapping_params = None      # 最优映射参数
    best_fusion_params = None       # 最优fusion参数
    best_step = -1                  # 最优解出现的步数
    best_metrics = None             # 最优解的详细指标

    # 打印初始状态
    print_graph_structure(graph, "初始计算图结构")
    print_fusion_parameters(fusion_params, graph, "初始Fusion参数")
    # print_mapping_parameters(mapping, "初始Mapping参数")

    # 主优化循环
    for step in range(NUM_OPTIMIZATION_STEPS):
        mapping_optimizer.zero_grad()  # 清零梯度
        fusion_optimizer.zero_grad()

        # 前向传播：计算性能指标
        latency, energy, area, mismatch, compat, mapping_invalid_penalty, penalty = perf_model(
            graph=graph,
            hw_params=hw_params,
            mapping=mapping,
            fusion_params=fusion_params
        )

        # 计算总损失：性能损失 + 映射无效惩罚
        loss = (latency * energy) + MAPPING_PENALTY_WEIGHT * mapping_invalid_penalty
        current_loss = loss.item()
        
        # Best-so-far 策略：检查并更新最优解
        if current_loss < best_loss:
            best_loss = current_loss
            best_step = step
            # 深拷贝当前最优的映射参数
            best_mapping_params = {name: param.data.clone() for name, param in mapping.named_parameters()}
            # 深拷贝当前最优的fusion参数
            best_fusion_params = {name: param.data.clone() for name, param in fusion_params.named_parameters()}
            # 保存最优解的详细指标
            best_metrics = {
                'latency': latency.item(),
                'energy': energy.item(),
                'area': area.item(),
                'mapping_invalid_penalty': mapping_invalid_penalty.item(),
                'penalty': penalty.item()
            }
            print(f"🎯 发现更优解！步数: {step}, 损失: {best_loss:.6f}")
            print_fusion_parameters(fusion_params, graph, f"步数 {step} - 最优Fusion参数")
        
        # 打印当前步骤的优化信息
        print_optimization_step(step, latency, energy, penalty, loss, current_loss, best_loss)
        
        # 反向传播：计算梯度
        loss.backward()
        
        # 打印参数梯度和更新信息
        # print_parameter_gradients(mapping, LEARNING_RATE)
        
        # 打印fusion参数的梯度信息
        print_fusion_gradients(fusion_params, f"Step {step} - Fusion参数梯度")
        
        # 分别更新mapping和fusion参数
        mapping_optimizer.step()
        fusion_optimizer.step()
        
        # 打印更新后的mapping参数
        # print_mapping_parameters(mapping, f"Step {step+1} - 更新后的Mapping参数")

    # 优化结束后的最优解恢复
    if best_mapping_params is not None:
        # 打印最优解摘要
        # print_best_solution_summary(best_step, best_loss, best_metrics)
        
        # 恢复最优参数到mapping对象
        for name, param in mapping.named_parameters():
            param.data.copy_(best_mapping_params[name])
        
        # 恢复最优参数到fusion_params对象
        if best_fusion_params is not None:
            for name, param in fusion_params.named_parameters():
                param.data.copy_(best_fusion_params[name])
        
        print(f"\n✅ 已将最优解参数恢复到mapping和fusion对象中")
        
        # 打印最优解的详细参数信息
        # print_mapping_parameters(mapping, "🏆 最优解的详细参数", show_projected=True)
        print_fusion_parameters(fusion_params, graph, "🏆 最优解的Fusion参数")
    else:
        print(f"\n⚠️ 未找到有效的最优解")

# =============== 主程序入口 ===============
def main():
    """
    主程序入口函数
    
    执行完整的性能模型优化流程：
    1. 初始化系统组件
    2. 运行优化循环
    3. 输出最终结果
    """
    print("🎯 DOSA 性能模型优化开始")
    print("=" * 70)
    
    # 初始化系统组件
    config, perf_model, mapping, hw_params, fusion_params,graph = initialize_system()
    
    # 运行优化
    run_optimization(perf_model, mapping, hw_params, fusion_params, graph)
    
    print("\n🎉 DOSA 性能模型优化完成")
    print("=" * 70)

if __name__ == "__main__":
    main()
