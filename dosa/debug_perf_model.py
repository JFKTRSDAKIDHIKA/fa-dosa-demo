import torch
import torch.nn as nn
import torch.optim as optim

# 正确的import路径
from dosa.performance_model import HighFidelityPerformanceModel
from dosa.mapping import FineGrainedMapping
from dosa.config import Config
from dosa.hardware_parameters import HardwareParameters

# 1. 获取config单例实例
config = Config.get_instance()

# 2. 创建性能模型
perf_model = HighFidelityPerformanceModel(config, debug_latency=True)

# 3. 构造简单的问题维度（模拟卷积层）
problem_dims = {
    "N": 1,    # batch size
    "C": 64,   # input channels
    "K": 128,  # output channels
    "P": 32,   # output height
    "Q": 32,   # output width
    "R": 3,    # kernel height
    "S": 3     # kernel width
}

# 4. 创建映射对象
mapping = FineGrainedMapping(problem_dims, config.MEMORY_HIERARCHY)

# Initialize mapping parameters with empirically good starting points
for i, p in enumerate(mapping.parameters()):
    if i == 0:  # Temporal mapping parameter
        p.data.fill_(1.0)  # Favor temporal reuse
    elif i == 1:  # Spatial mapping parameter  
        p.data.fill_(2.0)  # Moderate spatial parallelism
    else:  # Memory hierarchy parameter
        p.data.fill_(1.0)  # Balanced memory utilization

# 5. 创建硬件参数
hw_params = HardwareParameters(
    initial_num_pes=16.0,
    initial_l0_kb=2.0,
    initial_l1_kb=4.0,
    initial_l2_kb=64.0
)

# 6. 创建一个简单的模拟图对象
class MockGraph:
    def __init__(self, dims):
        self.problem_dims = dims
        self.layers = {}
        # 创建一个简单的卷积层（使用字典结构）
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

print("开始性能模型前向传播...")

# 查看mapping参数
named = dict(mapping.named_parameters())
print("has_temporal_param?",
      "factors.L0_Registers.K.temporal" in named)

print("Available parameters:")
for name, param in named.items():
    print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

# =============== 新增：优化器部分 ===============
learning_rate = 1e-8
optimizer = optim.SGD(mapping.parameters(), lr=learning_rate)

num_steps = 10  # 迭代步数，可以改大

for step in range(num_steps):
    optimizer.zero_grad()

    # 前向传播
    latency, energy, area, mismatch, compat = perf_model(
        graph=graph,
        hw_params=hw_params,
        mapping=mapping,
        fusion_params=None
    )

    # 构造loss（简单：EDP）
    loss = latency * energy

    # 反向传播
    loss.backward()

    print(f"\n=== Step {step+1} ===")
    print(f"Loss = {loss.item():.6e}")

    # 打印每个参数的更新公式
    for i, (name, p) in enumerate(mapping.named_parameters()):
        if p.grad is None:
            continue
        old_val = p.data.clone()
        grad_val = p.grad.clone()
        update_val = old_val -learning_rate * grad_val

        #  打印详细更新过程
        print(f"   update = old_val -lr * grad = {update_val.item():.6f}")
        print(f"param{i} {name}:")
        print(f"   old = {old_val.item():.6f}")
        print(f"   grad = {grad_val.item():.6f}")
        print(f"   lr = {learning_rate}")
        
    # 执行优化器更新
    optimizer.step()

    # 打印更新后的参数
    for i, (name, p) in enumerate(mapping.named_parameters()):
        print(f"   new {name} = {p.data.item():.6f}")
