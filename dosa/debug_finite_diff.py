import torch
from dosa.performance_model import HighFidelityPerformanceModel
from dosa.mapping import FineGrainedMapping
from dosa.config import Config
from dosa.hardware_parameters import HardwareParameters
import math

# === 配置 ===
config = Config.get_instance()
perf_model = HighFidelityPerformanceModel(config, debug_latency=False)

# 定义简单的卷积层
problem_dims = {"N": 1, "C": 64, "K": 128, "P": 32, "Q": 32, "R": 3, "S": 3}
mapping = FineGrainedMapping(problem_dims, config.MEMORY_HIERARCHY)

# 初始化参数 (log-space，1.0 = log(1) = 0)
for p in mapping.parameters():
    torch.nn.init.constant_(p, 0.0)

hw_params = HardwareParameters(initial_num_pes=16, initial_l0_kb=2.0, initial_l1_kb=4.0, initial_l2_kb=64.0)

# 定义 mock graph
class MockGraph:
    def __init__(self, dims):
        self.problem_dims = dims
        self.layers = {
            'conv1': {
                'type': 'Conv',
                'dims': dims,
                'input_shape': [dims['N'], dims['C'], dims['P']+dims['R']-1, dims['Q']+dims['S']-1],
                'output_shape': [dims['N'], dims['K'], dims['P'], dims['Q']],
                'weight_shape': [dims['K'], dims['C'], dims['R'], dims['S']]
            }
        }
        self.fusion_groups = [['conv1']]
        self.layer_order = ['conv1']
        self.adjacency = {}

graph = MockGraph(problem_dims)

# === 前向 ===
lat, en, area, mm, comp = perf_model(graph, hw_params, mapping, None)
loss = en * lat
loss.backward()

# === 选择一个测试参数 ===
param_name = "factors.L2_Scratchpad.S.temporal"
named_params = dict(mapping.named_parameters())
param = named_params[param_name]

print(f"\n[INFO] Testing gradient direction for {param_name}")
grad_val = param.grad.item()
print(f"Autograd grad = {grad_val:.6e}")

# === 验证方向 ===
eps = 1e-4   # 用大一点的步长，避免数值误差太大
with torch.no_grad():
    old_val = param.item()

    # 往正方向走
    param.copy_(torch.tensor(old_val + eps))
    lat_pos, en_pos, *_ = perf_model(graph, hw_params, mapping, None)
    loss_pos = (lat_pos * en_pos).item()

    # 往负方向走
    param.copy_(torch.tensor(old_val - eps))
    lat_neg, en_neg, *_ = perf_model(graph, hw_params, mapping, None)
    loss_neg = (lat_neg * en_neg).item()

    # 恢复原值
    param.copy_(torch.tensor(old_val))

print(f"Loss original = {loss.item():.6e}")
print(f"Loss(+eps)   = {loss_pos:.6e}")
print(f"Loss(-eps)   = {loss_neg:.6e}")

# === 判断方向是否正确 ===
if grad_val > 0:
    print(">> 梯度为正，理论上减少参数应降低Loss")
    print(f"   实际: Loss(-eps)={loss_neg:.6e}, Loss(+eps)={loss_pos:.6e}")
elif grad_val < 0:
    print(">> 梯度为负，理论上增加参数应降低Loss")
    print(f"   实际: Loss(+eps)={loss_pos:.6e}, Loss(-eps)={loss_neg:.6e}")
else:
    print(">> 梯度为0，参数变化对Loss无影响")
