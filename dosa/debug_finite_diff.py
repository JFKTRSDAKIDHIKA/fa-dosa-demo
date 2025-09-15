import torch
from dosa.performance_model import HighFidelityPerformanceModel
from dosa.mapping import FineGrainedMapping
from dosa.config import Config
from dosa.hardware_parameters import HardwareParameters

# === 配置 ===
config = Config.get_instance()
perf_model = HighFidelityPerformanceModel(config, debug_latency=False)

# 定义简单的卷积层
problem_dims = {"N": 1, "C": 64, "K": 128, "P": 32, "Q": 32, "R": 3, "S": 3}
mapping = FineGrainedMapping(problem_dims, config.MEMORY_HIERARCHY)

# 初始化参数
for p in mapping.parameters():
    torch.nn.init.constant_(p, 1.0)

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

print(f"\n[INFO] Testing finite difference for {param_name}")
print(f"Autograd grad: {param.grad}")

# === 有限差分 ===
eps = 1e-8
with torch.no_grad():
    old_val = param.item()
    param.copy_(torch.tensor(old_val + eps))
lat2, en2, area2, mm2, comp2 = perf_model(graph, hw_params, mapping, None)
loss2 = en2 * lat2
num_grad = (loss2.item() - loss.item()) / eps

print(f"Numerical grad ≈ {num_grad:.6e}")
