#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
薄适配层：不给你的分析模型文件做任何改动。
- 事先向 sys.modules 注入最小 "dosa.*" 依赖的桩（Config/HardwareParameters/Mapping/DMT/utils）
- 读取 HW/SW 配置（JSON）
- 导入你的模型文件，构建 Graph，对应地喂 direct_mapping_table
- 调用 forward()，打印结果，并可落盘 debug JSON
"""

import argparse, json, types, sys, importlib.util, pathlib
from types import SimpleNamespace
from functools import reduce
from operator import mul

import torch

# ---------- 1) 注入最小桩模块：dosa.config / hardware_parameters / mapping / dmt / utils ----------

def inject_dosa_shims(hw_cfg):
    """
    hw_cfg: 从 hw.json 读出的字典
    """
    # 顶级包
    dosa = types.ModuleType("dosa")
    sys.modules["dosa"] = dosa

    # ---- config ----
    dosacfg = types.ModuleType("dosa.config")
    sys.modules["dosa.config"] = dosacfg

    class _ConfigShim:
        """最小可用的 Config 单例，满足你的文件里用到的属性/方法。"""

        # 设备
        DEVICE = "cpu"

        # 维度集合（与你文件内期望一致）
        TENSOR_DIMENSIONS = {
            "W": {"K", "C", "R", "S"},
            "I": {"N", "C", "P", "Q"},
            "O": {"N", "K", "P", "Q"},
        }
        D_ALL = {"N", "K", "C", "R", "S", "P", "Q"}

        # 内存层级定义
        MEMORY_HIERARCHY = [
            {
                'index': 0,
                'name': 'L0_Registers', 
                'type': 'buffer', 
                'description': 'PE内部寄存器，暂存操作数'
            },
            {
                'index': 1,
                'name': 'L1_Accumulator', 
                'type': 'buffer', 
                'description': '累加器缓存，存储输出部分和'
            },
            {
                'index': 2,
                'name': 'L2_Scratchpad', 
                'type': 'buffer', 
                'description': '片上共享缓存，层级间数据交换'
            },
            {
                'index': 3,
                'name': 'L3_DRAM', 
                'type': 'dram', 
                'description': '主存，存储完整模型权重和数据'
            }
        ]

        # "哪个层存放哪些张量"的存储矩阵
        # 注意：根据你之前约定，REG 只存 W；ACC 只存 O；SPAD 存 I（也可放 W tile 预装载）
        STORAGE_MATRIX = {
            0: {"W": 1, "I": 1, "O": 0},  # L0_Registers: 权重+输入，无输出
            1: {"W": 0, "I": 0, "O": 1},  # L1_Accumulator: 仅输出
            2: {"W": 1, "I": 1, "O": 1},  # L2_Scratchpad: 全部张量
            3: {"W": 1, "I": 1, "O": 1}   # L3_DRAM: 全部张量
        }

        # 元素宽度与时钟（来自 hw.json）
        BYTES_PER_ELEMENT = None
        CLOCK_FREQUENCY_MHZ = None

        # 数据供给通路图
        DATA_SUPPLY_MAP = {
            'L0_Registers': {
                'Input': 'L2_Scratchpad',   # L0的Input数据来自L2
                'Weight': 'L2_Scratchpad',  # L0的Weight数据也来自L2
                'Output': 'PE'              # Output由PE计算单元产生，无需填充
            },
            'L1_Accumulator': {
                'Output': 'L0_Registers'    # L1的Output数据来自L0的写回
            },
            'L2_Scratchpad': {
                'Input': 'L3_DRAM',
                'Weight': 'L3_DRAM',
                'Output': 'L1_Accumulator'  # L2的Output数据来自L1的写回
            },
            'L3_DRAM': {
                # DRAM是最高层级，其数据被认为是"凭空而来"，主要接受来自下层的写回
                'Output': 'L2_Scratchpad'
            }
        }

        # 每次访问能量（pJ/word）
        L0_REG_BASE_EPA_PJ = None
        L1_ACCUM_BASE_EPA_PJ = None
        L1_ACCUM_CAPACITY_COEFF_PJ_PER_KB = None
        L2_SPM_BASE_EPA_PJ = None
        L2_SPM_CAPACITY_COEFF_PJ_PER_KB = None
        L3_DRAM_EPA_PJ = None
        PE_MAC_EPA_PJ = None

        # 供你的文件调用：Config.get_instance()
        _instance = None

        @classmethod
        def get_instance(cls):
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

        def __init__(self):
            # 把 JSON 里的值抬成 torch.Tensor，和你的模型保持张量口径
            self.BYTES_PER_ELEMENT = torch.tensor(
                float(hw_cfg["bytes_per_element"])
            )
            self.CLOCK_FREQUENCY_MHZ = torch.tensor(
                float(hw_cfg["clock_frequency_mhz"])
            )

            epa = hw_cfg.get("epa_pj", {})
            self.L0_REG_BASE_EPA_PJ = torch.tensor(float(epa.get("L0_REG_BASE", 0.02)))
            self.L1_ACCUM_BASE_EPA_PJ = torch.tensor(float(epa.get("L1_ACCUM_BASE", 0.5)))
            self.L1_ACCUM_CAPACITY_COEFF_PJ_PER_KB = torch.tensor(float(epa.get("L1_ACCUM_COEFF_PER_KB", 0.005)))
            self.L2_SPM_BASE_EPA_PJ = torch.tensor(float(epa.get("L2_SPM_BASE", 2.0)))
            self.L2_SPM_CAPACITY_COEFF_PJ_PER_KB = torch.tensor(float(epa.get("L2_SPM_COEFF_PER_KB", 0.02)))
            self.L3_DRAM_EPA_PJ = torch.tensor(float(epa.get("L3_DRAM", 300.0)))
            self.PE_MAC_EPA_PJ = torch.tensor(float(epa.get("PE_MAC", 1.0)))

    dosacfg.Config = _ConfigShim

    # ---- hardware_parameters ----
    dosahw = types.ModuleType("dosa.hardware_parameters")
    sys.modules["dosa.hardware_parameters"] = dosahw

    class _HardwareParametersShim:
        """最小可用硬件参数对象。"""
        def __init__(self, buffers_kb, pe_count):
            self._buffers = dict(buffers_kb)
            self._pe_count = int(pe_count)

        def get_projected_num_pes(self):
            return torch.tensor(float(self._pe_count))

        def get_buffer_size_kb(self, level_name: str):
            return torch.tensor(float(self._buffers.get(level_name, 0.0)))

        def get_area_cost(self):
            # 先返回 0；如果你有面积模型，可在 hw.json 里加字段并在此读取。
            return torch.tensor(0.0)

    dosahw.HardwareParameters = _HardwareParametersShim

    # ---- mapping（只是占位；我们走 direct_mapping_table，就不会用到）----
    dosamap = types.ModuleType("dosa.mapping")
    sys.modules["dosa.mapping"] = dosamap

    class _FineGrainedMappingShim:
        def get_all_factors(self):
            return {}
    dosamap.FineGrainedMapping = _FineGrainedMappingShim

    # ---- dmt（不启用 fusion 时不会走到；给无害桩以免导入失败）----
    dosadmt = types.ModuleType("dosa.dmt")
    sys.modules["dosa.dmt"] = dosadmt

    class _DMTZero:
        def __init__(self, *a, **k): pass
        def __call__(self, *a, **k):
            z = torch.tensor(0.0)
            return z, z, z, z, {}
    dosadmt.InPlaceFusionDMT = _DMTZero
    dosadmt.SkipConnectionDMT = _DMTZero

    # ---- utils.calculate_macs ----
    dosautil = types.ModuleType("dosa.utils")
    sys.modules["dosa.utils"] = dosautil

    def calculate_macs(dims: dict):
        # 只要 dims 含 {N,K,C,R,S,P,Q} 这些，乘起来就是 Conv MACs
        vals = []
        for k, v in dims.items():
            if v is None: continue
            vals.append(int(v))
        if not vals:
            return torch.tensor(0.0)
        prod = float(reduce(mul, vals, 1))
        return torch.tensor(prod)
    dosautil.calculate_macs = calculate_macs


# ---------- 2) 载入你的模型文件 ----------
def load_model_module(model_file: str):
    model_path = pathlib.Path(model_file).resolve()
    spec = importlib.util.spec_from_file_location("perf_model_mod", model_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, "Failed to create spec for model file."
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# ---------- 3) 简单 Graph 包装 ----------
class _GraphShim:
    def __init__(self, layers: dict, fusion_groups: list[list[str]]):
        self.layers = layers
        self.fusion_groups = fusion_groups


# ---------- 4) CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-file", required=True, help="你的分析模型 .py 文件路径")
    ap.add_argument("--hw", required=True, help="HW 配置 JSON")
    ap.add_argument("--sw", required=True, help="SW 配置 JSON（layers/fusion_groups/mapping）")
    ap.add_argument("--dump", default="debug_performance_model.json", help="调试 JSON 输出路径（可选）")
    args = ap.parse_args()

    with open(args.hw, "r", encoding="utf-8") as f:
        hw_cfg = json.load(f)

    with open(args.sw, "r", encoding="utf-8") as f:
        sw_cfg = json.load(f)

    # 注入 shims（不改你的模型文件）
    inject_dosa_shims(hw_cfg)

    # 导入你的模型模块
    mod = load_model_module(args.model_file)

    # 构造 Config/HardwareParameters/Graph
    from dosa.config import Config
    from dosa.hardware_parameters import HardwareParameters

    cfg = Config.get_instance()

    buffers = hw_cfg.get("buffers_kb", {
        "L0_Registers": 16,
        "L1_Accumulator": 256,
        "L2_Scratchpad": 2048,
        "L3_DRAM": 0
    })
    pe_count = hw_cfg.get("pe_count", 256)
    hw_params = HardwareParameters(buffers, pe_count)

    graph = _GraphShim(layers=sw_cfg["layers"], fusion_groups=sw_cfg["fusion_groups"])

    # 使用 direct_mapping_table（避免依赖真实 Mapping 类）
    direct_mapping = sw_cfg["mapping"]

    # 构建模型并执行 forward
    model = mod.HighFidelityPerformanceModel(cfg, debug_latency=True, fusion_aware=False)
    with torch.no_grad():
        total_latency, total_energy, area_cost, mismatch_loss, comp_penalty = model(
            graph=graph,
            hw_params=hw_params,
            mapping=None,
            fusion_params=None,
            direct_mapping_table=direct_mapping,
            debug_output_path=args.dump if args.dump else None
        )

    # 输出结果
    def _val(x): 
        return float(x.detach().cpu().item() if isinstance(x, torch.Tensor) else float(x))

    print("=== Results ===")
    print(f"Total latency (s): { _val(total_latency):.6e}")
    print(f"Total energy  (pJ): { _val(total_energy):.6e}")
    print(f"Area cost     (arb): { _val(area_cost):.6e}")
    print(f"Buffer mismatch loss: { _val(mismatch_loss):.6e}")
    print(f"Compatibility penalty: { _val(comp_penalty):.6e}")
    print(f"Debug JSON dumped to: { args.dump }")

if __name__ == "__main__":
    main()
