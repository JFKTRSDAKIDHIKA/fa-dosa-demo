# probe_fusion_chain.py
# 目的：手动指定输入（问题维度、融合链、映射、边界概率/Logit 等），只做一次前向，输出可对标的细粒度统计。
import math
import json
import argparse
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn

# ===== 按你的项目结构导入 =====
from dosa.config import Config
from dosa.hardware_parameters import HardwareParameters
from dosa.mapping import FineGrainedMapping
from dosa.performance_model import HighFidelityPerformanceModel

# =============== 1) 输入规范 ===============
# 说明：你可用命令行 --input 指定 JSON；若不指定，则使用下方 INPUT_SPEC。
# 注意：这里不放具体数值样例，以免干扰你自己的 Golden 配置；仅给出字段结构。
INPUT_SPEC = {
    "device": "cuda",                # 用GPU
    "problem_dims": {                # 全局问题维度（这里只是baseline，可按layer覆盖）
        "N": 1,
        "C": 64,
        "K": 256,
        "P": 32,
        "Q": 32,
        "R": 3,
        "S": 3
    },
    "hardware": {                    # 硬件配置
        "initial_num_pes": 64,
        "initial_l0_kb": 2,
        "initial_l1_kb": 4,
        "initial_l2_kb": 64
    },
    "graph": {
        "layers": [
            {
                "name": "conv1",
                "type": "Conv",
                "dims": {"N":1, "C":3, "K":16, "P":32, "Q":32, "R":3, "S":3}
            },
            {
                "name": "conv2",
                "type": "Conv",
                "dims": {"N":1, "C":16, "K":32, "P":32, "Q":32, "R":3, "S":3}
            }
        ],
        "fusion_groups": [
            ["conv1","conv2"]
        ],
        "layer_order": [
            "input","conv1","conv2","output"
        ]
    },
    "mappings": {
  "conv1": {
    "discrete_factors": {
      "L0_Registers": {
        "temporal": {
          "P": 1,
          "Q": 1,
          "R": 1,
          "S": 1,
          "K": 1,
          "C": 1,
          "N": 1
        },
        "spatial": {
          "K": 4,
          "C": 3
        }
      },
      "L1_Accumulator": {
        "temporal": {
          "P": 1,
          "Q": 1,
          "R": 3,
          "S": 3,
          "K": 1,
          "C": 1,
          "N": 1
        }
      },
      "L2_Scratchpad": {
        "temporal": {
          "P": 1,
          "Q": 1,
          "R": 1,
          "S": 1,
          "K": 1,
          "C": 1,
          "N": 1
        }
      }
    }
  },
  "conv2": {
    "discrete_factors": {
      "L0_Registers": {
        "temporal": {
          "P": 1,
          "Q": 1,
          "R": 1,
          "S": 1,
          "K": 1,
          "C": 1,
          "N": 1
        },
        "spatial": {
          "K": 8,
          "C": 4
        }
      },
      "L1_Accumulator": {
        "temporal": {
          "P": 1,
          "Q": 1,
          "R": 3,
          "S": 3,
          "K": 1,
          "C": 1,
          "N": 1
        }
      },
      "L2_Scratchpad": {
        "temporal": {
          "P": 1,
          "Q": 1,
          "R": 1,
          "S": 1,
          "K": 1,
          "C": 1,
          "N": 1
        }
      }
    }
  }
},
    "fusion_boundaries": {
        "conv1->conv2": {"s": 0.99},
        "conv2->conv3": {"s": 0.99}
    },
    "print_options": {
        "show_lb": True,
        "show_per_layer_traffic": True,
        "show_group_counters": True
    }
}


# =============== 2) 实用函数（构建图 / 设置参数 / 打印） ===============
def _safe_ln(x: float) -> float:
    if x is None:
        raise ValueError("离散 factor 缺失具体值")
    if x <= 0:
        raise ValueError(f"离散 factor 必须 > 0，收到 {x}")
    return math.log(float(x))

def build_graph_from_spec(spec: Dict[str, Any]):
    """按 spec 构建一个简单图对象；只要字段契合你现有 HighFidelityPerformanceModel 的需求即可。"""
    dims_global = spec["problem_dims"]
    gspec = spec["graph"]
    # 这里用最小实现（轻量 Mock），与你工程中 evaluate_* 的接口字段保持一致
    class G:
        def __init__(self, dims, gspec):
            self.problem_dims = dims
            self.layers = {}
            # 可选自动补 input/output
            if "layer_order" in gspec:
                self.layer_order = gspec["layer_order"]
            else:
                self.layer_order = [x["name"] for x in gspec["layers"]]

            for info in gspec["layers"]:
                name = info["name"]
                ltype = info.get("type", "Conv")
                ldims = info.get("dims", dims)  # 若单层覆盖 dims
                # 约定 Conv 的 shape 字段（与你工程一致即可）
                node = {"type": ltype, "dims": ldims}
                if ltype == "Conv":
                    node["input_shape"]  = [ldims["N"], ldims["C"], ldims["P"]+ldims["R"]-1, ldims["Q"]+ldims["S"]-1]
                    node["output_shape"] = [ldims["N"], ldims["K"], ldims["P"], ldims["Q"]]
                    node["weight_shape"] = [ldims["K"], ldims["C"], ldims["R"], ldims["S"]]
                self.layers[name] = node

            self.fusion_groups = gspec["fusion_groups"]
            self.adjacency = {}  # 如果你的实现用到，可在此构造
    return G(dims_global, gspec)

def create_mapping_for_layer(layer_dims: Dict[str, int], mem_hierarchy) -> FineGrainedMapping:
    m = FineGrainedMapping(layer_dims, mem_hierarchy)
    # 默认给个稳定初始化（不会优化，所以只为避免奇异值）
    for i, p in enumerate(m.parameters()):
        p.data.fill_(1.0 if i != 1 else 2.0)
    return m

def apply_log_params(mapping: FineGrainedMapping, log_param_dict: Dict[str, float]):
    """方式 A：直接把 log-param 名字精确写进来（与 named_parameters 对齐）"""
    named = dict(mapping.named_parameters())
    for k, v in log_param_dict.items():
        if k not in named:
            raise KeyError(f"未找到映射参数: {k}")
        named[k].data.fill_(float(v))

def apply_discrete_factors(mapping: FineGrainedMapping, disc: Dict[str, Any]):
    """
    方式 B：给出每层级/temporal|spatial/维度的离散值，脚本自动回填 log=ln(value)
    约束：值必须 > 0；未给到的项不改动。
    """
    named = dict(mapping.named_parameters())
    # 你的 FineGrainedMapping 的命名规则按你发来的风格：<level>.<dim>.<temporal|spatial>
    # 我们在此按照 disc 结构拼接键名并写入 ln(value)
    for level, tmap in disc.items():  # e.g. "L2_Scratchpad": {"temporal":{"K":4,...}, "spatial":{"P":1,...}}
        for kind in ("temporal", "spatial"):
            if kind not in tmap: 
                continue
            for dim, val in tmap[kind].items():
                key = f"factors.{level}.{dim}.{kind}"
                if key not in named:
                    # 有的实现用 <level>.<dim>.temporal 等在一个命名空间；有的以稍异名。必要时你在此对齐一下
                    raise KeyError(f"未找到映射参数: {key}")
                named[key].data.fill_(_safe_ln(val))

def build_layer_mappings(spec: Dict[str, Any], graph) -> Dict[str, FineGrainedMapping]:
    layer2mapping = {}
    for lname, node in graph.layers.items():
        if node["type"] != "Conv":
            continue
        m = create_mapping_for_layer(node["dims"], Config.get_instance().MEMORY_HIERARCHY)
        user = spec["mappings"].get(lname, {})
        if "log_params" in user and user["log_params"]:
            apply_log_params(m, user["log_params"])
        if "discrete_factors" in user and user["discrete_factors"]:
            apply_discrete_factors(m, user["discrete_factors"])
        layer2mapping[lname] = m
    return layer2mapping

def set_fusion_boundaries(perf: HighFidelityPerformanceModel, group_layers: List[str], fb: Dict[str, Any]):
    # 确保初始化过边界参数（若你的模型方法名不同，请替换）
    perf.init_fusion_boundaries(group_layers)
    for i in range(len(group_layers)-1):
        key = f"{group_layers[i]}->{group_layers[i+1]}"
        if key not in fb:
            continue
        item = fb[key]
        if "logit" in item and item["logit"] is not None:
            perf.fusion_boundary_logits[key].data.fill_(float(item["logit"]))
        elif "s" in item and item["s"] is not None:
            s = float(item["s"])
            # clip 避免 inf
            s = min(max(s, 1e-6), 1-1e-6)
            logit = math.log(s/(1.0-s))
            perf.fusion_boundary_logits[key].data.fill_(logit)

def tensor_to_float(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return float(x.item())
    if isinstance(x, dict):
        return {k: tensor_to_float(v) for k, v in x.items()}
    if isinstance(x, list):
        return [tensor_to_float(v) for v in x]
    return x

def run_single_forward(spec: Dict[str, Any]) -> Dict[str, Any]:
    # =============== 配置 ===============
    cfg = Config.get_instance()
    if spec.get("device"):
        cfg.DEVICE = spec["device"]

    perf = HighFidelityPerformanceModel(cfg)

    # =============== 硬件参数 ===============
    h = spec["hardware"]
    hw = HardwareParameters(
        initial_num_pes=float(h["initial_num_pes"]),
        initial_l0_kb=float(h["initial_l0_kb"]),
        initial_l1_kb=float(h["initial_l1_kb"]),
        initial_l2_kb=float(h["initial_l2_kb"])
    )

    # =============== 图与融合链 ===============
    graph = build_graph_from_spec(spec)
    if not spec["graph"]["fusion_groups"]:
        raise ValueError("fusion_groups 为空")
    group = spec["graph"]["fusion_groups"][0]  # 只用第一条链

    # =============== per-layer 映射 & 融合边界参数 ===============
    layer2mapping = build_layer_mappings(spec, graph)
    set_fusion_boundaries(perf, group, spec.get("fusion_boundaries", {}))

    # =============== 前向执行 ===============
    with torch.no_grad():
        # forward 接口返回更完整的指标
        latency, energy, area, mismatch, compat, invalid_penalty, penalty = perf(
            graph=graph,
            hw_params=hw,
            layer2mapping=layer2mapping,
            fusion_params=None
        )

    # =============== 结果结构化 ===============
    out: Dict[str, Any] = {
        "fusion_group": group,
        "fusion_boundaries": [
            {
                "key": f"{group[i]}->{group[i+1]}",
                "logit": float(perf.fusion_boundary_logits[f"{group[i]}->{group[i+1]}"].item()),
                "s": float(torch.sigmoid(perf.fusion_boundary_logits[f"{group[i]}->{group[i+1]}"]).item())
            }
            for i in range(len(group)-1)
        ],
        "chain_metrics": {
            "latency": float(latency.item()),
            "energy":  float(energy.item()),
            "area":    float(area.item()),
            "mismatch_loss": float(mismatch.item()) if isinstance(mismatch, torch.Tensor) else float(mismatch),
            "compat": float(compat.item()) if isinstance(compat, torch.Tensor) else float(compat),
            "mapping_invalid_penalty": float(invalid_penalty),
            "penalty": float(penalty.item())
        }
    }

    return out

# =============== 4) 人类可读打印 ===============
def pretty_print(out: Dict[str, Any], spec: Dict[str, Any]):
    print("\n" + "="*80)
    print("🔗 Fusion Boundaries")
    for b in out.get("fusion_boundaries", []):
        print(f"  {b['key']:>20s}  logit={b['logit']:+.6f}  s={b['s']:.6f}")

    print("\n" + "="*80)
    print("📊 Chain-level Metrics")
    cm = out.get("chain_metrics", {})
    print(f"  Latency={cm.get('latency', 0.0):.6e}  Energy={cm.get('energy', 0.0):.6e}  Area={cm.get('area', 0.0):.6e}")
    print(f"  mismatch={cm.get('mismatch_loss', 0.0):.6e}  compat={cm.get('compat', 0.0):.6e}")
    print(f"  invalid_penalty={cm.get('mapping_invalid_penalty', 0.0):.6e}  penalty={cm.get('penalty', 0.0):.6e}")

    if spec.get("print_options", {}).get("show_group_counters", True):
        gr = out.get("group_counters", {
            "L3_service_bytes": 0.0,
            "L2_service_bytes": 0.0,
            "W_L2_to_L0_bytes": 0.0
        })
        print("\n" + "="*80)
        print("📦 Group Counters")
        print(f"  L3_service_bytes : {gr['L3_service_bytes']:.6e}")
        print(f"  L2_service_bytes : {gr['L2_service_bytes']:.6e}")
        print(f"  W(L2->L0)_bytes  : {gr['W_L2_to_L0_bytes']:.6e}")

    if spec.get("print_options", {}).get("show_per_layer_traffic", True) and "layers" in out:
        print("\n" + "="*80)
        print("🧾 Per-layer Traffic (reads/writes/updates)")
        for L in out.get("layers", []):
            print(f"\n— {L['name']}")
            for cat in ("reads","writes","updates"):
                m = L.get(cat, {})
                if not m:
                    print(f"  {cat}: (none)")
                    continue
                print(f"  {cat}:")
                for lvl, kv in m.items():
                    total = sum(kv.values())
                    w = kv.get("Weight", 0.0); i = kv.get("Input", 0.0); o = kv.get("Output", 0.0)
                    print(f"    {lvl:<16s} total={total:.3e}  [W={w:.3e} I={i:.3e} O={o:.3e}]")
            print(f"  L3_service={L.get('L3_service_bytes',0.0):.3e} | "
                  f"L2_service={L.get('L2_service_bytes',0.0):.3e} | "
                  f"W(L2->L0)={L.get('W_L2_to_L0_bytes',0.0):.3e}")

# =============== 5) CLI ===============
def main():
    parser = argparse.ArgumentParser(description="Fusion Chain Probe (No-Optimize, Single Forward)")
    parser.add_argument("--input", type=str, default=None, help="JSON spec file (same schema as INPUT_SPEC)")
    parser.add_argument("--output", type=str, default=None, help="Save outputs as JSON")
    args = parser.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            spec = json.load(f)
    else:
        spec = INPUT_SPEC

    out = run_single_forward(spec)
    pretty_print(out, spec)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 已写入输出到: {args.output}")

if __name__ == "__main__":
    torch.set_printoptions(sci_mode=True, precision=6)
    main()
