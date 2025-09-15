"""Baseline runners for Act I experiment (skeleton).

Provides a unified interface `get_baseline_runner(name)` that returns an object
with a `.run(cfg: dict, seed: int, recorder: Recorder)` method.
Actual optimization logic will be implemented later.
"""
from __future__ import annotations

from typing import Protocol, Any


class Runner(Protocol):
    """Protocol for baseline runner objects."""

    def run(self, cfg: dict[str, Any], seed: int, recorder: "Recorder") -> None:  # noqa: D401
        ...


import random
from datetime import datetime
import torch

from dosa.config import Config
from dosa.hardware_parameters import HardwareParameters
from dosa.mapping import FineGrainedMapping
from dosa.performance_model import HighFidelityPerformanceModel
import os
from dosa.utils import ComputationGraph, FusionParameters
from dosa.searcher import FADOSASearcher
from run import parse_onnx_to_graph


class _BaseSearchRunner:
    """Common utilities for all real baseline runners."""

    def __init__(self, name: str) -> None:
        self.name = name

    def _build_components(self, cfg: dict[str, Any], recorder, model_name: str | None = None):
        # 使用绝对路径确保能找到ONNX文件
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        onnx_path = os.path.join(project_root, f"onnx_models/{model_name}.onnx") if model_name else None
        print(f"[BASELINE] 检查工作负载: {model_name}")
        print(f"[BASELINE] ONNX路径: {onnx_path}")
        print(f"[BASELINE] 当前工作目录: {os.getcwd()}")
        print(f"[BASELINE] ONNX文件存在: {os.path.exists(onnx_path) if onnx_path else False}")
        
        if model_name and onnx_path and os.path.exists(onnx_path):
            print(f"[BASELINE] 使用ONNX模型: {model_name}")
            graph = parse_onnx_to_graph(model_name)
        else:
            print(f"[BASELINE] 使用fallback图")
            graph = self._create_fallback_graph()
        config = Config.get_instance()
        
        # 设置优化参数（方案B）
        config.NUM_OUTER_STEPS = cfg.get("num_outer_steps", 2)
        config.NUM_MAPPING_STEPS = cfg.get("num_mapping_steps", 20)
        config.NUM_HARDWARE_STEPS = cfg.get("num_hardware_steps", 20)
        config.LR_MAPPING = cfg.get("lr_mapping", 0.01)
        config.LR_HARDWARE = cfg.get("lr_hardware", 0.01)
        
        device = config.DEVICE
        scenario = cfg.get("scenario")
        self.scenario = scenario
        if scenario:
            init_hw = config.SCENARIO_PRESETS.get(scenario, {}).get("initial_hw", {})
            hw = HardwareParameters(
                initial_num_pes=init_hw.get("num_pes", 4.0),
                initial_l0_kb=init_hw.get("l0_kb", 0.1),
                initial_l1_kb=init_hw.get("l1_kb", 0.2),
                initial_l2_kb=init_hw.get("l2_kb", 1.0),
            ).to(device)
        else:
            hw = HardwareParameters().to(device)
        mapping = FineGrainedMapping(graph.problem_dims, config.MEMORY_HIERARCHY).to(device)
        fusion = FusionParameters(graph).to(device)
        perf_model = HighFidelityPerformanceModel(config).to(device)
        searcher = FADOSASearcher(graph, hw, mapping, fusion, perf_model, config, recorder=recorder)
        return graph, searcher

    def _create_fallback_graph(self) -> ComputationGraph:
        graph = ComputationGraph()
        for i in range(2):
            dims_conv = {
                "N": 1,
                "C": 64 * (2 ** (i // 2)),
                "K": 64 * (2 ** (i // 2)),
                "P": 56 // (2 ** (i // 2)),
                "Q": 56 // (2 ** (i // 2)),
                "R": 3,
                "S": 3,
            }
            dims_relu = dims_conv.copy()
            graph.add_layer(f"conv_{i}", dims_conv, "Conv")
            graph.add_layer(f"relu_{i}", dims_relu, "ReLU")
            graph.add_fusion_group([f"conv_{i}", f"relu_{i}"])
            graph.add_fusion_group([f"conv_{i}"])
            graph.add_fusion_group([f"relu_{i}"])
        return graph

    # ---------------- Constraint helpers ----------------
    def _apply_constraints(self, hw_params, mapping, fusion_params, graph, kind: str) -> None:  # noqa: D401
        """Freeze parameters/initialize values according to baseline kind."""
        pass  # To be overridden by subclasses

    # Helper: initialize hardware params from current scenario preset
    def _init_hw_from_scenario(self, hw_params):
        config = Config.get_instance()
        scenario = getattr(self, "scenario", None)
        if not scenario:
            return
        init_hw = config.SCENARIO_PRESETS.get(scenario, {}).get("initial_hw", {})
        device = hw_params.log_num_pes.device
        with torch.no_grad():
            if "num_pes" in init_hw:
                hw_params.log_num_pes.data = torch.log(torch.tensor(float(init_hw["num_pes"]), device=device))
            if "l0_kb" in init_hw:
                hw_params.log_buffer_sizes_kb["L0_Registers"].data = torch.log(torch.tensor(float(init_hw["l0_kb"]), device=device))
            if "l1_kb" in init_hw:
                hw_params.log_buffer_sizes_kb["L1_Accumulator"].data = torch.log(torch.tensor(float(init_hw["l1_kb"]), device=device))
            if "l2_kb" in init_hw:
                hw_params.log_buffer_sizes_kb["L2_Scratchpad"].data = torch.log(torch.tensor(float(init_hw["l2_kb"]), device=device))

    # Helper: ensure hardware area does not exceed budget tolerance
    def _ensure_area_within_budget(self, hw_params):
        config = Config.get_instance()
        budget = getattr(config, "AREA_BUDGET_MM2", None)
        tolerance = getattr(config, "AREA_BUDGET_TOLERANCE", 0.0)
        if budget is None:
            return
        limit = budget * (1 + tolerance)
        area = hw_params.get_area_cost().item()
        if area <= limit:
            return
        base = config.AREA_BASE_MM2
        variable = area - base
        allowed = limit - base
        if variable <= 0 or allowed <= 0:
            print(f"[WARN] Base hardware area {base:.2f}mm² exceeds budget {limit:.2f}mm²")
            return
        scale = allowed / variable
        scale_tensor = torch.log(torch.tensor(scale, device=hw_params.log_num_pes.device))
        with torch.no_grad():
            hw_params.log_num_pes.data += scale_tensor
            for param in hw_params.log_buffer_sizes_kb.values():
                param.data += scale_tensor
        final_area = hw_params.get_area_cost().item()
        print(
            f"[WARN] Scaled hardware by {scale:.3f} to meet area budget: {final_area:.2f}mm² (limit {limit:.2f}mm²)"
        )

    # Deprecated method retained for compatibility (no-op)
    def _override_params(self, params: dict[str, Any], graph, kind: str) -> None:  # noqa: D401
        return

    def run(self, cfg: dict[str, Any], seed: int, recorder: "Recorder") -> None:  # noqa: D401
        import time
        random.seed(seed)
        torch.manual_seed(seed)
        # 从配置文件中正确获取工作负载名称
        workload_name = cfg.get("workload", {}).get("name", "resnet18")
        # 如果workload_name包含下划线，取第一部分作为模型名称
        model_name = workload_name.split("_")[0] if "_" in workload_name else workload_name
        
        print(f"\n{'='*60}")
        print(f"Starting {self.name.upper()} Baseline Experiment")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Seed: {seed}")
        print(f"Scenario: {cfg['shared'].get('scenario', 'default')}")
        
        graph, searcher = self._build_components(cfg["shared"], recorder, model_name)
        
        # 在 graph, searcher = self._build_components(...) 之后立刻加：
        setattr(searcher, "runner_name", self.name)
        
        # 只对 baselineB 生效：跳过恢复 + 禁用冻结离散映射
        if self.name == "baselineB":
            setattr(searcher, "skip_restore_best_mapping", True)
            setattr(searcher, "_freeze_discrete", False)
            setattr(searcher, "best_discrete_factors", None)
        
        print(f"Graph layers: {len(graph.layers)}")
        print(f"Fusion groups: {len(graph.fusion_groups)}")
        
        # 在搜索前按基准类型应用参数冻结/初始化约束
        self._apply_constraints(searcher.hw_params, searcher.mapping, searcher.fusion_params, graph, self.name)
        
        # 打印初始化后的两种口径，明确基线
        print("[DEBUG] runner.run() after build+apply_constraints")
        from dosa.searcher import _dump_mapping_raw, _dump_mapping_projected
        _dump_mapping_raw(searcher.mapping, tag="[RAW][runner_init]")
        proj0 = _dump_mapping_projected(searcher.mapping, tag="[PROJ][runner_init]")
        
        # 把"diff 的基线"就定在 **projected 的这份**，避免把"初始化重写"当变化
        try:
            setattr(searcher, "_prev_mapping_state_for_debug", proj0)
            print("[DEBUG] set prev_mapping_state baseline to projected(init)")
        except Exception:
            pass
        
        from pprint import pformat
        snap_after_apply = searcher._snapshot_mapping()
        print("[DEBUG] baseline init mapping snapshot:\n" + pformat(snap_after_apply)[:800])

        num_trials = cfg["shared"].get("num_trials", 30)
        print(f"Number of trials: {num_trials}")
        
        start_ts = datetime.now().isoformat(timespec="seconds")
        start_time = time.time()
        print(f"Search started at: {start_ts}")

        # 主动优化：调用 searcher.search()
        print(f"\n--- Running {self.name} Search ---")
        results = searcher.search(num_trials)

        # 完成后刷新 Recorder 最佳记录
        recorder.finalize_best()
        end_ts = datetime.now().isoformat(timespec="seconds")
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n--- Search Completed in {duration:.2f}s ---")
        
        # 输出详细的搜索结果摘要
        if hasattr(results, 'get') and results.get('best_edp_metrics'):
            best_metrics = results.get('best_edp_metrics', {})
            best_loss = results.get('best_loss', 'N/A')
            print(f"Best Loss: {best_loss:.4f}" if isinstance(best_loss, (int, float)) else f"Best Loss: {best_loss}")
            
            edp = best_metrics.get('edp', 'N/A')
            print(f"Best EDP: {edp:.2e}" if isinstance(edp, (int, float)) else f"Best EDP: {edp}")
            
            area = best_metrics.get('area_mm2', 'N/A')
            print(f"Best Area: {area:.2f}mm²" if isinstance(area, (int, float)) else f"Best Area: {area}")
            
            latency = best_metrics.get('latency_ms', 'N/A')
            print(f"Best Latency: {latency:.2f}ms" if isinstance(latency, (int, float)) else f"Best Latency: {latency}")
            
            energy = best_metrics.get('energy_mj', 'N/A')
            print(f"Best Energy: {energy:.2f}mJ" if isinstance(energy, (int, float)) else f"Best Energy: {energy}")
        elif hasattr(searcher, 'best_edp_metrics') and searcher.best_edp_metrics:
            best_metrics = searcher.best_edp_metrics
            best_loss = getattr(searcher, 'best_loss', 'N/A')
            print(f"Best Loss: {best_loss:.4f}" if isinstance(best_loss, (int, float)) else f"Best Loss: {best_loss}")
            
            edp = best_metrics.get('edp', 'N/A')
            print(f"Best EDP: {edp:.2e}" if isinstance(edp, (int, float)) else f"Best EDP: {edp}")
            
            area = best_metrics.get('area_mm2', 'N/A')
            print(f"Best Area: {area:.2f}mm²" if isinstance(area, (int, float)) else f"Best Area: {area}")
            
            latency = best_metrics.get('latency_ms', 'N/A')
            print(f"Best Latency: {latency:.2f}ms" if isinstance(latency, (int, float)) else f"Best Latency: {latency}")
            
            energy = best_metrics.get('energy_mj', 'N/A')
            print(f"Best Energy: {energy:.2f}mJ" if isinstance(energy, (int, float)) else f"Best Energy: {energy}")
        else:
            print("No valid solutions found or metrics unavailable.")
            
        print(f"Total Trials: {num_trials}")
        print(f"Duration: {start_ts} → {end_ts} ({duration:.2f}s)")
        print(f"[{self.name}] seed={seed} completed successfully.")
        print(f"{'='*60}\n")


class MappingOnlyA1Runner(_BaseSearchRunner):
    def __init__(self) -> None:
        super().__init__("baselineA_A1")

    # 仅优化映射/融合，硬件固定
    def _apply_constraints(self, hw_params, mapping, fusion_params, graph, kind: str) -> None:  # noqa: D401
        config = Config.get_instance()
        config.APPLY_MIN_HW_BOUNDS = False
        print("[DEBUG] A1 baseline: APPLY_MIN_HW_BOUNDS disabled; hardware fixed")
        # 初始化硬件为场景预设并冻结
        self._init_hw_from_scenario(hw_params)
        for p in hw_params.parameters():
            p.requires_grad = False
        # 解冻映射与融合参数（默认即可，无需显式设置）

        # 固定融合策略 (较少融合)
        if graph.fusion_groups:
            with torch.no_grad():
                fusion_params.fusion_logits.data = torch.full_like(fusion_params.fusion_logits, -2.0)

        # 确保初始硬件面积在预算容忍范围内
        self._ensure_area_within_budget(hw_params)


class MappingOnlyA2Runner(_BaseSearchRunner):
    def __init__(self) -> None:
        super().__init__("baselineA_A2")

    # 同 A1 但允许融合搜索
    def _apply_constraints(self, hw_params, mapping, fusion_params, graph, kind: str) -> None:  # noqa: D401
        # 初始化硬件为场景预设并冻结
        self._init_hw_from_scenario(hw_params)
        for p in hw_params.parameters():
            p.requires_grad = False

        # 确保初始硬件面积在预算容忍范围内
        self._ensure_area_within_budget(hw_params)


class HardwareOnlyRunner(_BaseSearchRunner):
    def __init__(self) -> None:
        super().__init__("baselineB")

    # 仅优化硬件，映射/融合固定
    def _apply_constraints(self, hw_params, mapping, fusion_params, graph, kind: str) -> None:  # noqa: D401
        # 冻结映射与融合参数
        for p in mapping.parameters():
            p.requires_grad = False
        for p in fusion_params.parameters():
            p.requires_grad = False

        # 设置映射参数为统一的无优化基准（所有分解因子=1，log(1)=0）
        with torch.no_grad():
            for level_factors in mapping.factors.values():
                for dim_dict in level_factors.values():
                    dim_dict["temporal"].data.fill_(0.0)
                    dim_dict["spatial"].data.fill_(0.0)
            # 可选: 提升 P/Q 的 spatial 至 2 用于提高并行度
            if "P" in graph.problem_dims and "L0_Registers" in mapping.factors:
                mapping.factors["L0_Registers"]["P"]["spatial"].data.fill_(torch.log(torch.tensor(2.0)))
            if "Q" in graph.problem_dims and "L0_Registers" in mapping.factors:
                mapping.factors["L0_Registers"]["Q"]["spatial"].data.fill_(torch.log(torch.tensor(2.0)))

        # 融合设置为极少融合
        if graph.fusion_groups:
            with torch.no_grad():
                fusion_params.fusion_logits.data = torch.full_like(fusion_params.fusion_logits, -2.0)

        # 调试打印
        print("[DEBUG] baselineB._apply_constraints() done.")
        from dosa.searcher import _dump_mapping_raw, _dump_mapping_projected
        _dump_mapping_raw(mapping, tag="[RAW][after_apply]")
        _dump_mapping_projected(mapping, tag="[PROJ][after_apply]")


class CooptRunner(_BaseSearchRunner):
    def __init__(self) -> None:
        super().__init__("coopt")

    # 协同优化：仍允许搜索，但从专家挑选的较优初始点出发
    def _apply_constraints(self, hw_params, mapping, fusion_params, graph, kind: str) -> None:  # noqa: D401
        device = hw_params.log_num_pes.device
        # 1) 使用场景预设的硬件初始值
        self._init_hw_from_scenario(hw_params)

        with torch.no_grad():
            # 2) 初始化映射因子为可行的基础方案
            for level_factors in mapping.factors.values():
                for dim_dict in level_factors.values():
                    dim_dict["temporal"].data.fill_(0.0)  # log(1)
                    dim_dict["spatial"].data.fill_(0.0)   # log(1)

            # 在寄存器层为输出空间引入适度并行度
            if "P" in graph.problem_dims and "L0_Registers" in mapping.factors:
                mapping.factors["L0_Registers"]["P"]["spatial"].data.fill_(torch.log(torch.tensor(2.0, device=device)))
            if "Q" in graph.problem_dims and "L0_Registers" in mapping.factors:
                mapping.factors["L0_Registers"]["Q"]["spatial"].data.fill_(torch.log(torch.tensor(2.0, device=device)))

            # 3) 融合概率初始化为中性值，鼓励搜索但不偏向任一方案
            if graph.fusion_groups:
                fusion_params.fusion_logits.data = torch.zeros_like(fusion_params.fusion_logits)

        # 确保初始硬件面积在预算容忍范围内
        self._ensure_area_within_budget(hw_params)


class ParetoFrontierRunner(_BaseSearchRunner):
    """Pareto Frontier Runner: Scans the area-performance trade-off space by varying loss weights."""
    
    def __init__(self, name: str = "ParetoFrontier") -> None:
        super().__init__(name)
        # Define area weight sweep range for Pareto frontier
        # Use class attribute if set, otherwise use default
        if hasattr(self.__class__, '_test_area_weights'):
            self.area_weights = self.__class__._test_area_weights
        else:
            self.area_weights = [0.0, 0.1, 0.5, 2.0, 10.0]
        self.pareto_results = []
        self.last_best_params = None  # Store best params from previous run
    
    def _apply_constraints(self, hw_params, mapping, fusion_params, graph, kind: str) -> None:  # noqa: D401
        """Apply constraints, starting from last best parameters if available."""
        if self.last_best_params:
            print("Loading parameters from previous best run.")
            hw_params.load_state_dict(self.last_best_params["hw"])
            mapping.load_state_dict(self.last_best_params["mapping"])
            fusion_params.load_state_dict(self.last_best_params["fusion"])
            return

        device = hw_params.log_num_pes.device
        with torch.no_grad():
            # Set initial hardware scale
            hw_params.log_num_pes.data = torch.log(torch.tensor(256.0, device=device))
            hw_params.log_buffer_sizes_kb["L0_Registers"].data = torch.log(torch.tensor(4.0, device=device))
            hw_params.log_buffer_sizes_kb["L1_Accumulator"].data = torch.log(torch.tensor(16.0, device=device))
            hw_params.log_buffer_sizes_kb["L2_Scratchpad"].data = torch.log(torch.tensor(512.0, device=device))
            
            # Initialize mapping factors to a reasonable baseline
            for level_factors in mapping.factors.values():
                for dim_dict in level_factors.values():
                    dim_dict["temporal"].data.fill_(0.0)  # log(1)
                    dim_dict["spatial"].data.fill_(0.0)   # log(1)
            
            # Initialize fusion probabilities to neutral values
            if graph.fusion_groups:
                fusion_params.fusion_logits.data = torch.zeros_like(fusion_params.fusion_logits)
    
    def run(self, cfg: dict[str, Any], seed: int, recorder: "Recorder") -> None:  # noqa: D401
        """Run Pareto frontier sweep by varying area weights."""
        import copy
        from dosa.config import Config
        
        random.seed(seed)
        torch.manual_seed(seed)
        
        print(f"Starting Pareto Frontier sweep with {len(self.area_weights)} area weight points...")
        
        for i, area_weight in enumerate(self.area_weights):
            print(f"\n=== Pareto Point {i+1}/{len(self.area_weights)}: Area Weight = {area_weight} ===")
            
            # Create fresh components for each weight point
            graph, searcher = self._build_components(cfg["shared"], recorder, cfg["shared"].get("model_name"))
            
            # Update area weight using the searcher's dynamic method
            searcher.update_loss_weights({'area_weight': area_weight})
            
            # Apply constraints
            self._apply_constraints(searcher.hw_params, searcher.mapping, searcher.fusion_params, graph, self.name)
            
            # Run search for this weight point
            num_trials = cfg["shared"].get("num_trials", 30)
            searcher.search(num_trials)

            # Store the best parameters for the next run
            self.last_best_params = {
                "hw": copy.deepcopy(searcher.hw_params.state_dict()),
                "mapping": copy.deepcopy(searcher.mapping.state_dict()),
                "fusion": copy.deepcopy(searcher.fusion_params.state_dict()),
            }
            
            # Store result with area weight info
            pareto_point = {
                'area_weight': area_weight,
                'seed': seed,
                'trials': num_trials
            }
            self.pareto_results.append(pareto_point)
        
        # Finalize recorder
        recorder.finalize_best()
        
        print(f"\n=== Pareto Frontier Sweep Complete ===")
        print(f"Generated {len(self.pareto_results)} Pareto points")


def get_baseline_runner(name: str) -> Runner:  # noqa: D401
    """Factory returning baseline runner by name."""
    mapping = {
        "baselineA_A1": MappingOnlyA1Runner,
        "baselineA_A2": MappingOnlyA2Runner,
        "baselineB": HardwareOnlyRunner,
        "coopt": CooptRunner,
        "pareto_frontier": ParetoFrontierRunner,
    }
    if name not in mapping:
        raise ValueError(f"Unknown baseline runner: {name}")
    return mapping[name]()
