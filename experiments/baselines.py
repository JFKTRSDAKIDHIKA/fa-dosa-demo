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
from dosa.utils import ComputationGraph, FusionParameters
from dosa.searcher import FADOSASearcher


class _BaseSearchRunner:
    """Common utilities for all real baseline runners."""

    def __init__(self, name: str) -> None:
        self.name = name

    def _build_components(self, cfg: dict[str, Any], recorder):
        graph = self._create_fallback_graph()
        config = Config.get_instance()
        device = config.DEVICE
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

    # Deprecated method retained for compatibility (no-op)
    def _override_params(self, params: dict[str, Any], graph, kind: str) -> None:  # noqa: D401
        return

    def run(self, cfg: dict[str, Any], seed: int, recorder: "Recorder") -> None:  # noqa: D401
        random.seed(seed)
        torch.manual_seed(seed)
        graph, searcher = self._build_components(cfg["shared"], recorder)
        # 在搜索前按基准类型应用参数冻结/初始化约束
        self._apply_constraints(searcher.hw_params, searcher.mapping, searcher.fusion_params, graph, self.name)

        num_trials = cfg["shared"].get("num_trials", 30)
        start_ts = datetime.now().isoformat(timespec="seconds")

        # 主动优化：调用 searcher.search()
        searcher.search(num_trials)

        # 完成后刷新 Recorder 最佳记录
        recorder.finalize_best()
        end_ts = datetime.now().isoformat(timespec="seconds")
        print(f"[{self.name}] seed={seed} completed {num_trials} trials in {start_ts}→{end_ts}.")


class MappingOnlyA1Runner(_BaseSearchRunner):
    def __init__(self) -> None:
        super().__init__("baselineA_A1")

    # 仅优化映射/融合，硬件固定
    def _apply_constraints(self, hw_params, mapping, fusion_params, graph, kind: str) -> None:  # noqa: D401
        # 冻结所有硬件参数
        for p in hw_params.parameters():
            p.requires_grad = False
        # 解冻映射与融合参数（默认即可，无需显式设置）

        # 将硬件参数设为固定基准值
        hw_params.log_num_pes.data = torch.log(torch.tensor(128.0, device=hw_params.log_num_pes.device))
        hw_params.log_buffer_sizes_kb["L0_Registers"].data = torch.log(torch.tensor(2.0, device=hw_params.log_buffer_sizes_kb["L0_Registers"].device))
        hw_params.log_buffer_sizes_kb["L1_Accumulator"].data = torch.log(torch.tensor(8.0, device=hw_params.log_buffer_sizes_kb["L1_Accumulator"].device))
        hw_params.log_buffer_sizes_kb["L2_Scratchpad"].data = torch.log(torch.tensor(256.0, device=hw_params.log_buffer_sizes_kb["L2_Scratchpad"].device))

        # 固定融合策略 (较少融合)
        if graph.fusion_groups:
            with torch.no_grad():
                fusion_params.fusion_logits.data = torch.full_like(fusion_params.fusion_logits, -2.0)


class MappingOnlyA2Runner(_BaseSearchRunner):
    def __init__(self) -> None:
        super().__init__("baselineA_A2")

    # 同 A1 但允许融合搜索
    def _apply_constraints(self, hw_params, mapping, fusion_params, graph, kind: str) -> None:  # noqa: D401
        # 冻结硬件参数
        for p in hw_params.parameters():
            p.requires_grad = False

        # 设置固定硬件规模
        hw_params.log_num_pes.data = torch.log(torch.tensor(128.0, device=hw_params.log_num_pes.device))
        hw_params.log_buffer_sizes_kb["L0_Registers"].data = torch.log(torch.tensor(2.0, device=hw_params.log_buffer_sizes_kb["L0_Registers"].device))
        hw_params.log_buffer_sizes_kb["L1_Accumulator"].data = torch.log(torch.tensor(8.0, device=hw_params.log_buffer_sizes_kb["L1_Accumulator"].device))
        hw_params.log_buffer_sizes_kb["L2_Scratchpad"].data = torch.log(torch.tensor(256.0, device=hw_params.log_buffer_sizes_kb["L2_Scratchpad"].device))


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


class CooptRunner(_BaseSearchRunner):
    def __init__(self) -> None:
        super().__init__("coopt")

    # 协同优化：仍允许搜索，但从专家挑选的较优初始点出发
    def _apply_constraints(self, hw_params, mapping, fusion_params, graph, kind: str) -> None:  # noqa: D401
        device = hw_params.log_num_pes.device
        with torch.no_grad():
            # 1) 设定一个经验上效果较好的硬件规模
            hw_params.log_num_pes.data = torch.log(torch.tensor(256.0, device=device))
            hw_params.log_buffer_sizes_kb["L0_Registers"].data = torch.log(torch.tensor(4.0, device=device))
            hw_params.log_buffer_sizes_kb["L1_Accumulator"].data = torch.log(torch.tensor(16.0, device=device))
            hw_params.log_buffer_sizes_kb["L2_Scratchpad"].data = torch.log(torch.tensor(512.0, device=device))

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
            graph, searcher = self._build_components(cfg["shared"], recorder)
            
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
