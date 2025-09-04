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

    def _build_components(self, cfg: dict[str, Any]):
        graph = self._create_fallback_graph()
        config = Config.get_instance()
        device = config.DEVICE
        hw = HardwareParameters().to(device)
        mapping = FineGrainedMapping(graph.problem_dims, config.MEMORY_HIERARCHY).to(device)
        fusion = FusionParameters(graph).to(device)
        perf_model = HighFidelityPerformanceModel(config).to(device)
        searcher = FADOSASearcher(graph, hw, mapping, fusion, perf_model, config)
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

    def _override_params(self, params: dict[str, Any], graph, kind: str) -> None:
        """Apply baseline-specific constraints to sampled parameters."""

    def run(self, cfg: dict[str, Any], seed: int, recorder: "Recorder") -> None:  # noqa: D401
        random.seed(seed)
        torch.manual_seed(seed)
        graph, searcher = self._build_components(cfg["shared"])
        space = searcher.space
        num_trials = cfg["shared"].get("num_trials", 30)
        start_ts = datetime.now().isoformat(timespec="seconds")

        for t in range(1, num_trials + 1):
            params = space.sample()
            self._override_params(params, graph, self.name)
            flat = space.to_flat(params)
            loss, metrics = searcher.evaluate(flat)
            row = {"trial": t, "seed": seed, **metrics}
            recorder.record_trial(row)
            recorder.update_best(metrics, key="edp")

        recorder.finalize_best()
        end_ts = datetime.now().isoformat(timespec="seconds")
        print(f"[{self.name}] seed={seed} completed {num_trials} trials in {start_ts}→{end_ts}.")


class MappingOnlyA1Runner(_BaseSearchRunner):
    def __init__(self) -> None:
        super().__init__("baselineA_A1")

    def _override_params(self, params: dict[str, Any], graph, kind: str) -> None:  # noqa: D401
        params["num_pes"] = 128
        params["l0_registers_size_kb"] = 2.0
        params["l1_accumulator_size_kb"] = 8.0
        params["l2_scratchpad_size_kb"] = 256.0
        if graph.fusion_groups:
            params["fusion_logits"] = [-2.0] * len(graph.fusion_groups)


class MappingOnlyA2Runner(_BaseSearchRunner):
    def __init__(self) -> None:
        super().__init__("baselineA_A2")

    def _override_params(self, params: dict[str, Any], graph, kind: str) -> None:  # noqa: D401
        params["num_pes"] = 128
        params["l0_registers_size_kb"] = 2.0
        params["l1_accumulator_size_kb"] = 8.0
        params["l2_scratchpad_size_kb"] = 256.0


class HardwareOnlyRunner(_BaseSearchRunner):
    def __init__(self) -> None:
        super().__init__("baselineB")

    def _override_params(self, params: dict[str, Any], graph, kind: str) -> None:  # noqa: D401
        for dim in graph.problem_dims.keys():
            for level in ["L0_Registers", "L1_Accumulator", "L2_Scratchpad"]:
                params[f"{dim}_{level}_temporal"] = 1
                params[f"{dim}_{level}_spatial"] = 1
        if "P" in graph.problem_dims:
            params["P_L0_Registers_spatial"] = 2
        if "Q" in graph.problem_dims:
            params["Q_L0_Registers_spatial"] = 2
        if graph.fusion_groups:
            params["fusion_logits"] = [-2.0] * len(graph.fusion_groups)


class CooptRunner(_BaseSearchRunner):
    def __init__(self) -> None:
        super().__init__("coopt")

    def _override_params(self, params: dict[str, Any], graph, kind: str) -> None:  # noqa: D401
        # Co-optimization searches the full space; no overrides required
        pass


def get_baseline_runner(name: str) -> Runner:  # noqa: D401
    """Factory returning baseline runner by name."""
    mapping = {
        "baselineA_A1": MappingOnlyA1Runner,
        "baselineA_A2": MappingOnlyA2Runner,
        "baselineB": HardwareOnlyRunner,
        "coopt": CooptRunner,
    }
    if name not in mapping:
        raise ValueError(f"Unknown baseline runner: {name}")
    return mapping[name]()
