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

class _BaseRandomRunner:
    """Simple random search baseline producing fake EDP values (placeholder)."""

    def __init__(self, name: str, key_space: list[str]) -> None:
        self.name = name
        self.key_space = key_space

    def _random_metrics(self) -> dict[str, Any]:
        # Produce fake EDP ~ log-uniform, loss correlated
        edp = 10 ** random.uniform(1, 6)
        loss = random.uniform(0.5, 5.0)
        return {"edp": edp, "loss": loss}

    def run(self, cfg: dict[str, Any], seed: int, recorder: "Recorder") -> None:  # noqa: D401
        random.seed(seed)
        num_trials = cfg["shared"].get("num_trials", 30)
        start_ts = datetime.now().isoformat(timespec="seconds")
        for t in range(1, num_trials + 1):
            metrics = self._random_metrics()
            row = {
                "trial": t,
                "seed": seed,
                **metrics,
            }
            recorder.record_trial(row)
            recorder.update_best(metrics, key="edp")
        recorder.finalize_best()
        end_ts = datetime.now().isoformat(timespec="seconds")
        print(f"[{self.name}] seed={seed} completed {num_trials} trials in {start_ts}→{end_ts}.")


class MappingOnlyA1Runner(_BaseRandomRunner):
    def __init__(self):
        super().__init__("baselineA_A1", ["mapping"])


class MappingOnlyA2Runner(_BaseRandomRunner):
    def __init__(self):
        super().__init__("baselineA_A2", ["mapping"])


class HardwareOnlyRunner(_BaseRandomRunner):
    def __init__(self):
        super().__init__("baselineB", ["hardware"])


class CooptRunner(_BaseRandomRunner):
    def __init__(self):
        super().__init__("coopt", ["mapping", "hardware"])


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