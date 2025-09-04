"""Unified recorder for Act I experiment (skeleton).

Handles CSV/JSON writing, best-so-far tracking, and final summary export.
Real implementation will be added later.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import csv
import json


class Recorder:  # noqa: D101
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trials_path = self.log_dir / "trials.csv"
        self.jsonl_path = self.log_dir / "trials.jsonl"
        self.best_path = self.log_dir / "best.json"
        # 专用协同优化调试日志路径 (JSON Lines)
        self.coopt_trace_path = self.log_dir / "coopt_debug_trace.jsonl"
        self._fieldnames: List[str] | None = None
        self._best_metrics: Dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Context manager helpers
    # ------------------------------------------------------------------
    def __enter__(self):  # noqa: D401
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: D401
        # Always flush best metrics if context closes
        self.close()
        # Do not suppress exceptions
        return False

    # Trial logging -----------------------------------------------------
    def record_trial(self, row: Dict[str, Any]) -> None:  # noqa: D401
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            with self.trials_path.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=self._fieldnames)
                writer.writeheader()
        with self.trials_path.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=self._fieldnames)
            writer.writerow(row)
        # JSONL append for easier downstream processing
        with self.jsonl_path.open("a", encoding="utf-8") as fp_json:
            fp_json.write(json.dumps(row) + "\n")

    # Best tracking -----------------------------------------------------
    def update_best(self, metrics: Dict[str, Any], key: str = "edp") -> None:  # noqa: D401
        if self._best_metrics is None or metrics[key] < self._best_metrics[key]:
            self._best_metrics = metrics.copy()

    def finalize_best(self) -> None:  # noqa: D401
        if self._best_metrics is not None:
            self.best_path.write_text(json.dumps(self._best_metrics, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Co-optimization debug logging
    # ------------------------------------------------------------------
    def log_coopt_debug_step(self, debug_data: Dict[str, Any]) -> None:  # noqa: D401
        """Append a single debug snapshot as JSONL.

        All tensors will be converted to native Python types recursively to
        avoid JSON serialization issues.
        """
        def _to_native(val: Any):  # noqa: D401
            if isinstance(val, dict):
                return {k: _to_native(v) for k, v in val.items()}
            if isinstance(val, (list, tuple)):
                return [_to_native(v) for v in val]
            # Tensor → item/ list
            try:
                import torch  # local import to avoid hard dep when not needed
                if isinstance(val, torch.Tensor):
                    return val.detach().cpu().tolist() if val.dim() > 0 else val.detach().item()
            except ModuleNotFoundError:  # torch may not be installed in some envs
                pass
            return val

        try:
            serializable = _to_native(debug_data)
            with self.coopt_trace_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(serializable) + "\n")
        except Exception as exc:  # noqa: BLE001
            # 静默失败，防止调试日志影响主流程
            print(f"[Recorder] DEBUG log write failed: {exc}")

    # Placeholder -------------------------------------------------------
    # Accessors ---------------------------------------------------------
    @property
    def best_metrics(self) -> Dict[str, Any] | None:  # noqa: D401
        return self._best_metrics

    def close(self) -> None:  # noqa: D401
        self.finalize_best()