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

    # Placeholder -------------------------------------------------------
    # Accessors ---------------------------------------------------------
    @property
    def best_metrics(self) -> Dict[str, Any] | None:  # noqa: D401
        return self._best_metrics

    def close(self) -> None:  # noqa: D401
        self.finalize_best()