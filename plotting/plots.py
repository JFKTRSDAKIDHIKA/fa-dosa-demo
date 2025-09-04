"""Plotting utilities for Act I experiment (skeleton)."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


plt.style.use("ggplot")


from collections import defaultdict
import numpy as np


def _extract_baseline_seed(csv_path: Path) -> tuple[str, str]:
    """Return (baseline_name, seed_str) from path .../baseline/seed_x/trials.csv"""
    baseline = csv_path.parent.parent.name
    seed = csv_path.parent.name  # seed_0
    return baseline, seed


def render_edp_bar(csv_paths: List[Path], out_path: Path) -> None:  # noqa: D401
    """Aggregate CSVs and draw EDP bar chart (mean ± 95%CI)."""
    edp_by_baseline: dict[str, list[float]] = defaultdict(list)
    for p in csv_paths:
        baseline, _ = _extract_baseline_seed(p)
        df = pd.read_csv(p)
        best_edp = df["edp"].min()
        edp_by_baseline[baseline].append(best_edp)

    if not edp_by_baseline:
        print("[Plots] No data to plot EDP bar chart.")
        return

    baselines = sorted(edp_by_baseline)
    means = []
    ci95 = []
    for b in baselines:
        arr = np.array(edp_by_baseline[b])
        mean = arr.mean()
        std_err = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        means.append(mean)
        ci95.append(1.96 * std_err)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(baselines))
    ax.bar(x, means, yerr=ci95, capsize=5, color="skyblue")
    ax.set_xticks(x)
    ax.set_xticklabels(baselines, rotation=15)
    ax.set_ylabel("Best EDP (lower is better)")
    ax.set_title("Act I Baseline Comparison – EDP")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("[Plots] Saved EDP bar chart to", out_path)


def render_convergence(csv_paths: List[Path], out_path: Path) -> None:  # noqa: D401
    """Draw best-so-far convergence curves for each baseline (seed-averaged)."""
    curves: dict[str, list[np.ndarray]] = defaultdict(list)
    max_trials = 0
    for p in csv_paths:
        baseline, _ = _extract_baseline_seed(p)
        df = pd.read_csv(p).sort_values("trial")
        edp = df["edp"].to_numpy()
        best_so_far = np.minimum.accumulate(edp)
        curves[baseline].append(best_so_far)
        max_trials = max(max_trials, len(best_so_far))

    if not curves:
        print("[Plots] No data to plot convergence curves.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for baseline, arrs in curves.items():
        # Pad arrays to same length with last value
        padded = [np.pad(a, (0, max_trials - len(a)), constant_values=a[-1]) for a in arrs]
        mean_curve = np.mean(padded, axis=0)
        ax.plot(range(1, max_trials + 1), mean_curve, label=baseline)

    ax.set_xlabel("Trial")
    ax.set_ylabel("Best-so-far EDP")
    ax.set_title("Convergence Curves (seed-averaged)")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("[Plots] Saved convergence plot to", out_path)