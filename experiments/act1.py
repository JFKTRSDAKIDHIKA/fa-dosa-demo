"""Act I experiment orchestrator (skeleton).

This module will coordinate three baselines (Mapping-only, Hardware-only, Co-optimization)
across multiple random seeds and record results with logging_utils.recorder.

Current version is a placeholder to enable CLI import; real logic will be filled later.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import importlib
import yaml

# Local imports will be available after corresponding modules are created later
# from logging_utils.recorder import Recorder  # noqa: E402  # pylint: disable=wrong-import-position
# from experiments.baselines import get_baseline_runner  # noqa: E402


DEFAULT_CFG_PATH = Path(__file__).resolve().parent.parent / "configs" / "act1.yaml"


def run_act1(cfg_path: str | Path | None = None, scenario: str | None = None) -> None:  # noqa: D401
    """Entry point for Act I experiment.

    Args:
        cfg_path: Path to YAML configuration. If *None*, fallbacks to default sample.
        scenario: Scenario preset name. Overrides config file if provided.
    """
    if isinstance(cfg_path, str):
        cfg: dict[str, Any] = yaml.safe_load(cfg_path)
    else:
        cfg_path = Path(cfg_path or DEFAULT_CFG_PATH)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        with cfg_path.open("r", encoding="utf-8") as fp:
            cfg: dict[str, Any] = yaml.safe_load(fp)

    # Resolve scenario from CLI or config
    scenario = scenario or cfg.get("scenario") or cfg.get("shared", {}).get("scenario")
    cfg.setdefault("shared", {})["scenario"] = scenario

    # Lazy import to avoid circular dependency before skeletons are complete
    baselines_mod = importlib.import_module("experiments.baselines")
    recorder_mod = importlib.import_module("logging_utils.recorder")
    from dosa.config import Config

    if scenario:
        Config.get_instance().apply_scenario_preset(scenario)

    # Build output root directory
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / "act1" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save effective configuration for reproducibility
    cfg_dump_path = output_dir / "cfg.yaml"
    cfg_dump_path.write_text(yaml.dump(cfg, sort_keys=False), encoding="utf-8")

    # ------------------------------------------------------------------
    # Execute baselines × seeds
    # ------------------------------------------------------------------
    baselines: list[str] = cfg.get("baselines", [
        "baselineA_A1",
        "baselineA_A2",
        "baselineB",
        "coopt",
    ])

    for baseline_name in baselines:
        baseline_dir = output_dir / baseline_name
        baseline_dir.mkdir(parents=True, exist_ok=True)

        runner = baselines_mod.get_baseline_runner(baseline_name)
        print(f"[Act1] >>> Baseline {baseline_name}")
        for seed in cfg["shared"]["seeds"]:
            seed_dir = baseline_dir / f"seed_{seed}"
            with recorder_mod.Recorder(seed_dir) as rec:
                print(f"[Act1]   • Seed {seed}")
                runner.run(cfg, seed, rec)

    # ------------------------------------------------------------------
    # Post-processing: plots + summary
    # ------------------------------------------------------------------
    from plotting.plots import render_edp_bar, render_convergence  # lazy import
    from logging_utils.summary import write_summary

    # Collect all CSV paths for plotting utilities
    csv_paths = list(output_dir.glob("*/seed_*/trials.csv"))

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    if csv_paths:
        render_edp_bar(csv_paths, plots_dir / "edp_bar.png")
        render_convergence(csv_paths, plots_dir / "convergence_all.png")

    write_summary(output_dir)
    print(f"[Act1] Experiment complete. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Act I experiment")
    parser.add_argument("--config", default=str(DEFAULT_CFG_PATH), help="Path to YAML config file")
    parser.add_argument("--scenario", default=None, help="Scenario preset name")
    args = parser.parse_args()
    run_act1(args.config, args.scenario)