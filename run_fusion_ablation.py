import json
import time
from pathlib import Path
import argparse

from run import run_experiment


def run_fusion_ablation(model_name: str = "resnet18", num_trials: int = 100, **kwargs):
    """Run fusion-aware and fusion-unaware experiments back to back.

    Args:
        model_name: Name of the model to evaluate.
        num_trials: Number of trials for each DSE run.
        **kwargs: Additional parameters forwarded to ``run_experiment``.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root_dir = Path("output") / f"fusion_ablation_{timestamp}"
    root_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, float]] = {}
    for aware in (True, False):
        mode = "fusion_aware" if aware else "fusion_unaware"
        mode_dir = root_dir / mode
        results = run_experiment(
            model_name=model_name,
            searcher_type="fa-dosa",
            num_trials=num_trials,
            fusion_aware=aware,
            log_dir=str(mode_dir),
            **kwargs,
        )
        summary[mode] = results.get("best_metrics", {})
        with open(root_dir / f"{mode}_results.json", "w", encoding="utf-8") as fp:
            json.dump(results, fp, indent=2)

    with open(root_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    try:
        import matplotlib.pyplot as plt

        labels = ["Fusion Aware", "Fusion Unaware"]
        edp_vals = [summary.get("fusion_aware", {}).get("edp", 0), summary.get("fusion_unaware", {}).get("edp", 0)]
        area_vals = [summary.get("fusion_aware", {}).get("area_mm2", 0), summary.get("fusion_unaware", {}).get("area_mm2", 0)]

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].bar(labels, edp_vals, color=["#4CAF50", "#F44336"])
        axes[0].set_title("EDP")
        axes[1].bar(labels, area_vals, color=["#4CAF50", "#F44336"])
        axes[1].set_title("Area (mm$^2$)")
        for ax in axes:
            ax.set_xticklabels(labels, rotation=15)
        plt.tight_layout()
        plt.savefig(root_dir / "comparison.png")
        plt.close(fig)
    except Exception as exc:
        print(f"Plotting failed: {exc}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fusion-aware vs fusion-unaware ablation experiment")
    parser.add_argument("--model", default="resnet18", help="Model name to evaluate")
    parser.add_argument("--num-trials", type=int, default=100, help="Number of trials per run")
    # 添加轻量级测试参数
    parser.add_argument("--num-outer-steps", type=int, default=5, help="Number of outer optimization steps")
    parser.add_argument("--num-mapping-steps", type=int, default=50, help="Number of mapping optimization steps")
    parser.add_argument("--num-hardware-steps", type=int, default=100, help="Number of hardware optimization steps")
    parser.add_argument("--lr-mapping", type=float, default=0.01, help="Learning rate for mapping optimization")
    parser.add_argument("--lr-hardware", type=float, default=0.05, help="Learning rate for hardware optimization")
    
    args = parser.parse_args()
    
    # 将额外参数传递给run_fusion_ablation
    run_fusion_ablation(
        model_name=args.model, 
        num_trials=args.num_trials,
        num_outer_steps=args.num_outer_steps,
        num_mapping_steps=args.num_mapping_steps,
        num_hardware_steps=args.num_hardware_steps,
        lr_mapping=args.lr_mapping,
        lr_hardware=args.lr_hardware
    )
