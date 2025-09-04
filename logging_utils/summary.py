"""Generate markdown summary for Act I experiment (skeleton)."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import pandas as pd  # type: ignore


def generate_summary(exp_dir: Path) -> str:  # noqa: D401
    """Aggregate results and return markdown string (placeholder)."""
    # Iterate over baseline directories and load best.json if exists
    rows: list[dict[str, Any]] = []
    for baseline_dir in exp_dir.iterdir():
        if not baseline_dir.is_dir():
            continue
        best_files = list(baseline_dir.glob("seed_*/best.json"))
        for best_file in best_files:
            data = json.loads(best_file.read_text(encoding="utf-8"))
            data["baseline"] = baseline_dir.name
            rows.append(data)
    if not rows:
        return "# Summary\n\n_No results found yet._\n"

    df = pd.DataFrame(rows)
    try:
        table_md = df.to_markdown(index=False)  # requires tabulate
    except ImportError:
        # Fallback to simple CSV display when tabulate is missing
        table_md = "\n".join([", ".join(df.columns)] + [", ".join(map(str, r)) for r in df.values])
    md_lines = ["# Act I Experiment Summary", "", table_md]
    return "\n".join(md_lines)


def write_summary(exp_dir: Path) -> None:  # noqa: D401
    md = generate_summary(exp_dir)
    summary_path = exp_dir / "summary.md"
    summary_path.write_text(md, encoding="utf-8")