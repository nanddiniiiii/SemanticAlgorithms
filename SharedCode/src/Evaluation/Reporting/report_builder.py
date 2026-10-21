"""Short transparent markdown report emitted for each Stage 1A run."""

from pathlib import Path
from typing import Any


def write_run_report(summary: dict[str, Any], path: str | Path) -> Path:
    lines = ["# Stage 1A run report", "", "Research status: corrected modern PyTorch port; not paper-exact replication.", ""]
    lines.extend(f"- {key}: {value}" for key, value in summary.items())
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
