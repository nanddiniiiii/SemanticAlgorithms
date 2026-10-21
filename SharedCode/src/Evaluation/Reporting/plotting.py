"""Plots only homogeneous curves; callers must not mix SNR definitions."""

from pathlib import Path
from typing import Any


def plot_quality_vs_channel_uses(rows: list[dict[str, Any]], path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    methods = sorted(set(str(row["method"]) for row in rows))
    figure, axis = plt.subplots(figsize=(8, 5))
    for method in methods:
        subset = sorted((row for row in rows if row["method"] == method), key=lambda row: row["channel_uses"])
        axis.plot([row["channel_uses"] for row in subset], [row["psnr"] for row in subset], marker="o", label=method)
    axis.set_xlabel("Complex channel uses per image")
    axis.set_ylabel("PSNR (dB); failed digital frames are excluded and counted separately")
    axis.legend()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target
