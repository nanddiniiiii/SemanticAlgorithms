"""Simple presentation plots derived only from persisted benchmark results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


QUALITY_METHODS = (
    "JPEG Q25", "JPEG Q50", "JPEG Q75", "JPEG Q90",
    "DeepJSCC", "NTSCC", "DeepJSCC codebook", "NTSCC codebook",
)

TRANSMITTED_METHODS = (
    "RAW + LDPC/QPSK/AWGN",
    "JPEG Q25 + LDPC/QPSK/AWGN",
    "JPEG Q50 + LDPC/QPSK/AWGN",
    "JPEG Q75 + LDPC/QPSK/AWGN",
    "JPEG Q90 + LDPC/QPSK/AWGN",
    "DeepJSCC", "NTSCC", "DeepJSCC codebook", "NTSCC codebook",
)


def load_measured_comparison(output_dir: str | Path) -> pd.DataFrame:
    path = Path(output_dir) / "final_comparison.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Validated comparison output not found: {path}")
    table = pd.read_csv(path, na_values=["N/A", ""], keep_default_na=True)
    if table.empty:
        raise ValueError(f"Measured comparison is empty: {path}")
    if table["Method"].duplicated().any():
        raise ValueError("Measured comparison contains duplicate method names")
    return table.set_index("Method", drop=False)


def _require_rows(table: pd.DataFrame, methods: tuple[str, ...]) -> pd.DataFrame:
    missing = [method for method in methods if method not in table.index]
    if missing:
        raise ValueError(f"Measured comparison is missing methods: {missing}")
    return table.loc[list(methods)].copy()


def _vertical_bars(axis, labels, values, ylabel: str, title: str, color: str) -> None:
    positions = np.arange(len(labels))
    bars = axis.bar(positions, values, color=color, edgecolor="white", linewidth=0.6)
    axis.set_xticks(positions, labels, rotation=32, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8, rotation=90)


def plot_reconstruction_quality(table: pd.DataFrame, output_dir: str | Path) -> Path:
    """PSNR and SSIM for codec-only JPEG and the four learned methods."""
    rows = _require_rows(table, QUALITY_METHODS)
    psnr = pd.to_numeric(rows["PSNR mean (dB)"], errors="raise").to_numpy(float)
    ssim = pd.to_numeric(rows["SSIM mean"], errors="raise").to_numpy(float)
    destination = Path(output_dir) / "presentation_reconstruction_quality.png"

    figure, axes = plt.subplots(2, 1, figsize=(13, 12))
    _vertical_bars(
        axes[0], rows["Method"], psnr, "PSNR mean (dB)",
        "PSNR — JPEG entries are codec-only", "#2E8B57",
    )
    _vertical_bars(
        axes[1], rows["Method"], ssim, "SSIM mean", "SSIM — higher is better", "#4C78A8",
    )
    axes[1].set_ylim(0, 1.08)
    figure.suptitle(
        "Reconstruction Quality\nRAW: exact reconstruction (PSNR = ∞, SSIM = 1; not drawn as a finite bar)",
        fontsize=15, y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_channel_usage(table: pd.DataFrame, output_dir: str | Path) -> Path:
    """Measured complex channel uses for every method that used a channel."""
    rows = _require_rows(table, TRANSMITTED_METHODS)
    uses = pd.to_numeric(rows["Complex channel uses"], errors="raise").to_numpy(float)
    destination = Path(output_dir) / "presentation_channel_usage.png"

    figure, axis = plt.subplots(figsize=(15, 8))
    _vertical_bars(
        axis, rows["Method"], uses, "Complex channel uses (log scale)",
        "Channel Usage — JPEG entries include LDPC/QPSK/AWGN", "#D17A22",
    )
    axis.set_yscale("log")
    axis.margins(y=0.16)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_channel_resource_saving(table: pd.DataFrame, output_dir: str | Path) -> Path:
    """Reduction in measured complex channel uses relative to RAW digital PHY."""
    rows = _require_rows(table, TRANSMITTED_METHODS)
    uses = pd.to_numeric(rows["Complex channel uses"], errors="raise")
    raw_uses = float(uses.loc["RAW + LDPC/QPSK/AWGN"])
    savings = 100.0 * (1.0 - uses.to_numpy(float) / raw_uses)
    destination = Path(output_dir) / "presentation_channel_resource_saving.png"

    figure, axis = plt.subplots(figsize=(15, 8))
    _vertical_bars(
        axis, rows["Method"], savings,
        "Reduction in complex channel uses vs RAW + LDPC (%)",
        "Channel Resource Saving — measured relative to RAW + LDPC/QPSK/AWGN", "#6F4E9C",
    )
    axis.set_ylim(0, 108)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def generate_measured_comparison_plots(output_dir: str | Path) -> dict[str, str]:
    """Generate presentation plots without invoking models, metrics or PHY."""
    output_dir = Path(output_dir)
    table = load_measured_comparison(output_dir)
    paths = {
        "reconstruction_quality": str(plot_reconstruction_quality(table, output_dir)),
        "channel_usage": str(plot_channel_usage(table, output_dir)),
        "channel_resource_saving": str(plot_channel_resource_saving(table, output_dir)),
    }
    methodology = {
        "input": str(output_dir / "final_comparison.csv"),
        "measured_values_only": True,
        "models_or_phy_rerun": False,
        "plot_orientation": "vertical bars",
        "reconstruction_quality": (
            "Codec-only JPEG Q25/Q50/Q75/Q90 and learned methods use measured PSNR/SSIM. "
            "RAW is exact and is annotated rather than assigned a finite PSNR bar."
        ),
        "channel_usage": (
            "Only transmitted methods with measured complex channel uses are included; log scale is used."
        ),
        "channel_resource_saving": (
            "Computed from measured complex channel uses as 100*(1-method/RAW+LDPC)."
        ),
        "supplementary_metrics": "LPIPS and CLIP remain in final_comparison.csv/json and are not main plots.",
        "outputs": paths,
    }
    (output_dir / "presentation_plot_methodology.json").write_text(
        json.dumps(methodology, indent=2), encoding="utf-8"
    )
    return paths
