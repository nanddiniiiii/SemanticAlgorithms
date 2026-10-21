"""Plot only information actually available for the SwinJSCC codebook branch.

The codebook notebook is unexecuted, so reconstruction-quality results are not
available.  The first two plots reproduce deterministic accounting implemented
in notebook cells 62-65.  The third plot is the separately executed native
SwinJSCC reference curve and is not labeled as a codebook result.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TEAM_NATIVE_NB = (
    ROOT
    / "references"
    / "SemanticSchedulerEnd2End_team"
    / "Notebooks"
    / "notebookbf0f6f4129.ipynb"
)

IMAGE_SIZE = 256
LATENT_TOKENS = 256
LATENT_DIM = 320
C_VALUES = [32, 64, 96, 128, 192]
K_VALUES = [64, 256, 1024]
TRAINED_K = 256


def accounting_table() -> pd.DataFrame:
    original_bits = IMAGE_SIZE * IMAGE_SIZE * 3 * 8
    rows = [
        {
            "representation": "Original RGB",
            "setting": "256x256x3x8",
            "per_image_bits": original_bits,
            "shared_bits": 0,
            "status": "deterministic input size",
        },
        {
            "representation": "Continuous latent FP32",
            "setting": "256x320 values",
            "per_image_bits": LATENT_TOKENS * LATENT_DIM * 32,
            "shared_bits": 0,
            "status": "memory diagnostic; not over-air bitrate",
        },
    ]
    for c in C_VALUES:
        rows.append(
            {
                "representation": "Active latent FP32",
                "setting": f"C={c}",
                "per_image_bits": LATENT_TOKENS * c * 32,
                "shared_bits": 0,
                "status": "memory diagnostic; not over-air bitrate",
            }
        )
    for k in K_VALUES:
        bits_per_index = math.ceil(math.log2(k))
        rows.append(
            {
                "representation": "Codebook indices",
                "setting": f"K={k}",
                "per_image_bits": LATENT_TOKENS * bits_per_index,
                "shared_bits": k * LATENT_DIM * 32,
                "status": "implementation-derived payload; excludes packet/PHY overhead",
            }
        )
    frame = pd.DataFrame(rows)
    frame["per_image_KiB"] = frame["per_image_bits"] / 8 / 1024
    frame["shared_KiB"] = frame["shared_bits"] / 8 / 1024
    frame["payload_compression_vs_raw"] = original_bits / frame["per_image_bits"]
    return frame


def native_reference() -> pd.DataFrame:
    nb = json.loads(TEAM_NATIVE_NB.read_text(encoding="utf-8"))
    pattern = re.compile(
        r"SNR=\s*(?P<snr>\d+)\s*dB\s*\|\s*C=\s*(?P<c>\d+)\s*\|\s*"
        r"CBR=(?P<cbr>[0-9.]+)\s*\|\s*PSNR=(?P<psnr>[0-9.]+)\s*dB"
    )
    rows = []
    for cell in nb["cells"]:
        for output in cell.get("outputs", []):
            text = "".join(output.get("text", []))
            for match in pattern.finditer(text):
                if int(match.group("snr")) == 10:
                    rows.append(
                        {
                            "C": int(match.group("c")),
                            "CBR": float(match.group("cbr")),
                            "PSNR_dB": float(match.group("psnr")),
                        }
                    )
    frame = pd.DataFrame(rows).sort_values("CBR")
    if len(frame) != 5:
        raise RuntimeError(f"Expected five native 10 dB points, found {len(frame)}")
    return frame


def representation_plot(frame: pd.DataFrame) -> None:
    subset = frame[
        (frame["representation"] != "Codebook indices")
        | (frame["setting"] == f"K={TRAINED_K}")
    ].copy()
    labels = [
        f"{row.representation}\n{row.setting}" for row in subset.itertuples(index=False)
    ]
    colors = [
        "#7f7f7f" if r == "Original RGB" else
        "#1f77b4" if r == "Continuous latent FP32" else
        "#4c78a8" if r == "Active latent FP32" else
        "#9467bd"
        for r in subset["representation"]
    ]
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    bars = ax.bar(labels, subset["per_image_KiB"], color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Per-image representation size (KiB, log scale)")
    ax.set_title(
        "SwinJSCC codebook implementation: deterministic representation accounting\n"
        "FP32 latent sizes are memory diagnostics, not radio bitrate"
    )
    ax.tick_params(axis="x", rotation=35)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    for bar, value in zip(bars, subset["per_image_KiB"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.12,
            f"{value:g} KiB",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.text(
        0.99,
        0.05,
        "Configured K=256 index payload: 256 indices x 8 bits = 0.25 KiB/image\n"
        "Shared K=256 x D=320 FP32 codebook: 320 KiB (one-time model state)",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#bbbbbb"},
    )
    fig.tight_layout()
    fig.savefig(OUT / "01_representation_size_accounting.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def payload_plot(frame: pd.DataFrame) -> None:
    codebooks = frame[frame["representation"] == "Codebook indices"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    labels = codebooks["setting"]
    colors = ["#9ecae1", "#9467bd", "#6baed6"]
    bars = axes[0].bar(labels, codebooks["per_image_bits"], color=colors)
    axes[0].set_ylabel("Index payload (bits/image)")
    axes[0].set_title("Index payload defined by the notebook")
    for bar, bits in zip(bars, codebooks["per_image_bits"]):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bits + 35,
            f"{int(bits):,} bits\n({bits/8:.0f} B)",
            ha="center",
            fontsize=9,
        )
    bars = axes[1].bar(labels, codebooks["shared_KiB"], color=colors)
    axes[1].set_ylabel("Shared FP32 codebook (KiB)")
    axes[1].set_title("One-time shared model state")
    for bar, value in zip(bars, codebooks["shared_KiB"]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 18,
            f"{value:.0f} KiB",
            ha="center",
            fontsize=9,
        )
    fig.suptitle(
        "SwinJSCC codebook size options in the reporting utility\n"
        "Only K=256 is configured for training; K=64/1024 are accounting scenarios, not trained results",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(OUT / "02_codebook_payload_accounting.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def native_plot(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.plot(frame["CBR"], frame["PSNR_dB"], "o-", color="#1f77b4", linewidth=2)
    for row in frame.itertuples(index=False):
        ax.annotate(
            f"C={row.C}\n{row.PSNR_dB:.3f} dB",
            (row.CBR, row.PSNR_dB),
            xytext=(3, 8 if row.C in (32, 96, 128) else -25),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("Native SwinJSCC CBR")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(
        "Available native SwinJSCC reference — Kodak24, AWGN 10 dB\n"
        "No executed K=256 codebook PSNR exists in the branch"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "03_native_swinjscc_reference_only.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    accounting = accounting_table()
    native = native_reference()
    accounting.to_csv(OUT / "implementation_accounting.csv", index=False)
    native.to_csv(OUT / "native_swinjscc_reference_10db.csv", index=False)
    representation_plot(accounting)
    payload_plot(accounting)
    native_plot(native)
    print(f"Wrote SwinJSCC codebook accounting plots to {OUT}")


if __name__ == "__main__":
    main()
