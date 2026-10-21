"""Create presentation plots from saved notebook/repository outputs only.

No model inference is performed. Cross-protocol results are placed in separate
panels or visibly marked and are never used for a direct numerical ranking.
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
USER_CSV = ROOT / "benchmark_outputs" / "final_repaired" / "final_comparison.csv"
ACTIVE_PREFIX_CSV = (
    ROOT
    / "benchmark_outputs"
    / "ntscc_aware_codebook_experiment"
    / "measured_comparison.csv"
)
ACTIVE_PREFIX_JSON = (
    ROOT
    / "benchmark_outputs"
    / "ntscc_aware_codebook_experiment"
    / "noisy_result.json"
)
TEAM_NB = (
    ROOT
    / "references"
    / "SemanticSchedulerEnd2End_team"
    / "Notebooks"
    / "notebookbf0f6f4129.ipynb"
)

COLORS = {
    "JPEG": "#7f7f7f",
    "DeepJSCC": "#d62728",
    "NTSCC": "#2ca02c",
    "DeepJSCC CB": "#ff7f0e",
    "NTSCC CB": "#9467bd",
    "NTSCC PQ16": "#8c564b",
    "SwinJSCC": "#1f77b4",
}


def num(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def user_rows() -> pd.DataFrame:
    raw = pd.read_csv(USER_CSV)
    names = {
        "JPEG Q25 + LDPC/QPSK/AWGN": ("JPEG Q25", "JPEG"),
        "JPEG Q50 + LDPC/QPSK/AWGN": ("JPEG Q50", "JPEG"),
        "JPEG Q75 + LDPC/QPSK/AWGN": ("JPEG Q75", "JPEG"),
        "JPEG Q90 + LDPC/QPSK/AWGN": ("JPEG Q90", "JPEG"),
        "DeepJSCC": ("DeepJSCC", "DeepJSCC"),
        "NTSCC": ("NTSCC", "NTSCC"),
        "DeepJSCC codebook": ("DeepJSCC K=256", "DeepJSCC CB"),
        "NTSCC codebook": ("NTSCC K=256", "NTSCC CB"),
    }
    rows: list[dict[str, object]] = []
    for _, item in raw.iterrows():
        if item["Method"] not in names:
            continue
        label, family = names[item["Method"]]
        rows.append(
            {
                "label": label,
                "family": family,
                "psnr": num(item["PSNR mean (dB)"]),
                "ssim": num(item["SSIM mean"]),
                "cbr": num(item["CBR"]),
                "uses": num(item["Complex channel uses"]),
                "payload_bits": num(item["Payload bits"]),
                "coded_bits": num(item["Coded bits"]),
                "lpips": num(item["LPIPS mean"]),
                "clip": num(item["CLIP cosine mean"]),
                "source": "Nandini measured: kodim08, AWGN 10 dB",
            }
        )
    active = pd.read_csv(ACTIVE_PREFIX_CSV)
    item = active[active["method"] == "NTSCC active-prefix PQ16 K=256"].iloc[0]
    active_packet = json.loads(ACTIVE_PREFIX_JSON.read_text(encoding="utf-8"))
    rows.append(
        {
            "label": "NTSCC PQ16 K=256",
            "family": "NTSCC PQ16",
            "psnr": float(item["psnr_db"]),
            "ssim": float(item["ssim"]),
            "cbr": float(item["cbr"]),
            "uses": float(item["complex_channel_uses"]),
            "payload_bits": float(active_packet["packet_bits"]),
            "coded_bits": float(active_packet["coded_bits"]),
            "lpips": float(item["lpips"]),
            "clip": float(item["clip_cosine"]),
            "source": "Nandini measured follow-up: kodim08, AWGN 10 dB",
        }
    )
    return pd.DataFrame(rows)


def teammate_rows() -> pd.DataFrame:
    nb = json.loads(TEAM_NB.read_text(encoding="utf-8"))
    pattern = re.compile(
        r"SNR=\s*(?P<snr>\d+)\s*dB\s*\|\s*C=\s*(?P<c>\d+)\s*\|\s*"
        r"CBR=(?P<cbr>[0-9.]+)\s*\|\s*PSNR=(?P<psnr>[0-9.]+)\s*dB"
    )
    rows = []
    for cell in nb["cells"]:
        for output in cell.get("outputs", []):
            text = "".join(output.get("text", []))
            for match in pattern.finditer(text):
                rows.append(
                    {
                        "label": f"C={int(match.group('c'))}",
                        "C": int(match.group("c")),
                        "snr": int(match.group("snr")),
                        "cbr": float(match.group("cbr")),
                        "psnr": float(match.group("psnr")),
                        "ssim": math.nan,
                        "lpips": math.nan,
                        "clip": math.nan,
                        "source": "Teammate measured: Kodak24 mean",
                    }
                )
    frame = pd.DataFrame(rows)
    if len(frame) != 25:
        raise RuntimeError(f"Expected 25 saved SwinJSCC results; found {len(frame)}")
    return frame


def label_horizontal_bars(ax: plt.Axes, bars, values, fmt: str) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + 0.012 * ax.get_xlim()[1],
            bar.get_y() + bar.get_height() / 2,
            format(value, fmt),
            va="center",
            fontsize=9,
        )


def psnr_bars(user: pd.DataFrame, team: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), gridspec_kw={"width_ratios": [1.35, 1]})
    ordered = user.sort_values("psnr")
    bars = axes[0].barh(
        ordered["label"], ordered["psnr"], color=[COLORS[x] for x in ordered["family"]]
    )
    axes[0].set_xlim(0, 40)
    axes[0].set_xlabel("PSNR (dB) — higher is better")
    axes[0].set_title("Your measured results\nkodim08, AWGN 10 dB, 5 seeds")
    label_horizontal_bars(axes[0], bars, ordered["psnr"], ".2f")

    team10 = team[team["snr"] == 10].sort_values("C", ascending=False)
    bars = axes[1].barh(team10["label"], team10["psnr"], color=COLORS["SwinJSCC"])
    axes[1].set_xlim(0, 28)
    axes[1].set_xlabel("PSNR (dB) — higher is better")
    axes[1].set_title("Teammate SwinJSCC SA+RA\nKodak24 mean, AWGN 10 dB")
    label_horizontal_bars(axes[1], bars, team10["psnr"], ".3f")

    fig.suptitle("PSNR comparison — separate protocols; do not rank across panels", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "01_psnr_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def ssim_bars(user: pd.DataFrame) -> None:
    ordered = user.sort_values("ssim")
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    bars = ax.barh(
        ordered["label"], ordered["ssim"], color=[COLORS[x] for x in ordered["family"]]
    )
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("SSIM — higher is better")
    ax.set_title(
        "SSIM — your measured kodim08 results at AWGN 10 dB\n"
        "Teammate SwinJSCC notebook did not save SSIM"
    )
    label_horizontal_bars(ax, bars, ordered["ssim"], ".3f")
    fig.tight_layout()
    fig.savefig(OUT / "02_ssim_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def cbr_bars(user: pd.DataFrame, team: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7), gridspec_kw={"width_ratios": [1.4, 1]})
    ordered = user.sort_values("cbr", ascending=False)
    bars = axes[0].barh(
        ordered["label"], ordered["cbr"], color=[COLORS[x] for x in ordered["family"]]
    )
    axes[0].set_xscale("log")
    axes[0].set_xlim(0.01, 2.2)
    axes[0].set_xlabel("CBR = complex channel uses / (3HW), log scale — lower is better")
    axes[0].set_title("Your measured channel usage")
    for bar, (_, row) in zip(bars, ordered.iterrows()):
        uses = f"; {int(row['uses']):,} uses" if np.isfinite(row["uses"]) else ""
        axes[0].text(
            row["cbr"] * 1.07,
            bar.get_y() + bar.get_height() / 2,
            f"{row['cbr']:.5f}{uses}",
            va="center",
            fontsize=8.5,
        )

    team10 = team[team["snr"] == 10].sort_values("C", ascending=False)
    bars = axes[1].barh(team10["label"], team10["cbr"], color=COLORS["SwinJSCC"])
    axes[1].set_xlim(0, 0.155)
    axes[1].set_xlabel("CBR — lower is better")
    axes[1].set_title("Teammate SwinJSCC saved rate levels")
    for bar, value in zip(bars, team10["cbr"]):
        axes[1].text(
            value + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.5f}",
            va="center",
            fontsize=9,
        )

    fig.suptitle("Measured CBR/channel usage — separate protocols", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "03_channel_usage_cbr_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def rate_quality(user: pd.DataFrame, team: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.3))
    jpeg = user[user["family"] == "JPEG"].sort_values("cbr")
    axes[0].plot(jpeg["cbr"], jpeg["psnr"], "s--", color=COLORS["JPEG"], label="Your JPEG + LDPC/QPSK")
    for _, row in jpeg.iterrows():
        axes[0].annotate(row["label"].replace("JPEG ", ""), (row["cbr"], row["psnr"]), xytext=(4, 5), textcoords="offset points", fontsize=8)

    neural = user[user["family"] != "JPEG"]
    for _, row in neural.iterrows():
        axes[0].scatter(row["cbr"], row["psnr"], s=80, marker="D", color=COLORS[row["family"]], label=row["label"])

    team10 = team[team["snr"] == 10].sort_values("cbr")
    axes[0].plot(team10["cbr"], team10["psnr"], "o-", color=COLORS["SwinJSCC"], label="Teammate SwinJSCC (Kodak24)")
    for _, row in team10.iterrows():
        axes[0].annotate(row["label"], (row["cbr"], row["psnr"]), xytext=(3, -12), textcoords="offset points", fontsize=8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("CBR (log scale) — lower is better")
    axes[0].set_ylabel("PSNR (dB) — higher is better")
    axes[0].set_title("PSNR versus measured rate\nCross-source curves are context only")
    axes[0].legend(fontsize=7.2, loc="best")

    axes[1].plot(jpeg["cbr"], jpeg["ssim"], "s--", color=COLORS["JPEG"], label="JPEG + LDPC/QPSK")
    for _, row in jpeg.iterrows():
        axes[1].annotate(row["label"].replace("JPEG ", ""), (row["cbr"], row["ssim"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    for _, row in neural.iterrows():
        axes[1].scatter(row["cbr"], row["ssim"], s=80, marker="D", color=COLORS[row["family"]], label=row["label"])
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 1.04)
    axes[1].set_xlabel("CBR (log scale) — lower is better")
    axes[1].set_ylabel("SSIM — higher is better")
    axes[1].set_title("SSIM versus measured rate\nTeammate SSIM unavailable")
    axes[1].legend(fontsize=7.2, loc="lower right")

    fig.suptitle("Compression/rate versus reconstruction quality — AWGN 10 dB", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "04_rate_vs_quality.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_tables(user: pd.DataFrame, team: pd.DataFrame) -> None:
    user_out = user[
        [
            "source",
            "label",
            "psnr",
            "ssim",
            "lpips",
            "clip",
            "cbr",
            "uses",
            "payload_bits",
            "coded_bits",
        ]
    ].copy()
    user_out.to_csv(OUT / "nandini_measured_values.csv", index=False)
    team_out = team[["source", "snr", "C", "cbr", "psnr", "ssim", "lpips", "clip"]].copy()
    team_out.to_csv(OUT / "teammate_swinjscc_measured_values.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    user = user_rows()
    team = teammate_rows()
    psnr_bars(user, team)
    ssim_bars(user)
    cbr_bars(user, team)
    rate_quality(user, team)
    save_tables(user, team)
    print(f"Created four presentation plots in {OUT}")


if __name__ == "__main__":
    main()
