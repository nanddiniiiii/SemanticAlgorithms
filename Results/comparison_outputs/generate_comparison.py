"""Build the DeepJSCC/NTSCC/SwinJSCC comparison from saved experiment outputs.

This script deliberately does not digitize paper figures.  Paper-reported aggregate
numbers are kept separate from locally measured points because the protocols differ.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
USER_RESULTS = ROOT / "benchmark_outputs" / "final_repaired" / "final_comparison.csv"
TEAM_NOTEBOOK = (
    ROOT
    / "references"
    / "SemanticSchedulerEnd2End_team"
    / "Notebooks"
    / "notebookbf0f6f4129.ipynb"
)
NTSCC_SWEEP = ROOT / "NTSCC_JSAC22" / "benchmark_results"
NTSCC_AWARE = (
    ROOT
    / "benchmark_outputs"
    / "ntscc_aware_codebook_experiment"
    / "measured_comparison.csv"
)


def parse_number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_user_results() -> pd.DataFrame:
    frame = pd.read_csv(USER_RESULTS)
    frame = frame.rename(
        columns={
            "Method": "method",
            "Category": "category",
            "CBR": "cbr",
            "PSNR mean (dB)": "psnr_db",
            "SSIM mean": "ssim",
            "LPIPS mean": "lpips",
            "CLIP cosine mean": "clip_cosine",
        }
    )
    for column in ["cbr", "psnr_db", "ssim", "lpips", "clip_cosine"]:
        frame[column] = frame[column].map(parse_number)
    frame["source"] = "Nandini final repaired notebook"
    frame["snr_db"] = 10.0
    aware = pd.read_csv(NTSCC_AWARE)
    aware = aware[aware["method"] == "NTSCC active-prefix PQ16 K=256"].copy()
    aware["category"] = "K=256 product quantizer + Digital PHY"
    aware["lpips"] = aware["lpips"]
    aware["clip_cosine"] = aware["clip_cosine"]
    aware["source"] = "Nandini final notebook follow-up"
    aware["snr_db"] = 10.0
    for column in frame.columns:
        if column not in aware.columns:
            aware[column] = math.nan
    return pd.concat([frame, aware[frame.columns]], ignore_index=True)


def load_team_swinjscc() -> pd.DataFrame:
    notebook = json.loads(TEAM_NOTEBOOK.read_text(encoding="utf-8"))
    pattern = re.compile(
        r"SNR=\s*(?P<snr>\d+)\s*dB\s*\|\s*C=\s*(?P<c>\d+)\s*\|\s*"
        r"CBR=(?P<cbr>[0-9.]+)\s*\|\s*PSNR=(?P<psnr>[0-9.]+)\s*dB"
    )
    rows: list[dict[str, object]] = []
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            text = "".join(output.get("text", []))
            for match in pattern.finditer(text):
                rows.append(
                    {
                        "method": "Teammate SwinJSCC SA+RA",
                        "category": "Analog JSCC",
                        "cbr": float(match.group("cbr")),
                        "psnr_db": float(match.group("psnr")),
                        "ssim": math.nan,
                        "lpips": math.nan,
                        "clip_cosine": math.nan,
                        "source": "Teammate executed SwinJSCC notebook",
                        "snr_db": float(match.group("snr")),
                        "C": int(match.group("c")),
                    }
                )
    if len(rows) != 25:
        raise RuntimeError(f"Expected 25 SwinJSCC grid points, found {len(rows)}")
    return pd.DataFrame(rows)


def load_ntscc_sweep() -> pd.DataFrame:
    rows = []
    for path in sorted(NTSCC_SWEEP.glob("ntscc_kodim08_*db.json")):
        snr_match = re.search(r"_(\d+)db\.json$", path.name)
        if not snr_match:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "method": "Nandini NTSCC sweep",
                "category": "Analog JSCC",
                "cbr": float(data["cbr"]),
                "psnr_db": float(data["psnr"]),
                "ssim": math.nan,
                "lpips": math.nan,
                "clip_cosine": math.nan,
                "source": "Nandini NTSCC standalone sweep",
                # Some saved JSON metadata says 10 dB for every file; the filename
                # and measured curve are the actual sweep labels used by the runner.
                "snr_db": float(snr_match.group(1)),
            }
        )
    return pd.DataFrame(rows)


def write_combined_csv(user: pd.DataFrame, team: pd.DataFrame, sweep: pd.DataFrame) -> None:
    columns = [
        "source",
        "method",
        "category",
        "snr_db",
        "cbr",
        "psnr_db",
        "ssim",
        "lpips",
        "clip_cosine",
    ]
    pd.concat([user, team, sweep], ignore_index=True)[columns].to_csv(
        OUT / "comparison_data.csv", index=False
    )


def make_measured_plot(user: pd.DataFrame, team: pd.DataFrame, sweep: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))

    ax = axes[0]
    team_10 = team[team["snr_db"] == 10].sort_values("cbr")
    ax.plot(
        team_10["cbr"],
        team_10["psnr_db"],
        "o-",
        linewidth=2.2,
        label="Teammate SwinJSCC SA+RA (Kodak24 mean)",
    )
    selected = user[
        user["method"].isin(
            [
                "DeepJSCC",
                "NTSCC",
                "DeepJSCC codebook",
                "NTSCC codebook",
                "NTSCC active-prefix PQ16 K=256",
                "JPEG Q90 + LDPC/QPSK/AWGN",
                "JPEG Q75 + LDPC/QPSK/AWGN",
                "JPEG Q50 + LDPC/QPSK/AWGN",
                "JPEG Q25 + LDPC/QPSK/AWGN",
            ]
        )
    ].dropna(subset=["cbr", "psnr_db"])
    colors = {
        "DeepJSCC": "#d62728",
        "NTSCC": "#2ca02c",
        "DeepJSCC codebook": "#ff7f0e",
        "NTSCC codebook": "#9467bd",
        "NTSCC active-prefix PQ16 K=256": "#8c564b",
    }
    for _, row in selected.iterrows():
        method = str(row["method"])
        if method.startswith("JPEG"):
            continue
        ax.scatter(
            row["cbr"], row["psnr_db"], s=85, marker="D", color=colors[method], label=method
        )
    jpeg = selected[selected["method"].str.startswith("JPEG")].sort_values("cbr")
    ax.plot(jpeg["cbr"], jpeg["psnr_db"], "s--", color="#7f7f7f", label="JPEG + LDPC/QPSK")
    ax.set_xscale("log")
    ax.set_xlabel("CBR = complex channel uses / RGB source values (log scale)")
    ax.set_ylabel("PSNR (dB), higher is better")
    ax.set_title("A. Saved 10 dB AWGN results")
    ax.legend(fontsize=7.8, loc="best")

    ax = axes[1]
    ssim = user[
        user["method"].isin(
            [
                "DeepJSCC",
                "NTSCC",
                "DeepJSCC codebook",
                "NTSCC codebook",
                "NTSCC active-prefix PQ16 K=256",
            ]
        )
    ].dropna(subset=["cbr", "ssim"])
    for _, row in ssim.iterrows():
        method = str(row["method"])
        ax.scatter(row["cbr"], row["ssim"], s=95, marker="D", color=colors[method], label=method)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("CBR (log scale)")
    ax.set_ylabel("SSIM, higher is better")
    ax.set_title("B. Structural quality (your final notebook)")
    ax.legend(fontsize=8, loc="best")
    ax.text(
        0.03,
        0.05,
        "Teammate notebook did not save SSIM,\nso no SwinJSCC SSIM point is invented.",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#bbbbbb"},
    )

    ax = axes[2]
    sweep = sweep.sort_values("snr_db")
    ax.plot(
        sweep["snr_db"], sweep["psnr_db"], "D-", linewidth=2.2, color="#2ca02c",
        label="Nandini NTSCC (CBR ≈ 0.0313; one image)",
    )
    for c, style in [(32, "o-"), (64, "s-")]:
        subset = team[team["C"] == c].sort_values("snr_db")
        ax.plot(
            subset["snr_db"], subset["psnr_db"], style, linewidth=1.8,
            label=f"Teammate SwinJSCC C={c} (CBR={subset['cbr'].iloc[0]:.4f}; Kodak24)",
        )
    ax.set_xlabel("Channel SNR (dB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("C. Noise robustness (protocols differ)")
    ax.legend(fontsize=7.8, loc="best")

    fig.suptitle(
        "Measured compression–reconstruction comparison\n"
        "Solid lines connect operating points; cross-protocol gaps are descriptive, not significance tests",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(OUT / "measured_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_paper_plot() -> None:
    # These are aggregate values explicitly reported in the papers, not values
    # digitized from figure pixels.  Both use BPG+LDPC as 0, but the exact
    # experiments/baselines are not identical, hence the separate plot.
    labels = ["NTSCC\n(Kodak)", "SwinJSCC SA+RA\n(Kodak)", "SwinJSCC fixed\n(Kodak)"]
    savings = [17.96, 25.10, 29.71]
    colors = ["#2ca02c", "#1f77b4", "#4c78a8"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(labels, savings, color=colors)
    ax.bar_label(bars, labels=[f"{v:.2f}%" for v in savings], padding=4)
    ax.set_ylabel("Paper-reported average bandwidth/rate saving vs BPG+LDPC (%)")
    ax.set_ylim(0, 35)
    ax.set_title("Original-paper aggregate rate savings (context only)")
    ax.text(
        0.5,
        -0.23,
        "Do not treat the bar heights as a controlled head-to-head test: paper protocols and BD metrics differ.\n"
        "DeepJSCC (2019) did not report a directly comparable Kodak BD-rate aggregate.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(OUT / "paper_reported_savings.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(user: pd.DataFrame, team: pd.DataFrame) -> None:
    ntscc = user[user["method"] == "NTSCC"].iloc[0]
    deep = user[user["method"] == "DeepJSCC"].iloc[0]
    team_10 = team[team["snr_db"] == 10].sort_values("cbr")
    # Linear interpolation is only a visual aid between the two neighboring
    # tested SwinJSCC points, not an additional measurement.
    target = float(ntscc["cbr"])
    low, high = team_10.iloc[0], team_10.iloc[1]
    interpolated = float(
        low["psnr_db"]
        + (target - low["cbr"])
        * (high["psnr_db"] - low["psnr_db"])
        / (high["cbr"] - low["cbr"])
    )
    summary = {
        "local_10db": {
            "ntscc": {"cbr": float(ntscc["cbr"]), "psnr_db": float(ntscc["psnr_db"])},
            "deepjscc": {"cbr": float(deep["cbr"]), "psnr_db": float(deep["psnr_db"])},
            "teammate_swinjscc_interpolated_at_ntscc_cbr": {
                "cbr": target,
                "psnr_db": interpolated,
                "ntscc_minus_swinjscc_db": float(ntscc["psnr_db"]) - interpolated,
            },
        },
        "paper_reported_context": {
            "ntscc_kodak_bandwidth_saving_vs_bpg_ldpc_percent": 17.96,
            "ntscc_kodak_bd_psnr_gain_vs_bpg_ldpc_db": 0.81,
            "swinjscc_sa_ra_bd_rate_vs_bpg_ldpc_percent": -25.10,
            "swinjscc_fixed_bd_rate_vs_bpg_ldpc_percent": -29.71,
        },
    }
    (OUT / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    user = load_user_results()
    team = load_team_swinjscc()
    sweep = load_ntscc_sweep()
    write_combined_csv(user, team, sweep)
    make_measured_plot(user, team, sweep)
    make_paper_plot()
    write_summary(user, team)
    print(f"Wrote comparison artifacts to {OUT}")


if __name__ == "__main__":
    main()
