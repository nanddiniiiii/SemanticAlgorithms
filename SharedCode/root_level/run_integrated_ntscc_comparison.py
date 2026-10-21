"""Evaluate selected rate-conditioned NTSCC VQ through PHY and rebuild all comparisons."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from semcom_benchmark.digital_phy import TeamDigitalPHY
from semcom_benchmark.metrics import MetricSuite, as_bchw01, to_pil
from semcom_benchmark.model_wrappers import NTSCCWindowsAdapter
from semcom_benchmark.ntscc_rate_conditioned_vq import RateConditionedConfig, RateConditionedPQ


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "benchmark_outputs" / "final_repaired"
SWEEP = ROOT / "benchmark_outputs" / "ntscc_rate_conditioned_sweep"
OUTPUT = ROOT / "benchmark_outputs" / "integrated_rate_conditioned_comparison"
SEEDS = (42, 43, 44, 45, 46)
SNR_DB = 10.0
RATE_GROUPS = [[0], [1], [2], [3, 4, 5, 6, 7], list(range(8, 16))]


def safe(value):
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(name, value):
    (OUTPUT / name).write_text(json.dumps(safe(value), indent=2), encoding="utf-8")


def evaluate_new_ntscc_codebook():
    device = torch.device("cpu")
    adapter = NTSCCWindowsAdapter(
        ROOT / "references" / "NTSCC_JSAC22_official",
        ROOT / "NTSCC_JSAC22" / "checkpoints" / "ntscc_hyperprior_quality_1_psnr.pth",
        device,
        SNR_DB,
    )
    if any(parameter.requires_grad for parameter in adapter.model.parameters()):
        raise RuntimeError("Frozen NTSCC invariant failed")
    cached = torch.load(SWEEP / "calibration_500_cifar_plus_23_kodak.pt", map_location="cpu", weights_only=False)
    calibration = cached["calibration"]
    config = RateConditionedConfig(block_size=4, codebook_size=256, normalize=False, rvq_stages=1)
    fit_started = time.perf_counter()
    quantizer = RateConditionedPQ(adapter.config.multiple_rate, config, RATE_GROUPS)
    reference = as_bchw01(Image.open(ROOT / "Deep-JSCC-PyTorch" / "demo" / "kodim08.png").convert("RGB"))
    target = adapter.encode(reference)
    required_rates = set(map(int, target.rate_indexes.cpu().numpy().reshape(-1)))
    quantizer.fit(calibration, required_rates=required_rates)
    fit_latency = time.perf_counter() - fit_started
    encode_started = time.perf_counter()
    packet = quantizer.encode(target.s_masked, target.rate_indexes)
    encode_latency = time.perf_counter() - encode_started

    phy = TeamDigitalPHY(ROOT / "references" / "SemanticSchedulerEnd2End_team")
    metric_suite = MetricSuite(device, include_optional=True)
    trials = []
    successful_metrics = []
    first_reconstruction = None
    for seed in SEEDS:
        trial_started = time.perf_counter()
        physical = phy.transmit(packet.bits, SNR_DB, seed, str(device))
        trial = {
            "seed": seed,
            "frame_success": bool(physical.frame_success),
            "failure_reason": physical.failure_reason,
            "information_bits": physical.information_bits,
            "framed_bits": physical.framed_bits,
            "padding_bits": physical.padding_bits,
            "coded_bits": physical.coded_bits,
            "complex_channel_uses": physical.channel_uses,
        }
        if physical.frame_success and physical.recovered_bits is not None:
            try:
                latent, rates = quantizer.decode(
                    type(packet)(**{**packet.__dict__, "bits": physical.recovered_bits}), device
                )
                reconstruction = adapter.decode_channel_latent(latent, rates)
                metrics = metric_suite(reference, reconstruction) if not successful_metrics else successful_metrics[0]
                trial.update(metrics)
                successful_metrics.append(metrics)
                if first_reconstruction is None:
                    first_reconstruction = reconstruction
            except Exception as error:
                trial["frame_success"] = False
                trial["failure_reason"] = f"packet_decode_error: {error}"
        trial["latency_s"] = time.perf_counter() - trial_started
        trials.append(trial)

    if not successful_metrics or first_reconstruction is None:
        raise RuntimeError(f"All selected-wrapper PHY trials failed: {trials}")
    metric_keys = successful_metrics[0]
    means = {f"{key}_mean": float(np.mean([item[key] for item in successful_metrics])) for key in metric_keys}
    stds = {f"{key}_std": float(np.std([item[key] for item in successful_metrics])) for key in metric_keys}
    first = trials[0]
    source_pixels = int(reference.shape[-2] * reference.shape[-1])
    source_rgb_values = int(reference.numel())
    reconstruction_path = OUTPUT / "ntscc_rate_conditioned_codebook_reconstruction.png"
    to_pil(first_reconstruction).save(reconstruction_path)
    result = {
        "method": "NTSCC codebook",
        "algorithm": "Rate-conditioned PQ4 K=256 (exact partial dimensions) + LDPC/QPSK/AWGN",
        "configuration": config.label,
        "official_ntscc_frozen": True,
        "evaluation_snr_db": SNR_DB,
        "channel_seeds": list(SEEDS),
        "frame_success_rate": sum(row["frame_success"] for row in trials) / len(trials),
        "latent_units": len(packet.keys),
        "active_value_count": packet.active_value_count,
        "latent_mse_active": packet.latent_mse,
        "latent_index_bits": packet.index_bits,
        "rate_map_bits": packet.rate_bits,
        "metadata_bits": packet.metadata_bits,
        "crc_bits": packet.crc_bits,
        "payload_bits": packet.total_bits,
        "bpp": packet.total_bits / source_pixels,
        "coded_bits": first["coded_bits"],
        "complex_channel_uses": first["complex_channel_uses"],
        "cbr": first["complex_channel_uses"] / source_rgb_values,
        "fit_latency_s": fit_latency,
        "encode_latency_s": encode_latency,
        "mean_trial_latency_s": float(np.mean([row["latency_s"] for row in trials])),
        "total_evaluation_latency_s": float(sum(row["latency_s"] for row in trials)),
        "reconstruction_path": str(reconstruction_path),
        **means,
        **stds,
        "trials": trials,
        "codebook_diagnostics": quantizer.diagnostics(),
        "calibration_provenance": cached["provenance"],
    }
    write_json("ntscc_rate_conditioned_codebook_phy_result.json", result)
    pd.DataFrame(trials).to_csv(OUTPUT / "ntscc_rate_conditioned_codebook_phy_trials.csv", index=False)
    return result


def build_kpi_table(new_result):
    table = pd.read_csv(BASE / "consolidated_measured_comparison.csv", na_values=["N/A", ""])
    table = table.rename(columns={"Model": "Model"})
    table["Coded bits"] = np.nan
    table["BPP"] = np.nan
    table["Latent MSE"] = np.nan
    table["Frame success rate"] = np.nan
    table["Fit latency (s)"] = np.nan

    with Image.open(ROOT / "Deep-JSCC-PyTorch" / "demo" / "kodim08.png") as source_image:
        source_pixels = int(source_image.width * source_image.height)
        reconstruction_bits = int(source_pixels * 3 * 8)
    json_by_model = {
        "RAW": BASE / "digital_raw_result.json",
        "JPEG Q90": BASE / "digital_jpeg_q90_result.json",
        "JPEG Q75": BASE / "digital_jpeg_q75_result.json",
        "JPEG Q50": BASE / "digital_jpeg_q50_result.json",
        "JPEG Q25": BASE / "digital_jpeg_q25_result.json",
        "DeepJSCC": BASE / "deepjscc_result.json",
        "NTSCC": BASE / "ntscc_result.json",
        "DeepJSCC codebook": BASE / "deepjscc_codebook_result.json",
    }
    for model, path in json_by_model.items():
        result = json.loads(path.read_text(encoding="utf-8"))
        selector = table.Model == model
        table.loc[selector, "Coded bits"] = result.get("coded_bits")
        payload = result.get("payload_bits")
        table.loc[selector, "BPP"] = (
            result.get("total_entropy_bpp") if model == "NTSCC"
            else (None if payload is None else payload / source_pixels)
        )
        table.loc[selector, "Latent MSE"] = result.get("quantization_mse")
        table.loc[selector, "Frame success rate"] = result.get("frame_success_rate")

    selector = table.Model == "NTSCC codebook"
    replacements = {
        "Algorithm": new_result["algorithm"],
        "Original image packets": 1,
        "Latent representation units": new_result["latent_units"],
        "CBR": new_result["cbr"],
        "PSNR (dB)": new_result["psnr_db_mean"],
        "SSIM": new_result["ssim_mean"],
        "LPIPS": new_result["lpips_mean"],
        "Reconstruction bits": reconstruction_bits,
        "Latent bits": new_result["latent_index_bits"],
        "Compression/payload bits": new_result["payload_bits"],
        "Complex channel uses": new_result["complex_channel_uses"],
        "Computational complexity": "Rate-conditioned nearest-centroid PQ4 O(N_active*K*4) plus iterative LDPC",
        "Measured latency (s)": new_result["total_evaluation_latency_s"],
        "Notes": "Selected measured Pareto-knee wrapper; payload includes indices, rate map, metadata, and CRC.",
        "Coded bits": new_result["coded_bits"],
        "BPP": new_result["bpp"],
        "Latent MSE": new_result["latent_mse_active"],
        "Frame success rate": new_result["frame_success_rate"],
        "Fit latency (s)": new_result["fit_latency_s"],
    }
    for column, value in replacements.items():
        table.loc[selector, column] = value
    table.to_csv(OUTPUT / "all_methods_kpi_table.csv", index=False, na_rep="N/A")
    write_json("all_methods_kpi_table.json", table.to_dict(orient="records"))
    return table


def labeled_bars(axis, labels, values, title, ylabel, color, log=False):
    positions = np.arange(len(labels))
    bars = axis.bar(positions, values, color=color)
    axis.set_xticks(positions, labels, rotation=32, ha="right")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    if log:
        axis.set_yscale("log")
    axis.bar_label(bars, fmt="%.3g", padding=3, fontsize=8, rotation=90)


def make_plots(table):
    learned_and_jpeg = table[table.Model != "RAW"].copy()
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    labeled_bars(axes[0], learned_and_jpeg.Model, learned_and_jpeg["PSNR (dB)"].astype(float),
                 "Reconstruction quality: PSNR", "PSNR (dB), higher is better", "#2E8B57")
    labeled_bars(axes[1], table.Model, table.SSIM.astype(float),
                 "Structural similarity", "SSIM, higher is better", "#4C78A8")
    labeled_bars(axes[2], table.Model, table.LPIPS.astype(float),
                 "Perceptual distance", "LPIPS, lower is better", "#8E5A9F")
    fig.suptitle("Measured reconstruction comparison (RAW PSNR = infinity and omitted from PSNR panel)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUTPUT / "all_methods_quality_ssim_lpips.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(15, 8))
    labeled_bars(axis, table.Model, table["Complex channel uses"].astype(float),
                 "Measured channel usage", "Complex channel uses (log scale)", "#D17A22", log=True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "all_methods_channel_usage.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    image_paths = [
        ("Original", ROOT / "Deep-JSCC-PyTorch" / "demo" / "kodim08.png"),
        ("DeepJSCC", BASE / "deepjscc_reconstruction.png"),
        ("NTSCC", BASE / "ntscc_reconstruction.png"),
        ("DeepJSCC codebook", BASE / "deepjscc_codebook_reconstruction.png"),
        ("New NTSCC codebook", OUTPUT / "ntscc_rate_conditioned_codebook_reconstruction.png"),
    ]
    fig, axes = plt.subplots(1, len(image_paths), figsize=(22, 5))
    for axis, (label, path) in zip(axes, image_paths):
        axis.imshow(Image.open(path).convert("RGB"))
        axis.set_title(label)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT / "all_learned_reconstructions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = evaluate_new_ntscc_codebook()
    table = build_kpi_table(result)
    make_plots(table)
    print(table.to_string(index=False))
    return result, table


if __name__ == "__main__":
    main()
