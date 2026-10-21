"""Run a frozen-NTSCC rate-conditioned PQ/RVQ sweep without changing baseline code."""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import to_tensor

from semcom_benchmark.metrics import MetricSuite, as_bchw01, to_pil
from semcom_benchmark.model_wrappers import NTSCCWindowsAdapter
from semcom_benchmark.ntscc_rate_conditioned_vq import (
    RateConditionedConfig,
    RateConditionedPQ,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "benchmark_outputs" / "ntscc_rate_conditioned_sweep"
CACHE = OUTPUT / "calibration_500_cifar_plus_23_kodak.pt"
TEST_IMAGE = ROOT / "Deep-JSCC-PyTorch" / "demo" / "kodim08.png"
KODAK = ROOT / "references" / "SemanticSchedulerEnd2End_team" / "data" / "Kodak24"
CALIBRATION_CIFAR_COUNT = 500
CALIBRATION_SIZE = 128
BATCH_SIZE = 32
RATE_GROUPS = [[0], [1], [2], [3, 4, 5, 6, 7], list(range(8, 16))]


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(name: str, payload) -> None:
    (OUTPUT / name).write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


def calibration_data(adapter: NTSCCWindowsAdapter):
    if CACHE.is_file():
        cached = torch.load(CACHE, map_location="cpu", weights_only=False)
        return cached["calibration"], cached["provenance"]

    dataset = CIFAR10(root=str(ROOT / "data"), train=True, download=False)
    calibration = []
    rate_histogram: Counter[int] = Counter()
    started = time.perf_counter()
    for start in range(0, CALIBRATION_CIFAR_COUNT, BATCH_SIZE):
        tensors = []
        for index in range(start, min(start + BATCH_SIZE, CALIBRATION_CIFAR_COUNT)):
            image = dataset[index][0].convert("RGB").resize(
                (CALIBRATION_SIZE, CALIBRATION_SIZE), Image.Resampling.BICUBIC
            )
            tensors.append(to_tensor(image))
        encoding = adapter.encode(torch.stack(tensors).to(adapter.device))
        calibration.append((encoding.s_masked.cpu(), encoding.rate_indexes.cpu()))
        rate_histogram.update(map(int, encoding.rate_indexes.cpu().numpy().reshape(-1)))
        print(f"calibration CIFAR: {min(start + BATCH_SIZE, CALIBRATION_CIFAR_COUNT)}/{CALIBRATION_CIFAR_COUNT}", flush=True)

    kodak_sources = sorted(path for path in KODAK.glob("kodim*.png") if path.name != TEST_IMAGE.name)
    for number, path in enumerate(kodak_sources, 1):
        image = Image.open(path).convert("RGB")
        encoding = adapter.encode(as_bchw01(image).to(adapter.device))
        calibration.append((encoding.s_masked.cpu(), encoding.rate_indexes.cpu()))
        rate_histogram.update(map(int, encoding.rate_indexes.cpu().numpy().reshape(-1)))
        print(f"calibration Kodak: {number}/{len(kodak_sources)} {path.name}", flush=True)

    provenance = {
        "calibration_sample_count": CALIBRATION_CIFAR_COUNT + len(kodak_sources),
        "cifar10_training_indices": list(range(CALIBRATION_CIFAR_COUNT)),
        "cifar_resize": [CALIBRATION_SIZE, CALIBRATION_SIZE],
        "additional_full_resolution_kodak_sources": [path.name for path in kodak_sources],
        "test_image": str(TEST_IMAGE),
        "test_image_excluded_from_calibration": TEST_IMAGE.name not in {path.name for path in kodak_sources},
        "rate_index_histogram": dict(sorted(rate_histogram.items())),
        "encoding_latency_s": time.perf_counter() - started,
    }
    torch.save({"calibration": calibration, "provenance": provenance}, CACHE)
    return calibration, provenance


def identity_verification(adapter, target, reference, metrics):
    direct = adapter.decode_channel_latent(target.s_masked, target.rate_indexes)
    bypass_latent = target.s_masked.clone()
    bypass_rates = target.rate_indexes.clone()
    wrapped = adapter.decode_channel_latent(bypass_latent, bypass_rates)
    result = {
        "latent_max_abs_error": float((bypass_latent - target.s_masked).abs().max()),
        "rate_map_equal": bool(torch.equal(bypass_rates, target.rate_indexes)),
        "reconstruction_max_abs_error": float((wrapped - direct).abs().max()),
        "direct_metrics": metrics(reference, direct, include_optional=False),
        "wrapped_metrics": metrics(reference, wrapped, include_optional=False),
    }
    result["passed"] = (
        result["latent_max_abs_error"] == 0.0
        and result["rate_map_equal"]
        and result["reconstruction_max_abs_error"] == 0.0
    )
    if not result["passed"]:
        raise RuntimeError(f"No-quantization wrapper identity check failed: {result}")
    return result, direct


def evaluate_configuration(adapter, calibration, target, reference, metrics, config, required_rates):
    fit_started = time.perf_counter()
    quantizer = RateConditionedPQ(adapter.config.multiple_rate, config, RATE_GROUPS)
    quantizer.fit(calibration, required_rates=required_rates)
    fit_latency = time.perf_counter() - fit_started
    evaluation_started = time.perf_counter()
    packet = quantizer.encode(target.s_masked, target.rate_indexes)
    latent, rates = quantizer.decode(packet, adapter.device)
    reconstruction = adapter.decode_channel_latent(latent, rates)
    quality = metrics(reference, reconstruction, include_optional=False)
    evaluation_latency = time.perf_counter() - evaluation_started
    row = {
        "configuration": config.label,
        "block_size": config.block_size,
        "K": config.codebook_size,
        "normalized": config.normalize,
        "rvq_stages": config.rvq_stages,
        "psnr_db": quality["psnr_db"],
        "ssim": quality["ssim"],
        "latent_mse_active": packet.latent_mse,
        "total_transmitted_bits": packet.total_bits,
        "bpp": packet.total_bits / (reference.shape[-2] * reference.shape[-1]),
        "index_bits": packet.index_bits,
        "rate_map_bits": packet.rate_bits,
        "metadata_bits": packet.metadata_bits,
        "crc_bits": packet.crc_bits,
        "active_value_count": packet.active_value_count,
        "fit_latency_s": fit_latency,
        "evaluation_latency_s": evaluation_latency,
        "status": "ok",
        **quantizer.diagnostics(),
    }
    return row, reconstruction


def pareto_and_best(table: pd.DataFrame):
    valid = table.loc[table["status"] == "ok"].copy()
    valid = valid.sort_values(["bpp", "psnr_db"], ascending=[True, False])
    frontier_rows = []
    best_psnr = -float("inf")
    for index, row in valid.iterrows():
        if float(row.psnr_db) > best_psnr:
            frontier_rows.append(index)
            best_psnr = float(row.psnr_db)
    frontier = valid.loc[frontier_rows].copy()
    if len(frontier) == 1:
        return frontier, frontier.iloc[0]
    bpp_span = float(frontier.bpp.max() - frontier.bpp.min()) or 1.0
    psnr_span = float(frontier.psnr_db.max() - frontier.psnr_db.min()) or 1.0
    distance = np.sqrt(
        ((frontier.bpp - frontier.bpp.min()) / bpp_span) ** 2
        + ((frontier.psnr_db.max() - frontier.psnr_db) / psnr_span) ** 2
    )
    return frontier, frontier.loc[distance.idxmin()]


def plots(table, frontier, best, reference, reconstruction):
    valid = table.loc[table.status == "ok"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for block, rows in valid.groupby("block_size"):
        axes[0].scatter(rows.bpp, rows.psnr_db, label=f"block {block}", s=55)
        axes[1].scatter(rows.bpp, rows.ssim, label=f"block {block}", s=55)
    axes[0].plot(frontier.bpp, frontier.psnr_db, "k--", alpha=0.6, label="Pareto frontier")
    axes[0].scatter([best.bpp], [best.psnr_db], marker="*", s=220, color="red", label="selected knee")
    axes[0].set(xlabel="Transmitted BPP", ylabel="PSNR (dB)", title="Rate–distortion sweep")
    axes[1].set(xlabel="Transmitted BPP", ylabel="SSIM", title="SSIM versus transmitted rate")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "rate_distortion_sweep.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    original = reference[0].cpu().permute(1, 2, 0).numpy()
    rebuilt = reconstruction[0].cpu().permute(1, 2, 0).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(original); axes[0].set_title("Original")
    axes[1].imshow(rebuilt); axes[1].set_title(f"Best rate-conditioned wrapper\n{best.configuration}")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT / "best_reconstruction.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    adapter = NTSCCWindowsAdapter(
        ROOT / "references" / "NTSCC_JSAC22_official",
        ROOT / "NTSCC_JSAC22" / "checkpoints" / "ntscc_hyperprior_quality_1_psnr.pth",
        device,
        10.0,
    )
    frozen = all(not parameter.requires_grad for parameter in adapter.model.parameters())
    if not frozen:
        raise RuntimeError("Official NTSCC must remain frozen")

    calibration, calibration_provenance = calibration_data(adapter)
    reference = as_bchw01(Image.open(TEST_IMAGE).convert("RGB")).to(device)
    target = adapter.encode(reference)
    required_rates = set(map(int, target.rate_indexes.cpu().numpy().reshape(-1)))
    metrics = MetricSuite(device, include_optional=False)
    identity, native_noiseless = identity_verification(adapter, target, reference, metrics)
    write_json("identity_verification.json", identity)
    write_json("calibration_provenance.json", calibration_provenance)

    primary = [RateConditionedConfig(block, k, True, 1) for block in (4, 8) for k in (256, 512, 1024)]
    remaining = [RateConditionedConfig(block, k, True, 1) for block in (2, 16) for k in (256, 512, 1024)]
    normalization_ablation = [RateConditionedConfig(block, k, False, 1) for block in (4, 8) for k in (256, 512)]
    rows, reconstructions = [], {}

    def run(config):
        print(f"START {config.label}", flush=True)
        try:
            row, reconstruction = evaluate_configuration(
                adapter, calibration, target, reference, metrics, config, required_rates
            )
            reconstructions[config.label] = reconstruction.cpu()
            print(
                f"DONE {config.label}: PSNR={row['psnr_db']:.4f}, SSIM={row['ssim']:.4f}, "
                f"MSE={row['latent_mse_active']:.6f}, bits={row['total_transmitted_bits']}",
                flush=True,
            )
        except Exception as error:
            row = {**asdict(config), "configuration": config.label, "status": "invalid", "error": str(error)}
            print(f"INVALID {config.label}: {error}", flush=True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUTPUT / "sweep_results_in_progress.csv", index=False, na_rep="N/A")

    for config in primary:
        run(config)

    primary_table = pd.DataFrame(rows)
    for block in (4, 8):
        candidates = primary_table[(primary_table.status == "ok") & (primary_table.block_size == block)]
        if not candidates.empty:
            best_primary = candidates.sort_values(["psnr_db", "bpp"], ascending=[False, True]).iloc[0]
            run(RateConditionedConfig(block, int(best_primary.K), True, 2))

    for config in remaining + normalization_ablation:
        run(config)

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "sweep_results.csv", index=False, na_rep="N/A")
    write_json("sweep_results.json", rows)
    frontier, best = pareto_and_best(table)
    frontier.to_csv(OUTPUT / "pareto_frontier.csv", index=False)
    best_reconstruction = reconstructions[str(best.configuration)]
    to_pil(best_reconstruction).save(OUTPUT / "best_reconstruction_only.png")
    plots(table, frontier, best, reference, best_reconstruction)
    summary = {
        "official_ntscc_frozen": frozen,
        "checkpoint_validation": adapter.validation,
        "identity_verification": identity,
        "calibration": calibration_provenance,
        "rate_groups": RATE_GROUPS,
        "tested_configuration_count": len(rows),
        "valid_configuration_count": int((table.status == "ok").sum()),
        "selection_rule": "Euclidean knee nearest minimum-BPP/maximum-PSNR ideal on the measured Pareto frontier",
        "best_configuration": best.to_dict(),
    }
    write_json("summary.json", summary)
    print(json.dumps(json_safe(summary), indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
