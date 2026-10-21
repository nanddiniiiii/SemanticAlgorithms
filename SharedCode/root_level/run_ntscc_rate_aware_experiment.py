"""Isolated NTSCC-aware codebook experiment; does not run unrelated baselines."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.datasets import CIFAR10

from semcom_benchmark.digital_phy import TeamDigitalPHY
from semcom_benchmark.metrics import MetricSuite, as_bchw01, summarize_trials, to_pil
from semcom_benchmark.model_wrappers import NTSCCWindowsAdapter
from semcom_benchmark.ntscc_rate_codebook import NTSCCActivePrefixK256


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "benchmark_outputs" / "ntscc_aware_codebook_experiment"
VALIDATED = ROOT / "benchmark_outputs" / "final_repaired"
SEEDS = (42, 43, 44, 45, 46)
SNR_DB = 10.0
CALIBRATION_INDICES = tuple(range(64))


def write_json(name: str, payload) -> Path:
    path = OUTPUT / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def rate_histogram(indexes: torch.Tensor) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(map(int, indexes.cpu().numpy().reshape(-1))).items())}


def main() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    adapter = NTSCCWindowsAdapter(
        ROOT / "references" / "NTSCC_JSAC22_official",
        ROOT / "NTSCC_JSAC22" / "checkpoints" / "ntscc_hyperprior_quality_1_psnr.pth",
        device,
        SNR_DB,
    )
    for parameter in adapter.model.parameters():
        if parameter.requires_grad:
            raise RuntimeError("Official NTSCC parameters must remain frozen")

    dataset = CIFAR10(root=str(ROOT / "data"), train=True, download=False)
    calibration = []
    calibration_rates: Counter[int] = Counter()
    for index in CALIBRATION_INDICES:
        image = dataset[index][0].convert("RGB").resize((128, 128), Image.Resampling.BICUBIC)
        encoding = adapter.encode(as_bchw01(image).to(device))
        calibration.append((encoding.s_masked.cpu(), encoding.rate_indexes.cpu()))
        calibration_rates.update(map(int, encoding.rate_indexes.cpu().numpy().reshape(-1)))

    codebook = NTSCCActivePrefixK256(
        ROOT / "references" / "SemanticSchedulerEnd2End_team",
        adapter.config.multiple_rate,
        block_size=16,
    ).fit(calibration)
    np.savez(OUTPUT / "ntscc_active_prefix_pq16_k256.npz", centers=codebook.model.cluster_centers_)

    test_image = Image.open(ROOT / "Deep-JSCC-PyTorch" / "demo" / "kodim08.png").convert("RGB")
    reference = as_bchw01(test_image).to(device)
    target = adapter.encode(reference)
    packet = codebook.encode(target.s_masked, target.rate_indexes)
    decoded_latent, decoded_indexes = codebook.decode(packet.bits, packet, device)
    noiseless_reconstruction = adapter.decode_channel_latent(decoded_latent, decoded_indexes)
    metric_suite = MetricSuite(device, include_optional=True)
    noiseless_metrics = metric_suite(reference, noiseless_reconstruction)
    to_pil(noiseless_reconstruction).save(OUTPUT / "ntscc_active_prefix_noiseless_reconstruction.png")

    native = json.loads((VALIDATED / "ntscc_result.json").read_text(encoding="utf-8"))
    generic = json.loads((VALIDATED / "ntscc_codebook_result.json").read_text(encoding="utf-8"))
    generic_noiseless_all = json.loads((VALIDATED / "codebook_noiseless_ablation.json").read_text(encoding="utf-8"))
    generic_noiseless = generic_noiseless_all["NTSCC codebook"]
    meaningful = (
        noiseless_metrics["psnr_db"] >= float(generic_noiseless["psnr_db"]) + 1.0
        and noiseless_metrics["ssim"] > float(generic_noiseless["ssim"])
    )

    provenance = {
        "design": "K=256 active-prefix product quantization with 16-D blocks",
        "official_rate_choices": list(map(int, adapter.config.multiple_rate)),
        "calibration_dataset": "CIFAR-10 training split",
        "calibration_indices": list(CALIBRATION_INDICES),
        "calibration_resize": [128, 128],
        "test_image_used_for_training": False,
        "calibration_rate_index_histogram": {str(k): int(v) for k, v in sorted(calibration_rates.items())},
        "calibration_active_block_count": codebook.calibration_block_count,
        "test_rate_index_histogram": rate_histogram(target.rate_indexes),
        "codebook_sha256": codebook.content_hash(),
        "shared_codebook_storage_bits": codebook.shared_storage_bits(),
        "team_packet_codec_reused": True,
        "team_ldpc_qpsk_awgn_reused": True,
        "official_ntscc_frozen": True,
        "checkpoint_validation": adapter.validation,
    }
    write_json("provenance.json", provenance)

    noiseless = {
        "method": "NTSCC active-prefix PQ16 K=256",
        "experiment": "noiseless quantization",
        "packet_bits": int(packet.bits.size),
        "index_count": packet.index_count,
        "active_value_count": packet.active_value_count,
        "active_quantization_mse": packet.active_quantization_mse,
        "shared_codebook_storage_bits": codebook.shared_storage_bits(),
        **packet.diagnostics,
        **noiseless_metrics,
        "meaningful_improvement_rule": "PSNR >= generic + 1 dB and SSIM > generic",
        "meaningful_improvement": bool(meaningful),
        "generic_noiseless_psnr_db": generic_noiseless["psnr_db"],
        "generic_noiseless_ssim": generic_noiseless["ssim"],
        "native_psnr_db_mean": native["psnr_db_mean"],
        "native_ssim_mean": native["ssim_mean"],
    }
    write_json("noiseless_result.json", noiseless)

    noisy = None
    if meaningful:
        phy = TeamDigitalPHY(ROOT / "references" / "SemanticSchedulerEnd2End_team")
        trials = []
        successful_metrics = []
        first_reconstruction = None
        for seed in SEEDS:
            physical = phy.transmit(packet.bits, SNR_DB, seed, str(device))
            trial = {
                "seed": seed,
                "frame_success": bool(physical.frame_success),
                "failure_reason": physical.failure_reason,
                "coded_bits": physical.coded_bits,
                "complex_channel_uses": physical.channel_uses,
            }
            if physical.frame_success and physical.recovered_bits is not None:
                try:
                    latent, indexes = codebook.decode(physical.recovered_bits, packet, device)
                    reconstruction = adapter.decode_channel_latent(latent, indexes)
                    metrics = metric_suite(reference, reconstruction)
                    trial.update(metrics)
                    successful_metrics.append(metrics)
                    if first_reconstruction is None:
                        first_reconstruction = reconstruction
                except Exception as error:
                    trial["frame_success"] = False
                    trial["failure_reason"] = f"packet_decode_error: {error}"
            trials.append(trial)
        first = trials[0]
        source_rgb_values = int(reference.numel())
        noisy = {
            "method": "NTSCC active-prefix PQ16 K=256 + LDPC/QPSK/AWGN",
            "evaluation_snr_db": SNR_DB,
            "channel_seeds": list(SEEDS),
            "packet_bits": int(packet.bits.size),
            "coded_bits": first["coded_bits"],
            "complex_channel_uses": first["complex_channel_uses"],
            "cbr": first["complex_channel_uses"] / source_rgb_values,
            "frame_success_rate": sum(row["frame_success"] for row in trials) / len(trials),
            "active_quantization_mse": packet.active_quantization_mse,
            **packet.diagnostics,
            **summarize_trials(successful_metrics),
            "trials": trials,
        }
        if first_reconstruction is not None:
            path = OUTPUT / "ntscc_active_prefix_noisy_reconstruction.png"
            to_pil(first_reconstruction).save(path)
            noisy["reconstruction_path"] = str(path)
        write_json("noisy_result.json", noisy)
        pd.DataFrame(trials).to_csv(OUTPUT / "noisy_trials.csv", index=False)
    else:
        write_json("noisy_transport_skipped.json", {
            "skipped": True,
            "reason": "Noiseless result did not pass the predefined meaningful-improvement rule.",
        })

    comparison = [
        {
            "method": "Native NTSCC",
            "experiment": "validated native baseline",
            "packet_bits": None,
            "complex_channel_uses": native["complex_channel_uses"],
            "cbr": native["cbr"],
            "psnr_db": native["psnr_db_mean"],
            "ssim": native["ssim_mean"],
            "lpips": native["lpips_mean"],
            "clip_cosine": native["clip_cosine_mean"],
            "quantization_mse": None,
            "quantization_mse_scope": "N/A: native NTSCC is not quantized by a codebook",
        },
        {
            "method": "Generic NTSCC K=256",
            "experiment": "validated generic codebook baseline",
            "packet_bits": generic["packet_bits"],
            "complex_channel_uses": generic["complex_channel_uses"],
            "cbr": generic["cbr"],
            "psnr_db": generic["psnr_db_mean"],
            "ssim": generic["ssim_mean"],
            "lpips": generic["lpips_mean"],
            "clip_cosine": generic["clip_cosine_mean"],
            "quantization_mse": generic["quantization_mse"],
            "quantization_mse_scope": "mean over the generic zero-padded 256-D tensor",
        },
        {
            "method": "NTSCC active-prefix PQ16 K=256",
            "experiment": "noiseless quantization" if noisy is None else "LDPC/QPSK/AWGN",
            "packet_bits": packet.bits.size,
            "complex_channel_uses": None if noisy is None else noisy["complex_channel_uses"],
            "cbr": None if noisy is None else noisy["cbr"],
            "psnr_db": noiseless_metrics["psnr_db"] if noisy is None else noisy["psnr_db_mean"],
            "ssim": noiseless_metrics["ssim"] if noisy is None else noisy["ssim_mean"],
            "lpips": noiseless_metrics["lpips"] if noisy is None else noisy["lpips_mean"],
            "clip_cosine": noiseless_metrics["clip_cosine"] if noisy is None else noisy["clip_cosine_mean"],
            "quantization_mse": packet.active_quantization_mse,
            "quantization_mse_scope": "mean over officially active prefix values only",
        },
    ]
    pd.DataFrame(comparison).to_csv(OUTPUT / "measured_comparison.csv", index=False, na_rep="N/A")
    write_json("measured_comparison.json", comparison)
    result = {"provenance": provenance, "noiseless": noiseless, "noisy": noisy, "comparison": comparison}
    write_json("experiment_summary.json", result)
    return result


if __name__ == "__main__":
    result = main()
    print(json.dumps({"noiseless": result["noiseless"], "noisy": result["noisy"]}, indent=2))
