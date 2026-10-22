"""Ordered, reusable end-to-end benchmark used by the repaired notebook."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.datasets import CIFAR10
from torchvision.transforms import functional as TF

from .codebooks import K256LatentCodebook
from .digital_phy import (
    TeamDigitalPHY,
    bits_to_bytes,
    bytes_to_bits,
    decode_jpeg,
    decode_raw,
    jpeg_payload,
    raw_payload,
)
from .metrics import MetricSuite, as_bchw01, summarize_trials, to_pil
from .model_wrappers import DeepJSCCAdapter, NTSCCWindowsAdapter
from .ntscc_rate_codebook import NTSCCActivePrefixK256


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class BenchmarkSession:
    def __init__(
        self,
        root: Path,
        output_dir: Path | None = None,
        snr_db: float = 10.0,
        seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
        jpeg_qualities: tuple[int, ...] = (90, 75, 50, 25),
        device: str = "cpu",
    ) -> None:
        self.root = Path(root).resolve()
        self.output_dir = (output_dir or self.root / "benchmark_outputs" / "final_repaired").resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snr_db = float(snr_db)
        self.seeds = tuple(int(seed) for seed in seeds)
        if len(self.seeds) < 5:
            raise ValueError("At least five channel-noise seeds are required")
        self.jpeg_qualities = jpeg_qualities
        self.device = torch.device(device)
        self.image_path = self.root / "data" / "Kodak24" / "kodim08.png"
        self.image = Image.open(self.image_path).convert("RGB")
        self.reference = as_bchw01(self.image).to(self.device)
        self.height, self.width = self.reference.shape[-2:]
        self.source_pixels = self.height * self.width
        self.source_rgb_values = self.source_pixels * 3
        self.raw_bits = self.source_rgb_values * 8
        self.metric_suite = MetricSuite(self.device, include_optional=True)
        self.deep = None
        self.ntscc = None
        self.phy = None
        self.native_results: dict[str, dict] = {}
        self.digital_results: dict[str, dict] = {}
        self.codec_results: dict[str, dict] = {}
        self.codebook_results: dict[str, dict] = {}
        self.noiseless_results: dict[str, dict] = {}
        self.reconstructions: dict[str, torch.Tensor] = {}

    def write_json(self, filename: str, payload) -> Path:
        path = self.output_dir / filename
        path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
        return path

    def save_reconstruction(self, name: str, value) -> Path:
        path = self.output_dir / f"{name}_reconstruction.png"
        to_pil(value).save(path)
        self.reconstructions[name] = as_bchw01(value).cpu()
        return path

    def load_models(self) -> dict:
        self.deep = DeepJSCCAdapter(
            self.root / "references" / "DeepJSCC_checkpoint_source",
            self.root.parent / "Deep-JSCC-PyTorch" / "out" / "imagenet_10_0.33_200.00_32_19.pth",
            self.device,
            self.snr_db,
        )
        self.ntscc = NTSCCWindowsAdapter(
            self.root / "references" / "NTSCC_JSAC22_official",
            self.root.parent / "NTSCC_JSAC22" / "checkpoints" / "ntscc_hyperprior_quality_1_psnr.pth",
            self.device,
            self.snr_db,
        )
        validation = {"DeepJSCC": self.deep.validation, "NTSCC": self.ntscc.validation}
        self.write_json("checkpoint_validation.json", validation)
        return validation

    def _ensure_models(self) -> None:
        if self.deep is None or self.ntscc is None:
            self.load_models()

    def run_native_models(self) -> pd.DataFrame:
        self._ensure_models()
        deep_started = time.perf_counter()
        padded, original_hw, deep_records = self.deep.encode_tiles(self.reference, 128)
        self._deep_target = (padded, original_hw, deep_records)
        deep_trials, deep_recon = [], None
        for seed in self.seeds:
            trial = self.deep.run_from_records(padded, original_hw, deep_records, seed, 128)
            metrics = self.metric_suite(self.reference, trial.reconstruction)
            deep_trials.append({"seed": seed, **metrics})
            deep_recon = deep_recon if deep_recon is not None else trial.reconstruction
        deep_real_values = sum(int(latent.numel()) for _, _, latent in deep_records)
        deep_complex_uses = math.ceil(deep_real_values / 2)
        deep_result = {
            "method": "DeepJSCC",
            "checkpoint_training_snr_db": self.deep.checkpoint_training_snr_db,
            "evaluation_snr_db": self.snr_db,
            "channel_seeds": list(self.seeds),
            "tile_size": 128,
            "tile_count": len(deep_records),
            "latent_shape_per_tile": list(deep_records[0][2].shape),
            "latent_real_values": deep_real_values,
            "continuous_channel_symbols": deep_real_values,
            "complex_channel_uses": deep_complex_uses,
            "cbr": deep_complex_uses / self.source_rgb_values,
            "repo_real_value_cbr": deep_real_values / self.source_rgb_values,
            "payload_bits": None,
            "latency_s": time.perf_counter() - deep_started,
            "warning": "Continuous JSCC symbols are not bits; evaluation SNR differs from the 200 dB checkpoint training SNR.",
            **summarize_trials(deep_trials),
            "trials": deep_trials,
        }
        deep_result["reconstruction_path"] = str(self.save_reconstruction("deepjscc", deep_recon))
        self.native_results["DeepJSCC"] = deep_result
        self.write_json("deepjscc_result.json", deep_result)

        ntscc_started = time.perf_counter()
        ntscc_encoding = self.ntscc.encode(self.reference)
        self._ntscc_target = ntscc_encoding
        ntscc_trials, ntscc_recon, uses = [], None, None
        for seed in self.seeds:
            trial = self.ntscc.run_from_encoding(self.reference, ntscc_encoding, seed)
            metrics = self.metric_suite(self.reference, trial.reconstruction)
            ntscc_trials.append({"seed": seed, **metrics})
            ntscc_recon = ntscc_recon if ntscc_recon is not None else trial.reconstruction
            uses = trial.complex_channel_uses
        token_count = int(ntscc_encoding.rate_indexes.numel())
        rate_bits = token_count * math.ceil(math.log2(len(self.ntscc.config.multiple_rate)))
        ntscc_result = {
            "method": "NTSCC",
            "checkpoint_training_snr_db": self.ntscc.checkpoint_training_snr_db,
            "evaluation_snr_db": self.snr_db,
            "channel_seeds": list(self.seeds),
            "analysis_latent_shape": list(ntscc_encoding.y.shape),
            "channel_latent_shape": list(ntscc_encoding.s_masked.shape),
            "bpp_y": ntscc_encoding.bpp_y,
            "bpp_z": ntscc_encoding.bpp_z,
            "total_entropy_bpp": ntscc_encoding.bpp_y + ntscc_encoding.bpp_z,
            "entropy_estimated_bits": (ntscc_encoding.bpp_y + ntscc_encoding.bpp_z) * self.source_pixels,
            "complex_channel_uses": uses,
            "cbr_y": uses / self.source_rgb_values,
            "cbr": uses / self.source_rgb_values,
            "rate_map_token_count": token_count,
            "rate_map_bits_untransmitted": rate_bits,
            "payload_bits": None,
            "latency_s": time.perf_counter() - ntscc_started,
            "warning": "Entropy BPP is diagnostic, not radio bits; native cbr_y excludes the untransmitted 4-bit/token rate map.",
            **summarize_trials(ntscc_trials),
            "trials": ntscc_trials,
        }
        ntscc_result["reconstruction_path"] = str(self.save_reconstruction("ntscc", ntscc_recon))
        self.native_results["NTSCC"] = ntscc_result
        self.write_json("ntscc_result.json", ntscc_result)
        return pd.DataFrame([deep_result, ntscc_result])

    def _ensure_phy(self) -> TeamDigitalPHY:
        if self.phy is None:
            self.phy = TeamDigitalPHY(self.root)
        return self.phy

    def run_digital_baselines(self) -> pd.DataFrame:
        phy = self._ensure_phy()
        payloads = [("RAW", raw_payload(self.image), "raw", None)]
        for quality in self.jpeg_qualities:
            payloads.append((f"JPEG Q{quality}", jpeg_payload(self.image, quality), "jpeg", quality))

        all_rows = []
        for name, payload, codec, quality in payloads:
            method_started = time.perf_counter()
            bits = bytes_to_bits(payload)
            codec_reconstruction = self.image if codec == "raw" else decode_jpeg(payload)
            codec_metrics = self.metric_suite(self.reference, codec_reconstruction)
            self.codec_results[name] = {
                "method": name,
                "payload_bits": int(bits.size),
                "compression_ratio_vs_raw": self.raw_bits / bits.size,
                "bit_reduction_vs_raw_pct": 100 * (1 - bits.size / self.raw_bits),
                "latency_s": time.perf_counter() - method_started,
                **{f"{key}_mean": value for key, value in codec_metrics.items()},
            }
            trials, successful_metrics, first_reconstruction = [], [], None
            # One full-resolution RAW trial is already ~9.4 million source
            # bits and takes >5 minutes on CPU Sionna. Learned models, JPEGs,
            # and both codebook variants retain all five requested seeds.
            trial_seeds = self.seeds[:1] if codec == "raw" else self.seeds
            for seed in trial_seeds:
                physical = phy.transmit(bits, self.snr_db, seed, str(self.device))
                row = {
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
                        recovered = bits_to_bytes(physical.recovered_bits)
                        reconstruction = decode_raw(recovered, self.image.size) if codec == "raw" else decode_jpeg(recovered)
                        metrics = codec_metrics
                        row.update(metrics)
                        successful_metrics.append(metrics)
                        first_reconstruction = reconstruction if first_reconstruction is None else first_reconstruction
                    except Exception as error:
                        row.update({"frame_success": False, "failure_reason": f"decode_error: {error}"})
                trials.append(row)
            first = trials[0]
            result = {
                "method": f"{name} + LDPC/QPSK/AWGN",
                "evaluation_snr_db": self.snr_db,
                "payload_bits": int(bits.size),
                "framed_bits": first["framed_bits"],
                "coded_bits": first["coded_bits"],
                "complex_channel_uses": first["complex_channel_uses"],
                "cbr": first["complex_channel_uses"] / self.source_rgb_values,
                "frame_success_rate": sum(row["frame_success"] for row in trials) / len(trials),
                "compression_ratio_vs_raw": self.raw_bits / bits.size,
                "bit_reduction_vs_raw_pct": 100 * (1 - bits.size / self.raw_bits),
                "latency_s": time.perf_counter() - method_started,
                "warning": (
                    "Metrics include successful CRC-valid frames only; LDPC adds redundancy and does not compress. "
                    + ("Full-resolution RAW used one CPU trial; JPEG used five." if codec == "raw" else "Five seeded channel trials.")
                ),
                **summarize_trials(successful_metrics),
                "trials": trials,
            }
            if first_reconstruction is not None:
                result["reconstruction_path"] = str(
                    self.save_reconstruction(f"digital_{name.lower().replace(' ', '_')}", first_reconstruction)
                )
            self.digital_results[name] = result
            self.write_json(f"digital_{name.lower().replace(' ', '_')}_result.json", result)
            pd.DataFrame(trials).to_csv(
                self.output_dir / f"digital_{name.lower().replace(' ', '_')}_trials.csv", index=False
            )
            all_rows.append(result)
        return pd.DataFrame(all_rows)

    def fit_codebooks(self, calibration_images: int = 8) -> dict:
        self._ensure_models()
        fit_started = time.perf_counter()
        dataset = CIFAR10(root=str(self.root.parent / "data"), train=True, download=False)
        selected_indices = list(range(calibration_images))
        deep_latents, ntscc_calibration = [], []
        for index in selected_indices:
            image = dataset[index][0].convert("RGB").resize((128, 128), Image.Resampling.BICUBIC)
            tensor = as_bchw01(image).to(self.device)
            _, _, records = self.deep.encode_tiles(tensor, 128)
            deep_latents.append(records[0][2])
            ntscc_encoding = self.ntscc.encode(tensor)
            ntscc_calibration.append((ntscc_encoding.s_masked, ntscc_encoding.rate_indexes))
        team_root = self.root
        self.deep_codebook = K256LatentCodebook("deepjscc_k256", team_root).fit(deep_latents)
        self.ntscc_codebook = NTSCCActivePrefixK256(
            team_root,
            self.ntscc.config.multiple_rate,
            block_size=16,
        ).fit(ntscc_calibration)
        np.savez(
            self.output_dir / "deepjscc_k256_codebook.npz",
            centers=self.deep_codebook.model.cluster_centers_,
        )
        np.savez(
            self.output_dir / "ntscc_k256_codebook.npz",
            centers=self.ntscc_codebook.model.cluster_centers_,
        )
        provenance = {
            "dataset": "CIFAR-10 training split",
            "dataset_root": str(self.root.parent / "data"),
            "selected_training_indices": selected_indices,
            "calibration_resize": [128, 128],
            "test_image_used_for_fit": False,
            "deepjscc_codebook_sha256": self.deep_codebook.content_hash(),
            "ntscc_codebook_sha256": self.ntscc_codebook.content_hash(),
            "ntscc_codebook_design": "K=256 active-prefix product quantization with 16-D blocks",
            "ntscc_calibration_active_block_count": self.ntscc_codebook.calibration_block_count,
            "separate_codebooks": True,
            "official_encoders_decoders_frozen": True,
            "fit_latency_s": time.perf_counter() - fit_started,
        }
        self.write_json("codebook_calibration_provenance.json", provenance)
        return provenance

    def _deep_codebook_reconstruction(self, packet_bits, packet):
        decoded, _ = self.deep_codebook.decode(packet_bits, packet, self.device)
        padded, (height, width), records = self._deep_target
        reconstruction = torch.empty_like(padded)
        with torch.inference_mode():
            for (top, left, _), latent in zip(records, decoded):
                reconstruction[:, :, top:top + 128, left:left + 128] = self.deep.model.decoder(latent).clamp(0, 1)
        return reconstruction[:, :, :height, :width]

    def _ntscc_codebook_reconstruction(self, packet_bits, packet):
        channel_latent, rate_indexes = self.ntscc_codebook.decode(packet_bits, packet, self.device)
        return self.ntscc.decode_channel_latent(channel_latent, rate_indexes)

    def prepare_codebook_packets(self) -> dict:
        if not hasattr(self, "deep_codebook"):
            self.fit_codebooks()
        if not hasattr(self, "_deep_target") or not hasattr(self, "_ntscc_target"):
            self.run_native_models()
        deep_started = time.perf_counter()
        deep_packet = self.deep_codebook.encode([record[2] for record in self._deep_target[2]])
        self._codebook_encode_latency_s = {"DeepJSCC codebook": time.perf_counter() - deep_started}
        ntscc_started = time.perf_counter()
        ntscc_packet = self.ntscc_codebook.encode(
            self._ntscc_target.s_masked,
            self._ntscc_target.rate_indexes,
        )
        self._codebook_encode_latency_s["NTSCC codebook"] = time.perf_counter() - ntscc_started
        self._codebook_packets = {"DeepJSCC codebook": deep_packet, "NTSCC codebook": ntscc_packet}
        return self._codebook_packets

    def run_noiseless_codebook_ablation(self) -> pd.DataFrame:
        if not hasattr(self, "_codebook_packets"):
            self.prepare_codebook_packets()
        deep_packet = self._codebook_packets["DeepJSCC codebook"]
        ntscc_packet = self._codebook_packets["NTSCC codebook"]
        noiseless_rows = []
        for name, packet, reconstruct, codebook in (
            ("DeepJSCC codebook", deep_packet, self._deep_codebook_reconstruction, self.deep_codebook),
            ("NTSCC codebook", ntscc_packet, self._ntscc_codebook_reconstruction, self.ntscc_codebook),
        ):
            method_started = time.perf_counter()
            reconstruction = reconstruct(packet.bits, packet)
            metrics = self.metric_suite(self.reference, reconstruction)
            result = {
                "method": name,
                "experiment": "noiseless quantization ablation",
                "packet_bits": int(packet.bits.size),
                "shared_codebook_storage_bits": codebook.shared_storage_bits(),
                "quantization_mse": packet.quantization_mse,
                "latency_s": self._codebook_encode_latency_s.get(name, 0.0) + (time.perf_counter() - method_started),
                **packet.diagnostics,
                **metrics,
            }
            self.noiseless_results[name] = result
            self.save_reconstruction(name.lower().replace(" ", "_") + "_noiseless", reconstruction)
            noiseless_rows.append(result)
        self.write_json("codebook_noiseless_ablation.json", self.noiseless_results)
        return pd.DataFrame(noiseless_rows)

    def run_noisy_codebook_transport(self) -> pd.DataFrame:
        if not hasattr(self, "_codebook_packets"):
            self.prepare_codebook_packets()
        deep_packet = self._codebook_packets["DeepJSCC codebook"]
        ntscc_packet = self._codebook_packets["NTSCC codebook"]
        phy = self._ensure_phy()
        digital_rows = []
        for name, packet, reconstruct, codebook in (
            ("DeepJSCC codebook", deep_packet, self._deep_codebook_reconstruction, self.deep_codebook),
            ("NTSCC codebook", ntscc_packet, self._ntscc_codebook_reconstruction, self.ntscc_codebook),
        ):
            transport_started = time.perf_counter()
            trials, successful_metrics, first_reconstruction = [], [], None
            cached_metrics = None
            for seed in self.seeds:
                physical = phy.transmit(packet.bits, self.snr_db, seed, str(self.device))
                row = {
                    "seed": seed,
                    "frame_success": bool(physical.frame_success),
                    "failure_reason": physical.failure_reason,
                    "coded_bits": physical.coded_bits,
                    "complex_channel_uses": physical.channel_uses,
                }
                if physical.frame_success and physical.recovered_bits is not None:
                    try:
                        reconstruction = reconstruct(physical.recovered_bits, packet)
                        cached_metrics = cached_metrics or self.metric_suite(self.reference, reconstruction)
                        row.update(cached_metrics)
                        successful_metrics.append(cached_metrics)
                        first_reconstruction = reconstruction if first_reconstruction is None else first_reconstruction
                    except Exception as error:
                        row.update({"frame_success": False, "failure_reason": f"packet_decode_error: {error}"})
                trials.append(row)
            first = trials[0]
            result = {
                "method": name,
                "evaluation_snr_db": self.snr_db,
                "packet_bits": int(packet.bits.size),
                "payload_bits": int(packet.bits.size),
                "coded_bits": first["coded_bits"],
                "complex_channel_uses": first["complex_channel_uses"],
                "cbr": first["complex_channel_uses"] / self.source_rgb_values,
                "frame_success_rate": sum(row["frame_success"] for row in trials) / len(trials),
                "compression_ratio_vs_raw": self.raw_bits / packet.bits.size,
                "bit_reduction_vs_raw_pct": 100 * (1 - packet.bits.size / self.raw_bits),
                "shared_codebook_storage_bits": codebook.shared_storage_bits(),
                "quantization_mse": packet.quantization_mse,
                "latency_s": time.perf_counter() - transport_started,
                **packet.diagnostics,
                **summarize_trials(successful_metrics),
                "warning": "Packet includes indices, shape metadata, NTSCC rate map where applicable, and CRC; failed frames retain N/A quality.",
                "trials": trials,
            }
            if first_reconstruction is not None:
                result["reconstruction_path"] = str(
                    self.save_reconstruction(name.lower().replace(" ", "_"), first_reconstruction)
                )
            self.codebook_results[name] = result
            self.write_json(name.lower().replace(" ", "_") + "_result.json", result)
            pd.DataFrame(trials).to_csv(
                self.output_dir / (name.lower().replace(" ", "_") + "_trials.csv"), index=False
            )
            digital_rows.append(result)
        return pd.DataFrame(digital_rows)

    def consolidated_table(self) -> pd.DataFrame:
        """Measured comparison with explicit units and N/A for undefined quantities."""
        if not self.codebook_results:
            raise RuntimeError("Run all native, digital, and codebook evaluations first")

        rows = []

        def add(method, algorithm, result, original_packets=1, latent_units=None, latent_bits=None,
                compression_bits=None, complexity="Measured CPU wall-clock", note=""):
            rows.append({
                "Model": method,
                "Algorithm": algorithm,
                "Original image packets": original_packets,
                "Latent representation units": latent_units,
                "CBR": result.get("cbr"),
                "PSNR (dB)": result.get("psnr_db_mean"),
                "SSIM": result.get("ssim_mean"),
                "LPIPS": result.get("lpips_mean"),
                "Reconstruction bits": self.raw_bits,
                "Latent bits": latent_bits,
                "Compression/payload bits": compression_bits,
                "Complex channel uses": result.get("complex_channel_uses"),
                "Computational complexity": complexity,
                "Measured latency (s)": result.get("latency_s"),
                "Notes": note,
            })

        raw = self.digital_results["RAW"]
        add("RAW", "8-bit RGB + CRC + rate-1/2 LDPC + QPSK + AWGN", raw,
            compression_bits=raw.get("payload_bits"),
            complexity="LDPC decoding is iterative in coded blocks and decoder iterations",
            note="No latent representation; one channel trial.")
        for quality in self.jpeg_qualities:
            name = f"JPEG Q{quality}"
            result = self.digital_results[name]
            add(name, f"JPEG quality {quality} + CRC + rate-1/2 LDPC + QPSK + AWGN", result,
                compression_bits=result.get("payload_bits"),
                complexity="Block-DCT JPEG plus iterative LDPC decoding",
                note="Compressed payload bits are not latent-model bits.")

        deep = self.native_results["DeepJSCC"]
        deep_latent_units = sum(
            int(latent.shape[-2] * latent.shape[-1]) for _, _, latent in self._deep_target[2]
        )
        add("DeepJSCC", "Tiled learned analog JSCC", deep,
            original_packets=deep.get("tile_count"), latent_units=deep_latent_units,
            complexity="Encoder/decoder neural inference per image tile",
            note="Continuous channel symbols have no mathematically defined payload-bit count.")
        ntscc = self.native_results["NTSCC"]
        add("NTSCC", "Rate-adaptive nonlinear transform source-channel coding", ntscc,
            latent_units=ntscc.get("rate_map_token_count"),
            latent_bits=ntscc.get("entropy_estimated_bits"),
            compression_bits=ntscc.get("entropy_estimated_bits"),
            complexity="Analysis/hyperprior/rate-adapter/decoder neural inference",
            note="Entropy-estimated bits are diagnostic, not transmitted radio bits.")

        deep_packet = self._codebook_packets["DeepJSCC codebook"]
        deep_cb = self.codebook_results["DeepJSCC codebook"]
        add("DeepJSCC codebook", "K=256 full-token vector quantization + LDPC/QPSK/AWGN", deep_cb,
            original_packets=deep.get("tile_count"),
            latent_units=int(deep_packet.header.index_count),
            latent_bits=int(deep_packet.header.index_count * deep_packet.header.index_bits),
            compression_bits=int(deep_packet.bits.size),
            complexity="K-means nearest-centroid search O(N*K*D) plus iterative LDPC",
            note="Payload also includes packet metadata and CRC.")
        ntscc_packet = self._codebook_packets["NTSCC codebook"]
        ntscc_cb = self.codebook_results["NTSCC codebook"]
        add("NTSCC codebook", "K=256 active-prefix PQ16 + rate map + LDPC/QPSK/AWGN", ntscc_cb,
            latent_units=int(ntscc_packet.index_count),
            latent_bits=int(ntscc_packet.index_count * ntscc_packet.header.index_bits),
            compression_bits=int(ntscc_packet.bits.size),
            complexity="Active-block K-means search O(N_active*K*16) plus iterative LDPC",
            note="Only active prefix blocks are clustered; payload also includes the rate map, metadata, and CRC.")

        table = pd.DataFrame(rows)
        table.to_csv(self.output_dir / "consolidated_measured_comparison.csv", index=False, na_rep="N/A")
        self.write_json("consolidated_measured_comparison.json", rows)
        return table

    def run_codebook_experiments(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Backward-compatible ordered entry point: ablation, then real PHY."""
        self.prepare_codebook_packets()
        noiseless = self.run_noiseless_codebook_ablation()
        digital = self.run_noisy_codebook_transport()
        return noiseless, digital

    def comparison_table(self) -> pd.DataFrame:
        rows = []

        def row(method, category, result, warning=""):
            payload = result.get("payload_bits")
            compression = result.get("compression_ratio_vs_raw")
            uses = result.get("complex_channel_uses")
            raw_digital_uses = self.digital_results.get("RAW", {}).get("complex_channel_uses")
            return {
                "Method": method,
                "Category": category,
                "Payload bits": payload,
                "Coded bits": result.get("coded_bits"),
                "Complex channel uses": uses,
                "CBR": result.get("cbr"),
                "Compression vs RAW": compression,
                "Bit reduction vs RAW (%)": result.get("bit_reduction_vs_raw_pct"),
                "Channel-use saving vs RAW (%)": None if uses is None or raw_digital_uses is None else 100 * (1 - uses / raw_digital_uses),
                "Frame success rate": result.get("frame_success_rate"),
                "PSNR mean (dB)": result.get("psnr_db_mean"),
                "PSNR std": result.get("psnr_db_std"),
                "SSIM mean": result.get("ssim_mean"),
                "LPIPS mean": result.get("lpips_mean"),
                "CLIP cosine mean": result.get("clip_cosine_mean"),
                "Warning": result.get("warning", warning),
            }

        rows.append(row("RAW image", "Source", {
            "payload_bits": self.raw_bits,
            "compression_ratio_vs_raw": 1.0,
            "bit_reduction_vs_raw_pct": 0.0,
            "psnr_db_mean": float("inf"), "psnr_db_std": 0.0, "ssim_mean": 1.0,
            "lpips_mean": 0.0, "clip_cosine_mean": 1.0,
        }, "Untransmitted source reference."))
        for name in (f"JPEG Q{q}" for q in self.jpeg_qualities):
            rows.append(row(name, "Codec only", self.codec_results[name], "Codec distortion only; no channel."))
        for name, result in self.digital_results.items():
            rows.append(row(result["method"], "Digital PHY", result))
        for name, result in self.native_results.items():
            rows.append(row(name, "Analog JSCC", result))
        for name, result in self.codebook_results.items():
            rows.append(row(name, "K=256 + Digital PHY", result))
        table = pd.DataFrame(rows)
        table.to_csv(self.output_dir / "final_comparison.csv", index=False, na_rep="N/A")
        self.write_json("final_comparison.json", rows)
        return table

    def plot_native_reconstructions(self) -> Path:
        return self._plot_reconstructions(
            [("DeepJSCC", self.reconstructions["deepjscc"]), ("NTSCC", self.reconstructions["ntscc"])],
            "native_reconstructions.png",
        )

    def plot_codebook_reconstructions(self) -> Path:
        return self._plot_reconstructions(
            [
                ("DeepJSCC codebook", self.reconstructions["deepjscc_codebook"]),
                ("NTSCC codebook", self.reconstructions["ntscc_codebook"]),
            ],
            "codebook_reconstructions.png",
        )

    def _plot_reconstructions(self, items, filename: str) -> Path:
        original = self.reference[0].cpu().permute(1, 2, 0).numpy()
        figure, axes = plt.subplots(len(items), 3, figsize=(14, 4.5 * len(items)))
        axes = np.asarray(axes).reshape(len(items), 3)
        for row_index, (name, reconstruction) in enumerate(items):
            reconstructed = reconstruction[0].permute(1, 2, 0).numpy()
            error = np.mean(np.abs(original - reconstructed), axis=-1)
            axes[row_index, 0].imshow(original)
            axes[row_index, 0].set_title("Original")
            axes[row_index, 1].imshow(reconstructed)
            axes[row_index, 1].set_title(name)
            image = axes[row_index, 2].imshow(error, cmap="magma", vmin=0, vmax=max(float(error.max()), 1e-6))
            axes[row_index, 2].set_title("Mean absolute error")
            figure.colorbar(image, ax=axes[row_index, 2], fraction=0.046)
            for axis in axes[row_index]:
                axis.axis("off")
        figure.tight_layout()
        path = self.output_dir / filename
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return path

    def plot_comparison(self, table: pd.DataFrame | None = None) -> Path:
        table = self.comparison_table() if table is None else table
        plotted = table[table["Complex channel uses"].notna()].copy()
        finite_quality = plotted[pd.to_numeric(plotted["PSNR mean (dB)"], errors="coerce").replace([np.inf], np.nan).notna()]
        figure, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes[0].barh(finite_quality["Method"], finite_quality["PSNR mean (dB)"].astype(float))
        axes[0].set_xlabel("PSNR mean (dB), successful frames only")
        axes[0].grid(axis="x", alpha=0.25)
        axes[1].barh(plotted["Method"], plotted["Complex channel uses"].astype(float))
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Complex channel uses (log scale)")
        axes[1].grid(axis="x", alpha=0.25)
        figure.tight_layout()
        path = self.output_dir / "quality_channel_use_comparison.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return path
