"""Repository-owned Stage 1A procedures called by the one Kaggle notebook."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any

from Baselines.DigitalPHY.channel_coding import LDPCQPSKAWGN
from Baselines.JPEG.jpeg_bitstream import bits_to_bytes
from Baselines.JPEG.jpeg_pipeline import encode_jpeg_for_transport
from Channels.awgn import AWGNChannel
from Channels.channel_utils import require_torch
from EncDecPipeline.Codebooks.codebook_codec import CodebookPacketHeader, decode_codebook_packet_bits
from EncDecPipeline.Codebooks.codebook_metrics import codebook_usage_metrics
from EncDecPipeline.Codebooks.minibatch_kmeans_vq import MiniBatchKMeansVQ
from EncDecPipeline.Models.SwinJSCC.adapter import SwinJSCCAdapter
from EncDecPipeline.Models.SwinJSCC.checkpoint_manager import export_stage1_artifacts
from EncDecPipeline.Models.SwinJSCC.swin_config import SwinJSCCConfig, expected_cbr
from EncDecPipeline.Models.SwinJSCC.trainer import (
    TrainingPhaseConfig,
    TrainingResult,
    build_base_then_sara,
    make_loader,
    train_phase,
    transfer_compatible_weights,
)
from EncDecPipeline.Models.SwinJSCC.training_utils import DataParallelPlan, verify_two_t4_data_parallel
from EncDecPipeline.Preprocessing.image_loader import dataset_provenance, discover_images, pil_to_tensor
from EncDecPipeline.Preprocessing.resize import center_crop_to_multiple
from EncDecPipeline.Postprocessing.tensor_to_image import tensor_to_pil
from Evaluation.Communication.cbr import native_cbr
from Evaluation.Image.psnr import psnr
from Evaluation.Image.ssim import ssim
from INFRA.Artifacts import ImageArtifact, TransmissionArtifact


def verify_kaggle_stage1a_runtime() -> dict[str, Any]:
    """Record—not replace—the CUDA-enabled Kaggle runtime selected by the user."""

    torch = require_torch()
    plan = verify_two_t4_data_parallel()
    return {
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "gpu_names": [torch.cuda.get_device_name(index) for index in plan.device_ids],
        "data_parallel": asdict(plan),
    }


def resolve_required_datasets(train_root: str | Path, validation_root: str | Path, kodak_root: str | Path) -> dict[str, Any]:
    """Validate real DIV2K/Kodak paths and record immutable file-list provenance."""

    return {
        "train": dataset_provenance(train_root),
        "validation": dataset_provenance(validation_root),
        "kodak": dataset_provenance(kodak_root),
    }


def run_base_then_sara_training(
    config: SwinJSCCConfig,
    upstream_root: str | Path,
    train_root: str | Path,
    validation_root: str | Path,
    output_dir: str | Path,
    base_epochs: int,
    sara_epochs: int,
    batch_size_per_gpu: int,
    early_stopping_patience: int,
) -> tuple[SwinJSCCAdapter, dict[str, Any]]:
    """Train Base at C=96/SNR=10, then transfer compatible state to SA+RA."""

    plan = verify_two_t4_data_parallel()
    base, sara = build_base_then_sara(config, upstream_root)
    base_result = train_phase(
        adapter=base,
        phase=TrainingPhaseConfig(
            name="base",
            epochs=base_epochs,
            batch_size_per_gpu=batch_size_per_gpu,
            early_stopping_patience=early_stopping_patience,
        ),
        train_root=train_root,
        validation_root=validation_root,
        snr_grid=config.snr_db_grid,
        rate_grid=config.rate_grid,
        data_parallel=plan,
        output_dir=Path(output_dir) / "checkpoints",
    )
    missing_keys, unexpected_keys = transfer_compatible_weights(base.build_training_module(), sara.build_training_module())
    sara_result = train_phase(
        adapter=sara,
        phase=TrainingPhaseConfig(
            name="sa_ra",
            epochs=sara_epochs,
            batch_size_per_gpu=batch_size_per_gpu,
            early_stopping_patience=early_stopping_patience,
        ),
        train_root=train_root,
        validation_root=validation_root,
        snr_grid=config.snr_db_grid,
        rate_grid=config.rate_grid,
        data_parallel=plan,
        output_dir=Path(output_dir) / "checkpoints",
    )
    summary = {
        "base": asdict(base_result),
        "sa_ra": asdict(sara_result),
        "transfer_missing_keys": missing_keys,
        "transfer_unexpected_keys": unexpected_keys,
        "two_t4_data_parallel": asdict(plan),
    }
    (Path(output_dir) / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return sara, summary


def _image_artifact(path: Path, device: str = "cuda:0") -> tuple[ImageArtifact, Any]:
    from PIL import Image

    image = center_crop_to_multiple(Image.open(path).convert("RGB"), divisor=16)
    tensor = pil_to_tensor(image).unsqueeze(0).to(device)
    return ImageArtifact(tensor=tensor, sample_ids=[path.name], source=str(path)), image


def _quality_row(
    method: str,
    image: ImageArtifact,
    reconstruction: ImageArtifact,
    snr_db: float,
    channel_uses: int,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "method": method,
        "snr_definition": "Es/N0_dB",
        "snr_db": float(snr_db),
        "channel_uses": int(channel_uses),
        "cbr": channel_uses / (3.0 * image.spatial_shape()[0] * image.spatial_shape()[1]),
        "psnr": psnr(image.tensor, reconstruction.tensor),
        "ssim": ssim(image.tensor, reconstruction.tensor),
        "frame_success": True,
        **extra,
    }


def evaluate_native_swinjscc(
    adapter: SwinJSCCAdapter,
    kodak_root: str | Path,
    output_dir: str | Path,
    snr_grid: tuple[int, ...],
    rate_grid: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Evaluate native JSCC under AWGN with actual image dimensions and channel uses."""

    torch = require_torch()
    output = Path(output_dir)
    reconstructions = output / "reconstructions" / "native_swinjscc"
    reconstructions.mkdir(parents=True, exist_ok=True)
    channel = AWGNChannel()
    rows: list[dict[str, Any]] = []
    adapter.encoder.eval()
    adapter.decoder.eval()
    with torch.no_grad():
        for path in discover_images(kodak_root):
            image, _ = _image_artifact(path)
            for rate in rate_grid:
                for snr_db in snr_grid:
                    latent = adapter.encode(image, snr_db=snr_db, rate=rate)
                    transmission = channel.transmit(latent, snr_db)
                    reconstruction = adapter.reconstruct_received(transmission, latent)
                    recorded_cbr = native_cbr(image, transmission)
                    expected = expected_cbr(rate, adapter.config.downsample_stages)
                    if abs(recorded_cbr - expected) > 1e-12:
                        raise AssertionError(f"Native CBR mismatch: {recorded_cbr} != {expected}")
                    rows.append(
                        _quality_row(
                            "native_swinjscc",
                            image,
                            reconstruction,
                            snr_db,
                            transmission.channel_uses_per_image[0],
                            rate=rate,
                            native_cbr=recorded_cbr,
                        )
                    )
                    tensor_to_pil(reconstruction.tensor[0]).save(reconstructions / f"{path.stem}_r{rate}_snr{snr_db}.png")
    return rows


def fit_k256_codebook(adapter: SwinJSCCAdapter, train_root: str | Path, rate: int = 96, max_images: int = 512) -> MiniBatchKMeansVQ:
    """Fit only from real SA+RA latent tokens. A collapsed codebook is surfaced later."""

    torch = require_torch()
    vq = MiniBatchKMeansVQ(codebook_size=256)
    paths = discover_images(train_root)[:max_images]
    if not paths:
        raise FileNotFoundError("No real images are available for codebook fitting.")
    adapter.encoder.eval()
    def real_latent_batches() -> Any:
        with torch.no_grad():
            for path in paths:
                image, _ = _image_artifact(path)
                yield adapter.encode(image, snr_db=10, rate=rate)

    vq.fit(real_latent_batches())
    return vq


def evaluate_vq_noiseless_ablation(
    adapter: SwinJSCCAdapter,
    codebook: MiniBatchKMeansVQ,
    kodak_root: str | Path,
    output_dir: str | Path,
    rate: int = 96,
) -> list[dict[str, Any]]:
    """Separate no-channel ablation: encoder -> codebook -> decoder, not a PHY result."""

    torch = require_torch()
    output = Path(output_dir)
    reconstructions = output / "reconstructions" / "vq_noiseless"
    reconstructions.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    usage: list[Any] = []
    with torch.no_grad():
        for path in discover_images(kodak_root):
            image, _ = _image_artifact(path)
            latent = adapter.encode(image, snr_db=10, rate=rate)
            packet = codebook.encode(latent)
            quantized = codebook.decode(packet, latent)
            reconstruction = adapter.decode(quantized, snr_db=10)
            usage.append(packet.transmitted)
            row = _quality_row(
                "vq_noiseless_ablation",
                image,
                reconstruction,
                10,
                0,
                rate=rate,
                vq_packet_bits=packet.payload_bits_per_image[0],
                shared_codebook_hash=codebook.content_hash(),
                experiment="no_channel_ablation_not_a_fair_air_interface_curve",
            )
            rows.append(row)
            tensor_to_pil(reconstruction.tensor[0]).save(reconstructions / f"{path.stem}_r{rate}.png")
    metrics = codebook_usage_metrics(__import__("numpy").concatenate(usage), codebook.codebook_size)
    (output / "vq_noiseless_usage.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return rows


def evaluate_vq_over_digital_awgn(
    adapter: SwinJSCCAdapter,
    codebook: MiniBatchKMeansVQ,
    kodak_root: str | Path,
    output_dir: str | Path,
    snr_grid: tuple[int, ...],
    rate: int = 96,
) -> list[dict[str, Any]]:
    """Send complete VQ packets through 5G-LDPC/QPSK/AWGN and retain frame failures."""

    torch = require_torch()
    digital = LDPCQPSKAWGN()
    rows: list[dict[str, Any]] = []
    output = Path(output_dir)
    reconstructions = output / "reconstructions" / "vq_digital_awgn"
    reconstructions.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for path in discover_images(kodak_root):
            image, _ = _image_artifact(path)
            latent = adapter.encode(image, snr_db=10, rate=rate)
            packet = codebook.encode(latent)
            header_values = dict(packet.metadata["header"])
            header_values.pop("payload_bits")
            header_values["latent_shape"] = tuple(header_values["latent_shape"])
            header = CodebookPacketHeader(**header_values)
            for snr_db in snr_grid:
                physical = digital.transmit(packet.metadata["packet_bits_per_image"][0], snr_db)
                common = {
                    "method": "vq_k256_ldpc_qpsk",
                    "snr_definition": "Es/N0_dB",
                    "snr_db": float(snr_db),
                    "channel_uses": physical.channel_uses,
                    "cbr": physical.channel_uses / (3.0 * image.spatial_shape()[0] * image.spatial_shape()[1]),
                    "rate": rate,
                    "vq_packet_bits": packet.payload_bits_per_image[0],
                    "ldpc_coded_bits": physical.coded_bits,
                    "frame_success": physical.frame_success,
                    "frame_failure_reason": physical.failure_reason,
                }
                if not physical.frame_success or physical.recovered_bits is None:
                    rows.append({**common, "psnr": None, "ssim": None})
                    continue
                try:
                    indices, _ = decode_codebook_packet_bits(physical.recovered_bits, header)
                except ValueError as error:
                    rows.append({**common, "frame_success": False, "frame_failure_reason": str(error), "psnr": None, "ssim": None})
                    continue
                received_packet = TransmissionArtifact(
                    transmitted=packet.transmitted,
                    received=indices.reshape(packet.received.shape),
                    channel_name="ldpc_qpsk_awgn",
                    snr_db=float(snr_db),
                    channel_uses_per_image=[physical.channel_uses],
                    payload_bits_per_image=packet.payload_bits_per_image,
                    metadata=packet.metadata,
                )
                quantized = codebook.decode(received_packet, latent)
                reconstruction = adapter.decode(quantized, snr_db=10)
                rows.append(
                    _quality_row(
                        "vq_k256_ldpc_qpsk",
                        image,
                        reconstruction,
                        snr_db,
                        physical.channel_uses,
                        rate=rate,
                        vq_packet_bits=packet.payload_bits_per_image[0],
                        ldpc_coded_bits=physical.coded_bits,
                        frame_failure_reason=physical.failure_reason,
                    )
                )
                tensor_to_pil(reconstruction.tensor[0]).save(reconstructions / f"{path.stem}_r{rate}_snr{snr_db}.png")
    return rows


def evaluate_jpeg_over_digital_awgn(
    kodak_root: str | Path,
    output_dir: str | Path,
    snr_grid: tuple[int, ...],
    quality: int,
) -> list[dict[str, Any]]:
    """JPEG quality is global per sweep; failed frames are never converted to quality scores."""

    torch = require_torch()
    from Baselines.JPEG.jpeg_codec import JPEGCodec

    rows: list[dict[str, Any]] = []
    digital = LDPCQPSKAWGN()
    reconstructions = Path(output_dir) / "reconstructions" / "jpeg_ldpc_qpsk"
    reconstructions.mkdir(parents=True, exist_ok=True)
    for path in discover_images(kodak_root):
        image, pil_image = _image_artifact(path)
        stream = encode_jpeg_for_transport(pil_image, quality)
        for snr_db in snr_grid:
            physical = digital.transmit(stream.bits, snr_db)
            common = {
                "method": "jpeg_ldpc_qpsk",
                "snr_definition": "Es/N0_dB",
                "snr_db": float(snr_db),
                "channel_uses": physical.channel_uses,
                "cbr": physical.channel_uses / (3.0 * image.spatial_shape()[0] * image.spatial_shape()[1]),
                "jpeg_quality": quality,
                "jpeg_source_bits": int(stream.bits.size),
                "ldpc_coded_bits": physical.coded_bits,
                "frame_success": physical.frame_success,
                "frame_failure_reason": physical.failure_reason,
            }
            if not physical.frame_success or physical.recovered_bits is None:
                rows.append({**common, "psnr": None, "ssim": None})
                continue
            decoded_pil = JPEGCodec.decode(bits_to_bytes(physical.recovered_bits))
            reconstruction = ImageArtifact(
                tensor=pil_to_tensor(decoded_pil).unsqueeze(0).to(image.tensor.device),
                sample_ids=[path.name],
                source="jpeg_ldpc_qpsk",
            )
            rows.append(
                _quality_row(
                    "jpeg_ldpc_qpsk",
                    image,
                    reconstruction,
                    snr_db,
                    physical.channel_uses,
                    jpeg_quality=quality,
                    jpeg_source_bits=int(stream.bits.size),
                    ldpc_coded_bits=physical.coded_bits,
                    frame_failure_reason=physical.failure_reason,
                )
            )
            decoded_pil.save(reconstructions / f"{path.stem}_q{quality}_snr{snr_db}.png")
    return rows


def export_final_stage1_artifacts(
    adapter: SwinJSCCAdapter,
    model_dir: str | Path,
    data_provenance: dict[str, Any],
    training_summary: dict[str, Any],
) -> None:
    export_stage1_artifacts(adapter, model_dir, data_provenance, training_summary)
