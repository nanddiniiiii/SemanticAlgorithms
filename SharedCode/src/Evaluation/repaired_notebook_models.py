"""Adapters for the repaired DeepJSCC/NTSCC Windows benchmark assets.

The external research repositories and checkpoints remain outside this Git
repository.  Set ``SEMCOM_EXPERIMENT_ROOT`` to the directory containing
``semcom_benchmark/``, checkpoint-compatible sources, ``models/`` and ``data/``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from Evaluation.kodak_benchmark import EncodedSample, module_complexity


def _seed_from_image(image_id: str, base_seed: int) -> int:
    try:
        return int(base_seed + int(image_id[-2:]) - 1)
    except ValueError:
        return int(base_seed)


def _digital_channel_uses(payload_bits: int) -> int:
    """Team PHY: CRC32, 1024/2048 LDPC blocks, then QPSK."""
    framed_bits = int(payload_bits) + 32
    code_blocks = math.ceil(framed_bits / 1024)
    coded_bits = code_blocks * 2048
    return coded_bits // 2


def _module_with_storage(module: Any, extra_bytes: int = 0) -> dict[str, int | float | None]:
    result = module_complexity([module])
    result["model_size_mb"] = float(result["model_size_mb"] or 0.0) + extra_bytes / (1024**2)
    return result


def _add_cifar_partial_block_fallbacks(quantizer: Any, calibration: list[Any]) -> list[int]:
    """Fit missing exact-width global fallbacks from CIFAR active values only.

    The cached CIFAR records contain rates 0--5, whose active channel counts
    are divisible by four.  Other valid NTSCC rates can end in a two-value
    block.  Pooling adjacent pairs from the already-active CIFAR values gives
    that exact-dimensional fallback without fitting on any Kodak image.
    """
    from sklearn.cluster import MiniBatchKMeans

    required_lengths = {
        min(quantizer.config.block_size, int(active) - start)
        for active in quantizer.rate_choices
        for start in range(0, int(active), quantizer.config.block_size)
    }
    missing_lengths = sorted(length for length in required_lengths if (-1, length) not in quantizer.models)
    fitted: list[int] = []
    for length in missing_lengths:
        blocks = []
        for latent, rate_indexes in calibration:
            values = latent.detach().float().cpu().permute(0, 2, 3, 1).reshape(-1, latent.shape[1]).numpy()
            rates = rate_indexes.detach().cpu().numpy().reshape(-1).astype(np.int64)
            for rate in np.unique(rates):
                active = int(quantizer.rate_choices[rate])
                selected = values[rates == rate, :active]
                usable = (active // length) * length
                if usable:
                    blocks.append(selected[:, :usable].reshape(-1, length))
        if not blocks:
            raise RuntimeError(f"Cannot build CIFAR fallback codebook for block width {length}")
        population = np.concatenate(blocks, axis=0).astype(np.float32)
        cluster_count = min(quantizer.config.codebook_size, int(population.shape[0]))
        model = MiniBatchKMeans(
            n_clusters=cluster_count,
            batch_size=min(16384, max(cluster_count * 4, 1024)),
            n_init=1,
            max_iter=100,
            reassignment_ratio=0.01,
            random_state=quantizer.config.random_state,
        ).fit(population)
        quantizer.models[(-1, length)] = SimpleNamespace(
            mean=np.zeros(length, np.float32),
            scale=np.ones(length, np.float32),
            stages=[model],
            sample_count=int(population.shape[0]),
        )
        fitted.append(length)
    return fitted


class _BaseCodec:
    device = "cpu"

    def __init__(self, model: Any, *, name: str, version: str, operating_point: str, notes: str) -> None:
        self.model = model
        self.name = name
        self.version = version
        self.operating_point = operating_point
        self.notes = notes

    def complexity(self) -> Mapping[str, int | float | None]:
        return _module_with_storage(self.model)


class RepairedDeepJSCCCodec(_BaseCodec):
    def __init__(self, adapter: Any, snr_db: float, base_seed: int) -> None:
        super().__init__(
            adapter.model,
            name="DeepJSCC",
            version=adapter.validation["checkpoint_sha256"],
            operating_point=f"tile=128;snr_db={snr_db:g};snr=Es/N0;seed_per_image={base_seed}+index",
            notes=(
                "Official repaired-notebook checkpoint; tiled 128x128 analog JSCC. "
                "One deterministic AWGN seed per Kodak image. Continuous symbols have no payload-bit count."
            ),
        )
        self.adapter = adapter
        self.base_seed = int(base_seed)

    def encode(self, tensor: Any, image: Any, image_id: str) -> EncodedSample:
        padded, original_hw, records = self.adapter.encode_tiles(tensor, 128)
        uses = math.ceil(sum(int(latent.numel()) for _, _, latent in records) / 2)
        return EncodedSample(
            state=(padded, original_hw, records, _seed_from_image(image_id, self.base_seed)),
            channel_uses=uses,
        )

    def decode(self, encoded: EncodedSample) -> Any:
        padded, original_hw, records, seed = encoded.state
        return self.adapter.run_from_records(padded, original_hw, records, seed, 128).reconstruction


class RepairedNTSCCCodec(_BaseCodec):
    def __init__(self, adapter: Any, snr_db: float, base_seed: int) -> None:
        super().__init__(
            adapter.model,
            name="NTSCC",
            version=adapter.validation["checkpoint_sha256"],
            operating_point=f"quality=1;snr_db={snr_db:g};snr=Es/N0;seed_per_image={base_seed}+index",
            notes=(
                "Official repaired-notebook checkpoint. Native CBR counts complex channel uses. "
                "Entropy BPP is diagnostic and is not written as transmitted payload bits."
            ),
        )
        self.adapter = adapter
        self.base_seed = int(base_seed)

    def encode(self, tensor: Any, image: Any, image_id: str) -> EncodedSample:
        analysis = self.adapter.encode(tensor)
        active_real_values = int(analysis.mask.sum().item())
        if active_real_values % 2:
            raise RuntimeError("NTSCC produced an odd number of active real symbols")
        return EncodedSample(
            state=(tensor, analysis, _seed_from_image(image_id, self.base_seed)),
            channel_uses=active_real_values // 2,
            metadata={"entropy_bpp": analysis.bpp_y + analysis.bpp_z},
        )

    def decode(self, encoded: EncodedSample) -> Any:
        reference, analysis, seed = encoded.state
        return self.adapter.run_from_encoding(reference, analysis, seed).reconstruction


class RepairedDeepJSCCCodebookCodec(_BaseCodec):
    def __init__(self, adapter: Any, codebook: Any, phy: Any, snr_db: float, base_seed: int) -> None:
        super().__init__(
            adapter.model,
            name="DeepJSCC codebook",
            version=f"{adapter.validation['checkpoint_sha256']}+{codebook.content_hash()}",
            operating_point=f"K=256;tile=128;ldpc=1/2;qpsk;snr_db={snr_db:g};seed_per_image={base_seed}+index",
            notes=(
                "K=256 full-token codebook fitted on CIFAR-10 training images 0-7, matching the repaired notebook. "
                "Packet includes indices, metadata and CRC and is actually transmitted through team LDPC/QPSK/AWGN."
            ),
        )
        self.adapter = adapter
        self.codebook = codebook
        self.phy = phy
        self.snr_db = float(snr_db)
        self.base_seed = int(base_seed)

    def complexity(self) -> Mapping[str, int | float | None]:
        return _module_with_storage(self.model, int(self.codebook.model.cluster_centers_.nbytes))

    def encode(self, tensor: Any, image: Any, image_id: str) -> EncodedSample:
        padded, original_hw, records = self.adapter.encode_tiles(tensor, 128)
        packet = self.codebook.encode([record[2] for record in records])
        return EncodedSample(
            state=(padded, original_hw, records, packet, _seed_from_image(image_id, self.base_seed)),
            payload_bits=int(packet.bits.size),
            channel_uses=_digital_channel_uses(int(packet.bits.size)),
        )

    def decode(self, encoded: EncodedSample) -> Any:
        import torch

        padded, (height, width), records, packet, seed = encoded.state
        physical = self.phy.transmit(packet.bits, self.snr_db, seed, self.device)
        encoded.channel_uses = int(physical.channel_uses)
        if not physical.frame_success or physical.recovered_bits is None:
            raise RuntimeError(f"DeepJSCC codebook frame failed for seed {seed}: {physical.failure_reason}")
        decoded, _ = self.codebook.decode(physical.recovered_bits, packet, self.adapter.device)
        reconstruction = torch.empty_like(padded)
        with torch.inference_mode():
            for (top, left, _), latent in zip(records, decoded, strict=True):
                reconstruction[:, :, top:top + 128, left:left + 128] = self.adapter.model.decoder(latent).clamp(0, 1)
        return reconstruction[:, :, :height, :width]


class RepairedNTSCCCodebookCodec(_BaseCodec):
    def __init__(self, adapter: Any, quantizer: Any, phy: Any, snr_db: float, base_seed: int) -> None:
        super().__init__(
            adapter.model,
            name="NTSCC codebook",
            version=f"{adapter.validation['checkpoint_sha256']}+{quantizer.model_hash()}",
            operating_point=f"rc_pq4_k256_raw;cifar500_only;ldpc=1/2;qpsk;snr_db={snr_db:g};seed_per_image={base_seed}+index",
            notes=(
                "Rate-conditioned PQ4 K=256 design from the repaired notebook. For unbiased Kodak-24 evaluation, "
                "the quantizer is fitted only on cached CIFAR-10 training latents; all Kodak calibration entries are excluded. "
                "Its exact-width-2 global fallback is fitted from adjacent active CIFAR latent values. "
                "The packet is actually transmitted through team LDPC/QPSK/AWGN."
            ),
        )
        self.adapter = adapter
        self.quantizer = quantizer
        self.phy = phy
        self.snr_db = float(snr_db)
        self.base_seed = int(base_seed)

    def complexity(self) -> Mapping[str, int | float | None]:
        extra_bytes = 0
        for population in self.quantizer.models.values():
            extra_bytes += int(population.mean.nbytes + population.scale.nbytes)
            extra_bytes += sum(int(stage.cluster_centers_.nbytes) for stage in population.stages)
        return _module_with_storage(self.model, extra_bytes)

    def encode(self, tensor: Any, image: Any, image_id: str) -> EncodedSample:
        analysis = self.adapter.encode(tensor)
        packet = self.quantizer.encode(analysis.s_masked, analysis.rate_indexes)
        return EncodedSample(
            state=(packet, _seed_from_image(image_id, self.base_seed)),
            payload_bits=int(packet.total_bits),
            channel_uses=_digital_channel_uses(int(packet.total_bits)),
            metadata={"latent_mse": packet.latent_mse},
        )

    def decode(self, encoded: EncodedSample) -> Any:
        from dataclasses import replace

        packet, seed = encoded.state
        physical = self.phy.transmit(packet.bits, self.snr_db, seed, self.device)
        encoded.channel_uses = int(physical.channel_uses)
        if not physical.frame_success or physical.recovered_bits is None:
            raise RuntimeError(f"NTSCC codebook frame failed for seed {seed}: {physical.failure_reason}")
        latent, rates = self.quantizer.decode(replace(packet, bits=physical.recovered_bits), self.adapter.device)
        return self.adapter.decode_channel_latent(latent, rates)


def build_repaired_notebook_codecs(
    experiment_root: str | Path,
    team_root: str | Path,
    *,
    snr_db: float = 10.0,
    base_seed: int = 42,
) -> tuple[list[Any], dict[str, Any]]:
    """Load the repaired notebook assets and build its four benchmark methods."""
    import torch
    from PIL import Image
    from torchvision.datasets import CIFAR10

    experiment_root = Path(experiment_root).resolve()
    team_root = Path(team_root).resolve()
    required = (
        experiment_root / "semcom_benchmark",
        experiment_root / "references" / "DeepJSCC_checkpoint_source",
        experiment_root / "references" / "NTSCC_JSAC22_official",
        experiment_root / "models" / "DeepJSCC" / "production",
        experiment_root / "models" / "NTSCC" / "production",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Repaired-notebook experiment root is incomplete: {missing}")
    if str(experiment_root) not in sys.path:
        sys.path.insert(0, str(experiment_root))

    from semcom_benchmark.codebooks import K256LatentCodebook
    from semcom_benchmark.digital_phy import TeamDigitalPHY
    from semcom_benchmark.metrics import as_bchw01
    from semcom_benchmark.model_wrappers import DeepJSCCAdapter, NTSCCWindowsAdapter
    from semcom_benchmark.ntscc_rate_conditioned_vq import RateConditionedConfig, RateConditionedPQ

    device = torch.device("cpu")
    deep = DeepJSCCAdapter(
        experiment_root / "references" / "DeepJSCC_checkpoint_source",
        experiment_root / "models" / "DeepJSCC" / "production" / "imagenet_10_0.33_200.00_32_19.pth",
        device,
        snr_db,
    )
    ntscc = NTSCCWindowsAdapter(
        experiment_root / "references" / "NTSCC_JSAC22_official",
        experiment_root / "models" / "NTSCC" / "production" / "ntscc_hyperprior_quality_1_psnr.pth",
        device,
        snr_db,
    )

    cifar = CIFAR10(root=str(experiment_root / "data"), train=True, download=False)
    calibration_latents = []
    for index in range(8):
        image = cifar[index][0].convert("RGB").resize((128, 128), Image.Resampling.BICUBIC)
        _, _, records = deep.encode_tiles(as_bchw01(image), 128)
        calibration_latents.append(records[0][2])
    deep_codebook = K256LatentCodebook("deepjscc_k256", team_root).fit(calibration_latents)

    cache = experiment_root / "models" / "NTSCC" / "calibration" / "calibration_500_cifar_plus_23_kodak.pt"
    cached = torch.load(cache, map_location="cpu", weights_only=False)
    # The cache contains 16 CIFAR batches followed by 23 Kodak records. Only
    # CIFAR is retained so no Kodak test image contributes to codebook fitting.
    cifar_calibration = list(cached["calibration"][:16])
    rate_groups = [[0], [1], [2], [3, 4, 5, 6, 7], list(range(8, 16))]
    config = RateConditionedConfig(block_size=4, codebook_size=256, normalize=False, rvq_stages=1)
    ntscc_quantizer = RateConditionedPQ(ntscc.config.multiple_rate, config, rate_groups).fit(cifar_calibration)
    fallback_widths = _add_cifar_partial_block_fallbacks(ntscc_quantizer, cifar_calibration)

    phy = TeamDigitalPHY(team_root)
    methods = [
        RepairedDeepJSCCCodec(deep, snr_db, base_seed),
        RepairedNTSCCCodec(ntscc, snr_db, base_seed),
        RepairedDeepJSCCCodebookCodec(deep, deep_codebook, phy, snr_db, base_seed),
        RepairedNTSCCCodebookCodec(ntscc, ntscc_quantizer, phy, snr_db, base_seed),
    ]
    provenance = {
        "experiment_root": str(experiment_root),
        "deepjscc_checkpoint_sha256": deep.validation["checkpoint_sha256"],
        "ntscc_checkpoint_sha256": ntscc.validation["checkpoint_sha256"],
        "deepjscc_codebook_sha256": deep_codebook.content_hash(),
        "ntscc_codebook_sha256": ntscc_quantizer.model_hash(),
        "deepjscc_calibration": "CIFAR-10 training indices 0-7, resized to 128x128",
        "ntscc_calibration": (
            "CIFAR-10 training indices 0-499 only; cached Kodak entries excluded; "
            f"global exact-width fallbacks added for widths {fallback_widths} from CIFAR active-value blocks"
        ),
        "snr_db": float(snr_db),
        "snr_definition": "Es/N0_dB",
        "base_seed": int(base_seed),
    }
    return methods, provenance
