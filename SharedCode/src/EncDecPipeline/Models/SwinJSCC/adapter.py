"""Adapter that exposes upstream SwinJSCC through the project EncDec contract."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

from Channels.awgn import AWGNChannel
from Channels.channel_utils import require_torch
from EncDecPipeline.Models.SwinJSCC.swin_config import BASE_MODEL_NAME, SwinJSCCConfig
from INFRA.Artifacts import ImageArtifact, LatentArtifact, TransmissionArtifact
from INFRA.Interfaces import EncDecInterface
from INFRA.Registries import ENCDEC_REGISTRY


class SwinJSCCAdapter(EncDecInterface):
    """Owns only upstream encoder/decoder modules; the project owns channel selection."""

    def __init__(self, config: SwinJSCCConfig, upstream_root: str | Path = "external/SwinJSCC") -> None:
        self.config = config
        self.upstream_root = Path(upstream_root)
        self.encoder: Any | None = None
        self.decoder: Any | None = None
        self._resolution: tuple[int, int] | None = None

    def _validate_upstream_patch(self) -> None:
        patch_file = self.upstream_root / ".stage1a_patch.json"
        if not patch_file.exists():
            raise RuntimeError("SwinJSCC source is not bootstrapped. Run: python scripts/bootstrap_swinjscc.py")
        patch = json.loads(patch_file.read_text(encoding="utf-8"))
        if patch.get("upstream_commit") != self.config.upstream_commit:
            raise RuntimeError("Upstream SwinJSCC commit differs from the model configuration.")
        if patch.get("patch_version") != self.config.patch_version:
            raise RuntimeError("Upstream SwinJSCC patch version differs from the model configuration.")

    def build(self, device: str | None = None) -> "SwinJSCCAdapter":
        torch = require_torch()
        self._validate_upstream_patch()
        source_parent = str(self.upstream_root.resolve())
        if source_parent not in sys.path:
            sys.path.insert(0, source_parent)
        encoder_module = importlib.import_module("net.encoder")
        decoder_module = importlib.import_module("net.decoder")
        norm_layer = torch.nn.LayerNorm
        encoder_kwargs = self.config.encoder_kwargs()
        decoder_kwargs = self.config.decoder_kwargs()
        encoder_kwargs["norm_layer"] = norm_layer
        decoder_kwargs["norm_layer"] = norm_layer
        self.encoder = encoder_module.create_encoder(**encoder_kwargs)
        self.decoder = decoder_module.create_decoder(**decoder_kwargs)
        if device is not None:
            self.encoder.to(device)
            self.decoder.to(device)
        return self

    def _require_built(self) -> tuple[Any, Any]:
        if self.encoder is None or self.decoder is None:
            raise RuntimeError("Build or load SwinJSCCAdapter before encoding or decoding.")
        return self.encoder, self.decoder

    def _update_resolution(self, image_tensor: Any) -> None:
        encoder, decoder = self._require_built()
        height, width = int(image_tensor.shape[-2]), int(image_tensor.shape[-1])
        required_divisor = 2 ** self.config.downsample_stages
        if height % required_divisor or width % required_divisor:
            raise ValueError(f"Image dimensions {(height, width)} must be divisible by {required_divisor}.")
        if self._resolution != (height, width):
            encoder.update_resolution(height, width)
            decoder.update_resolution(height // required_divisor, width // required_divisor)
            self._resolution = (height, width)

    def encode(self, image: ImageArtifact, **options: object) -> LatentArtifact:
        encoder, _ = self._require_built()
        snr_db = float(options.get("snr_db", 10.0))
        rate = int(options.get("rate", self.config.base_fixed_c))
        self._update_resolution(image.tensor)
        encoded = encoder(image.tensor, snr_db, rate, self.config.variant)
        if self.config.variant == BASE_MODEL_NAME or self.config.variant == "SwinJSCC_w/_SA":
            tensor, rate_mask = encoded, None
        else:
            tensor, rate_mask = encoded
        return LatentArtifact(
            tensor=tensor,
            rate_mask=rate_mask,
            source_model="SwinJSCC",
            rate_tokens=rate,
            metadata={"snr_db": snr_db, "variant": self.config.variant},
        )

    def decode(self, latent: LatentArtifact, **options: object) -> ImageArtifact:
        _, decoder = self._require_built()
        snr_db = float(options.get("snr_db", latent.metadata.get("snr_db", 10.0)))
        reconstruction = decoder(latent.tensor, snr_db, self.config.variant)
        return ImageArtifact(
            tensor=reconstruction,
            sample_ids=[],
            source="SwinJSCCDecoder",
            metadata={"snr_db": snr_db, "variant": self.config.variant},
        )

    def reconstruct_received(self, transmission: TransmissionArtifact, latent: LatentArtifact) -> ImageArtifact:
        received_latent = LatentArtifact(
            tensor=transmission.received,
            rate_mask=latent.rate_mask,
            source_model=latent.source_model,
            rate_tokens=latent.rate_tokens,
            metadata={**latent.metadata, "snr_db": transmission.snr_db},
        )
        if received_latent.rate_mask is not None:
            received_latent.tensor = received_latent.tensor * received_latent.rate_mask
        output = self.decode(received_latent, snr_db=transmission.snr_db)
        output.sample_ids = list(latent.metadata.get("sample_ids", []))
        return output

    def build_training_module(self) -> Any:
        """Return a tensor-only module that can be wrapped by two-GPU DataParallel."""

        torch = require_torch()
        encoder, decoder = self._require_built()
        variant = self.config.variant

        config = self.config

        class TensorOnlySwinJSCC(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
                self.channel = AWGNChannel()
                self.config = config

            def forward(self, images: Any, snr_db: float, rate: int) -> tuple[Any, Any | None]:
                artifact = ImageArtifact(tensor=images, sample_ids=[], source="training_batch")
                latent = SwinJSCCAdapter.encode(
                    _TensorAdapterView(self.encoder, self.decoder, variant, self.config),
                    artifact,
                    snr_db=snr_db,
                    rate=rate,
                )
                transmission = self.channel.transmit(latent, snr_db)
                received = transmission.received
                if latent.rate_mask is not None:
                    received = received * latent.rate_mask
                reconstruction = self.decoder(received, snr_db, variant)
                return reconstruction, latent.rate_mask

        return TensorOnlySwinJSCC()


class _TensorAdapterView(SwinJSCCAdapter):
    """Internal adapter view used only inside a tensor-only DataParallel forward."""

    def __init__(self, encoder: Any, decoder: Any, variant: str, config: SwinJSCCConfig) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.upstream_root = Path(".")
        self._resolution: tuple[int, int] | None = None

    def _require_built(self) -> tuple[Any, Any]:
        return self.encoder, self.decoder


def register_swinjscc() -> None:
    if not ENCDEC_REGISTRY.contains("swinjscc"):
        ENCDEC_REGISTRY.register("swinjscc", SwinJSCCAdapter)
