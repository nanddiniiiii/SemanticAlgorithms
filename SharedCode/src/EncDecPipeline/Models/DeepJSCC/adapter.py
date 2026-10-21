"""Adapter for Nandini's historical DeepJSCC checkpoint.

The external implementation stays outside this repository.  This module only
loads its encoder/decoder and exposes the repository EncDecInterface contract.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

from Channels.channel_utils import require_torch
from INFRA.Artifacts import ImageArtifact, LatentArtifact, TransmissionArtifact
from INFRA.Interfaces import EncDecInterface
from INFRA.Registries import ENCDEC_REGISTRY


def _unwrap_state_dict(checkpoint: Any) -> dict[str, Any]:
    """Return a plain state dict and remove an optional DataParallel prefix."""
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("DeepJSCC checkpoint must contain a PyTorch state_dict.")
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
    }


class DeepJSCCAdapter(EncDecInterface):
    """Load ``old_model_6.py`` without copying the external model into the repo."""

    def __init__(self, checkpoint_path, source_root, c: int = 19, device=None):
        self.checkpoint_path = Path(checkpoint_path)
        self.source_root = Path(source_root)
        self.c = int(c)
        self.device = device
        self.encoder = None
        self.decoder = None

    def build(self, device=None):
        torch = require_torch()
        target_device = device or self.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model_path = self.source_root / "old_model_6.py"
        if not model_path.is_file():
            raise FileNotFoundError(f"DeepJSCC model file not found: {model_path}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"DeepJSCC checkpoint not found: {self.checkpoint_path}"
            )

        source_root = str(self.source_root.resolve())
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        spec = importlib.util.spec_from_file_location(
            "external_deepjscc_old_model_6", model_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load DeepJSCC module from {model_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.encoder = module._Encoder(c=self.c)
        self.decoder = module._Decoder(c=self.c)
        raw_checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        state_dict = _unwrap_state_dict(raw_checkpoint)
        encoder_state = {
            key.removeprefix("encoder."): value
            for key, value in state_dict.items()
            if key.startswith("encoder.")
        }
        decoder_state = {
            key.removeprefix("decoder."): value
            for key, value in state_dict.items()
            if key.startswith("decoder.")
        }
        if not encoder_state or not decoder_state:
            raise RuntimeError(
                "Checkpoint must contain both 'encoder.*' and 'decoder.*' weights."
            )
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.decoder.load_state_dict(decoder_state, strict=True)
        self.encoder.to(target_device).eval()
        self.decoder.to(target_device).eval()
        self.device = target_device
        return self

    def _require_built(self):
        if self.encoder is None or self.decoder is None:
            raise RuntimeError("Call DeepJSCCAdapter.build() before inference.")
        return self.encoder, self.decoder

    def encode(self, image: ImageArtifact, **options: object) -> LatentArtifact:
        torch = require_torch()
        encoder, _ = self._require_built()
        image_tensor = image.tensor.to(self.device).float()
        if image_tensor.ndim != 4 or image_tensor.shape[1] != 3:
            raise ValueError(f"Expected BCHW RGB input, got {tuple(image_tensor.shape)}")
        if image_tensor.numel() and (
            image_tensor.min().item() < 0.0 or image_tensor.max().item() > 1.0
        ):
            raise ValueError("DeepJSCC repository input must be in [0, 1].")
        with torch.no_grad():
            latent = encoder(image_tensor)
        return LatentArtifact(
            tensor=latent,
            rate_mask=None,
            source_model="DeepJSCC",
            rate_tokens=self.c,
            metadata={
                "sample_ids": list(image.sample_ids),
                "latent_shape": tuple(latent.shape),
                "c": self.c,
                "input_scale": "[0,1]",
            },
        )

    def decode(self, latent: LatentArtifact, **options: object) -> ImageArtifact:
        torch = require_torch()
        _, decoder = self._require_built()
        with torch.no_grad():
            reconstruction = decoder(latent.tensor.to(self.device))
        reconstruction = torch.clamp(reconstruction, 0.0, 1.0)
        return ImageArtifact(
            tensor=reconstruction,
            sample_ids=list(latent.metadata.get("sample_ids", [])),
            source="DeepJSCCDecoder",
            metadata={"c": self.c, "output_scale": "[0,1]"},
        )

    def reconstruct_received(
        self, transmission: TransmissionArtifact, latent: LatentArtifact
    ) -> ImageArtifact:
        received_latent = LatentArtifact(
            tensor=transmission.received,
            rate_mask=latent.rate_mask,
            source_model=latent.source_model,
            rate_tokens=latent.rate_tokens,
            metadata={**latent.metadata, "snr_db": transmission.snr_db},
        )
        return self.decode(received_latent)

    def parameter_summary(self) -> dict[str, int | str]:
        encoder, decoder = self._require_built()
        encoder_parameters = sum(parameter.numel() for parameter in encoder.parameters())
        decoder_parameters = sum(parameter.numel() for parameter in decoder.parameters())
        digest = hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest()
        return {
            "encoder_parameters": int(encoder_parameters),
            "decoder_parameters": int(decoder_parameters),
            "total_parameters": int(encoder_parameters + decoder_parameters),
            "checkpoint_sha256": digest,
        }


def register_deepjscc() -> None:
    if not ENCDEC_REGISTRY.contains("deepjscc"):
        ENCDEC_REGISTRY.register("deepjscc", DeepJSCCAdapter)
