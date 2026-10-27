"""Non-invasive capture of SwinJSCC encoder state before channel/rate modulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from Infrastructure.Artifacts import ImageArtifact, LatentArtifact


@dataclass(slots=True)
class SwinSemanticCapture:
    """Pre-modulation representation and ordinary Stage 1A latent from one encoder pass."""

    representation: Any
    latent: LatentArtifact


def capture_pre_modulation_representation(
    adapter: Any,
    image: ImageArtifact,
    *,
    snr_db: float,
    rate: int,
) -> SwinSemanticCapture:
    """Hook ``encoder.norm`` without changing the pinned upstream SwinJSCC source."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Swin semantic representation capture requires PyTorch.") from error
    encoder, _ = adapter._require_built()
    norm_layer = getattr(encoder, "norm", None)
    if norm_layer is None or not hasattr(norm_layer, "register_forward_hook"):
        raise RuntimeError("The configured SwinJSCC encoder has no hookable final normalization layer.")
    captured: dict[str, Any] = {}

    def _capture(_: Any, __: tuple[Any, ...], output: Any) -> None:
        captured["representation"] = output.detach().clone()

    hook = norm_layer.register_forward_hook(_capture)
    try:
        with torch.no_grad():
            latent = adapter.encode(image, snr_db=snr_db, rate=rate)
    finally:
        hook.remove()
    representation = captured.get("representation")
    if representation is None:
        raise RuntimeError("SwinJSCC encoder hook did not observe the pre-modulation representation.")
    return SwinSemanticCapture(representation=representation, latent=latent)


class SwinFeatureProvider:
    def __init__(self, adapter):
        self.adapter = adapter

    def capture(self, image, **options):
        return capture_pre_modulation_representation(self.adapter, image, **options)
