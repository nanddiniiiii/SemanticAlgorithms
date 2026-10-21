"""DINOv2 loader used by the public semantic encoder."""

from __future__ import annotations

import os

import torch


@torch.no_grad()
def load_encoders(enc_type: str, device: str, resolution: int = 256):
    """Load the DINOv2 image encoder expected by released DiT-JSCC models."""

    if resolution not in (256, 512):
        raise ValueError("resolution must be 256 or 512")

    encoder_type, architecture, model_config = enc_type.split("-")
    if encoder_type != "dinov2" or architecture != "vit":
        raise NotImplementedError(
            "The public inference path currently supports DINOv2 ViT encoders only."
        )

    local_repo = os.environ.get("DINOV2_REPO")
    if local_repo:
        if not os.path.isdir(local_repo):
            raise FileNotFoundError(
                f"DINOV2_REPO does not point to a directory: {local_repo}"
            )
        encoder = torch.hub.load(
            local_repo,
            f"dinov2_vit{model_config}14",
            source="local",
        )
    else:
        encoder = torch.hub.load(
            "facebookresearch/dinov2",
            f"dinov2_vit{model_config}14",
            trust_repo=True,
        )

    import timm

    if hasattr(encoder, "head"):
        del encoder.head
    patch_resolution = 16 * (resolution // 256)
    encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
        encoder.pos_embed.data,
        [patch_resolution, patch_resolution],
    )
    encoder.head = torch.nn.Identity()
    encoder = encoder.to(device).eval()
    return encoder, encoder_type, architecture
