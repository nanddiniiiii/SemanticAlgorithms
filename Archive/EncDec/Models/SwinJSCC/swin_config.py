"""Frozen Stage 1A architecture configuration for the official SwinJSCC source."""

from dataclasses import asdict, dataclass, field
from typing import Any


UPSTREAM_COMMIT = "a6d0e6da53548976acbe9317839a077ef31f190f"
BASE_MODEL_NAME = "SwinJSCC_w/o_SAandRA"
SA_RA_MODEL_NAME = "SwinJSCC_w/_SAandRA"


@dataclass(frozen=True, slots=True)
class SwinJSCCConfig:
    """Only architecture and grid choices that should be manifest-validated."""

    variant: str = SA_RA_MODEL_NAME
    initial_image_size: tuple[int, int] = (1024, 1024)
    base_fixed_c: int = 96
    rate_grid: tuple[int, ...] = (32, 64, 96, 128, 192)
    snr_db_grid: tuple[int, ...] = (1, 4, 7, 10, 13)
    channel_type: str = "awgn"
    model_size: str = "base"
    downsample_stages: int = 4
    window_size: int = 8
    encoder_embed_dims: tuple[int, ...] = (128, 192, 256, 320)
    encoder_depths: tuple[int, ...] = (2, 2, 6, 2)
    encoder_heads: tuple[int, ...] = (4, 6, 8, 10)
    decoder_embed_dims: tuple[int, ...] = (320, 256, 192, 128)
    decoder_depths: tuple[int, ...] = (2, 6, 2, 2)
    decoder_heads: tuple[int, ...] = (10, 8, 6, 4)
    upstream_commit: str = UPSTREAM_COMMIT
    # v2 fixes the DataParallel-safe attention-mask device lookup in addition
    # to the original per-image complex-symbol correction.
    patch_version: str = "stage1a-per-image-symbols-v2"
    objective: str = "mse"
    metadata: dict[str, Any] = field(default_factory=dict)

    def encoder_kwargs(self) -> dict[str, Any]:
        channels = self.base_fixed_c if self.variant in (BASE_MODEL_NAME, "SwinJSCC_w/_SA") else None
        return {
            "model": self.variant,
            "img_size": self.initial_image_size,
            "patch_size": 2,
            "in_chans": 3,
            "embed_dims": list(self.encoder_embed_dims),
            "depths": list(self.encoder_depths),
            "num_heads": list(self.encoder_heads),
            "C": channels,
            "window_size": self.window_size,
            "mlp_ratio": 4.0,
            "qkv_bias": True,
            "qk_scale": None,
            "patch_norm": True,
        }

    def decoder_kwargs(self) -> dict[str, Any]:
        channels = self.base_fixed_c if self.variant in (BASE_MODEL_NAME, "SwinJSCC_w/_SA") else None
        return {
            "model": self.variant,
            "img_size": self.initial_image_size,
            "embed_dims": list(self.decoder_embed_dims),
            "depths": list(self.decoder_depths),
            "num_heads": list(self.decoder_heads),
            "C": channels,
            "window_size": self.window_size,
            "mlp_ratio": 4.0,
            "qkv_bias": True,
            "qk_scale": None,
            "patch_norm": True,
        }

    def to_manifest_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_manifest_dict(cls, values: dict[str, Any]) -> "SwinJSCCConfig":
        normalized = dict(values)
        for key in (
            "initial_image_size",
            "rate_grid",
            "snr_db_grid",
            "encoder_embed_dims",
            "encoder_depths",
            "encoder_heads",
            "decoder_embed_dims",
            "decoder_depths",
            "decoder_heads",
        ):
            if key in normalized:
                normalized[key] = tuple(normalized[key])
        return cls(**normalized)


def expected_cbr(rate: int, downsample_stages: int = 4) -> float:
    """SwinJSCC SA+RA CBR: transmitted complex uses / raw RGB samples."""

    return float(rate) / (2.0 * 3.0 * (2 ** (2 * downsample_stages)))
