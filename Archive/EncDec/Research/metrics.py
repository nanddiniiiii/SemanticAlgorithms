"""Shared image metrics with optional models loaded once per clean run."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


def as_bchw01(value) -> torch.Tensor:
    if isinstance(value, Image.Image):
        tensor = TF.to_tensor(value).unsqueeze(0)
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
        if tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        if tensor.dtype == torch.uint8:
            tensor = tensor.float().div(255)
    elif torch.is_tensor(value):
        tensor = value.detach()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
    else:
        raise TypeError(f"Unsupported image type: {type(value)!r}")
    return tensor.float().clamp(0, 1)


def to_pil(value) -> Image.Image:
    return TF.to_pil_image(as_bchw01(value)[0].cpu())


@dataclass
class MetricSuite:
    device: torch.device
    include_optional: bool = True

    def __post_init__(self) -> None:
        self._lpips = None
        self._clip = None
        self._clip_preprocess = None

    def load_optional_models(self) -> None:
        if not self.include_optional:
            return
        if self._lpips is None:
            import lpips

            self._lpips = lpips.LPIPS(net="alex").to(self.device).eval()
        if self._clip is None:
            import clip

            self._clip, self._clip_preprocess = clip.load("ViT-B/32", device=str(self.device))
            self._clip.eval()

    def __call__(self, reference, reconstruction, include_optional: bool | None = None) -> dict:
        ref = as_bchw01(reference)
        rec = as_bchw01(reconstruction)
        if ref.shape != rec.shape:
            raise ValueError(f"Metric shape mismatch: {tuple(ref.shape)} != {tuple(rec.shape)}")
        mse = float(torch.mean((ref - rec) ** 2))
        psnr = float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)

        from skimage.metrics import structural_similarity

        ref_np = ref[0].permute(1, 2, 0).cpu().numpy()
        rec_np = rec[0].permute(1, 2, 0).cpu().numpy()
        ssim = float(structural_similarity(ref_np, rec_np, channel_axis=-1, data_range=1.0))
        result = {"mse": mse, "psnr_db": psnr, "ssim": ssim, "lpips": None, "clip_cosine": None}

        optional = self.include_optional if include_optional is None else bool(include_optional)
        if not optional:
            return result
        self.load_optional_models()
        with torch.inference_mode():
            result["lpips"] = float(
                self._lpips(ref.to(self.device) * 2 - 1, rec.to(self.device) * 2 - 1).mean().cpu()
            )
            a = self._clip_preprocess(to_pil(ref)).unsqueeze(0).to(self.device)
            b = self._clip_preprocess(to_pil(rec)).unsqueeze(0).to(self.device)
            ea = torch.nn.functional.normalize(self._clip.encode_image(a).float(), dim=-1)
            eb = torch.nn.functional.normalize(self._clip.encode_image(b).float(), dim=-1)
            result["clip_cosine"] = float((ea * eb).sum(dim=-1).mean().cpu())
        return result


def image_metrics(reference, reconstruction, include_optional: bool = False, device: str = "cpu") -> dict:
    """Compatibility helper retained from the source notebooks.

    Core MSE, PSNR and SSIM are always evaluated.  ``include_optional`` only
    controls the heavyweight LPIPS and CLIP models.
    """
    suite = MetricSuite(torch.device(device), include_optional=include_optional)
    return suite(reference, reconstruction, include_optional=include_optional)


def summarize_trials(rows: list[dict]) -> dict:
    summary: dict[str, float | None] = {}
    for key in ("mse", "psnr_db", "ssim", "lpips", "clip_cosine"):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if not values:
            mean = std = None
        elif key == "psnr_db" and all(math.isinf(value) for value in values):
            mean, std = float("inf"), 0.0
        else:
            finite = [value for value in values if math.isfinite(value)]
            mean = float(np.mean(finite)) if finite else None
            std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0 if finite else None
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
    return summary
