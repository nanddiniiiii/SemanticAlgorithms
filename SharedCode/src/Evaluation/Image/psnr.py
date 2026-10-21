"""PSNR with explicit [0, 1] image-value assumption."""

from typing import Any


def psnr(reference: Any, reconstruction: Any, data_range: float = 1.0) -> float:
    import torch

    mse = torch.mean((reference.detach().float() - reconstruction.detach().float().clamp(0.0, data_range)) ** 2)
    if float(mse) == 0.0:
        return float("inf")
    return float(10.0 * torch.log10(torch.tensor(data_range**2, device=mse.device) / mse).cpu())
