"""Image SSIM evaluated sample by sample to retain original image dimensions."""

from typing import Any


def ssim(reference: Any, reconstruction: Any) -> float:
    try:
        from skimage.metrics import structural_similarity
    except ImportError as error:
        raise RuntimeError("SSIM requires scikit-image; install the project metrics extra.") from error
    reference_images = reference.detach().float().cpu().permute(0, 2, 3, 1).numpy()
    reconstruction_images = reconstruction.detach().float().clamp(0.0, 1.0).cpu().permute(0, 2, 3, 1).numpy()
    scores = [
        structural_similarity(original, restored, channel_axis=-1, data_range=1.0)
        for original, restored in zip(reference_images, reconstruction_images, strict=True)
    ]
    return float(sum(scores) / len(scores))
