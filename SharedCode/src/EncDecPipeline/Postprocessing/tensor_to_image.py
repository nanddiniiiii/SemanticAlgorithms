"""Tensor-to-PIL conversion for saved reconstructions."""

from typing import Any


def tensor_to_pil(tensor: Any) -> Any:
    from PIL import Image

    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255.0 + 0.5).astype("uint8"))
