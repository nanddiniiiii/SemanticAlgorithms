"""Raw RGB size based on the actual evaluated tensor, never a training crop constant."""

from typing import Any


def raw_rgb_bits(image_tensor: Any, bits_per_channel: int = 8) -> int:
    if image_tensor.ndim != 4 or int(image_tensor.shape[1]) != 3:
        raise ValueError(f"Expected [batch, 3, height, width] RGB tensor, got {tuple(image_tensor.shape)}.")
    _, channels, height, width = (int(value) for value in image_tensor.shape)
    return channels * height * width * bits_per_channel
