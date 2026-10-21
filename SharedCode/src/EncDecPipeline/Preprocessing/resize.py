"""Image resize/crop helpers kept separate from model code."""

from typing import Any


def center_crop_to_multiple(image: Any, divisor: int = 16) -> Any:
    width, height = image.size
    target_width = width - (width % divisor)
    target_height = height - (height % divisor)
    if target_width == 0 or target_height == 0:
        raise ValueError(f"Image {(width, height)} is smaller than divisor {divisor}.")
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    return image.crop((left, top, left + target_width, top + target_height))
