"""Full-image boundary adapter; never changes an author's network.

Padding is an explicitly reported benchmark adapter for unsupported geometry.
Aligned Kodak images pass through unchanged. Communication counts must include
the entire padded input; quality metrics see only the original pixels.
"""
import numpy as np


def pad_image(image, multiple=128):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or multiple <= 0:
        raise ValueError("Expected HWC RGB and a positive alignment")
    h, w = image.shape[:2]
    bottom, right = (-h) % multiple, (-w) % multiple
    padded = np.pad(image, ((0, bottom), (0, right), (0, 0)), mode="edge")
    return padded, {"original_height": h, "original_width": w,
                    "padded_height": h + bottom, "padded_width": w + right,
                    "padding_bottom": bottom, "padding_right": right,
                    "padding_mode": "edge", "alignment": multiple,
                    "resize": False, "tiling": False,
                    "author_preprocessing": bottom == 0 and right == 0}


def unpad_image(reconstruction, geometry):
    reconstruction = np.asarray(reconstruction)
    expected = (geometry["padded_height"], geometry["padded_width"], 3)
    if reconstruction.shape != expected or not np.isfinite(reconstruction).all():
        raise ValueError("Wrong reconstruction dimensions or nonfinite pixels")
    return reconstruction[:geometry["original_height"], :geometry["original_width"]].copy()


def measured_cbr(complex_channel_uses, geometry):
    if isinstance(complex_channel_uses, bool) or int(complex_channel_uses) != complex_channel_uses or complex_channel_uses <= 0:
        raise ValueError("Positive measured complex channel uses required")
    return complex_channel_uses / (3 * geometry["original_height"] * geometry["original_width"])
