"""Square QAM modem for future digital baselines."""

import math
from typing import Any

from Channels.channel_utils import require_torch


def _gray_to_binary(values: Any) -> Any:
    result = values.clone()
    shifted = values.clone()
    while bool((shifted > 0).any()):
        shifted = shifted >> 1
        result = result ^ shifted
    return result


def qam_map(bits: Any, order: int) -> Any:
    """Gray-labelled square-QAM mapper with unit average symbol energy."""

    torch = require_torch()
    axis_levels = int(math.sqrt(order))
    bits_per_axis = int(math.log2(axis_levels))
    if axis_levels * axis_levels != order or order < 4 or bits.shape[-1] % (2 * bits_per_axis) != 0:
        raise ValueError("order must be a square power-of-two QAM order compatible with the bit count.")
    grouped = bits.reshape(*bits.shape[:-1], -1, 2, bits_per_axis).to(torch.long)
    weights = 2 ** torch.arange(bits_per_axis - 1, -1, -1, device=bits.device)
    gray = (grouped * weights).sum(dim=-1)
    binary = _gray_to_binary(gray)
    amplitudes = 2 * binary.to(torch.float32) - (axis_levels - 1)
    normalization = math.sqrt((2.0 / 3.0) * (order - 1))
    return torch.complex(amplitudes[..., 0], amplitudes[..., 1]) / normalization


def qam_hard_demap(symbols: Any, order: int) -> Any:
    """Hard-decision inverse mapping for controlled baseline use."""

    torch = require_torch()
    axis_levels = int(math.sqrt(order))
    bits_per_axis = int(math.log2(axis_levels))
    normalization = math.sqrt((2.0 / 3.0) * (order - 1))
    scaled = symbols * normalization
    coordinates = torch.stack((scaled.real, scaled.imag), dim=-1)
    binary = torch.round((coordinates + axis_levels - 1) / 2).clamp(0, axis_levels - 1).to(torch.long)
    gray = binary ^ (binary >> 1)
    shifts = torch.arange(bits_per_axis - 1, -1, -1, device=symbols.device)
    bits = ((gray.unsqueeze(-1) >> shifts) & 1).to(torch.uint8)
    return bits.reshape(*symbols.shape[:-1], -1)
