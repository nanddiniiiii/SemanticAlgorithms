"""Numerically explicit utilities for complex symbol channels."""

import math
from typing import Any


def require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for neural-channel simulation. Use the documented Python 3.12 runtime."
        ) from error
    return torch


def esn0_to_noise_variance(snr_db: float) -> float:
    """Return complex AWGN variance N0 for unit-average-energy symbols."""

    return 10.0 ** (-float(snr_db) / 10.0)


def _validate_even_feature_count(tensor: Any) -> None:
    per_image_values = int(tensor[0].numel())
    if per_image_values % 2 != 0:
        raise ValueError(
            "SwinJSCC channel conversion requires an even number of active real values per image; "
            f"received {per_image_values}."
        )


def real_to_complex_per_image(real_features: Any) -> tuple[Any, Any]:
    """Pair real and imaginary halves *inside each image*, never across a batch.

    The official legacy implementation flattened the entire minibatch before splitting
    real and imaginary halves. That silently crossed image boundaries when DataParallel
    supplied more than one local sample. This function preserves the batch dimension.
    """

    torch = require_torch()
    if real_features.ndim < 2:
        raise ValueError("Expected a batched feature tensor with at least two dimensions.")
    _validate_even_feature_count(real_features)
    batch_size = int(real_features.shape[0])
    flattened = real_features.reshape(batch_size, -1)
    split = flattened.shape[1] // 2
    # Explicitly promote each half before complex construction.  AMP can make
    # the latent tensor float16; constructing ComplexHalf triggers incomplete
    # operator coverage in current PyTorch releases.  Complex64 is supported
    # by the channel and decoder path and preserves the intended values.
    real_part = flattened[:, :split].float()
    imaginary_part = flattened[:, split:].float()
    complex_symbols = torch.complex(real_part, imaginary_part)
    return complex_symbols, real_features.shape


def complex_to_real_per_image(complex_symbols: Any, original_shape: Any) -> Any:
    """Inverse of :func:`real_to_complex_per_image` with the original batch shape."""

    torch = require_torch()
    flattened = torch.cat((complex_symbols.real, complex_symbols.imag), dim=1)
    return flattened.reshape(original_shape)


def normalize_complex_per_image(symbols: Any) -> tuple[Any, Any]:
    """Normalize each image to unit complex-symbol energy and return its scale."""

    torch = require_torch()
    power = symbols.abs().square().mean(dim=1, keepdim=True)
    scale = torch.sqrt(torch.clamp(power, min=torch.finfo(symbols.real.dtype).eps))
    return symbols / scale, scale


def qpsk_symbols_from_bits(bits: Any) -> Any:
    """Map boolean/0-1 bits of shape [batch, even_bits] to unit-energy QPSK."""

    torch = require_torch()
    if bits.shape[-1] % 2 != 0:
        raise ValueError("QPSK requires an even number of bits.")
    pairs = bits.reshape(*bits.shape[:-1], -1, 2).to(dtype=torch.float32)
    in_phase = 1.0 - 2.0 * pairs[..., 0]
    quadrature = 1.0 - 2.0 * pairs[..., 1]
    return torch.complex(in_phase, quadrature) / math.sqrt(2.0)


def qpsk_bits_from_symbols(symbols: Any) -> Any:
    """Hard-decision QPSK demapper matching :func:`qpsk_symbols_from_bits`."""

    torch = require_torch()
    in_phase = (symbols.real < 0).to(dtype=torch.uint8)
    quadrature = (symbols.imag < 0).to(dtype=torch.uint8)
    return torch.stack((in_phase, quadrature), dim=-1).reshape(*symbols.shape[:-1], -1)
