"""Shared QPSK mapping and Es/N0-consistent LLR demapping."""

from __future__ import annotations

from typing import Any

from Channels.channel_utils import qpsk_symbols_from_bits, require_torch


def qpsk_map(coded_bits: Any) -> Any:
    return qpsk_symbols_from_bits(coded_bits)


def qpsk_llr(received_symbols: Any, noise_variance_n0: float) -> Any:
    """Return Sionna's positive-for-bit-one LLR convention for unit-energy QPSK."""

    torch = require_torch()
    if noise_variance_n0 <= 0:
        raise ValueError("AWGN noise variance N0 must be positive.")
    scale = -2.0 * (2.0**0.5) / noise_variance_n0
    return torch.stack((scale * received_symbols.real, scale * received_symbols.imag), dim=-1).reshape(
        *received_symbols.shape[:-1], -1
    )
