"""Exact conversion between JPEG bytes and one-dimensional binary payloads."""

from __future__ import annotations

import numpy as np


def bytes_to_bits(payload: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big").astype(np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if values.size % 8:
        raise ValueError("A byte payload requires a multiple of eight bits.")
    return np.packbits(values, bitorder="big").tobytes()
