"""Transport headers, CRC and padding accounting for digital baselines."""

from __future__ import annotations

import binascii
import numpy as np


LENGTH_HEADER_BITS = 32
CRC_BITS = 32


def integer_to_bits(value: int, width: int) -> np.ndarray:
    if value < 0 or value >= 2**width:
        raise ValueError(f"Value {value} does not fit in {width} bits.")
    return np.fromiter((int(bit) for bit in format(value, f"0{width}b")), dtype=np.uint8)


def bits_to_integer(bits: np.ndarray) -> int:
    return int("".join(str(int(bit)) for bit in np.asarray(bits).reshape(-1)), 2)


def crc32_bits(payload_bits: np.ndarray) -> np.ndarray:
    packed = np.packbits(np.asarray(payload_bits, dtype=np.uint8), bitorder="big")
    checksum = binascii.crc32(packed.tobytes()) & 0xFFFFFFFF
    return integer_to_bits(checksum, CRC_BITS)


def frame_payload_bits(source_bits: np.ndarray) -> np.ndarray:
    source = np.asarray(source_bits, dtype=np.uint8).reshape(-1)
    return np.concatenate((integer_to_bits(int(source.size), LENGTH_HEADER_BITS), source, crc32_bits(source)))


def recover_payload_bits(framed_bits: np.ndarray) -> tuple[np.ndarray | None, bool, str]:
    values = np.asarray(framed_bits, dtype=np.uint8).reshape(-1)
    if values.size < LENGTH_HEADER_BITS + CRC_BITS:
        return None, False, "frame_too_short"
    length = bits_to_integer(values[:LENGTH_HEADER_BITS])
    expected = LENGTH_HEADER_BITS + length + CRC_BITS
    if length % 8 or expected > values.size:
        return None, False, "invalid_length_header"
    payload = values[LENGTH_HEADER_BITS : LENGTH_HEADER_BITS + length]
    crc = values[LENGTH_HEADER_BITS + length : expected]
    if not np.array_equal(crc, crc32_bits(payload)):
        return None, False, "crc_mismatch"
    return payload, True, "ok"
