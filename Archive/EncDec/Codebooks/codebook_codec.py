"""Packet schema and bit accounting for shared-codebook VQ packets."""

from __future__ import annotations

import math
import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class CodebookPacketHeader:
    """Fields that must travel with each VQ packet; codebook storage is separate."""

    codebook_id: str
    codebook_size: int
    latent_shape: tuple[int, int, int]
    active_channels: int
    index_bits: int
    rate_mask_bits: int
    metadata_bits: int = 128
    crc_bits: int = 32

    @property
    def index_count(self) -> int:
        batch, tokens, _ = self.latent_shape
        return batch * tokens

    @property
    def payload_bits(self) -> int:
        return self.index_count * self.index_bits + self.rate_mask_bits + self.metadata_bits + self.crc_bits

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["payload_bits"] = self.payload_bits
        return values


def index_width(codebook_size: int) -> int:
    if codebook_size < 2:
        raise ValueError("A vector codebook needs at least two entries.")
    return math.ceil(math.log2(codebook_size))


def pack_fixed_width_indices(indices: list[int], width: int) -> bytes:
    """Pack codebook indices exactly; padding bits are reported separately by callers."""

    if width <= 0:
        raise ValueError("Index width must be positive.")
    bit_string = "".join(format(index, f"0{width}b") for index in indices)
    padding = (-len(bit_string)) % 8
    return int(bit_string + "0" * padding or "0", 2).to_bytes((len(bit_string) + padding) // 8, "big")


def unpack_fixed_width_indices(payload: bytes, count: int, width: int) -> list[int]:
    bits = "".join(f"{byte:08b}" for byte in payload)
    return [int(bits[index * width : (index + 1) * width], 2) for index in range(count)]


def _integer_bits(value: int, width: int) -> np.ndarray:
    if not 0 <= value < 2**width:
        raise ValueError(f"Cannot represent {value} in {width} bits.")
    return np.fromiter((int(bit) for bit in format(value, f"0{width}b")), dtype=np.uint8)


def _crc32_bits(bits: np.ndarray) -> np.ndarray:
    import binascii

    padding = (-bits.size) % 8
    packed = np.packbits(np.pad(bits, (0, padding)), bitorder="big")
    return _integer_bits(binascii.crc32(packed.tobytes()) & 0xFFFFFFFF, 32)


def _metadata_bits(header: CodebookPacketHeader) -> np.ndarray:
    """128-bit schema: codebook fingerprint, K, active rate, latent dimensions, version."""

    identifier = int.from_bytes(hashlib.sha256(header.codebook_id.encode("utf-8")).digest()[:8], "big")
    _, tokens, channels = header.latent_shape
    fields = np.concatenate(
        (
            _integer_bits(identifier, 64),
            _integer_bits(header.codebook_size, 9),
            _integer_bits(header.active_channels, 9),
            _integer_bits(tokens, 16),
            _integer_bits(channels, 16),
            _integer_bits(1, 4),
            _integer_bits(0, 10),
        )
    )
    if fields.size != header.metadata_bits:
        raise AssertionError("Codebook metadata schema no longer matches declared metadata bits.")
    return fields


def encode_codebook_packet_bits(indices: np.ndarray, header: CodebookPacketHeader, rate_mask: list[int] | None) -> np.ndarray:
    """Build one complete VQ packet: indices, explicit mask, metadata, and a real CRC32."""

    flattened = np.asarray(indices, dtype=np.int64).reshape(-1)
    if flattened.size != header.index_count:
        raise ValueError("Index count disagrees with the packet header.")
    index_bits = np.concatenate([_integer_bits(int(index), header.index_bits) for index in flattened])
    if header.rate_mask_bits:
        if rate_mask is None or len(rate_mask) != header.rate_mask_bits:
            raise ValueError("The VQ packet must include one full rate mask with the declared bit length.")
        mask_bits = np.asarray(rate_mask, dtype=np.uint8)
    else:
        mask_bits = np.empty(0, dtype=np.uint8)
    content = np.concatenate((index_bits, mask_bits, _metadata_bits(header)))
    return np.concatenate((content, _crc32_bits(content)))


def decode_codebook_packet_bits(bits: np.ndarray, header: CodebookPacketHeader) -> tuple[np.ndarray, list[int] | None]:
    """Verify packet CRC and recover indices/mask for one VQ image."""

    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if values.size != header.payload_bits:
        raise ValueError("Received VQ packet length does not match its declared header.")
    content, received_crc = values[:-header.crc_bits], values[-header.crc_bits:]
    if not np.array_equal(_crc32_bits(content), received_crc):
        raise ValueError("VQ packet CRC mismatch.")
    index_bit_count = header.index_count * header.index_bits
    index_values = [
        int("".join(str(int(bit)) for bit in content[start : start + header.index_bits]), 2)
        for start in range(0, index_bit_count, header.index_bits)
    ]
    mask_start = index_bit_count
    mask_end = mask_start + header.rate_mask_bits
    mask = content[mask_start:mask_end].tolist() if header.rate_mask_bits else None
    if not np.array_equal(content[mask_end:], _metadata_bits(header)):
        raise ValueError("VQ packet metadata does not match the selected codebook/rate/shape.")
    return np.asarray(index_values, dtype=np.uint16), mask
