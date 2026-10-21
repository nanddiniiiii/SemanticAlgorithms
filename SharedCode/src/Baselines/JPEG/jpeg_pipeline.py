"""JPEG source coding only; transport is injected separately through DigitalPHY."""

from dataclasses import dataclass
from typing import Any

from Baselines.JPEG.jpeg_bitstream import bytes_to_bits
from Baselines.JPEG.jpeg_codec import JPEGCodec


@dataclass(slots=True)
class JPEGBitstream:
    jpeg_bytes: bytes
    bits: Any
    quality: int


def encode_jpeg_for_transport(image: Any, quality: int) -> JPEGBitstream:
    codec = JPEGCodec(quality)
    payload = codec.encode(image)
    return JPEGBitstream(jpeg_bytes=payload, bits=bytes_to_bits(payload), quality=quality)
