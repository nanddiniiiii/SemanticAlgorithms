"""Actual team 5G-LDPC/QPSK/AWGN transport with module caching."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def bytes_to_bits(payload: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")


def bits_to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="big").tobytes()


class TeamDigitalPHY:
    def __init__(self, team_root: Path, information_bits: int = 1024, codeword_bits: int = 2048, iterations: int = 20):
        source = str((Path(team_root) / "src").resolve())
        if source not in sys.path:
            sys.path.insert(0, source)
        from Baselines.DigitalPHY.bit_utils import frame_payload_bits, recover_payload_bits
        from Baselines.DigitalPHY.channel_coding import DigitalTransmissionResult, LDPCQPSKAWGN
        from Baselines.DigitalPHY.qpsk_modem import qpsk_llr, qpsk_map
        from Channels.channel_utils import esn0_to_noise_variance

        class CachedLDPC(LDPCQPSKAWGN):
            def _sionna_modules(instance, device: str):
                key = str(device)
                if not hasattr(instance, "_module_cache"):
                    instance._module_cache = {}
                if key not in instance._module_cache:
                    instance._module_cache[key] = super(CachedLDPC, instance)._sionna_modules(device)
                return instance._module_cache[key]

        self.impl = CachedLDPC(information_bits, codeword_bits, iterations)
        self._frame_payload_bits = frame_payload_bits
        self._recover_payload_bits = recover_payload_bits
        self._qpsk_map = qpsk_map
        self._qpsk_llr = qpsk_llr
        self._noise_variance = esn0_to_noise_variance
        self._result_type = DigitalTransmissionResult

    def transmit(self, bits: np.ndarray, snr_db: float, seed: int, device: str = "cpu", chunk_blocks: int = 1024):
        torch.manual_seed(seed)
        source = np.asarray(bits, dtype=np.uint8).reshape(-1)
        framed = self._frame_payload_bits(source)
        k, n = self.impl.information_block_bits, self.impl.codeword_bits
        block_count = int(np.ceil(framed.size / k))
        padded_size = block_count * k
        padded = np.pad(framed, (0, padded_size - framed.size)).astype(np.float32).reshape(block_count, k)
        encoder, decoder = self.impl._sionna_modules(device)
        n0 = self._noise_variance(snr_db)
        standard_deviation = (n0 / 2.0) ** 0.5
        decoded_chunks = []
        for start in range(0, block_count, chunk_blocks):
            information = torch.from_numpy(padded[start : start + chunk_blocks]).to(device)
            coded = encoder(information)
            symbols = self._qpsk_map(coded)
            noise = torch.complex(
                torch.randn_like(symbols.real) * standard_deviation,
                torch.randn_like(symbols.imag) * standard_deviation,
            )
            decoded = decoder(self._qpsk_llr(symbols + noise, n0))
            decoded_chunks.append(decoded.detach().round().to(torch.uint8).cpu().numpy().reshape(-1))
        decoded_bits = np.concatenate(decoded_chunks)[: framed.size]
        recovered, success, reason = self._recover_payload_bits(decoded_bits)
        return self._result_type(
            recovered_bits=recovered,
            frame_success=success,
            failure_reason=reason,
            information_bits=int(source.size),
            framed_bits=int(framed.size),
            padding_bits=int(padded_size - framed.size),
            coded_bits=block_count * n,
            channel_uses=block_count * (n // 2),
            esn0_db=float(snr_db),
        )


def raw_payload(image: Image.Image) -> bytes:
    return np.asarray(image.convert("RGB"), dtype=np.uint8).tobytes()


def jpeg_payload(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality))
    return buffer.getvalue()


def decode_raw(payload: bytes, size: tuple[int, int]) -> Image.Image:
    width, height = size
    expected = width * height * 3
    if len(payload) != expected:
        raise ValueError(f"RAW payload has {len(payload)} bytes; expected {expected}")
    return Image.fromarray(np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3), "RGB")


def decode_jpeg(payload: bytes) -> Image.Image:
    return Image.open(io.BytesIO(payload)).convert("RGB")
