"""Separate K=256 latent codebooks and auditable packet construction."""
from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans


def _team_codec(team_root: Path):
    source = str((team_root / "src").resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from EncDecPipeline.Codebooks.codebook_codec import (
        CodebookPacketHeader,
        decode_codebook_packet_bits,
        encode_codebook_packet_bits,
    )

    return CodebookPacketHeader, encode_codebook_packet_bits, decode_codebook_packet_bits


def tensor_tokens(tensor: torch.Tensor) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    value = tensor.detach().float().cpu()
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4:
        raise ValueError(f"Expected BCHW/CHW latent, got {tuple(value.shape)}")
    batch, channels, height, width = value.shape
    tokens = value.permute(0, 2, 3, 1).reshape(-1, channels).numpy()
    return tokens, (batch, channels, height, width)


def usage_metrics(indices: np.ndarray, size: int = 256) -> dict:
    counts = np.bincount(np.asarray(indices).reshape(-1), minlength=size)
    probabilities = counts / max(int(counts.sum()), 1)
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())
    return {
        "codebook_utilization": float((counts > 0).sum() / size),
        "used_entries": int((counts > 0).sum()),
        "dead_entries": int((counts == 0).sum()),
        "perplexity": float(2**entropy),
    }


@dataclass
class EncodedCodebookPacket:
    bits: np.ndarray
    header: object
    indices: np.ndarray
    rate_indexes: np.ndarray | None
    token_shapes: list[tuple[int, int, int, int]]
    quantization_mse: float
    diagnostics: dict


class K256LatentCodebook:
    def __init__(self, identifier: str, team_root: Path, random_state: int = 42):
        self.identifier = identifier
        self.team_root = Path(team_root)
        self.size = 256
        self.model = MiniBatchKMeans(
            n_clusters=256,
            batch_size=8192,
            n_init=10,
            max_iter=200,
            random_state=random_state,
        )
        self.channels: int | None = None

    def fit(self, calibration_latents: list[torch.Tensor]) -> "K256LatentCodebook":
        batches = []
        for latent in calibration_latents:
            tokens, shape = tensor_tokens(latent)
            channels = shape[1]
            if self.channels is None:
                self.channels = channels
            if channels != self.channels:
                raise ValueError("Latents with different channel dimensions cannot share a codebook")
            batches.append(tokens)
        training = np.concatenate(batches, axis=0)
        if training.shape[0] < self.size:
            raise ValueError("K=256 requires at least 256 real calibration tokens")
        self.model.fit(training)
        return self

    def content_hash(self) -> str:
        return hashlib.sha256(np.asarray(self.model.cluster_centers_, dtype=np.float32).tobytes()).hexdigest()

    @staticmethod
    def _rate_bits(rate_indexes: np.ndarray | None) -> list[int] | None:
        if rate_indexes is None:
            return None
        flat = np.asarray(rate_indexes, dtype=np.uint8).reshape(-1)
        return np.concatenate(
            [np.fromiter((int(bit) for bit in format(int(value), "04b")), dtype=np.uint8) for value in flat]
        ).tolist()

    def encode(self, latents: list[torch.Tensor], rate_indexes: np.ndarray | None = None) -> EncodedCodebookPacket:
        if self.channels is None:
            raise RuntimeError("Fit the codebook before encoding")
        token_batches, shapes = zip(*(tensor_tokens(latent) for latent in latents))
        all_tokens = np.concatenate(token_batches, axis=0)
        indices = self.model.predict(all_tokens).astype(np.uint16)
        quantized = self.model.cluster_centers_[indices]
        quantization_mse = float(np.mean((all_tokens - quantized) ** 2))
        rate_bits = self._rate_bits(rate_indexes)
        Header, encode_bits, _ = _team_codec(self.team_root)
        header = Header(
            codebook_id=self.identifier,
            codebook_size=256,
            latent_shape=(1, int(indices.size), int(self.channels)),
            active_channels=int(self.channels),
            index_bits=8,
            rate_mask_bits=0 if rate_bits is None else len(rate_bits),
        )
        bits = encode_bits(indices, header, rate_bits)
        return EncodedCodebookPacket(
            bits=bits,
            header=header,
            indices=indices,
            rate_indexes=None if rate_indexes is None else np.asarray(rate_indexes, dtype=np.uint8).reshape(-1),
            token_shapes=list(shapes),
            quantization_mse=quantization_mse,
            diagnostics=usage_metrics(indices),
        )

    def decode(self, received_bits: np.ndarray, packet: EncodedCodebookPacket, device: torch.device):
        _, _, decode_bits = _team_codec(self.team_root)
        indices, rate_bits = decode_bits(received_bits, packet.header)
        decoded = self.model.cluster_centers_[indices.astype(np.int64)]
        outputs = []
        offset = 0
        for batch, channels, height, width in packet.token_shapes:
            count = batch * height * width
            values = decoded[offset : offset + count].reshape(batch, height, width, channels)
            outputs.append(torch.from_numpy(values).permute(0, 3, 1, 2).float().to(device))
            offset += count
        decoded_rates = None
        if rate_bits is not None:
            bit_array = np.asarray(rate_bits, dtype=np.uint8).reshape(-1, 4)
            decoded_rates = np.asarray(
                [int("".join(str(int(bit)) for bit in row), 2) for row in bit_array], dtype=np.int64
            )
        return outputs, decoded_rates

    def shared_storage_bits(self, precision_bits: int = 32) -> int:
        return int(self.size * int(self.channels) * precision_bits)
