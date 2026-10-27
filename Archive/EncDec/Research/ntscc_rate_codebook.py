"""NTSCC-aware K=256 product quantization over active channel prefixes only."""
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
    source = str((Path(team_root) / "src").resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from EncDecPipeline.Codebooks.codebook_codec import (
        CodebookPacketHeader,
        decode_codebook_packet_bits,
        encode_codebook_packet_bits,
    )
    return CodebookPacketHeader, encode_codebook_packet_bits, decode_codebook_packet_bits


def _rate_bits(rate_indexes: np.ndarray) -> list[int]:
    flat = np.asarray(rate_indexes, dtype=np.uint8).reshape(-1)
    return np.concatenate([
        np.fromiter((int(bit) for bit in format(int(value), "04b")), dtype=np.uint8)
        for value in flat
    ]).tolist()


def _decode_rate_bits(bits: list[int]) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8).reshape(-1, 4)
    weights = np.asarray([8, 4, 2, 1], dtype=np.uint8)
    return (values * weights).sum(axis=1).astype(np.int64)


def _tokens(tensor: torch.Tensor) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    value = tensor.detach().float().cpu()
    if value.ndim != 4:
        raise ValueError(f"Expected BCHW NTSCC channel latent, got {tuple(value.shape)}")
    batch, channels, height, width = value.shape
    return value.permute(0, 2, 3, 1).reshape(-1, channels).numpy(), (batch, channels, height, width)


@dataclass
class RateAwarePacket:
    bits: np.ndarray
    header: object
    rate_indexes: np.ndarray
    latent_shape: tuple[int, int, int, int]
    index_count: int
    active_value_count: int
    active_quantization_mse: float
    diagnostics: dict

    @property
    def quantization_mse(self) -> float:
        """Compatibility alias; the denominator is active NTSCC values only."""
        return self.active_quantization_mse


class NTSCCActivePrefixK256:
    """One K=256 codebook for 16-D blocks drawn only from active prefixes.

    Official NTSCC selects a prefix length per spatial token.  This wrapper
    product-quantizes that prefix and never includes inactive zero dimensions
    in training or nearest-centroid distance.  A partial final block compares
    only its real active coordinates.
    """

    def __init__(self, team_root: Path, rate_choices: list[int], block_size: int = 16, random_state: int = 42):
        self.team_root = Path(team_root)
        self.rate_choices = np.asarray(rate_choices, dtype=np.int64)
        self.block_size = int(block_size)
        self.size = 256
        if self.block_size != 16:
            raise ValueError("This experiment locks the active-prefix block size to 16")
        self.model = MiniBatchKMeans(
            n_clusters=self.size,
            batch_size=8192,
            n_init=10,
            max_iter=200,
            random_state=random_state,
        )

    def _active_blocks(self, latent: torch.Tensor, rate_indexes: torch.Tensor) -> np.ndarray:
        tokens, _ = _tokens(latent)
        indexes = rate_indexes.detach().cpu().numpy().reshape(-1).astype(np.int64)
        if len(tokens) != len(indexes):
            raise ValueError("Rate-index count does not match spatial token count")
        blocks = []
        for token, rate_index in zip(tokens, indexes):
            active = int(self.rate_choices[rate_index])
            for start in range(0, active, self.block_size):
                length = min(self.block_size, active - start)
                if length == self.block_size:
                    block = token[start:start + self.block_size].astype(np.float32, copy=False)
                    blocks.append(block)
        return np.asarray(blocks, dtype=np.float32)

    def fit(self, calibration: list[tuple[torch.Tensor, torch.Tensor]]) -> "NTSCCActivePrefixK256":
        blocks = np.concatenate([self._active_blocks(latent, indexes) for latent, indexes in calibration], axis=0)
        if len(blocks) < self.size:
            raise ValueError(f"K=256 needs at least 256 active blocks, received {len(blocks)}")
        self.model.fit(blocks)
        self.calibration_block_count = int(len(blocks))
        return self

    def _nearest(self, block: np.ndarray, length: int) -> int:
        centers = self.model.cluster_centers_[:, :length]
        difference = centers - np.asarray(block[:length], dtype=np.float32)
        return int(np.argmin(np.einsum("ij,ij->i", difference, difference)))

    def encode(self, latent: torch.Tensor, rate_indexes: torch.Tensor) -> RateAwarePacket:
        tokens, shape = _tokens(latent)
        indexes = rate_indexes.detach().cpu().numpy().reshape(-1).astype(np.int64)
        encoded_indices: list[int] = []
        squared_error = 0.0
        active_values = 0
        for token, rate_index in zip(tokens, indexes):
            active = int(self.rate_choices[rate_index])
            for start in range(0, active, self.block_size):
                length = min(self.block_size, active - start)
                index = self._nearest(token[start:start + length], length)
                encoded_indices.append(index)
                error = token[start:start + length] - self.model.cluster_centers_[index, :length]
                squared_error += float(np.dot(error, error))
                active_values += length
        encoded = np.asarray(encoded_indices, dtype=np.uint16)
        rate_bits = _rate_bits(indexes)
        Header, encode_bits, _ = _team_codec(self.team_root)
        header = Header(
            codebook_id="ntscc_active_prefix_pq16_k256",
            codebook_size=256,
            latent_shape=(1, int(encoded.size), self.block_size),
            active_channels=self.block_size,
            index_bits=8,
            rate_mask_bits=len(rate_bits),
        )
        bits = encode_bits(encoded, header, rate_bits)
        counts = np.bincount(encoded.astype(np.int64), minlength=self.size)
        probabilities = counts[counts > 0] / max(int(counts.sum()), 1)
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        return RateAwarePacket(
            bits=bits,
            header=header,
            rate_indexes=indexes,
            latent_shape=shape,
            index_count=int(encoded.size),
            active_value_count=int(active_values),
            active_quantization_mse=float(squared_error / active_values),
            diagnostics={
                "used_entries": int((counts > 0).sum()),
                "dead_entries": int((counts == 0).sum()),
                "codebook_utilization": float((counts > 0).sum() / self.size),
                "perplexity": float(2**entropy),
            },
        )

    def decode(self, received_bits: np.ndarray, packet: RateAwarePacket, device: torch.device):
        _, _, decode_bits = _team_codec(self.team_root)
        encoded, rate_bits = decode_bits(received_bits, packet.header)
        if rate_bits is None:
            raise ValueError("NTSCC-aware packet omitted rate indexes")
        indexes = _decode_rate_bits(rate_bits)
        batch, channels, height, width = packet.latent_shape
        if indexes.size != batch * height * width:
            raise ValueError("Decoded rate-map size does not match latent spatial shape")
        tokens = np.zeros((indexes.size, channels), dtype=np.float32)
        cursor = 0
        for token_number, rate_index in enumerate(indexes):
            active = int(self.rate_choices[rate_index])
            for start in range(0, active, self.block_size):
                if cursor >= len(encoded):
                    raise ValueError("Packet ended before all active NTSCC blocks were decoded")
                length = min(self.block_size, active - start)
                tokens[token_number, start:start + length] = self.model.cluster_centers_[int(encoded[cursor]), :length]
                cursor += 1
        if cursor != len(encoded):
            raise ValueError("Packet contains unused NTSCC block indices")
        tensor = torch.from_numpy(tokens.reshape(batch, height, width, channels)).permute(0, 3, 1, 2).to(device)
        return tensor, torch.from_numpy(indexes).long().to(device)

    def content_hash(self) -> str:
        return hashlib.sha256(np.asarray(self.model.cluster_centers_, dtype=np.float32).tobytes()).hexdigest()

    def shared_storage_bits(self) -> int:
        return int(self.size * self.block_size * 32)
