"""Experimental rate-conditioned PQ/RVQ wrapper for a frozen NTSCC model.

This module is intentionally independent from the existing benchmark codebook.
It clusters only active values, trains an exact-dimensional model for partial
blocks, optionally standardizes each rate/length population, and optionally
adds a second residual codebook stage.
"""
from __future__ import annotations

import binascii
import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans


def _tokens(value: torch.Tensor) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    tensor = value.detach().float().cpu()
    if tensor.ndim != 4:
        raise ValueError(f"Expected BCHW latent, received {tuple(tensor.shape)}")
    b, c, h, w = map(int, tensor.shape)
    return tensor.permute(0, 2, 3, 1).reshape(-1, c).numpy(), (b, c, h, w)


def _pack_uint(values: np.ndarray, width: int) -> np.ndarray:
    flat = np.asarray(values, dtype=np.uint64).reshape(-1)
    shifts = np.arange(width - 1, -1, -1, dtype=np.uint64)
    return ((flat[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)


def _unpack_uint(bits: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8).reshape(-1, width)
    weights = (1 << np.arange(width - 1, -1, -1, dtype=np.uint64))
    return (values * weights).sum(axis=1, dtype=np.uint64)


def _crc32(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    packed = np.packbits(np.pad(values, (0, (-values.size) % 8)), bitorder="big")
    return _pack_uint(np.asarray([binascii.crc32(packed.tobytes()) & 0xFFFFFFFF]), 32)


@dataclass(frozen=True)
class RateConditionedConfig:
    block_size: int
    codebook_size: int
    normalize: bool = True
    rvq_stages: int = 1
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.block_size not in (2, 4, 8, 16):
            raise ValueError("block_size must be one of 2, 4, 8, 16")
        if self.codebook_size not in (256, 512, 1024):
            raise ValueError("codebook_size must be one of 256, 512, 1024")
        if self.rvq_stages not in (1, 2):
            raise ValueError("rvq_stages must be 1 or 2")

    @property
    def index_bits(self) -> int:
        return math.ceil(math.log2(self.codebook_size))

    @property
    def label(self) -> str:
        norm = "norm" if self.normalize else "raw"
        rvq = f"rvq{self.rvq_stages}" if self.rvq_stages > 1 else "pq"
        return f"rc_{rvq}{self.block_size}_k{self.codebook_size}_{norm}"


@dataclass
class _PopulationModel:
    mean: np.ndarray
    scale: np.ndarray
    stages: list[MiniBatchKMeans]
    sample_count: int


@dataclass
class RateConditionedPacket:
    bits: np.ndarray
    latent_shape: tuple[int, int, int, int]
    rate_indexes: np.ndarray
    index_streams: list[np.ndarray]
    keys: list[tuple[int, int]]
    active_value_count: int
    latent_mse: float
    index_bits: int
    rate_bits: int
    metadata_bits: int
    crc_bits: int

    @property
    def total_bits(self) -> int:
        return int(self.bits.size)


class RateConditionedPQ:
    """Separate codebooks per NTSCC rate group and exact block dimensionality.

    Rates may be grouped only when sparse high-rate populations cannot support
    the requested K. Exact-dimensional global fallback models are trained for
    partial blocks; a partial block is never compared with a full-D centroid.
    """

    METADATA_BITS = 256
    CRC_BITS = 32

    def __init__(
        self,
        rate_choices: list[int],
        config: RateConditionedConfig,
        rate_groups: list[list[int]] | None = None,
    ):
        self.rate_choices = np.asarray(rate_choices, dtype=np.int64)
        self.config = config
        groups = rate_groups or [[index] for index in range(len(rate_choices))]
        self.rate_to_group = {int(rate): group for group, rates in enumerate(groups) for rate in rates}
        if set(self.rate_to_group) != set(range(len(rate_choices))):
            raise ValueError("rate_groups must cover every rate index exactly once")
        self.rate_groups = [list(map(int, rates)) for rates in groups]
        self.models: dict[tuple[int, int], _PopulationModel] = {}

    def _populations(self, calibration: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[tuple[int, int], np.ndarray]:
        pools: dict[tuple[int, int], list[np.ndarray]] = {}
        global_pools: dict[int, list[np.ndarray]] = {}
        for latent, rate_indexes in calibration:
            tokens, _ = _tokens(latent)
            rates = rate_indexes.detach().cpu().numpy().reshape(-1).astype(np.int64)
            if tokens.shape[0] != rates.size:
                raise ValueError("Calibration token/rate count mismatch")
            for rate_index in np.unique(rates):
                active = int(self.rate_choices[rate_index])
                selected = tokens[rates == rate_index, :active]
                for start in range(0, active, self.config.block_size):
                    length = min(self.config.block_size, active - start)
                    group = self.rate_to_group[int(rate_index)]
                    block = selected[:, start:start + length]
                    pools.setdefault((group, length), []).append(block)
                    global_pools.setdefault(length, []).append(block)
        combined = {key: np.concatenate(parts, axis=0).astype(np.float32) for key, parts in pools.items()}
        combined.update({(-1, length): np.concatenate(parts, axis=0).astype(np.float32) for length, parts in global_pools.items()})
        return combined

    def _model_key(self, rate: int, length: int) -> tuple[int, int]:
        conditioned = (self.rate_to_group[int(rate)], int(length))
        return conditioned if conditioned in self.models else (-1, int(length))

    def fit(self, calibration: list[tuple[torch.Tensor, torch.Tensor]], required_rates: set[int] | None = None):
        populations = self._populations(calibration)
        if required_rates is not None:
            missing = sorted(rate for rate in required_rates if self.rate_to_group[rate] not in {group for group, _ in populations})
            if missing:
                raise ValueError(f"Calibration does not cover required NTSCC rate groups for indexes: {missing}")
        for key, values in populations.items():
            cluster_count = min(self.config.codebook_size, int(values.shape[0]))
            if cluster_count < 1:
                continue
            mean = values.mean(axis=0, dtype=np.float64).astype(np.float32) if self.config.normalize else np.zeros(values.shape[1], np.float32)
            scale = values.std(axis=0, dtype=np.float64).astype(np.float32) if self.config.normalize else np.ones(values.shape[1], np.float32)
            scale = np.maximum(scale, np.float32(1e-6))
            residual = (values - mean) / scale
            stages = []
            for stage in range(self.config.rvq_stages):
                model = MiniBatchKMeans(
                    n_clusters=cluster_count,
                    batch_size=min(16384, max(cluster_count * 4, 1024)),
                    n_init=1,
                    max_iter=100,
                    reassignment_ratio=0.01,
                    random_state=self.config.random_state + stage,
                ).fit(residual)
                stages.append(model)
                residual = residual - model.cluster_centers_[model.predict(residual)]
            self.models[key] = _PopulationModel(mean, scale, stages, int(values.shape[0]))
        if required_rates is not None:
            unavailable = []
            for rate in sorted(required_rates):
                active = int(self.rate_choices[rate])
                lengths = {min(self.config.block_size, active - start) for start in range(0, active, self.config.block_size)}
                unavailable.extend(
                    (rate, length) for length in lengths
                    if self._model_key(rate, length) not in self.models
                )
            if unavailable:
                raise ValueError(
                    f"Insufficient calibration samples for K={self.config.codebook_size} populations: {unavailable}"
                )
        return self

    def _quantize(self, block: np.ndarray, key: tuple[int, int]) -> tuple[list[int], np.ndarray]:
        if key not in self.models:
            raise KeyError(f"No exact rate/dimensionality codebook for {key}")
        population = self.models[key]
        normalized = (np.asarray(block, np.float32) - population.mean) / population.scale
        residual = normalized.reshape(1, -1)
        reconstructed = np.zeros_like(residual)
        indices = []
        for model in population.stages:
            index = int(model.predict(residual)[0])
            indices.append(index)
            centroid = model.cluster_centers_[index:index + 1]
            reconstructed += centroid
            residual -= centroid
        return indices, (reconstructed[0] * population.scale + population.mean).astype(np.float32)

    def encode(self, latent: torch.Tensor, rate_indexes: torch.Tensor) -> RateConditionedPacket:
        tokens, shape = _tokens(latent)
        rates = rate_indexes.detach().cpu().numpy().reshape(-1).astype(np.int64)
        index_streams = [[] for _ in range(self.config.rvq_stages)]
        keys: list[tuple[int, int]] = []
        decoded = np.zeros_like(tokens)
        squared_error = 0.0
        active_values = 0
        for token_number, (token, rate) in enumerate(zip(tokens, rates)):
            active = int(self.rate_choices[rate])
            for start in range(0, active, self.config.block_size):
                length = min(self.config.block_size, active - start)
                key = self._model_key(int(rate), int(length))
                indices, reconstruction = self._quantize(token[start:start + length], key)
                keys.append(key)
                for stage, index in enumerate(indices):
                    index_streams[stage].append(index)
                decoded[token_number, start:start + length] = reconstruction
                difference = token[start:start + length] - reconstruction
                squared_error += float(np.dot(difference, difference))
                active_values += length

        streams = [np.asarray(values, dtype=np.uint16) for values in index_streams]
        rate_width = math.ceil(math.log2(len(self.rate_choices)))
        content = np.concatenate([
            *[_pack_uint(values, self.config.index_bits) for values in streams],
            _pack_uint(rates, rate_width),
            np.zeros(self.METADATA_BITS, dtype=np.uint8),
        ])
        bits = np.concatenate([content, _crc32(content)])
        return RateConditionedPacket(
            bits=bits,
            latent_shape=shape,
            rate_indexes=rates,
            index_streams=streams,
            keys=keys,
            active_value_count=active_values,
            latent_mse=float(squared_error / active_values),
            index_bits=self.config.index_bits * sum(len(stream) for stream in streams),
            rate_bits=rate_width * rates.size,
            metadata_bits=self.METADATA_BITS,
            crc_bits=self.CRC_BITS,
        )

    def decode(self, packet: RateConditionedPacket, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        content, crc = packet.bits[:-self.CRC_BITS], packet.bits[-self.CRC_BITS:]
        if not np.array_equal(_crc32(content), crc):
            raise ValueError("Rate-conditioned packet CRC mismatch")
        index_count = len(packet.keys)
        offset = 0
        streams = []
        for _ in range(self.config.rvq_stages):
            end = offset + index_count * self.config.index_bits
            streams.append(_unpack_uint(content[offset:end], self.config.index_bits).astype(np.int64))
            offset = end
        rate_width = math.ceil(math.log2(len(self.rate_choices)))
        rate_end = offset + packet.rate_indexes.size * rate_width
        rates_from_bits = _unpack_uint(content[offset:rate_end], rate_width).astype(np.int64)
        if not np.array_equal(rates_from_bits, packet.rate_indexes):
            raise ValueError("Serialized rate map disagrees with packet metadata")
        b, channels, h, w = packet.latent_shape
        tokens = np.zeros((b * h * w, channels), dtype=np.float32)
        cursors = [0] * self.config.rvq_stages
        key_cursor = 0
        for token_number, rate in enumerate(rates_from_bits):
            active = int(self.rate_choices[rate])
            for start in range(0, active, self.config.block_size):
                length = min(self.config.block_size, active - start)
                key = packet.keys[key_cursor]
                if key != self._model_key(int(rate), int(length)):
                    raise ValueError("Packet block key disagrees with decoded rate map")
                population = self.models[key]
                normalized = np.zeros(length, dtype=np.float32)
                for stage, model in enumerate(population.stages):
                    index = int(streams[stage][cursors[stage]])
                    cursors[stage] += 1
                    normalized += model.cluster_centers_[index]
                tokens[token_number, start:start + length] = normalized * population.scale + population.mean
                key_cursor += 1
        tensor = torch.from_numpy(tokens.reshape(b, h, w, channels)).permute(0, 3, 1, 2).to(device)
        rates = torch.from_numpy(rates_from_bits.copy()).long().to(device)
        return tensor, rates

    def model_hash(self) -> str:
        digest = hashlib.sha256()
        for key in sorted(self.models):
            model = self.models[key]
            digest.update(np.asarray(key, dtype=np.int64).tobytes())
            digest.update(model.mean.tobytes())
            digest.update(model.scale.tobytes())
            for stage in model.stages:
                digest.update(np.asarray(stage.cluster_centers_, np.float32).tobytes())
        return digest.hexdigest()

    def diagnostics(self) -> dict:
        return {
            "configuration": self.config.label,
            "rate_groups": self.rate_groups,
            "population_count": len(self.models),
            "population_samples": {f"rate{r}_dim{d}": m.sample_count for (r, d), m in sorted(self.models.items())},
            "population_clusters": {
                f"rate{r}_dim{d}": [int(stage.n_clusters) for stage in m.stages]
                for (r, d), m in sorted(self.models.items())
            },
            "model_sha256": self.model_hash(),
        }
