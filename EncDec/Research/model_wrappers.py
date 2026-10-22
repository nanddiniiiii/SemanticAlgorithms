"""Checkpoint-strict wrappers around untouched official repositories."""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .compressai_1_2_compat import install_compressai_1_2_compat


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_checkpoint(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    regenerated_buffer_predicate=None,
) -> dict:
    expected = model.state_dict()
    predicate = regenerated_buffer_predicate or (lambda _key: False)
    skipped = sorted(key for key in set(expected) | set(state) if predicate(key))
    expected_keys = {key for key in expected if not predicate(key)}
    state_keys = {key for key in state if not predicate(key)}
    missing = sorted(expected_keys - state_keys)
    unexpected = sorted(state_keys - expected_keys)
    shape_mismatches = {
        key: {"model": list(expected[key].shape), "checkpoint": list(state[key].shape)}
        for key in sorted(expected_keys & state_keys)
        if tuple(expected[key].shape) != tuple(state[key].shape)
    }
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            f"Checkpoint validation failed: missing={missing}, unexpected={unexpected}, "
            f"shape_mismatches={shape_mismatches}"
        )
    # Replace only officially regenerated buffers with this instance's values,
    # then retain a genuinely strict load for the complete model state.
    strict_state = {key: value for key, value in state.items() if not predicate(key)}
    strict_state.update({key: expected[key] for key in expected if predicate(key)})
    model.load_state_dict(strict_state, strict=True)
    return {
        "checkpoint_key_count": len(state),
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": {},
        "skipped_regenerated_buffers": skipped,
        "strict_load": True,
    }


@contextlib.contextmanager
def _source_import(path: Path, clear_prefixes: tuple[str, ...]):
    old_path = list(sys.path)
    old_cwd = Path.cwd()
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in clear_prefixes):
            sys.modules.pop(name, None)
    sys.path.insert(0, str(path))
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


@dataclass
class DeepTrial:
    reconstruction: torch.Tensor
    latent_shapes: list[tuple[int, ...]]
    latent_real_values: int
    tile_count: int


class DeepJSCCAdapter:
    source_commit = "e1cef6aeadedde9ff359760936a2ddf2540dd821"
    checkpoint_training_snr_db = 200.0

    def __init__(self, source_dir: Path, checkpoint: Path, device: torch.device, evaluation_snr_db: float = 10.0):
        self.source_dir = Path(source_dir)
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.evaluation_snr_db = float(evaluation_snr_db)
        with _source_import(self.source_dir, ("model", "channel")):
            module = importlib.import_module("model")
            model = module.DeepJSCC(c=19, channel_type="AWGN", snr=self.evaluation_snr_db)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self.validation = validate_checkpoint(model, state)
        self.validation.update({"checkpoint_sha256": sha256(self.checkpoint), "source_commit": self.source_commit})
        self.model = model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _pad(x: torch.Tensor, tile: int) -> tuple[torch.Tensor, tuple[int, int]]:
        h, w = x.shape[-2:]
        pad_h, pad_w = (-h) % tile, (-w) % tile
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), (h, w)

    def encode_tiles(self, x: torch.Tensor, tile: int = 128):
        padded, original_hw = self._pad(x, tile)
        records = []
        with torch.inference_mode():
            for top in range(0, padded.shape[-2], tile):
                for left in range(0, padded.shape[-1], tile):
                    patch = padded[:, :, top:top + tile, left:left + tile]
                    records.append((top, left, self.model.encoder(patch)))
        return padded, original_hw, records

    def run_once(self, x: torch.Tensor, seed: int, tile: int = 128) -> DeepTrial:
        seed_everything(seed)
        padded, (h, w), records = self.encode_tiles(x, tile)
        return self.run_from_records(padded, (h, w), records, seed, tile)

    def run_from_records(self, padded, original_hw, records, seed: int, tile: int = 128) -> DeepTrial:
        seed_everything(seed)
        h, w = original_hw
        reconstructed = torch.empty_like(padded)
        with torch.inference_mode():
            for top, left, latent in records:
                received = self.model.channel(latent)
                reconstructed[:, :, top:top + tile, left:left + tile] = self.model.decoder(received).clamp(0, 1)
        return DeepTrial(
            reconstruction=reconstructed[:, :, :h, :w],
            latent_shapes=[tuple(latent.shape) for _, _, latent in records],
            latent_real_values=sum(int(latent.numel()) for _, _, latent in records),
            tile_count=len(records),
        )


@dataclass
class NTSCCEncoding:
    y: torch.Tensor
    s_masked: torch.Tensor
    mask: torch.Tensor
    rate_indexes: torch.Tensor
    bpp_y: float
    bpp_z: float


@dataclass
class NTSCCTrial:
    reconstruction: torch.Tensor
    encoding: NTSCCEncoding
    complex_channel_uses: int
    cbr_y: float


class NTSCCWindowsAdapter:
    source_commit = "dd3c745374023af6d01f465c379122be785d936f"
    checkpoint_training_snr_db = 10.0

    def __init__(self, source_dir: Path, checkpoint: Path, device: torch.device, evaluation_snr_db: float = 10.0):
        install_compressai_1_2_compat()
        self.source_dir = Path(source_dir)
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.evaluation_snr_db = float(evaluation_snr_db)
        prefixes = ("config", "net", "loss", "layer", "channel", "utils")
        with _source_import(self.source_dir, prefixes):
            config_module = importlib.import_module("config")
            layers_module = importlib.import_module("layer.layers")
            encoder_module = importlib.import_module("layer.jscc_encoder")

            def device_safe_attention_forward(attention, x, add_token=True, mask=None):
                batch_windows, tokens, channels = x.shape
                qkv = attention.qkv(x).reshape(
                    batch_windows, tokens, 3, attention.num_heads, channels // attention.num_heads
                ).permute(2, 0, 3, 1, 4)
                query, key, value = qkv[0], qkv[1], qkv[2]
                scores = (query * attention.scale) @ key.transpose(-2, -1)
                bias = attention.relative_position_bias_table[
                    attention.relative_position_index.view(-1)
                ].view(
                    attention.window_size[0] * attention.window_size[1],
                    attention.window_size[0] * attention.window_size[1],
                    -1,
                ).permute(2, 0, 1).contiguous()
                if add_token:
                    scores[:, :, 2:, 2:] = scores[:, :, 2:, 2:] + bias.unsqueeze(0)
                else:
                    scores = scores + bias.unsqueeze(0)
                if mask is not None:
                    if add_token:
                        mask = F.pad(mask, (2, 0, 2, 0), "constant", 0)
                    mask = mask.to(scores.device)
                    window_count = mask.shape[0]
                    scores = scores.view(
                        batch_windows // window_count,
                        window_count,
                        attention.num_heads,
                        tokens,
                        tokens,
                    ) + mask.unsqueeze(1).unsqueeze(0)
                    scores = scores.view(-1, attention.num_heads, tokens, tokens)
                probabilities = attention.attn_drop(attention.softmax(scores))
                output = (probabilities @ value).transpose(1, 2).reshape(batch_windows, tokens, channels)
                return attention.proj_drop(attention.proj(output))

            def device_safe_rate_update(rate_adapter, height, width, _device):
                rate_adapter.H = height
                rate_adapter.W = width
                rate_adapter.num_patches = height * width
                actual_device = next(rate_adapter.parameters()).device
                rate_adapter.mask = torch.arange(
                    0, max(rate_adapter.rate_choice), device=actual_device
                ).repeat(rate_adapter.num_patches, 1)

            layers_module.WindowAttention.forward = device_safe_attention_forward
            encoder_module.RateAdaptionEncoder.update_resolution = device_safe_rate_update

            def device_safe_update_mask(block):
                if block.shift_size <= 0:
                    return
                height, width = block.input_resolution
                height += (block.window_size - height % block.window_size) % block.window_size
                width += (block.window_size - width % block.window_size) % block.window_size
                device_for_mask = next(block.parameters()).device
                image_mask = torch.zeros((1, height, width, 1), device=device_for_mask)
                h_slices = (
                    slice(0, -block.window_size),
                    slice(-block.window_size, -block.shift_size),
                    slice(-block.shift_size, None),
                )
                w_slices = h_slices
                count = 0
                for h_slice in h_slices:
                    for w_slice in w_slices:
                        image_mask[:, h_slice, w_slice, :] = count
                        count += 1
                windows = layers_module.window_partition(image_mask, block.window_size)
                windows = windows.view(-1, block.window_size * block.window_size)
                mask = windows.unsqueeze(1) - windows.unsqueeze(2)
                block.attn_mask = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

            # Official code hard-codes .cuda() only while regenerating a
            # resolution-dependent mask. The equations are unchanged.
            layers_module.SwinTransformerBlock.update_mask = device_safe_update_mask
            model_module = importlib.import_module("net.NTSCC_Hyperior")
            # Use an instance so the cloned official module/class is never
            # mutated. The checkpoint's learned relative-position tensors
            # prove the repository's default 256x256 construction config.
            config = config_module.config()
            config.ga_kwargs = dict(config.ga_kwargs)
            config.gs_kwargs = dict(config.gs_kwargs)
            config.fe_kwargs = dict(config.fe_kwargs)
            config.fd_kwargs = dict(config.fd_kwargs)
            config.device = str(device)
            config.channel = dict(config.channel)
            config.channel["chan_param"] = self.evaluation_snr_db
            model = model_module.NTSCC_Hyperprior(config)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self.validation = validate_checkpoint(
            model,
            state,
            regenerated_buffer_predicate=lambda key: key.endswith("attn_mask"),
        )
        self.validation.update({"checkpoint_sha256": sha256(self.checkpoint), "source_commit": self.source_commit})
        self.config = config
        self.model = model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        # Official CUDA-era code passes integer device=-1 on CPU.  This wrapper
        # changes only device selection; the two independent Gaussian draws and
        # their standard deviation remain identical.
        def cpu_safe_noise(input_layer: torch.Tensor, std: float):
            real = torch.normal(0.0, std, size=input_layer.shape, device=input_layer.device)
            imag = torch.normal(0.0, std, size=input_layer.shape, device=input_layer.device)
            return input_layer + (real + 1j * imag)

        self.model.channel.gaussian_noise_layer = cpu_safe_noise

    def encode(self, x: torch.Tensor) -> NTSCCEncoding:
        with torch.inference_mode():
            values = self.model.forward_NTC(x, require_probs=True)
            _, bpp_y, bpp_z, _, y, _, scales_hat, means_hat = values
            probabilities = self.model.feature_probs_based_Gaussian(y, means_hat, scales_hat)
            s_masked, mask, indexes = self.model.fe(y, probabilities.detach(), eta=self.model.eta)
        return NTSCCEncoding(
            y=y,
            s_masked=s_masked,
            mask=mask.bool(),
            rate_indexes=indexes,
            bpp_y=float(bpp_y.detach().cpu()),
            bpp_z=float(bpp_z.detach().cpu()),
        )

    def decode_channel_latent(self, s_hat: torch.Tensor, rate_indexes: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            # ``forward`` normally performs this update before invoking fd/gs.
            # Codebook decoding deliberately enters at the channel latent, so
            # restore the official resolution update explicitly.  This matters
            # after fitting on 128x128 calibration images and then decoding the
            # full-resolution test latent.
            self.model.update_resolution(s_hat.shape[-2] * 16, s_hat.shape[-1] * 16)
            y_hat = self.model.fd(s_hat, rate_indexes)
            return self.model.gs(y_hat).clamp(0, 1)

    def run_once(self, x: torch.Tensor, seed: int) -> NTSCCTrial:
        seed_everything(seed)
        encoding = self.encode(x)
        return self.run_from_encoding(x, encoding, seed)

    def run_from_encoding(self, x: torch.Tensor, encoding: NTSCCEncoding, seed: int) -> NTSCCTrial:
        seed_everything(seed)
        selected = torch.masked_select(encoding.s_masked, encoding.mask)
        if selected.numel() % 2:
            raise RuntimeError("Official NTSCC channel requires an even number of selected real symbols")
        with torch.inference_mode():
            received, uses = self.model.channel.forward(selected)
            s_hat = torch.zeros_like(encoding.s_masked)
            s_hat[encoding.mask] = received
            reconstruction = self.decode_channel_latent(s_hat, encoding.rate_indexes)
        source_rgb_values = int(x.numel())
        return NTSCCTrial(
            reconstruction=reconstruction,
            encoding=encoding,
            complex_channel_uses=int(uses),
            cbr_y=float(uses / source_rgb_values),
        )
