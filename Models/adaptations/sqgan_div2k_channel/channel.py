from __future__ import annotations

import torch
from torch import nn


class ComplexLatentChannel(nn.Module):
    """Per-example unit-power complex AWGN or flat Rayleigh channel.

    Input follows the author VQ module layout [batch, real_feature_dim, tokens].
    A 256-D real code vector is serialized as 128 complex channel uses.
    This is a DIV2K/AWGN/Rayleigh adaptation, not the original SQ-GAN pipeline.
    """
    def __init__(self, channel: str, snr_db: float = 10.0, perfect_csi: bool = True):
        super().__init__()
        self.channel = channel.lower()
        if self.channel not in {"awgn", "rayleigh"}:
            raise ValueError(f"channel must be awgn or rayleigh, got {channel!r}")
        self.snr_db = float(snr_db)
        self.perfect_csi = bool(perfect_csi)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError(f"Expected [B,C,N] latent tensor, got {tuple(z.shape)}")
        b, c, n = z.shape
        if c % 2:
            raise ValueError(f"Real latent dimension must be even, got {c}")
        pairs = z.transpose(1, 2).contiguous().reshape(b, n, c // 2, 2)
        power = pairs.square().sum(dim=-1).mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        scaled = pairs / power.sqrt().unsqueeze(-1)
        x = torch.complex(scaled[..., 0], scaled[..., 1])
        noise_power = 10.0 ** (-self.snr_db / 10.0)
        sigma = (noise_power / 2.0) ** 0.5
        noise = torch.complex(torch.randn_like(x.real), torch.randn_like(x.real)) * sigma
        if self.channel == "rayleigh":
            h = torch.complex(torch.randn_like(x.real), torch.randn_like(x.real)) / (2.0 ** 0.5)
            y = h * x + noise
            if self.perfect_csi:
                x_hat = y * h.conj() / h.abs().square().clamp_min(1e-8)
            else:
                x_hat = y
        else:
            x_hat = x + noise
        received = torch.stack((x_hat.real, x_hat.imag), dim=-1) * power.sqrt().unsqueeze(-1)
        return received.reshape(b, n, c).transpose(1, 2).contiguous()


class ChannelizedVectorQuantizer(nn.Module):
    """Keep author VQ and insert a differentiable channel after its STE output."""
    def __init__(self, quantizer_config: dict, channel: str, snr_db: float = 10.0,
                 perfect_csi: bool = True):
        super().__init__()
        from utils.utils import instantiate_from_config
        self.quantizer = instantiate_from_config(quantizer_config)
        self.channel = ComplexLatentChannel(channel, snr_db, perfect_csi)

    def forward(self, *args, **kwargs):
        quantized, commitment_loss, info = self.quantizer(*args, **kwargs)
        return self.channel(quantized), commitment_loss, info
