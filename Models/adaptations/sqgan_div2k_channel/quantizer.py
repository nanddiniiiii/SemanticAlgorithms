"""Wrapper that inserts a differentiable noisy channel after author SQ-GAN VQ."""
from __future__ import annotations

from torch import nn
from utils.utils import instantiate_from_config
from adaptations.sqgan_div2k_channel.channel import ComplexLatentChannel


class ChannelizedVectorQuantizer(nn.Module):
    def __init__(self, quantizer_config: dict, channel: str, snr_db: float = 10.0, perfect_csi: bool = True):
        super().__init__()
        self.quantizer = instantiate_from_config(quantizer_config)
        self.channel = ComplexLatentChannel(channel, snr_db, perfect_csi)

    def forward(self, *args, **kwargs):
        quantized, commitment_loss, info = self.quantizer(*args, **kwargs)
        return self.channel(quantized), commitment_loss, info
