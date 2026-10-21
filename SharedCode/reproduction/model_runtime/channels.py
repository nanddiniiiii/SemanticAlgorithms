"""Explicit matched transport extension; never presented as author channel code."""
import math
import torch
from torch import nn


class MatchedChannel(nn.Module):
    """Unit complex symbol power; CN(0,10**(-SNR/10)) noise.

    Rayleigh uses ONE CN(0,1) fade per image/codeword, with perfect receiver
    CSI and exact zero forcing. No clipping/outage threshold hides deep fades.
    Real latents are packed as first-half I, second-half Q per image. Decoder
    scale is restored after unit-power transport so author decoder units survive.
    """
    def __init__(self, kind="awgn", snr_db=10):
        super().__init__()
        if kind not in {"awgn", "rayleigh"}:
            raise ValueError("Unknown matched channel")
        self.kind, self.snr_db = kind, snr_db
        self.last_complex_uses = 0
        self.last_gain = None

    def forward(self, features, snr_db=None, avg_pwr=False, gain=None):
        del avg_pwr
        original_shape = features.shape
        if features.ndim < 2:
            raise ValueError("Transport requires a batch dimension")
        flat = features.reshape(features.shape[0], -1)
        was_complex = flat.is_complex()
        if was_complex:
            symbols = flat
        else:
            if flat.shape[1] % 2:
                raise ValueError("An even number of real channel coordinates is required")
            half = flat.shape[1] // 2
            symbols = torch.complex(flat[:, :half], flat[:, half:])
        self.last_complex_uses = symbols.shape[1]
        scale = symbols.abs().square().mean(1, keepdim=True).sqrt().clamp_min(1e-12)
        tx = symbols / scale
        snr = self.snr_db if snr_db is None else snr_db
        std = math.sqrt(10.0 ** (-float(snr) / 10.0) / 2.0)
        noise = torch.complex(torch.randn_like(tx.real), torch.randn_like(tx.real)) * std
        if self.kind == "rayleigh":
            fade = (torch.complex(torch.randn_like(tx.real[:, :1]), torch.randn_like(tx.real[:, :1])) / math.sqrt(2)
                    if gain is None else torch.as_tensor(gain,device=tx.device,dtype=tx.dtype).reshape(-1,1))
            if fade.shape[0] not in {1,tx.shape[0]}:
                raise ValueError("Supplied image fade batch differs from symbols")
            self.last_gain=fade
            rx = (fade * tx + noise) / fade
        else:
            self.last_gain=torch.ones_like(tx[:,:1])
            rx = tx + noise
        decoded = rx * scale
        if not was_complex:
            decoded = torch.cat((decoded.real, decoded.imag), dim=1)
        return decoded.reshape(original_shape)

