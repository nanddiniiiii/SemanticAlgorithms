"""Flat Rayleigh channel reserved for controlled ablations beyond Stage 1A."""

from dataclasses import dataclass

from Channels.channel_utils import (
    complex_to_real_per_image,
    esn0_to_noise_variance,
    normalize_complex_per_image,
    real_to_complex_per_image,
    require_torch,
)
from INFRA.Artifacts import LatentArtifact, TransmissionArtifact
from INFRA.Interfaces import ChannelInterface


@dataclass(slots=True)
class RayleighChannel(ChannelInterface):
    """Perfect-CSI flat Rayleigh channel; not part of the locked Stage 1A results."""

    name: str = "rayleigh_perfect_csi"

    def transmit(self, latent: LatentArtifact, snr_db: float, **options: object) -> TransmissionArtifact:
        torch = require_torch()
        complex_symbols, original_shape = real_to_complex_per_image(latent.tensor)
        normalized, scale = normalize_complex_per_image(complex_symbols)
        fading = torch.complex(torch.randn_like(normalized.real), torch.randn_like(normalized.imag)) / (2.0**0.5)
        standard_deviation = (esn0_to_noise_variance(snr_db) / 2.0) ** 0.5
        noise = torch.complex(
            torch.randn_like(normalized.real) * standard_deviation,
            torch.randn_like(normalized.imag) * standard_deviation,
        )
        equalized = (fading * normalized + noise) / fading
        received = complex_to_real_per_image(equalized * scale, original_shape)
        return TransmissionArtifact(
            transmitted=latent.tensor,
            received=received,
            channel_name=self.name,
            snr_db=float(snr_db),
            channel_uses_per_image=[complex_symbols.shape[1]] * int(latent.tensor.shape[0]),
            metadata={"snr_definition": "Es/N0_dB", "receiver": "perfect_csi_zero_forcing"},
        )
