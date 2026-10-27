"""Explicit Rayleigh fading with perfect-CSI zero-forcing equalization."""

from dataclasses import dataclass

from Infrastructure.Channels.channel_utils import (
    active_symbol_counts,
    complex_to_real_per_image,
    esn0_to_noise_variance,
    normalize_complex_per_image,
    real_to_complex_per_image,
    require_torch,
)
from Infrastructure.Artifacts import LatentArtifact, TransmissionArtifact
from Infrastructure.Interfaces import ChannelInterface


@dataclass(slots=True)
class RayleighChannel(ChannelInterface):
    """Perfect-CSI flat Rayleigh channel; not part of the locked Stage 1A results."""

    name: str = "rayleigh_perfect_csi"
    coherence: str = "symbol"

    def __post_init__(self):
        if self.coherence not in {'symbol', 'image'}:
            raise ValueError('Rayleigh coherence must be symbol or image')

    def transmit(self, latent: LatentArtifact, snr_db: float, **options: object) -> TransmissionArtifact:
        torch = require_torch()
        clean = latent.tensor if latent.rate_mask is None else latent.tensor * latent.rate_mask
        complex_symbols, original_shape = real_to_complex_per_image(clean)
        channel_uses = active_symbol_counts(latent, complex_symbols)
        normalized, scale = normalize_complex_per_image(complex_symbols, channel_uses)
        shape = normalized.real if self.coherence == 'symbol' else normalized.real[:, :1]
        fading = torch.complex(torch.randn_like(shape), torch.randn_like(shape)) / (2.0**0.5)
        standard_deviation = (esn0_to_noise_variance(snr_db) / 2.0) ** 0.5
        noise = torch.complex(
            torch.randn_like(normalized.real) * standard_deviation,
            torch.randn_like(normalized.imag) * standard_deviation,
        )
        equalized = (fading * normalized + noise) / fading
        received = complex_to_real_per_image(equalized * scale if options.get("restore_scale", True) else equalized, original_shape)
        if latent.rate_mask is not None:
            received = received * latent.rate_mask
        return TransmissionArtifact(
            transmitted=clean,
            received=received,
            channel_name=self.name,
            snr_db=float(snr_db),
            channel_uses_per_image=channel_uses,
            metadata={"snr_definition": "Es/N0_dB", "receiver": "perfect_csi_zero_forcing",
                      "coherence": self.coherence, "fading_distribution": "CN(0,1)",
                      "restore_scale": bool(options.get("restore_scale", True)),
                      "deep_fade_clipping": False},
        )
