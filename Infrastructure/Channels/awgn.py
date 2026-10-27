"""AWGN channel whose complex conversion is DataParallel-safe."""

from dataclasses import dataclass
from typing import Any

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
class AWGNChannel(ChannelInterface):
    """Unit-energy complex AWGN channel using Es/N0 in dB.

    `rate_mask` is applied before channel use counting. The values themselves are
    retained as a dense tensor because the SwinJSCC decoder needs its expected shape.
    """

    name: str = "awgn"

    def transmit(self, latent: LatentArtifact, snr_db: float, **options: object) -> TransmissionArtifact:
        torch = require_torch()
        clean = latent.tensor
        if latent.rate_mask is not None:
            clean = clean * latent.rate_mask

        complex_symbols, original_shape = real_to_complex_per_image(clean)
        channel_uses = active_symbol_counts(latent, complex_symbols)
        normalized_symbols, scale = normalize_complex_per_image(complex_symbols, channel_uses)
        noise_variance = esn0_to_noise_variance(snr_db)
        standard_deviation = (noise_variance / 2.0) ** 0.5
        noise = torch.complex(
            torch.randn_like(normalized_symbols.real) * standard_deviation,
            torch.randn_like(normalized_symbols.imag) * standard_deviation,
        )
        received_symbols = normalized_symbols + noise
        received_real = complex_to_real_per_image(received_symbols * scale if options.get("restore_scale", True) else received_symbols, original_shape)

        if latent.rate_mask is not None:
            received_real = received_real * latent.rate_mask

        return TransmissionArtifact(
            transmitted=clean,
            received=received_real,
            channel_name=self.name,
            snr_db=float(snr_db),
            channel_uses_per_image=channel_uses,
            metadata={
                "snr_definition": "Es/N0_dB",
                "noise_variance_n0": noise_variance,
                "per_image_complex_mapping": True,
                "restore_scale": bool(options.get("restore_scale", True)),
            },
        )
