"""AWGN channel whose complex conversion is DataParallel-safe."""

from dataclasses import dataclass
from typing import Any

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
        normalized_symbols, scale = normalize_complex_per_image(complex_symbols)
        noise_variance = esn0_to_noise_variance(snr_db)
        standard_deviation = (noise_variance / 2.0) ** 0.5
        noise = torch.complex(
            torch.randn_like(normalized_symbols.real) * standard_deviation,
            torch.randn_like(normalized_symbols.imag) * standard_deviation,
        )
        received_symbols = normalized_symbols + noise
        received_real = complex_to_real_per_image(received_symbols * scale, original_shape)

        if latent.rate_mask is not None:
            active_per_image = latent.rate_mask.reshape(latent.rate_mask.shape[0], -1).sum(dim=1)
            channel_uses = [int(value.item()) // 2 for value in active_per_image]
        else:
            channel_uses = [complex_symbols.shape[1]] * int(clean.shape[0])

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
            },
        )
