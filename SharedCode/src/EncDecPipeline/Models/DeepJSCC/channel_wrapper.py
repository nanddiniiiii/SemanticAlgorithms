"""DeepJSCC channel adapter using the repository's canonical AWGN semantics."""

from __future__ import annotations

from Channels.awgn import AWGNChannel
from INFRA.Artifacts import LatentArtifact, TransmissionArtifact
from INFRA.Interfaces import ChannelInterface


class DeepJSCCChannelWrapper(ChannelInterface):
    """Delegate I/Q pairing, power normalization and Es/N0 AWGN to AWGNChannel.

    Channel uses are complex symbols.  Pairing is performed independently for
    every image, so values from two different batch items can never form one
    complex symbol.
    """

    def __init__(self, channel: ChannelInterface | None = None):
        self.channel = channel or AWGNChannel()

    def transmit(
        self, latent: LatentArtifact, snr_db: float, **options: object
    ) -> TransmissionArtifact:
        transmission = self.channel.transmit(latent, snr_db, **options)
        transmission.channel_name = "deepjscc_awgn"
        transmission.metadata = {
            **transmission.metadata,
            "model": "DeepJSCC",
            "channel_use_unit": "complex_symbol",
        }
        return transmission
