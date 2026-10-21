"""Rician support is deliberately deferred until its CSI assumptions are specified."""

from INFRA.Artifacts import LatentArtifact, TransmissionArtifact
from INFRA.Interfaces import ChannelInterface
from INFRA.errors import ComponentUnavailableError


class RicianChannel(ChannelInterface):
    def transmit(self, latent: LatentArtifact, snr_db: float, **options: object) -> TransmissionArtifact:
        raise ComponentUnavailableError(
            component="RicianChannel",
            stage="Stage 1A",
            reason="Rician K-factor and receiver CSI assumptions are not locked.",
        )
