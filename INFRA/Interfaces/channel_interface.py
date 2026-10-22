"""Channel contract; implementations belong to the top-level Channels package."""

from abc import ABC, abstractmethod

from INFRA.Artifacts import LatentArtifact, TransmissionArtifact


class ChannelInterface(ABC):
    @abstractmethod
    def transmit(self, latent: LatentArtifact, snr_db: float, **options: object) -> TransmissionArtifact:
        raise NotImplementedError
