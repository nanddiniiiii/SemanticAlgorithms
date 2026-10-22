"""Contract that keeps model internals out of downstream components."""

from abc import ABC, abstractmethod

from INFRA.Artifacts import ImageArtifact, LatentArtifact, TransmissionArtifact


class EncDecInterface(ABC):
    """Communication encoder/decoder contract."""

    @abstractmethod
    def encode(self, image: ImageArtifact, **options: object) -> LatentArtifact:
        raise NotImplementedError

    @abstractmethod
    def decode(self, latent: LatentArtifact, **options: object) -> ImageArtifact:
        raise NotImplementedError

    @abstractmethod
    def reconstruct_received(self, transmission: TransmissionArtifact, latent: LatentArtifact) -> ImageArtifact:
        raise NotImplementedError
