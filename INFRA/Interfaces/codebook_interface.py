"""Contract for vector quantization without coupling the encoder to K-means."""

from abc import ABC, abstractmethod

from INFRA.Artifacts import LatentArtifact, TransmissionArtifact


class CodebookInterface(ABC):
    @abstractmethod
    def fit(self, latent_batches: object) -> None:
        raise NotImplementedError

    @abstractmethod
    def encode(self, latent: LatentArtifact) -> TransmissionArtifact:
        raise NotImplementedError

    @abstractmethod
    def decode(self, packet: TransmissionArtifact, template: LatentArtifact) -> LatentArtifact:
        raise NotImplementedError
