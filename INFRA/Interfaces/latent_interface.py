"""Minimal behavior expected from a latent transformer such as a codebook."""

from abc import ABC, abstractmethod

from INFRA.Artifacts import LatentArtifact


class LatentTransformInterface(ABC):
    @abstractmethod
    def transform(self, latent: LatentArtifact) -> LatentArtifact:
        raise NotImplementedError
