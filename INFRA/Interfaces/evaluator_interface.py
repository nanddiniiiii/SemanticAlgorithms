"""Contract for independent evaluation components."""

from abc import ABC, abstractmethod

from INFRA.Artifacts import EvaluationArtifact, ImageArtifact


class EvaluatorInterface(ABC):
    @abstractmethod
    def evaluate(self, reference: ImageArtifact, reconstruction: ImageArtifact) -> EvaluationArtifact:
        raise NotImplementedError
