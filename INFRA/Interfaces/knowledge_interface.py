"""Contract for historical/context evidence stores."""

from abc import ABC, abstractmethod

from INFRA.Artifacts import KnowledgeArtifact, ObservationArtifact, SemanticParameterBundle


class KnowledgeInterface(ABC):
    @abstractmethod
    def query(self, observation: ObservationArtifact) -> KnowledgeArtifact:
        raise NotImplementedError

    @abstractmethod
    def record(self, observation: ObservationArtifact, parameters: SemanticParameterBundle) -> None:
        raise NotImplementedError
