"""Contract for future scheduling decisions."""

from abc import ABC, abstractmethod

from INFRA.Artifacts import KnowledgeArtifact, SchedulingArtifact, SemanticParametersArtifact


class SchedulerInterface(ABC):
    @abstractmethod
    def schedule(self, semantic: SemanticParametersArtifact, knowledge: KnowledgeArtifact) -> SchedulingArtifact:
        raise NotImplementedError
