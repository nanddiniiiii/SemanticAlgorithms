"""Contract for independent current-input semantic parameter extractors."""

from abc import ABC, abstractmethod

from datetime import datetime

from INFRA.Artifacts import KnowledgeArtifact, ObservationArtifact, ParameterValue, TaskContext


class ParameterExtractorInterface(ABC):
    @abstractmethod
    def extract(
        self,
        observation: ObservationArtifact,
        task: TaskContext,
        knowledge: KnowledgeArtifact,
        now: datetime,
    ) -> ParameterValue:
        raise NotImplementedError
