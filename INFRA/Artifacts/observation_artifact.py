"""Observation and task-context objects shared by semantic extractors."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Mapping

from INFRA.Artifacts.semantic_params_artifact import ParameterValue


def ensure_utc(value: datetime) -> datetime:
    """Reject ambiguous timestamps instead of silently assuming a timezone."""

    if value.tzinfo is None:
        raise ValueError("Semantic timestamps must be timezone-aware.")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class TaskContext:
    """Explicit task policy and metadata; it never invents urgency or priority."""

    task_id: str
    task_text: str
    concept_importance: Mapping[str, float] = field(default_factory=dict)
    deadline_at: datetime | None = None
    urgency_horizon_seconds: float = 60.0
    context_text: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provided_parameters: Mapping[str, ParameterValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.task_text.strip():
            raise ValueError("TaskContext requires non-empty task_id and task_text.")
        if self.urgency_horizon_seconds <= 0:
            raise ValueError("urgency_horizon_seconds must be positive.")
        if self.deadline_at is not None:
            object.__setattr__(self, "deadline_at", ensure_utc(self.deadline_at))
        invalid = {name: weight for name, weight in self.concept_importance.items() if not 0.0 <= weight <= 1.0}
        if invalid:
            raise ValueError(f"Concept importance weights must be in [0, 1]: {invalid}")
        invalid_parameters = [name for name, value in self.provided_parameters.items() if not isinstance(value, ParameterValue)]
        if invalid_parameters:
            raise TypeError(f"Provided semantic values must be ParameterValue instances: {invalid_parameters}")


@dataclass(slots=True)
class ObservationArtifact:
    """One image's reusable semantic evidence, produced before parameter extraction."""

    observation_id: str
    image_hash: str
    source_path: str
    image_embedding: Any
    concept_scores: dict[str, float]
    task_embeddings: dict[str, Any] = field(default_factory=dict)
    captured_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.observation_id or not self.image_hash or not self.source_path:
            raise ValueError("ObservationArtifact requires an id, image hash, and source path.")
        if self.captured_at is not None:
            self.captured_at = ensure_utc(self.captured_at)
