"""Machine-readable evaluation result artifact."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvaluationArtifact:
    metrics: dict[str, float | int | str | None]
    evaluator_name: str
    samples_evaluated: int
    metadata: dict[str, Any] = field(default_factory=dict)
