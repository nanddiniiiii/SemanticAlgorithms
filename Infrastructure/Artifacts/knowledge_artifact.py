"""Historical and contextual semantic evidence artifact."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class KnowledgeArtifact:
    evidence: dict[str, Any]
    knowledge_source: str
    metadata: dict[str, Any] = field(default_factory=dict)
