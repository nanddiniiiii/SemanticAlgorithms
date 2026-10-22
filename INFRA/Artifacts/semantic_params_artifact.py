"""Typed semantic-parameter artifacts shared by every Stage 1B extractor.

The communication latent remains unchanged. These objects hold the rich,
host-side evidence needed to build a compact semantic-header sidecar later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from numbers import Real
from typing import Any, Mapping


class ParameterProducerLevel(str, Enum):
    """Where a semantic value was produced or is intended to be produced."""

    UE = "ue"
    ORAN = "oran"
    EXTERNAL = "external"


def _json_safe(value: Any) -> Any:
    """Convert common provenance values without making NumPy a core dependency."""

    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return _json_safe(to_list())
    if isinstance(value, Real):
        return float(value)
    return value


@dataclass(frozen=True, slots=True)
class ParameterDefinition:
    """Stable catalog entry for a configurable semantic-header field."""

    name: str
    producer_level: ParameterProducerLevel
    units: str | None
    header_encoding: str
    description: str


@dataclass(frozen=True, slots=True)
class ParameterValue:
    """One auditable semantic value.

    ``value`` is deliberately ``None`` for unavailable fields. A zero would
    be ambiguous: it could mean a measured zero or a fabricated placeholder.
    """

    value: float | None
    raw_value: Any | None
    available: bool
    method: str
    source: str
    version: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    units: str | None = None
    producer_level: ParameterProducerLevel = ParameterProducerLevel.UE

    def __post_init__(self) -> None:
        if self.available and self.value is None:
            raise ValueError("An available ParameterValue must have a numeric value.")
        if not self.available and self.value is not None:
            raise ValueError("An unavailable ParameterValue must not carry a numeric value.")
        if self.value is not None and not isinstance(self.value, Real):
            raise TypeError("ParameterValue.value must be numeric or None.")
        if not self.method or not self.source or not self.version:
            raise ValueError("ParameterValue method, source, and version are required.")

    @classmethod
    def unavailable(
        cls,
        *,
        method: str,
        source: str,
        version: str,
        reason: str,
        units: str | None = None,
        producer_level: ParameterProducerLevel = ParameterProducerLevel.UE,
        provenance: Mapping[str, Any] | None = None,
    ) -> "ParameterValue":
        details = {"reason": reason}
        if provenance:
            details.update(dict(provenance))
        return cls(
            value=None,
            raw_value=None,
            available=False,
            method=method,
            source=source,
            version=version,
            provenance=details,
            units=units,
            producer_level=producer_level,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": float(self.value) if self.value is not None else None,
            "raw_value": _json_safe(self.raw_value),
            "available": self.available,
            "method": self.method,
            "source": self.source,
            "version": self.version,
            "provenance": _json_safe(self.provenance),
            "units": self.units,
            "producer_level": self.producer_level.value,
        }


@dataclass(slots=True)
class SemanticParametersArtifact:
    """Dynamic semantic bundle with host-side provenance for every field."""

    values: dict[str, ParameterValue]
    extractor_name: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "semantic-parameter-bundle/v1"
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.extractor_name:
            raise ValueError("Semantic bundle requires an extractor name.")
        invalid = [name for name, value in self.values.items() if not isinstance(value, ParameterValue)]
        if invalid:
            raise TypeError(f"Semantic bundle values must be ParameterValue instances: {invalid}")

    def get(self, name: str) -> ParameterValue:
        return self.values[name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "extractor_name": self.extractor_name,
            "confidence": self.confidence,
            "generated_at": self.generated_at.isoformat(),
            "metadata": _json_safe(self.metadata),
            "values": {name: value.to_dict() for name, value in self.values.items()},
        }


# This is the Stage 1B public name. Keep the previous artifact name as the
# concrete class so older imports continue to work without an adapter layer.
SemanticParameterBundle = SemanticParametersArtifact
