"""Logical semantic-header sidecar paired with existing Stage 1A payload artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from INFRA.Artifacts.latent_artifact import LatentArtifact
from INFRA.Artifacts.semantic_params_artifact import SemanticParameterBundle
from INFRA.Artifacts.transmission_artifact import TransmissionArtifact


@dataclass(slots=True)
class SemanticPacketArtifact:
    """A semantic sidecar; it does not alter Stage 1A latent or VQ packet framing."""

    header_bytes: bytes
    parameter_bundle: SemanticParameterBundle
    native_latent: LatentArtifact | None = None
    communication_vq_packet: TransmissionArtifact | None = None
    header_schema_version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def semantic_header_bits(self) -> int:
        return len(self.header_bytes) * 8
