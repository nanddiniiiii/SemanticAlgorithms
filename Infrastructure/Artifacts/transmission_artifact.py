"""Artifact crossing a communication channel."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TransmissionArtifact:
    """Channel input/output and accounting for one independently mapped batch."""

    transmitted: Any
    received: Any
    channel_name: str
    snr_db: float
    channel_uses_per_image: list[int]
    payload_bits_per_image: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def channel_uses_total(self) -> int:
        return sum(self.channel_uses_per_image)
