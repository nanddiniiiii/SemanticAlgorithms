"""Encoder-output runtime artifact."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class LatentArtifact:
    """Latent tensor and its rate-allocation mask without transport assumptions."""

    tensor: Any
    rate_mask: Any | None
    source_model: str
    rate_tokens: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def active_real_values(self) -> int:
        if self.rate_mask is None:
            return int(self.tensor[0].numel())
        return int(self.rate_mask[0].sum().item())
