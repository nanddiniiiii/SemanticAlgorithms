"""Input-image runtime artifact."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ImageArtifact:
    """A batched image tensor plus provenance that must survive a pipeline run."""

    tensor: Any
    sample_ids: list[str]
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def batch_size(self) -> int:
        return int(self.tensor.shape[0])

    def spatial_shape(self) -> tuple[int, int]:
        return int(self.tensor.shape[-2]), int(self.tensor.shape[-1])
