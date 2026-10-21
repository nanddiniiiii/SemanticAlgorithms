"""Explicit encoder-only inference facade."""

from INFRA.Artifacts import ImageArtifact, LatentArtifact


class EncoderRuntime:
    def __init__(self, adapter: object) -> None:
        self.adapter = adapter

    def __call__(self, image: ImageArtifact, snr_db: float, rate: int) -> LatentArtifact:
        return self.adapter.encode(image, snr_db=snr_db, rate=rate)
