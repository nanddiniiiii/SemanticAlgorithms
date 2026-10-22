"""Explicit decoder-only inference facade."""

from INFRA.Artifacts import ImageArtifact, LatentArtifact


class DecoderRuntime:
    def __init__(self, adapter: object) -> None:
        self.adapter = adapter

    def __call__(self, latent: LatentArtifact, snr_db: float) -> ImageArtifact:
        return self.adapter.decode(latent, snr_db=snr_db)
