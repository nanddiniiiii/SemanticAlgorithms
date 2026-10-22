"""Named semantic boundary for consumers that need latent tensors only."""

from INFRA.Artifacts import ImageArtifact, LatentArtifact


def extract_latent(adapter: object, image: ImageArtifact, snr_db: float, rate: int) -> LatentArtifact:
    return adapter.encode(image, snr_db=snr_db, rate=rate)
