"""Storage diagnostic for a float latent; explicitly not an over-air bit count."""

from INFRA.Artifacts import LatentArtifact


def fp32_latent_storage_bits(latent: LatentArtifact) -> int:
    return int(latent.tensor[0].numel()) * 32
