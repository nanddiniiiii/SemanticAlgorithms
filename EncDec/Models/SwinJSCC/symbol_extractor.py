"""Per-image native channel-use accounting for a SwinJSCC latent."""

from INFRA.Artifacts import LatentArtifact


def native_channel_uses_per_image(latent: LatentArtifact) -> list[int]:
    if latent.rate_mask is None:
        uses = int(latent.tensor[0].numel()) // 2
    else:
        uses = int(latent.rate_mask[0].sum().item()) // 2
    return [uses] * int(latent.tensor.shape[0])
