"""Communication bandwidth ratio based on complex channel uses."""

from INFRA.Artifacts import ImageArtifact, TransmissionArtifact


def native_cbr(image: ImageArtifact, transmission: TransmissionArtifact, index: int = 0) -> float:
    height, width = image.spatial_shape()
    channel_uses = transmission.channel_uses_per_image[index]
    return channel_uses / (3.0 * height * width)
