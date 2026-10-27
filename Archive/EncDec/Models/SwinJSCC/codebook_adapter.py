"""Stage 1A K=256 codebook wrapper for a manifest-loaded SwinJSCC model."""

from __future__ import annotations

from typing import Any

from Archive.EncDec.Codebooks.minibatch_kmeans_vq import MiniBatchKMeansVQ
from Infrastructure.Artifacts import ImageArtifact, LatentArtifact, TransmissionArtifact
from Infrastructure.Interfaces import EncDecInterface


class SwinJSCCCodebookAdapter(EncDecInterface):
    """Quantize SwinJSCC latent tokens before the normal channel/decoder path.

    The codebook is shared receiver state. Each encoded latent records its exact
    fixed-width index packet size in metadata; the benchmark reports that value
    as the per-image payload while the caller remains responsible for channel
    selection.
    """

    def __init__(self, base_adapter: Any, codebook: MiniBatchKMeansVQ) -> None:
        self.base_adapter = base_adapter
        self.codebook = codebook
        self.encoder = base_adapter.encoder
        self.decoder = base_adapter.decoder

    def encode(self, image: ImageArtifact, **options: object) -> LatentArtifact:
        continuous_latent = self.base_adapter.encode(image, **options)
        packet = self.codebook.encode(continuous_latent)
        quantized_latent = self.codebook.decode(packet, continuous_latent)
        quantized_latent.metadata = {
            **quantized_latent.metadata,
            "codebook_id": self.codebook.codebook_id,
            "codebook_payload_bits": int(packet.payload_bits_per_image[0]),
            "codebook_packet": packet.metadata,
        }
        return quantized_latent

    def decode(self, latent: LatentArtifact, **options: object) -> ImageArtifact:
        return self.base_adapter.decode(latent, **options)

    def reconstruct_received(self, transmission: TransmissionArtifact, latent: LatentArtifact) -> ImageArtifact:
        return self.base_adapter.reconstruct_received(transmission, latent)

    def codebook_storage_bytes(self) -> int:
        return int(self.codebook.model.cluster_centers_.nbytes)
