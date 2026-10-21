"""K=256 MiniBatchKMeans vector quantization with auditable packet accounting."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from EncDecPipeline.Codebooks.codebook_codec import CodebookPacketHeader, encode_codebook_packet_bits, index_width
from INFRA.Artifacts import LatentArtifact, TransmissionArtifact
from INFRA.Interfaces import CodebookInterface
from INFRA.Registries import CODEBOOK_REGISTRY


class _TorchMiniBatchKMeans:
    """Small dependency-free MiniBatch K-means state used by Stage 1A.

    The implementation deliberately exposes the same two attributes the codec
    needs from scikit-learn: ``cluster_centers_`` and ``predict``.  This keeps
    Kaggle's compiled scientific stack out of the training-critical path.
    """

    def __init__(self, centers: Any, counts: Any) -> None:
        self.centers = centers
        self.counts = counts

    @property
    def cluster_centers_(self) -> np.ndarray:
        return self.centers.detach().cpu().numpy()

    def predict(self, tokens: np.ndarray) -> np.ndarray:
        torch = __import__("torch")
        values = torch.as_tensor(tokens, dtype=torch.float32, device=self.centers.device)
        distances = torch.cdist(values, self.centers)
        return distances.argmin(dim=1).cpu().numpy()


class MiniBatchKMeansVQ(CodebookInterface):
    """A shared K=256 codebook over latent tokens, not a fake compressed tensor."""

    def __init__(
        self,
        codebook_size: int = 256,
        batch_size: int = 8192,
        random_state: int = 20260906,
        codebook_id: str = "swinjscc_tx_k256",
    ) -> None:
        if codebook_size != 256:
            raise ValueError("Stage 1A locks the ordinary MiniBatchKMeans codebook at K=256.")
        self.codebook_size = codebook_size
        self.batch_size = batch_size
        self.random_state = random_state
        self.codebook_id = codebook_id
        self.model: Any | None = None

    @staticmethod
    def _tokens(latent: LatentArtifact) -> np.ndarray:
        values = latent.tensor.detach().float().cpu().numpy()
        if values.ndim != 3:
            raise ValueError(f"Expected [batch, tokens, channels] latent, received {values.shape}.")
        if latent.rate_mask is not None:
            values = values * latent.rate_mask.detach().float().cpu().numpy()
        return values.reshape(-1, values.shape[-1])

    def fit(self, latent_batches: Iterable[LatentArtifact]) -> None:
        torch = __import__("torch")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_state)
        model: _TorchMiniBatchKMeans | None = None
        batches_seen = 0
        for latent in latent_batches:
            tokens = self._tokens(latent)
            if tokens.shape[0] < self.codebook_size:
                continue
            values = torch.as_tensor(tokens, dtype=torch.float32, device="cpu")
            if model is None:
                initial_indices = torch.randperm(values.shape[0], generator=generator)[: self.codebook_size]
                model = _TorchMiniBatchKMeans(
                    centers=values[initial_indices].clone(),
                    counts=torch.ones(self.codebook_size, dtype=torch.float32),
                )
            for start in range(0, values.shape[0], self.batch_size):
                batch = values[start : start + self.batch_size]
                nearest = torch.cdist(batch, model.centers).argmin(dim=1)
                sums = torch.zeros_like(model.centers)
                sums.index_add_(0, nearest, batch)
                counts = torch.bincount(nearest, minlength=self.codebook_size).to(torch.float32)
                updated = counts > 0
                total_counts = model.counts[updated] + counts[updated]
                model.centers[updated] = (
                    model.centers[updated] * model.counts[updated].unsqueeze(1)
                    + sums[updated]
                ) / total_counts.unsqueeze(1)
                model.counts[updated] = total_counts
            batches_seen += 1
        if batches_seen == 0 or model is None:
            raise ValueError("No real latent batch contained at least K=256 tokens; codebook was not trained.")
        self.model = model

    def _require_model(self) -> Any:
        if self.model is None:
            raise RuntimeError("Fit or load the codebook before quantization.")
        return self.model

    def encode(self, latent: LatentArtifact) -> TransmissionArtifact:
        model = self._require_model()
        tokens = self._tokens(latent)
        indices = model.predict(tokens).astype(np.uint16)
        batch, token_count, channels = (int(value) for value in latent.tensor.shape)
        if latent.rate_mask is None:
            active_channels = channels
            mask_bits = 0
            mask_vector = None
        else:
            mask_vector = latent.rate_mask[0, 0].detach().to(dtype=__import__("torch").uint8).cpu().tolist()
            active_channels = int(sum(mask_vector))
            mask_bits = channels
        # Each image has an independently decodable packet, including its own header,
        # mask and CRC. Do not amortize protocol bits across a minibatch by accident.
        header = CodebookPacketHeader(
            codebook_id=self.codebook_id,
            codebook_size=self.codebook_size,
            latent_shape=(1, token_count, channels),
            active_channels=active_channels,
            index_bits=index_width(self.codebook_size),
            rate_mask_bits=mask_bits,
        )
        per_image_bits = header.payload_bits
        packet_bits = [
            encode_codebook_packet_bits(indices.reshape(batch, token_count)[index], header, mask_vector)
            for index in range(batch)
        ]
        return TransmissionArtifact(
            transmitted=indices.reshape(batch, token_count),
            received=indices.reshape(batch, token_count).copy(),
            channel_name="untransmitted_vq_packet",
            snr_db=float("inf"),
            channel_uses_per_image=[0] * batch,
            payload_bits_per_image=[per_image_bits] * batch,
            metadata={
                "header": header.to_dict(),
                "rate_mask_vector": mask_vector,
                "packet_bits_per_image": packet_bits,
                "batch_size": batch,
            },
        )

    def decode(self, packet: TransmissionArtifact, template: LatentArtifact) -> LatentArtifact:
        model = self._require_model()
        indices = np.asarray(packet.received, dtype=np.int64)
        batch, token_count, channels = (int(value) for value in template.tensor.shape)
        if indices.shape != (batch, token_count):
            raise ValueError(f"VQ packet index shape {indices.shape} does not match latent template {(batch, token_count)}.")
        decoded = model.cluster_centers_[indices.reshape(-1)].reshape(batch, token_count, channels)
        torch = __import__("torch")
        tensor = torch.from_numpy(decoded).to(device=template.tensor.device, dtype=template.tensor.dtype)
        if template.rate_mask is not None:
            tensor = tensor * template.rate_mask
        return LatentArtifact(
            tensor=tensor,
            rate_mask=template.rate_mask,
            source_model=f"{template.source_model}+{self.codebook_id}",
            rate_tokens=template.rate_tokens,
            metadata={**template.metadata, "vq_packet": packet.metadata},
        )

    def fit_from_latents(self, latents: list[LatentArtifact]) -> None:
        self.fit(latents)

    def save(self, path: str | Path) -> Path:
        model = self._require_model()
        try:
            import joblib
        except ImportError as error:
            raise RuntimeError("Saving a VQ codebook requires joblib.") from error
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "codebook_size": self.codebook_size,
                "batch_size": self.batch_size,
                "random_state": self.random_state,
                "codebook_id": self.codebook_id,
                "model": model,
            },
            target,
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "MiniBatchKMeansVQ":
        import joblib

        payload = joblib.load(path)
        instance = cls(
            codebook_size=int(payload["codebook_size"]),
            batch_size=int(payload["batch_size"]),
            random_state=int(payload["random_state"]),
            codebook_id=str(payload["codebook_id"]),
        )
        instance.model = payload["model"]
        return instance

    def content_hash(self) -> str:
        model = self._require_model()
        return hashlib.sha256(np.asarray(model.cluster_centers_, dtype=np.float32).tobytes()).hexdigest()


def register_minibatch_kmeans_vq() -> None:
    if not CODEBOOK_REGISTRY.contains("minibatch_kmeans_vq_k256"):
        CODEBOOK_REGISTRY.register("minibatch_kmeans_vq_k256", MiniBatchKMeansVQ)
