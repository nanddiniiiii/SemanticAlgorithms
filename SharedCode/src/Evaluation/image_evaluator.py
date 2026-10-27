"""Typed image evaluator registered for dependency-injected pipeline use."""

from Infrastructure.Artifacts import EvaluationArtifact, ImageArtifact
from Infrastructure.Interfaces import EvaluatorInterface
from Infrastructure.Registries import EVALUATOR_REGISTRY
from Evaluation.Image.psnr import psnr
from Evaluation.Image.ssim import ssim


class ImageQualityEvaluator(EvaluatorInterface):
    def evaluate(self, reference: ImageArtifact, reconstruction: ImageArtifact) -> EvaluationArtifact:
        return EvaluationArtifact(
            metrics={"psnr_db": psnr(reference.tensor, reconstruction.tensor), "ssim": ssim(reference.tensor, reconstruction.tensor)},
            evaluator_name="image_quality",
            samples_evaluated=reference.batch_size(),
        )


def register_image_quality_evaluator() -> None:
    if not EVALUATOR_REGISTRY.contains("image_quality"):
        EVALUATOR_REGISTRY.register("image_quality", ImageQualityEvaluator)
