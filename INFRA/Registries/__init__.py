"""Named component registries. Registration happens in concrete package modules."""

from INFRA.Registries.channel_registry import CHANNEL_REGISTRY
from INFRA.Registries.codebook_registry import CODEBOOK_REGISTRY
from INFRA.Registries.encdec_registry import ENCDEC_REGISTRY
from INFRA.Registries.evaluator_registry import EVALUATOR_REGISTRY
from INFRA.Registries.knowledge_registry import KNOWLEDGE_REGISTRY
from INFRA.Registries.parameter_codebook_registry import PARAMETER_CODEBOOK_REGISTRY
from INFRA.Registries.scheduler_registry import SCHEDULER_REGISTRY
from INFRA.Registries.transformer_registry import TRANSFORMER_REGISTRY

__all__ = [
    "CHANNEL_REGISTRY",
    "CODEBOOK_REGISTRY",
    "ENCDEC_REGISTRY",
    "EVALUATOR_REGISTRY",
    "KNOWLEDGE_REGISTRY",
    "PARAMETER_CODEBOOK_REGISTRY",
    "SCHEDULER_REGISTRY",
    "TRANSFORMER_REGISTRY",
]
