"""Registry for semantic parameter codebooks, separate from communication VQ."""

from INFRA.Registries.base_registry import ComponentRegistry


PARAMETER_CODEBOOK_REGISTRY: ComponentRegistry[object] = ComponentRegistry("semantic parameter codebook")
