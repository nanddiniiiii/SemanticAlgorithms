"""Registry for semantic parameter codebooks, separate from communication VQ."""

from Infrastructure.Registries.base_registry import ComponentRegistry


PARAMETER_CODEBOOK_REGISTRY: ComponentRegistry[object] = ComponentRegistry("semantic parameter codebook")
