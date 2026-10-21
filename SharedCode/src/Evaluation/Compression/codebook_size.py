"""Separate shared-codebook storage from per-image VQ packet cost."""


def shared_codebook_storage_bits(codebook_size: int, vector_dimension: int, bits_per_value: int = 32) -> int:
    return codebook_size * vector_dimension * bits_per_value
