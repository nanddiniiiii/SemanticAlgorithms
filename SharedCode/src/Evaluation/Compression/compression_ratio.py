"""Source compression ratio using only the payload actually sent per image."""


def compression_ratio(raw_bits: int, transmitted_bits: int) -> float:
    if raw_bits <= 0 or transmitted_bits <= 0:
        raise ValueError("Raw and transmitted bits must be positive.")
    return raw_bits / transmitted_bits
