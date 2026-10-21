"""Effective source bits per complex channel use."""


def source_bits_per_channel_use(source_bits: int, channel_uses: int) -> float:
    if channel_uses <= 0:
        raise ValueError("Channel uses must be positive.")
    return source_bits / channel_uses
