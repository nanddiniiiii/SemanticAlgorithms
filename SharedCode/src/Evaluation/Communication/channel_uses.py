"""Channel-use accounting helpers."""

from INFRA.Artifacts import TransmissionArtifact


def channel_uses_from_transmission(transmission: TransmissionArtifact, index: int = 0) -> int:
    return int(transmission.channel_uses_per_image[index])


def qpsk_channel_uses(coded_bits: int) -> int:
    if coded_bits % 2:
        raise ValueError("QPSK coded-bit count must be even.")
    return coded_bits // 2
