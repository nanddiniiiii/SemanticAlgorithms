"""QPSK modem wrappers used by digital baselines."""

from Channels.channel_utils import qpsk_bits_from_symbols, qpsk_symbols_from_bits

__all__ = ["qpsk_bits_from_symbols", "qpsk_symbols_from_bits"]
