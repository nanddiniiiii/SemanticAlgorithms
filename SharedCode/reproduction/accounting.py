"""Measured communication ledger for the released NTSCC+ streams.

The author evaluates ``results['k']`` plus a losslessly encoded index map;
see NTSCC_plus/main.py, lines 169-209 in the pinned release.  The caller must
provide actual encoded and channel-coded side-information sizes.  This module
does not silently assume an ideal entropy coder or a fractional QAM symbol.
"""
from __future__ import annotations

import math


def _count(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def ntscc_plus_ledger(*, primary_complex_uses: int, index_payload_bits: int,
                      coded_index_bits: int, modulation_bits_per_symbol: int,
                      original_height: int, original_width: int,
                      padded_height: int, padded_width: int,
                      index_codec: str, channel_code: str, modulation: str):
    """Count all released NTSCC+ streams after real index serialization.

    ``coded_index_bits`` includes FEC and framing, as actually transmitted.
    QAM symbols are rounded up; a partial symbol still consumes one channel use.
    The index codec and transport names are mandatory audit metadata.
    """
    primary = _count(primary_complex_uses, "primary_complex_uses", positive=True)
    payload = _count(index_payload_bits, "index_payload_bits")
    coded = _count(coded_index_bits, "coded_index_bits")
    bits_per_symbol = _count(modulation_bits_per_symbol, "modulation_bits_per_symbol", positive=True)
    h, w = _count(original_height, "original_height", positive=True), _count(original_width, "original_width", positive=True)
    ph, pw = _count(padded_height, "padded_height", positive=True), _count(padded_width, "padded_width", positive=True)
    if ph < h or pw < w or coded < payload:
        raise ValueError("Padding or coded side-information length is inconsistent")
    if not all(isinstance(s, str) and s.strip() for s in (index_codec, channel_code, modulation)):
        raise ValueError("Index codec, channel code and modulation must be identified")
    side_uses = math.ceil(coded / bits_per_symbol)
    total = primary + side_uses
    return {"primary_complex_uses": primary, "index_payload_bits": payload,
            "coded_index_bits": coded, "sideinfo_complex_uses": side_uses,
            "complex_channel_uses": total, "cbr": total / (3 * h * w),
            "original_shape": [h, w], "padded_shape": [ph, pw],
            "index_codec": index_codec, "channel_code": channel_code,
            "modulation": modulation, "modulation_bits_per_symbol": bits_per_symbol,
            "all_transmitted_streams_counted": True,
            "source": "pinned NTSCC_plus/main.py:169-209; primary k plus coded index map"}
