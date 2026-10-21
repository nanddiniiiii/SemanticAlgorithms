"""Synthetic packet tests; no learned model or image benchmark claims."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from digital_transport import encode_packet, decode_packet, transmit_bytes


def test_frame_rejects_bit_corruption_and_wrong_length():
    packet = encode_packet(b'actual serialized index fixture')
    assert decode_packet(packet) == b'actual serialized index fixture'
    corrupt = bytearray(packet)
    corrupt[5] ^= 1
    assert decode_packet(bytes(corrupt)) is None
    assert decode_packet(packet[:-1]) is None


def test_qpsk_counts_include_frame_and_repetition():
    payload = bytes(range(256))
    decoded, ledger = transmit_bytes(payload, channel='awgn', snr_db=80, seed=10)
    assert decoded == payload
    assert ledger['payload_bits'] == len(payload)*8
    assert ledger['framed_bits'] == (len(payload)+8)*8
    assert ledger['coded_bits'] == ledger['framed_bits']*3
    assert ledger['complex_channel_uses']*2 == ledger['coded_bits']
    assert ledger['author_native_transport'] is False


def test_shared_rayleigh_gain_and_real_erasure():
    payload = bytes(range(128))
    decoded, ledger = transmit_bytes(payload, channel='rayleigh', snr_db=80, seed=6, gain=.3+.4j)
    assert decoded == payload
    assert ledger['gain_shared_with_primary'] is True
    assert ledger['gain_real'] == .3 and ledger['gain_imag'] == .4
    decoded, ledger = transmit_bytes(payload, channel='rayleigh', snr_db=10, seed=6, gain=0j)
    assert decoded is None and ledger['crc_ok'] is False
