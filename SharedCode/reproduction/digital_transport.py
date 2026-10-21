"""Explicit benchmark extension: byte framing, CRC32, repetition-3, QPSK.

This is not an author LDPC/capacity-achieving transport. All transmitted coded
bits/symbols are counted. CRC failure returns an erasure, never original bytes.
Optional gain shares a Rayleigh codeword coefficient with another stream.
"""
import struct
import zlib
import numpy as np


def encode_packet(payload):
    if not isinstance(payload, bytes):
        raise TypeError("Payload must be serialized bytes")
    content = struct.pack('>I', len(payload)) + payload
    return content + struct.pack('>I', zlib.crc32(content))


def decode_packet(packet):
    if len(packet) < 8:
        return None
    size = struct.unpack('>I', packet[:4])[0]
    if size + 8 != len(packet):
        return None
    expected = struct.unpack('>I', packet[-4:])[0]
    return packet[4:-4] if zlib.crc32(packet[:-4]) == expected else None


def transmit_bytes(payload, *, channel, snr_db, seed, gain=None):
    if channel not in {'awgn', 'rayleigh'} or not np.isfinite(snr_db):
        raise ValueError('Expected finite SNR and awgn/rayleigh')
    packet = encode_packet(payload)
    bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
    coded = np.repeat(bits, 3)
    pad = len(coded) % 2
    if pad:
        coded = np.pad(coded, (0, 1))
    pairs = coded.reshape(-1, 2).astype(np.float64)
    symbols = ((1 - 2*pairs[:, 0]) + 1j*(1 - 2*pairs[:, 1])) / np.sqrt(2)
    rng = np.random.default_rng(seed)
    if channel == 'rayleigh':
        h = complex(gain) if gain is not None else (rng.normal()+1j*rng.normal())/np.sqrt(2)
    else:
        h = 1+0j
    noise = (rng.normal(size=len(symbols))+1j*rng.normal(size=len(symbols))) * np.sqrt(10.**(-snr_db/10)/2)
    received = symbols*h + noise
    equalized = received/h if h != 0 else np.full_like(received, complex(np.inf, np.inf))
    hard = np.stack([equalized.real < 0, equalized.imag < 0], axis=1).reshape(-1)
    hard = hard[:len(bits)*3].reshape(-1, 3).sum(axis=1) >= 2
    packet_hat = np.packbits(hard.astype(np.uint8)).tobytes()
    decoded = decode_packet(packet_hat)
    return decoded, {'payload_bits': len(payload)*8, 'framed_bits': len(packet)*8,
        'coded_bits': len(coded), 'modulation_padding_bits': pad,
        'complex_channel_uses': len(symbols), 'modulation': 'unit-energy QPSK',
        'fec': 'repetition-3 hard majority', 'crc': 'CRC32 IEEE',
        'crc_ok': decoded is not None, 'channel': channel, 'snr_db': float(snr_db),
        'receiver_csi': 'perfect' if channel == 'rayleigh' else 'not_applicable',
        'gain_real': float(h.real), 'gain_imag': float(h.imag),
        'gain_shared_with_primary': gain is not None, 'seed': int(seed),
        'author_native_transport': False, 'serialized': True}
