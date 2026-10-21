"""5G-NR LDPC plus QPSK plus AWGN reference baseline using Sionna 2.0.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from Baselines.DigitalPHY.bit_utils import frame_payload_bits, recover_payload_bits
from Baselines.DigitalPHY.qpsk_modem import qpsk_llr, qpsk_map
from Channels.channel_utils import esn0_to_noise_variance, require_torch


@dataclass(slots=True)
class DigitalTransmissionResult:
    recovered_bits: np.ndarray | None
    frame_success: bool
    failure_reason: str
    information_bits: int
    framed_bits: int
    padding_bits: int
    coded_bits: int
    channel_uses: int
    esn0_db: float


class LDPCQPSKAWGN:
    """Packetized reference chain. A failed CRC is a failure, never a hidden PSNR."""

    def __init__(self, information_block_bits: int = 1024, codeword_bits: int = 2048, iterations: int = 20) -> None:
        if information_block_bits <= 0 or codeword_bits <= information_block_bits:
            raise ValueError("LDPC needs 0 < information_block_bits < codeword_bits.")
        if codeword_bits % 2:
            raise ValueError("QPSK needs an even LDPC codeword length.")
        self.information_block_bits = information_block_bits
        self.codeword_bits = codeword_bits
        self.iterations = iterations

    def _sionna_modules(self, device: str) -> tuple[Any, Any]:
        try:
            import sionna
            from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
        except ImportError as error:
            raise RuntimeError(
                "The digital baseline requires sionna==2.0.1. Install it in Kaggle without replacing torch."
            ) from error
        if str(sionna.__version__) != "2.0.1":
            raise RuntimeError(f"Expected sionna==2.0.1, found {sionna.__version__!r}.")
        encoder = LDPC5GEncoder(
            self.information_block_bits,
            self.codeword_bits,
            num_bits_per_symbol=2,
            device=device,
        )
        decoder = LDPC5GDecoder(encoder, hard_out=True, num_iter=self.iterations, device=device)
        return encoder, decoder

    def transmit(self, source_bits: np.ndarray, esn0_db: float, device: str = "cuda:0") -> DigitalTransmissionResult:
        torch = require_torch()
        encoder, decoder = self._sionna_modules(device)
        framed = frame_payload_bits(source_bits)
        block_count = int(np.ceil(framed.size / self.information_block_bits))
        padded_size = block_count * self.information_block_bits
        padded = np.pad(framed, (0, padded_size - framed.size))
        information = torch.from_numpy(padded.astype(np.float32)).reshape(block_count, self.information_block_bits).to(device)
        coded = encoder(information)
        symbols = qpsk_map(coded)
        n0 = esn0_to_noise_variance(esn0_db)
        standard_deviation = (n0 / 2.0) ** 0.5
        noise = torch.complex(
            torch.randn_like(symbols.real) * standard_deviation,
            torch.randn_like(symbols.imag) * standard_deviation,
        )
        received = symbols + noise
        decoded = decoder(qpsk_llr(received, n0))
        decoded_bits = decoded.detach().round().to(torch.uint8).cpu().numpy().reshape(-1)[: framed.size]
        recovered, success, reason = recover_payload_bits(decoded_bits)
        return DigitalTransmissionResult(
            recovered_bits=recovered,
            frame_success=success,
            failure_reason=reason,
            information_bits=int(source_bits.size),
            framed_bits=int(framed.size),
            padding_bits=int(padded_size - framed.size),
            coded_bits=block_count * self.codeword_bits,
            channel_uses=block_count * (self.codeword_bits // 2),
            esn0_db=float(esn0_db),
        )
