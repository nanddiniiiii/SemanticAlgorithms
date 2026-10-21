import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accounting import ntscc_plus_ledger
from matched_evaluation import evaluate_native_image
from profiling import training_flops_receipt


def test_ntscc_plus_counts_index_transport_and_padding():
    row = ntscc_plus_ledger(primary_complex_uses=100, index_payload_bits=7,
                            coded_index_bits=11, modulation_bits_per_symbol=4,
                            original_height=256, original_width=256,
                            padded_height=256, padded_width=256,
                            index_codec="PNG", channel_code="measured FEC",
                            modulation="16QAM")
    assert row["sideinfo_complex_uses"] == 3
    assert row["complex_channel_uses"] == 103
    assert row["cbr"] == 103 / (3 * 256 * 256)
    with pytest.raises(ValueError):
        ntscc_plus_ledger(primary_complex_uses=100, index_payload_bits=7,
                          coded_index_bits=6, modulation_bits_per_symbol=4,
                          original_height=256, original_width=256,
                          padded_height=256, padded_width=256,
                          index_codec="PNG", channel_code="FEC", modulation="16QAM")


def test_flops_requires_real_backward_and_flags_missing_operators():
    with pytest.raises(ValueError):
        training_flops_receipt(forward_flops=10, backward_flops=0,
                               optimizer_flops=0, unsupported_ops=[], batch_size=2,
                               precision="fp32", profiler="CPU", actual_optimizer_step=False)
    row = training_flops_receipt(forward_flops=10, backward_flops=20,
                                 optimizer_flops=3, unsupported_ops=["fused_attention"],
                                 batch_size=2, precision="fp32", profiler="CPU",
                                 actual_optimizer_step=True)
    assert row["coverage"] == "partial_supported_operators"
    assert row["supported_operator_training_flops_per_image"] == 16.5


def test_native_full_image_preserves_portrait_and_counts_padding():
    pytest.importorskip("torch")
    pytest.importorskip("pytorch_msssim")
    image = np.full((257, 256, 3), 128, dtype=np.uint8)
    seen = {}

    def transmit(padded, context):
        seen["shape"] = padded.shape
        return {"reconstruction": padded, "complex_channel_uses": 200,
                "all_transmitted_streams_counted": True}

    row, restored = evaluate_native_image(image, transmit, image_id="synthetic",
                                          channel="AWGN", snr_db=10, trial=0,
                                          seed=1, alignment=128)
    assert seen["shape"] == (384, 256, 3)
    assert restored.shape == (257, 256, 3)
    assert row["cbr"] == 200 / (3 * 257 * 256)
    assert row["protocol"] == "native_full_image_v1"
