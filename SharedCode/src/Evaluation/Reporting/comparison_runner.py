"""Comparison rows preserve method, SNR, use count, quality, and failure semantics."""

from typing import Any


REQUIRED_COMPARISON_COLUMNS = (
    "method",
    "snr_definition",
    "snr_db",
    "channel_uses",
    "psnr",
    "ssim",
    "frame_success",
)


def validate_comparison_row(row: dict[str, Any]) -> None:
    missing = [column for column in REQUIRED_COMPARISON_COLUMNS if column not in row]
    if missing:
        raise ValueError(f"Comparison row is missing required fields: {missing}")
    if row["snr_definition"] != "Es/N0_dB":
        raise ValueError("Stage 1A comparisons must label their SNR convention as Es/N0_dB.")
    if row["channel_uses"] < 0:
        raise ValueError("Channel use count cannot be negative.")
