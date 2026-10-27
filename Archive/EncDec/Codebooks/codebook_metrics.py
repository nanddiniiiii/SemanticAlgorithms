"""Diagnostics that prevent a collapsed K=256 codebook from being misreported."""

from typing import Any

import numpy as np


def codebook_usage_metrics(indices: Any, codebook_size: int = 256) -> dict[str, float | int]:
    flat = np.asarray(indices).reshape(-1)
    counts = np.bincount(flat, minlength=codebook_size)
    probabilities = counts / max(int(counts.sum()), 1)
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())
    return {
        "codebook_size": codebook_size,
        "used_entries": int((counts > 0).sum()),
        "dead_entries": int((counts == 0).sum()),
        "perplexity": float(2**entropy),
        "max_usage_fraction": float(probabilities.max(initial=0.0)),
    }
