"""Symbol error rate for aligned symbol arrays."""

import numpy as np


def symbol_error_rate(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference_array = np.asarray(reference).reshape(-1)
    estimate_array = np.asarray(estimate).reshape(-1)
    if reference_array.shape != estimate_array.shape:
        raise ValueError("SER inputs must have equal shape.")
    return float(np.mean(reference_array != estimate_array))
