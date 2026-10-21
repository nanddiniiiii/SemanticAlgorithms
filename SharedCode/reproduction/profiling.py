"""Conservative audit of reported training FLOPs.

Profiler counts are partial until forward, backward, optimizer, and unsupported
operators have each been accounted for on the actual training path.
"""
from __future__ import annotations

import math


def training_flops_receipt(*, forward_flops, backward_flops, optimizer_flops,
                           unsupported_ops, batch_size, precision, profiler,
                           actual_optimizer_step):
    values = (forward_flops, backward_flops, optimizer_flops)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("FLOP counts must be finite nonnegative numbers")
    if backward_flops <= 0 or not actual_optimizer_step:
        raise ValueError("A real backward and applied optimizer step are required")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("Positive batch size required")
    if not isinstance(unsupported_ops, (list, tuple)) or not all(isinstance(x, str) and x for x in unsupported_ops):
        raise ValueError("Unsupported operators must be listed, even when empty")
    if not precision or not profiler:
        raise ValueError("Precision and profiler provenance required")
    total = sum(values)
    return {"supported_operator_training_flops_per_step": total,
            "supported_operator_training_flops_per_image": total / batch_size,
            "forward_flops": forward_flops, "backward_flops": backward_flops,
            "optimizer_flops": optimizer_flops, "unsupported_ops": list(unsupported_ops),
            # An empty list is not an independent audit of missing profiler ops.
            "coverage": "partial_supported_operators",
            "multiply_add_flops": 2, "batch_size": batch_size,
            "precision": precision, "profiler": profiler,
            "actual_optimizer_step": True}
