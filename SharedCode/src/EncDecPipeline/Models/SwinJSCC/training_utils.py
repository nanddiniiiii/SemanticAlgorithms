"""Training helpers shared by the one Kaggle Stage 1A runner notebook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from Channels.channel_utils import require_torch


@dataclass(frozen=True, slots=True)
class DataParallelPlan:
    requested_gpu_count: int = 2
    enabled: bool = False
    device_ids: tuple[int, ...] = ()


def verify_two_t4_data_parallel() -> DataParallelPlan:
    """Fail loudly rather than silently pretending the requested two-T4 plan ran."""

    torch = require_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 1A Kaggle training requires CUDA, but CUDA is unavailable.")
    device_count = torch.cuda.device_count()
    if device_count < 2:
        raise RuntimeError(f"Stage 1A requires two GPUs for DataParallel; Kaggle exposed {device_count}.")
    names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name.upper() for name in names):
        raise RuntimeError(f"Expected two T4 GPUs, found: {names}")
    return DataParallelPlan(enabled=True, device_ids=(0, 1))


def wrap_data_parallel(module: Any, plan: DataParallelPlan) -> Any:
    torch = require_torch()
    if not plan.enabled:
        return module
    return torch.nn.DataParallel(module, device_ids=list(plan.device_ids)).cuda(plan.device_ids[0])


def unwrap_data_parallel(module: Any) -> Any:
    return module.module if hasattr(module, "module") else module


def transfer_compatible_weights(source_module: Any, destination_module: Any) -> tuple[list[str], list[str]]:
    """Transfer Base weights to SA+RA and report deliberate missing ModNet parameters."""

    source_state = unwrap_data_parallel(source_module).state_dict()
    destination = unwrap_data_parallel(destination_module)
    result = destination.load_state_dict(source_state, strict=False)
    return list(result.missing_keys), list(result.unexpected_keys)


def mse_loss(reference: Any, reconstruction: Any) -> Any:
    torch = require_torch()
    return torch.nn.functional.mse_loss(reconstruction.clamp(0.0, 1.0), reference)
