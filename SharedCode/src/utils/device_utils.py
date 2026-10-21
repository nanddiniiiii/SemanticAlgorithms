"""Runtime checks are reporting utilities, not a replacement for Kaggle GPU validation."""

from typing import Any


def inspect_torch_runtime() -> dict[str, Any]:
    import torch

    return {
        "python": __import__("platform").python_version(),
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else [],
    }
