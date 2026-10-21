"""MS-SSIM is optional and deliberately not part of the locked MSE objective."""

from INFRA.errors import ComponentUnavailableError


def ms_ssim(*_: object, **__: object) -> float:
    raise ComponentUnavailableError("MS-SSIM", "Stage 1A", "Stage 1A is locked to MSE/PSNR/SSIM evaluation.")
