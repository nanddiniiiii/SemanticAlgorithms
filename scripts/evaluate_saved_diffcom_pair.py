"""Measure pixel fidelity of the existing saved DiffCom pilot, without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "Results/kaggle_pilot/diffcom_v4/Semantic_Lossless/Results/implementation_checks/diffcom_gpu")
    parser.add_argument("--output", type=Path, default=root / "Docs/validation/diffcom_saved_crop_quality.json")
    args = parser.parse_args()
    print(f"Evaluating existing saved RGB pair: {args.input_dir}", flush=True)
    arrays = []
    for name in ("input_crop.png", "reconstruction.png"):
        with Image.open(args.input_dir / name) as image:
            arrays.append(np.asarray(image.convert("RGB"), dtype=np.float64))
    reference, reconstruction = arrays
    if reference.shape != reconstruction.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} versus {reconstruction.shape}")
    mse = float(np.mean((reference - reconstruction) ** 2))
    result = {
        "status": "completed",
        "scope": "posthoc evaluation of one saved uint8 RGB Kodak center-crop reconstruction; no new diffusion inference or training",
        "images": 1,
        "shape": list(reference.shape),
        "rgb_mse_255": mse,
        "psnr_db": 10 * math.log10(255 ** 2 / mse) if mse else None,
        "psnr_infinite": mse == 0,
        "mae_255": float(np.mean(abs(reference - reconstruction))),
        "exact_pixel_fraction": float(np.mean(np.all(reference == reconstruction, axis=-1))),
        "exact_rgb": bool(np.array_equal(reference, reconstruction)),
        "unmeasured": ["SSIM", "MS-SSIM", "LPIPS", "FID", "semantic task accuracy"],
        "rounding": "saved clamp/round uint8 images; not raw floating reconstruction",
        "source_files": {name: hashlib.sha256((args.input_dir / name).read_bytes()).hexdigest()
                         for name in ("input_crop.png", "reconstruction.png", "status.json")},
        "runtime_receipt": json.loads((args.input_dir / "status.json").read_text()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Completed: PSNR={result['psnr_db']}; MAE255={result['mae_255']}; output={args.output}", flush=True)


if __name__ == "__main__":
    main()
