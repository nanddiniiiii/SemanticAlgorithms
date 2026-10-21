"""Matched-resource DeepJSCC/NTSCC/digital communication benchmark.

Run NTSCC in WSL first, then pass its JSON result to this script.  Every CBR
in the output uses the repository definition:

    complex channel uses / (3 * image height * image width)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Baselines.DigitalPHY.channel_coding import LDPCQPSKAWGN
from Baselines.JPEG.jpeg_bitstream import bits_to_bytes
from Baselines.JPEG.jpeg_codec import JPEGCodec
from Baselines.JPEG.jpeg_pipeline import encode_jpeg_for_transport
from EncDecPipeline.Preprocessing.image_loader import pil_to_tensor
from Evaluation.Image.psnr import psnr
from Evaluation.Image.ssim import ssim
from run_deepjscc import evaluate_deepjscc


def default_semcom_root() -> Path:
    return Path(os.environ.get("SEMCOM_ROOT", ROOT))


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metric_pair(reference: Image.Image, reconstruction: Image.Image) -> tuple[float, float]:
    a = pil_to_tensor(reference).unsqueeze(0)
    b = pil_to_tensor(reconstruction).unsqueeze(0)
    if a.shape != b.shape:
        raise ValueError(f"Metric shape mismatch: {tuple(a.shape)} versus {tuple(b.shape)}")
    return float(psnr(a, b)), float(ssim(a, b))


def digital_trials(
    image: Image.Image,
    method: str,
    source_bits: np.ndarray,
    quality: int | None,
    snr_db: float,
    trials: int,
    device: str,
    seed: int,
) -> dict:
    digital = LDPCQPSKAWGN(
        information_block_bits=1024, codeword_bits=2048, iterations=20
    )
    trial_rows = []
    for trial in range(trials):
        seed_everything(seed + trial)
        physical = digital.transmit(source_bits, esn0_db=snr_db, device=device)
        row = {
            "frame_success": bool(physical.frame_success),
            "failure_reason": physical.failure_reason,
            "psnr_db": None,
            "ssim": None,
        }
        if physical.frame_success and physical.recovered_bits is not None:
            try:
                if quality is None:
                    payload = np.packbits(physical.recovered_bits, bitorder="big")
                    reconstruction = Image.fromarray(
                        payload.reshape(image.height, image.width, 3), mode="RGB"
                    )
                else:
                    reconstruction = JPEGCodec.decode(bits_to_bytes(physical.recovered_bits))
                row["psnr_db"], row["ssim"] = metric_pair(image, reconstruction)
            except Exception as error:
                row["frame_success"] = False
                row["failure_reason"] = f"decode failure: {error}"
        trial_rows.append(row)

    successes = [row for row in trial_rows if row["frame_success"]]
    uses = physical.channel_uses
    source_values = 3 * image.height * image.width
    return {
        "Method": method,
        "SNR_dB": snr_db,
        "Complex_channel_uses": uses,
        "CBR": uses / source_values,
        "Information_bits": physical.information_bits,
        "Coded_bits": physical.coded_bits,
        "Frame_success_rate": len(successes) / trials,
        "PSNR_dB_mean": float(np.mean([row["psnr_db"] for row in successes]))
        if successes
        else None,
        "SSIM_mean": float(np.mean([row["ssim"] for row in successes]))
        if successes
        else None,
        "Qualification": "Quality is reported only for CRC-valid frames.",
        "trials": trial_rows,
    }


def load_ntscc_result(path: Path, image: Image.Image, snr_db: float) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "input_size_hw",
        "snr_db",
        "complex_channel_uses",
        "cbr",
        "psnr_db_mean",
        "ssim_mean",
    }
    missing = required.difference(result)
    if missing:
        raise ValueError(f"NTSCC result is missing fields: {sorted(missing)}")
    expected_size = [image.height, image.width]
    if result["input_size_hw"] != expected_size:
        raise ValueError(
            f"NTSCC evaluated {result['input_size_hw']}, but benchmark image is {expected_size}. "
            "Use the exact NTSCC reference image or rerun both models at one resolution."
        )
    if not math.isclose(float(result["snr_db"]), snr_db, abs_tol=1e-9):
        raise ValueError("NTSCC and benchmark SNR values do not match.")
    expected_cbr = result["complex_channel_uses"] / (3 * image.height * image.width)
    if not math.isclose(float(result["cbr"]), expected_cbr, rel_tol=2e-4, abs_tol=1e-8):
        raise ValueError("NTSCC JSON has inconsistent channel-use/CBR accounting.")
    return {
        "Method": "NTSCC Hyperprior",
        "SNR_dB": snr_db,
        "Complex_channel_uses": int(result["complex_channel_uses"]),
        "CBR": float(result["cbr"]),
        "Information_bits": None,
        "Coded_bits": None,
        "Frame_success_rate": 1.0,
        "PSNR_dB_mean": result["psnr_db_mean"],
        "SSIM_mean": result["ssim_mean"],
        "Qualification": (
            "Native cbr_y; rate-map side information is reported separately and is not "
            "included in the official model's cbr_y."
        ),
        "Rate_map_bits_not_in_CBR": result.get("rate_map_bits_not_in_native_cbr"),
    }


def parse_args() -> argparse.Namespace:
    semcom = default_semcom_root()
    deep_root = semcom / "references" / "DeepJSCC_checkpoint_source"
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=semcom / "data" / "Kodak24" / "kodim08.png")
    parser.add_argument(
        "--deep-checkpoint",
        type=Path,
        default=semcom / "models" / "DeepJSCC" / "production" / "imagenet_10_0.33_200.00_32_19.pth",
    )
    parser.add_argument("--deep-source-root", type=Path, default=deep_root)
    parser.add_argument("--ntscc-result", type=Path, default=None)
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg-qualities", type=int, nargs="+", default=[90, 75, 50, 25])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Image.open(args.image).convert("RGB")
    rows = []

    deep, _ = evaluate_deepjscc(
        args.image,
        args.deep_checkpoint,
        args.deep_source_root,
        snr_db=args.snr_db,
        trials=args.trials,
        tile_size=args.tile_size,
        seed=args.seed,
    )
    rows.append(
        {
            "Method": "DeepJSCC",
            "SNR_dB": args.snr_db,
            "Complex_channel_uses": deep["complex_channel_uses"],
            "CBR": deep["cbr"],
            "Information_bits": None,
            "Coded_bits": None,
            "Frame_success_rate": 1.0,
            "PSNR_dB_mean": deep["psnr_db_mean"],
            "SSIM_mean": deep["ssim_mean"],
            "Qualification": "Continuous JSCC symbols; no transmitted bitstream.",
        }
    )
    if args.ntscc_result is not None:
        rows.append(load_ntscc_result(args.ntscc_result, image, args.snr_db))

    raw = np.unpackbits(
        np.frombuffer(np.asarray(image, dtype=np.uint8).tobytes(), dtype=np.uint8),
        bitorder="big",
    )
    payloads = [("RAW + 5G-LDPC/QPSK/AWGN", raw, None)]
    for quality in args.jpeg_qualities:
        stream = encode_jpeg_for_transport(image, quality)
        payloads.append(
            (f"JPEG Q{quality} + 5G-LDPC/QPSK/AWGN", np.asarray(stream.bits), quality)
        )
    for method, bits, quality in payloads:
        rows.append(
            digital_trials(
                image,
                method,
                bits.reshape(-1).astype(np.uint8),
                quality,
                args.snr_db,
                args.trials,
                args.device,
                args.seed,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"communication_comparison_{args.snr_db:g}db.json"
    csv_path = args.output_dir / f"communication_comparison_{args.snr_db:g}db.csv"
    json_path.write_text(json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8")
    columns = [
        "Method",
        "SNR_dB",
        "Complex_channel_uses",
        "CBR",
        "Information_bits",
        "Coded_bits",
        "Frame_success_rate",
        "PSNR_dB_mean",
        "SSIM_mean",
        "Qualification",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['Method']}: uses={row['Complex_channel_uses']:,}, "
            f"CBR={row['CBR']:.6f}, PSNR={row['PSNR_dB_mean']}, "
            f"SSIM={row['SSIM_mean']}"
        )
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
