"""Evaluate the external DeepJSCC checkpoint through repository contracts."""

from __future__ import annotations

import argparse
import json
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

from EncDecPipeline.Models.DeepJSCC.adapter import DeepJSCCAdapter
from EncDecPipeline.Models.DeepJSCC.channel_wrapper import DeepJSCCChannelWrapper
from EncDecPipeline.Preprocessing.image_loader import pil_to_tensor
from Evaluation.Image.psnr import psnr
from Evaluation.Image.ssim import ssim
from INFRA.Artifacts import ImageArtifact


def default_semcom_root() -> Path:
    configured = os.environ.get("SEMCOM_ROOT")
    if configured:
        return Path(configured)
    return ROOT


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pad_to_tile(tensor, tile_size: int):
    import torch.nn.functional as functional

    height, width = tensor.shape[-2:]
    pad_h, pad_w = (-height) % tile_size, (-width) % tile_size
    if not pad_h and not pad_w:
        return tensor, (height, width)
    return functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect"), (
        height,
        width,
    )


def _single_trial(model, channel, tensor, snr_db: float, tile_size: int):
    import torch

    padded, original_shape = _pad_to_tile(tensor, tile_size)
    reconstructed = torch.empty_like(padded)
    channel_uses = 0
    latent_real_values = 0
    latent_shapes: list[tuple[int, ...]] = []
    tile_number = 0
    for top in range(0, padded.shape[-2], tile_size):
        for left in range(0, padded.shape[-1], tile_size):
            patch = padded[:, :, top : top + tile_size, left : left + tile_size]
            artifact = ImageArtifact(
                tensor=patch,
                sample_ids=[f"tile-{tile_number}"],
                source="DeepJSCCBenchmark",
            )
            latent = model.encode(artifact)
            transmission = channel.transmit(latent, snr_db=snr_db)
            decoded = model.reconstruct_received(transmission, latent).tensor
            reconstructed[
                :, :, top : top + tile_size, left : left + tile_size
            ] = decoded
            channel_uses += transmission.channel_uses_per_image[0]
            latent_real_values += int(latent.tensor[0].numel())
            latent_shapes.append(tuple(int(v) for v in latent.tensor.shape))
            tile_number += 1
    height, width = original_shape
    return (
        reconstructed[:, :, :height, :width].clamp(0.0, 1.0),
        channel_uses,
        latent_real_values,
        latent_shapes,
    )


def evaluate_deepjscc(
    image_path: Path,
    checkpoint_path: Path,
    source_root: Path,
    snr_db: float = 10.0,
    trials: int = 5,
    tile_size: int = 128,
    c: int = 19,
    device: str | None = None,
    seed: int = 42,
):
    import torch

    image = Image.open(image_path).convert("RGB")
    reference = pil_to_tensor(image).unsqueeze(0)
    model = DeepJSCCAdapter(
        checkpoint_path=checkpoint_path,
        source_root=source_root,
        c=c,
        device=device,
    ).build()
    channel = DeepJSCCChannelWrapper()
    rows = []
    first_reconstruction = None
    expected_accounting = None
    for trial in range(trials):
        seed_everything(seed + trial)
        reconstruction, uses, real_values, shapes = _single_trial(
            model, channel, reference.to(model.device), snr_db, tile_size
        )
        accounting = (uses, real_values, shapes)
        if expected_accounting is not None and accounting != expected_accounting:
            raise RuntimeError("DeepJSCC communication accounting changed across trials.")
        expected_accounting = accounting
        reconstruction_cpu = reconstruction.cpu()
        rows.append(
            {
                "trial": trial,
                "psnr_db": float(psnr(reference, reconstruction_cpu)),
                "ssim": float(ssim(reference, reconstruction_cpu)),
            }
        )
        if first_reconstruction is None:
            first_reconstruction = reconstruction_cpu

    assert expected_accounting is not None and first_reconstruction is not None
    uses, real_values, shapes = expected_accounting
    source_rgb_values = 3 * image.height * image.width
    result = {
        "schema_version": 1,
        "model": "DeepJSCC",
        "image": str(image_path),
        "input_size_hw": [image.height, image.width],
        "snr_db": float(snr_db),
        "snr_definition": "Es/N0_dB",
        "channel_use_unit": "complex_symbol",
        "complex_channel_uses": int(uses),
        "cbr": float(uses / source_rgb_values),
        "source_rgb_values": int(source_rgb_values),
        "source_payload_bits_rgb8": int(source_rgb_values * 8),
        "latent_real_values": int(real_values),
        "latent_shape_per_tile": list(shapes[0]),
        "tile_count": len(shapes),
        "tile_size": int(tile_size),
        "transmitted_bits": None,
        "psnr_db_mean": float(np.mean([row["psnr_db"] for row in rows])),
        "psnr_db_std": float(np.std([row["psnr_db"] for row in rows], ddof=1))
        if trials > 1
        else 0.0,
        "ssim_mean": float(np.mean([row["ssim"] for row in rows])),
        "ssim_std": float(np.std([row["ssim"] for row in rows], ddof=1))
        if trials > 1
        else 0.0,
        "trials": rows,
        "parameters": model.parameter_summary(),
        "notes": [
            "DeepJSCC sends continuous channel symbols, not a conventional bitstream.",
            "Tiling matches the 128x128 checkpoint sanity-test resolution and is reported explicitly.",
        ],
    }
    return result, first_reconstruction


def _save_tensor_png(tensor, path: Path) -> None:
    array = (
        tensor[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255.0
    ).round().astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def parse_args() -> argparse.Namespace:
    semcom = default_semcom_root()
    deep_root = semcom / "references" / "DeepJSCC_checkpoint_source"
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=semcom / "data" / "Kodak24" / "kodim08.png")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=semcom / "models" / "DeepJSCC" / "production" / "imagenet_10_0.33_200.00_32_19.pth",
    )
    parser.add_argument("--source-root", type=Path, default=deep_root)
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--c", type=int, default=19)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials < 1 or args.tile_size < 1:
        raise ValueError("--trials and --tile-size must be positive.")
    result, reconstruction = evaluate_deepjscc(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        source_root=args.source_root,
        snr_db=args.snr_db,
        trials=args.trials,
        tile_size=args.tile_size,
        c=args.c,
        device=args.device,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"deepjscc_{args.image.stem}_{args.snr_db:g}db"
    json_path = args.output_dir / f"{stem}.json"
    image_path = args.output_dir / f"{stem}.png"
    _save_tensor_png(reconstruction, image_path)
    result["reconstruction_path"] = str(image_path)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved result: {json_path}")


if __name__ == "__main__":
    main()
