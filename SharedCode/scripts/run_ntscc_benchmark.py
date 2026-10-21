"""Run NTSCC in its WSL environment and export an auditable result bundle."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scalar(value) -> float:
    return float(value.detach().float().mean().cpu()) if torch.is_tensor(value) else float(value)


def image_metrics(reference: torch.Tensor, reconstruction: torch.Tensor) -> dict[str, float]:
    reference = reference.detach().float().cpu().clamp(0, 1)
    reconstruction = reconstruction.detach().float().cpu().clamp(0, 1)
    mse = float(torch.mean((reference - reconstruction) ** 2))
    psnr_db = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
    try:
        from skimage.metrics import structural_similarity

        ref = reference[0].permute(1, 2, 0).numpy()
        rec = reconstruction[0].permute(1, 2, 0).numpy()
        ssim_value = float(
            structural_similarity(ref, rec, channel_axis=-1, data_range=1.0)
        )
    except ImportError:
        ssim_value = float("nan")
    return {"mse": mse, "psnr_db": psnr_db, "ssim": ssim_value}


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    array = (
        tensor[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255.0
    ).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def load_model(repo_dir: Path, checkpoint: Path, device: torch.device, snr_db: float):
    for name in list(sys.modules):
        if name == "channel" or name.startswith("channel."):
            del sys.modules[name]
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    os.chdir(repo_dir)
    from config import config
    from net.NTSCC_Hyperior import NTSCC_Hyperprior

    config.device = str(device)
    config.channel["chan_param"] = float(snr_db)
    model = NTSCC_Hyperprior(config).to(device)
    raw = torch.load(checkpoint, map_location=device)
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    if not isinstance(state, dict):
        raise TypeError("NTSCC checkpoint does not contain a state_dict.")
    state = dict(state)
    skipped = [
        key
        for key in state
        if "attn_mask" in key or key.endswith("rate_adaption.mask")
    ]
    for key in skipped:
        state.pop(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    illegal_missing = [
        key
        for key in missing
        if "attn_mask" not in key and not key.endswith("rate_adaption.mask")
    ]
    if illegal_missing or unexpected:
        raise RuntimeError(
            f"Unsafe checkpoint load: missing={illegal_missing}, unexpected={list(unexpected)}"
        )
    return model.eval(), config, skipped


def parse_args() -> argparse.Namespace:
    default_root = Path(os.environ.get("SEMCOM_ROOT", Path(__file__).resolve().parents[1]))
    ntscc_root = default_root / "references" / "NTSCC_JSAC22_official"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=Path,
        default=default_root / "data" / "Kodak24" / "kodim08.png",
    )
    parser.add_argument("--repo-dir", type=Path, default=ntscc_root)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_root / "models" / "NTSCC" / "production" / "ntscc_hyperprior_quality_1_psnr.pth",
    )
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Optional explicit evaluation resize. The saved reference uses this exact size.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path, label in (
        (args.repo_dir, "NTSCC repository"),
        (args.checkpoint, "NTSCC checkpoint"),
        (args.image, "input image"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.trials < 1:
        raise ValueError("--trials must be positive.")

    device = torch.device(args.device)
    image = Image.open(args.image).convert("RGB")
    if args.resize:
        image = image.resize(tuple(args.resize), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    reference = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    model, config, skipped = load_model(
        args.repo_dir, args.checkpoint, device, args.snr_db
    )

    captured: dict[str, torch.Tensor] = {}

    def capture_y(_module, _inputs, output):
        captured["y"] = output.detach()

    handle = model.ga.register_forward_hook(capture_y)
    rows = []
    first_reconstruction = None
    final_values = None
    with torch.no_grad():
        for trial in range(args.trials):
            seed_everything(args.seed + trial)
            output = model(reference)
            if len(output) < 7:
                raise RuntimeError(f"Unexpected NTSCC output length: {len(output)}")
            _, bpp_y, bpp_z, _, cbr_y, _, reconstruction = output[:7]
            reconstruction = reconstruction.clamp(0, 1)
            rows.append({"trial": trial, **image_metrics(reference, reconstruction)})
            if first_reconstruction is None:
                first_reconstruction = reconstruction.detach().cpu()
            final_values = (scalar(bpp_y), scalar(bpp_z), scalar(cbr_y))
    handle.remove()

    if first_reconstruction is None or final_values is None or "y" not in captured:
        raise RuntimeError("NTSCC inference produced no result.")
    bpp_y, bpp_z, cbr_y = final_values
    height, width = image.height, image.width
    source_rgb_values = 3 * height * width
    # Official NTSCC source: cbr_y = channel_usage / (3*H*W), where
    # channel_usage is the number of complex I/Q symbols.
    complex_channel_uses = int(round(cbr_y * source_rgb_values))
    y = captured["y"]
    patch_count = int(y.shape[-2] * y.shape[-1])
    rate_choices = list(config.fe_kwargs.get("rate_choice", []))
    if not rate_choices and hasattr(config, "multiple_rate"):
        rate_choices = list(config.multiple_rate)
    rate_map_bits = (
        patch_count * math.ceil(math.log2(len(rate_choices)))
        if len(rate_choices) > 1
        else 0
    )
    parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    result = {
        "schema_version": 1,
        "model": "NTSCC Hyperprior",
        "image": str(args.image),
        "input_size_hw": [height, width],
        "snr_db": float(args.snr_db),
        "snr_definition": "Es/N0_dB",
        "channel_use_unit": "complex_symbol",
        "complex_channel_uses": complex_channel_uses,
        "cbr": float(cbr_y),
        "source_rgb_values": source_rgb_values,
        "source_payload_bits_rgb8": source_rgb_values * 8,
        "y_latent_shape": [int(value) for value in y.shape],
        "bpp_y_entropy_estimate": bpp_y,
        "bpp_z_entropy_estimate": bpp_z,
        "entropy_estimated_bits": float((bpp_y + bpp_z) * height * width),
        "rate_map_bits_not_in_native_cbr": int(rate_map_bits),
        "rate_choices": rate_choices,
        "transmitted_bits": None,
        "psnr_db_mean": float(np.mean([row["psnr_db"] for row in rows])),
        "psnr_db_std": float(np.std([row["psnr_db"] for row in rows], ddof=1))
        if args.trials > 1
        else 0.0,
        "ssim_mean": float(np.nanmean([row["ssim"] for row in rows])),
        "ssim_std": float(np.nanstd([row["ssim"] for row in rows], ddof=1))
        if args.trials > 1
        else 0.0,
        "trials": rows,
        "parameters": {"total_parameters": parameters},
        "checkpoint_buffers_regenerated": skipped,
        "notes": [
            "cbr_y already counts complex channel uses per RGB source value; do not divide it by two again.",
            "bpp_y+bpp_z is an entropy-model diagnostic, not the analog JSCC radio payload.",
            "The official forward path passes rate indexes directly to the decoder; their fixed-width bit cost is reported separately and is not in native cbr_y.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ntscc_{args.image.stem}_{args.snr_db:g}db"
    reference_path = args.output_dir / f"{stem}_reference.png"
    reconstruction_path = args.output_dir / f"{stem}_reconstruction.png"
    result_path = args.output_dir / f"{stem}.json"
    image.save(reference_path)
    tensor_to_image(first_reconstruction).save(reconstruction_path)
    result.update(
        {
            "reference_path": str(reference_path),
            "reconstruction_path": str(reconstruction_path),
        }
    )
    result_path.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=True))
    print(f"Saved result: {result_path}")


if __name__ == "__main__":
    main()
