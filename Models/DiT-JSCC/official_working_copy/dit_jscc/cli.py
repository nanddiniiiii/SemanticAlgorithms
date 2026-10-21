"""Command-line interface for DiT-JSCC inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import DiTJSCCPipeline, check_environment


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct images with DiT-JSCC over a simulated channel."
    )
    parser.add_argument("--input", type=Path, help="Image file or directory.")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--resolution", type=int, choices=(256, 512), default=256)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--autoencoder", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--channel", choices=("awgn", "rayleigh"), default="awgn")
    parser.add_argument("--snr", type=float, default=0.0, help="Channel SNR in dB.")
    parser.add_argument("--semantic-rate", type=int)
    parser.add_argument("--detail-rate", type=int)
    parser.add_argument(
        "--sampler",
        choices=("dpm_solver", "euler_ode", "euler_sde"),
        default="dpm_solver",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print setup diagnostics without loading model weights.",
    )
    return parser


def _collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES
        )
    raise FileNotFoundError(path)


def main() -> None:
    args = build_parser().parse_args()
    if args.check:
        print(json.dumps(check_environment(args.resolution), indent=2, ensure_ascii=False))
        return
    if args.input is None:
        raise SystemExit("--input is required unless --check is used")

    images = _collect_images(args.input)
    if not images:
        raise SystemExit(f"No supported images found in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    pipeline = DiTJSCCPipeline(
        resolution=args.resolution,
        checkpoint=args.checkpoint,
        autoencoder=args.autoencoder,
        device=args.device,
        precision=args.precision,
    )
    records = []
    for index, source in enumerate(images):
        result = pipeline(
            source,
            channel_type=args.channel,
            snr_db=args.snr,
            semantic_rate=args.semantic_rate,
            detail_rate=args.detail_rate,
            sampler=args.sampler,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed + index,
        )
        destination = args.output / f"{source.stem}_ditjscc.png"
        result.image.save(destination)
        record = {"input": str(source), "output": str(destination), **result.metadata()}
        records.append(record)
        print(json.dumps(record, ensure_ascii=False))
    (args.output / "metadata.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
