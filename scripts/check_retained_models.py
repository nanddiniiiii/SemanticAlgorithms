"""Verify imported file identities or run a small, explicitly synthetic CPU smoke.

This does not measure Kodak performance or certify training/GPU compatibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if (ROOT / ".runtime_deps").is_dir():
    sys.path.insert(0, str(ROOT / ".runtime_deps"))


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["deepjscc", "ntscc", "swinjscc"])
    parser.add_argument("--verify-assets", action="store_true")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.model and not args.verify_assets:
        parser.error("Choose --verify-assets or --model")
    receipt: dict = {}
    if args.verify_assets:
        manifest = json.loads((ROOT / "Docs/encdec_import_manifest.json").read_text(encoding="utf-8"))
        failures = []
        for item in manifest["files"]:
            path = ROOT / item["path"]
            if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                failures.append(item["path"])
        if failures:
            raise RuntimeError(f"Asset verification failed: {failures}")
        receipt["asset_verification"] = {"files": len(manifest["files"]), "all_hashes_match": True}
    if args.model:
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(7)
        image = torch.rand(1, 3, args.size, args.size)
        started = time.perf_counter()
        weights = ROOT / "Models/trained_checkpoints"
        if args.model == "deepjscc":
            from Archive.EncDec.Research.model_wrappers import DeepJSCCAdapter
            adapter = DeepJSCCAdapter(ROOT / "Models/DeepJSCC/checkpoint_source",
                                     weights / "DeepJSCC/production/imagenet_10_0.33_200.00_32_19.pth",
                                     torch.device("cpu"), evaluation_snr_db=10)
            result = adapter.run_once(image, seed=7)
            reconstruction = result.reconstruction
            detail = {"strict_load": adapter.validation, "latent_real_values": result.latent_real_values,
                      "training_snr_db": adapter.checkpoint_training_snr_db,
                      "channel_note": "retained legacy channel convention; not a calibrated common benchmark"}
        elif args.model == "ntscc":
            from Archive.EncDec.Research.model_wrappers import NTSCCWindowsAdapter
            adapter = NTSCCWindowsAdapter(ROOT / "Models/NTSCC/official",
                                         weights / "NTSCC/production/ntscc_hyperprior_quality_1_psnr.pth",
                                         torch.device("cpu"), evaluation_snr_db=10)
            result = adapter.run_once(image, seed=7)
            reconstruction = result.reconstruction
            detail = {"strict_load": adapter.validation, "complex_channel_uses": result.complex_channel_uses,
                      "training_snr_db": adapter.checkpoint_training_snr_db,
                      "rate_note": "main latent uses only; not complete digital bitstream accounting"}
        else:
            from Archive.EncDec.Models.SwinJSCC.model_loader import load_stage1_swinjscc
            from Infrastructure.Artifacts import ImageArtifact
            from Infrastructure.Channels.awgn import AWGNChannel
            adapter = load_stage1_swinjscc(weights / "SwinJSCC/encdec_sa_ra", device="cpu",
                                          upstream_root=ROOT / "Models/SwinJSCC/local_source")
            adapter.encoder.eval()
            adapter.decoder.eval()
            with torch.inference_mode():
                latent = adapter.encode(ImageArtifact(tensor=image, sample_ids=["synthetic"], source="smoke"),
                                        snr_db=10, rate=96)
                transmission = AWGNChannel().transmit(latent, 10)
                reconstruction = adapter.reconstruct_received(transmission, latent).tensor
            detail = {"strict_load": True, "manifest_runtime_and_source_validation": True,
                      "variant": adapter.config.variant, "rate": 96,
                      "complex_channel_uses": transmission.channel_uses_per_image}
        if tuple(reconstruction.shape) != tuple(image.shape) or not torch.isfinite(reconstruction).all():
            raise RuntimeError(f"Invalid reconstruction: {tuple(reconstruction.shape)}")
        receipt["smoke"] = {"algorithm": args.model, "status": "passed", "device": "cpu",
                            "pytorch": str(torch.__version__), "python": sys.version.split()[0],
                            "input": "synthetic random tensor; no quality metric reported",
                            "input_shape": list(image.shape), "output_shape": list(reconstruction.shape),
                            "finite": True, "load_and_smoke_seconds": time.perf_counter() - started,
                            "benchmark_or_gpu_validation": False, **detail}
    print(json.dumps(receipt, indent=2), flush=True)
    if args.output:
        path = args.output if args.output.is_absolute() else ROOT / args.output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
