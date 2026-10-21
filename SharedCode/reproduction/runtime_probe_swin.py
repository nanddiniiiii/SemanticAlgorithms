"""Real upstream Swin checkpoint/forward/backward/round-trip diagnostic.

CPU mode directly exercises the unchanged encoder, decoder and distortion loss
at the initialized training geometry. The network resolution updater and channel
allocate CUDA tensors; neither is silently patched to run on CPU.
GPU mode runs the original native AWGN channel. Synthetic input is used only for
compatibility checks; no benchmark quality or trained epochs are claimed.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output or root / f"Results/reproduction_audit/swin_{args.device}_runtime_probe"
    out.mkdir(parents=True, exist_ok=True)
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "status": "RUNNING",
              "device": args.device, "synthetic_inputs_only": True,
              "benchmark_metrics_computed": False, "real_training_epochs_completed": 0,
              "source_model_edits": [], "native_channel_executed": args.device == "cuda"}

    def persist():
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    persist()
    start = time.perf_counter()
    try:
        deps = root / "SharedCode/reproduction/.cpu_dependencies"
        if deps.exists():
            sys.path.insert(0, str(deps))
        import torch
        import torch.nn as nn
        import timm
        torch.set_num_threads(4)
        torch.manual_seed(20261020)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("No CUDA device; GPU validation not performed")
        report.update(torch_version=torch.__version__, timm_version=timm.__version__,
                      cuda_available=torch.cuda.is_available())
        manifests = list((root / "Models/author_releases/SwinJSCC").glob("*.manifest.json"))
        if len(manifests) != 1:
            raise RuntimeError("Exactly one pinned Swin release is required")
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        source = root / manifest["source_directory"]
        for name, digest in manifest["files_sha256"].items():
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError("Changed upstream source: " + name)
        sys.path.insert(0, str(source))
        # Execute only the unchanged author configuration class. Importing the
        # entire main script would also execute its CUDA-only metric setup.
        tree = ast.parse((source / "main.py").read_text(encoding="utf-8"))
        config_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "config")
        model_args = SimpleNamespace(trainset="DIV2K", testset="kodak", model_size="base",
                                     model="SwinJSCC_w/o_SAandRA", C="96", multiple_snr="10",
                                     channel_type="awgn", distortion_metric="MSE")
        scope = {"torch": torch, "nn": nn, "datetime": datetime, "args": model_args}
        exec(compile(ast.Module(body=[config_node], type_ignores=[]), str(source / "main.py"), "exec"), scope)
        config = scope["config"]
        config.device = torch.device(args.device)
        config.CUDA = args.device == "cuda"
        config.pass_channel = args.device == "cuda"
        from net.network import SwinJSCC
        model = SwinJSCC(model_args, config).to(args.device)
        checkpoint_path = args.checkpoint or root / "Results/seven_model_results/SwinJSCC/AWGN/model.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint["model"]
        prefix = "module.core."
        if not all(key.startswith(prefix) for key in state):
            raise RuntimeError("Unexpected checkpoint wrapper; refusing a partial/guessing load")
        state = {key[len(prefix):]: value for key, value in state.items()}
        incompatible = model.load_state_dict(state, strict=True)
        report.update(source_commit=manifest["commit"], source_integrity="PASS",
                      checkpoint_epoch=checkpoint["epoch"], checkpoint_strict_load="PASS",
                      missing_keys=list(incompatible.missing_keys), unexpected_keys=list(incompatible.unexpected_keys),
                      checkpoint_parameter_tensors=len(state), parameter_count=sum(p.numel() for p in model.parameters()),
                      original_checkpoint_path=str(checkpoint_path))
        persist()
        print("STRICT CHECKPOINT LOAD PASS", flush=True)
        observed_features = []
        def observe_encoder(module, inputs, output):
            # Observation only: unchanged fixed-rate model sends this whole tensor.
            observed_features.append(output.numel())
        observer = model.encoder.register_forward_hook(observe_encoder)
        def diagnostic_forward(x):
            if args.device == "cuda":
                return model(x)
            feature = model.encoder(x, 10, 96, model_args.model)
            rec = model.decoder(feature, 10, model_args.model)
            mse = model.squared_difference(x * 255., rec.clamp(0., 1.) * 255.).mean()
            loss = model.distortion_loss.forward(x, rec.clamp(0., 1.)).mean()
            return rec, feature.numel() / 2 / x.numel(), 10, mse, loss

        shapes = [(256, 256), (512, 768), (768, 512)] if args.device == "cuda" else [(256, 256)]
        report["cpu_scope"] = "Direct unchanged encoder/decoder/loss at initialized 256x256; CUDA-only resolution updater and channel not tested" if args.device == "cpu" else None
        report["full_kodak_geometry_checked"] = False
        report["forward_checks"] = []
        model.eval()
        for height, width in shapes:
            x = torch.rand(1, 3, height, width, device=args.device)
            t0 = time.perf_counter()
            # Author resolution updates cache masks. Keep those reusable by
            # autograd when this diagnostic subsequently checks backward.
            with torch.no_grad():
                reconstructed, cbr, _, _, loss = diagnostic_forward(x)
            if args.device == "cuda":
                torch.cuda.synchronize()
            if reconstructed.shape != x.shape or not torch.isfinite(reconstructed).all() or not torch.isfinite(loss):
                raise RuntimeError("Invalid reconstruction shape or nonfinite output")
            item = {"input_shape": list(x.shape), "output_shape": list(reconstructed.shape),
                    "finite": True, "reported_cbr": float(cbr), "seconds": time.perf_counter()-t0,
                    "observed_complex_channel_uses": observed_features[-1] // 2,
                    "padding_pixels": 0, "tiling": False}
            assert observed_features[-1] % 2 == 0
            assert abs(item["observed_complex_channel_uses"] / x.numel() - float(cbr)) < 1e-12
            report["forward_checks"].append(item)
            persist()
            print("FORWARD PASS", item, flush=True)
            del x, reconstructed, loss
        report["full_kodak_geometry_checked"] = args.device == "cuda"
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer.zero_grad(set_to_none=True)
        x = torch.rand(1, 3, 256, 256, device=args.device)
        t0 = time.perf_counter()
        _, _, _, _, loss = diagnostic_forward(x)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite diagnostic training loss")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError("Missing or nonfinite gradients")
        optimizer.step()
        probe = out / "synthetic_roundtrip.pt"
        torch.save({"synthetic_probe_only": True, "production_epochs_completed": 0,
                    "source_commit": manifest["commit"], "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()}, probe)
        restored = torch.load(probe, map_location=args.device, weights_only=False)
        if not all(torch.equal(value, restored["model"][key]) for key, value in model.state_dict().items()):
            raise RuntimeError("Checkpoint round-trip differs")
        optimizer.load_state_dict(restored["optimizer"])
        report.update(backward_finite="PASS", optimizer_state_load="PASS", synthetic_optimizer_steps=1,
                      diagnostic_step_seconds=time.perf_counter()-t0, checkpoint_roundtrip="PASS",
                      source_checkpoint_overwritten=False, synthetic_checkpoint_bytes=probe.stat().st_size,
                      status="PASS" if args.device == "cuda" else "PASS_CPU_CORE_ONLY", limitation="CPU mode directly exercises unchanged encoder/decoder/loss, without CUDA-only resolution update and channel. New PyTorch CPU runtime is not the native paper environment. This is compatibility evidence only.")
        for name, digest in manifest["files_sha256"].items():
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError("Upstream source changed during validation")
    except Exception as exc:
        report.update(status="FAIL", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter()-start
        persist()


if __name__ == "__main__":
    main()
