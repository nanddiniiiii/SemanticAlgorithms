"""Measure released channel functions on synthetic CPU data, not image scores."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Results/reproduction_audit/channel_contract_probe.json"


def main():
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(20261020)
    z = torch.randn(4, 16, 64, 64) / np.sqrt(2)
    real, imag = torch.chunk(z, 2, dim=1)
    signal = torch.complex(real, imag).reshape(4, -1)
    report = {"checked_at": datetime.now(timezone.utc).isoformat(),
              "synthetic_channel_calibration_only": True, "image_training_or_evaluation": False,
              "nominal_snr_db": 10.0, "complex_samples": signal.numel(), "checks": {}}
    path = ROOT / "Models/author_releases/DeepJSCC-third-party/2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06/channel.py"
    spec = importlib.util.spec_from_file_location("deepjscc_reference_channel_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    y = module.Channel("AWGN", 10)(z)
    yr, yi = torch.chunk(y, 2, dim=1)
    noise = torch.complex(yr, yi).reshape(4, -1) - signal
    snr = 10*torch.log10(signal.abs().square().mean() / noise.abs().square().mean())
    report["checks"]["deepjscc_third_party_awgn"] = {"source": str(path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "measured_complex_snr_db": float(snr), "provenance": "third-party reference; historical run source revision was not pinned",
        "conclusion": "This inspected source uses a different real-noise convention from unit-power complex AWGN. Do not equate its nominal SNR labels with the common contract."}
    try:
        import tensorflow as tf
        tf.random.set_seed(20261020)
        path = ROOT / "Models/author_releases/ADJSCC/e5332e95faf592aab9f440992de96029162dc7dd/util_channel.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"awgn", "slow_fading_eq"}]
        # Execute exactly these unchanged author functions; unrelated tensorflow-
        # probability/burst dependencies are not needed for this function check.
        scope = {"tf": tf, "np": np}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), str(path), "exec"), scope)
        x = tf.convert_to_tensor(signal.numpy())
        y = scope["awgn"](x, tf.constant(10., tf.float32))
        noise = y - x
        measured = 10 * tf.math.log(tf.reduce_mean(tf.abs(x)**2) / tf.reduce_mean(tf.abs(noise)**2)) / tf.math.log(10.)
        report["checks"]["adjscc_author_awgn_function"] = {"source": str(path.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "measured_complex_snr_db": float(measured.numpy()), "tensorflow": tf.__version__,
            "scope": "Standalone unchanged author function; not full ADJSCC/TFC model or checkpoint validation"}
    except Exception as exc:
        report["checks"]["adjscc_author_awgn_function"] = {"status": "BLOCKED", "error": str(exc)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
