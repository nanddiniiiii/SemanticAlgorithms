"""Smoke-test DeepJSCC construction through the project registry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from EncDecPipeline.Models.DeepJSCC.adapter import register_deepjscc
from INFRA.Registries import ENCDEC_REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    register_deepjscc()
    model = ENCDEC_REGISTRY.create(
        "deepjscc",
        checkpoint_path=args.checkpoint,
        source_root=args.source_root,
        c=19,
        device=args.device,
    ).build()
    print(model.parameter_summary())
    print("DeepJSCC registry construction OK")


if __name__ == "__main__":
    main()
