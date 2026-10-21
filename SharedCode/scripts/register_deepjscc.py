"""Register and validate the DeepJSCC component."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from EncDecPipeline.Models.DeepJSCC.adapter import register_deepjscc
from INFRA.Registries import ENCDEC_REGISTRY


def main() -> None:
    register_deepjscc()
    if not ENCDEC_REGISTRY.contains("deepjscc"):
        raise RuntimeError("DeepJSCC was not registered.")
    print("DeepJSCC registry integration OK")


if __name__ == "__main__":
    main()
