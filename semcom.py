"""Run the project without installing it: python semcom.py --help."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
if (ROOT / ".runtime_deps").is_dir():
    sys.path.insert(0, str(ROOT / ".runtime_deps"))

from semantic_lossless.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
