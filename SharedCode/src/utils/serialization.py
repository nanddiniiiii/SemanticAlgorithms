"""JSON serialization with deterministic key ordering for run metadata."""

import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, value: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    return target
