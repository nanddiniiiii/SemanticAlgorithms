"""Small YAML loader with explicit one-level/multi-level inheritance."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("Configuration loading requires PyYAML.") from error
    source = Path(path)
    values = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    parent = values.pop("extends", None)
    if parent is None:
        return values
    parent_values = load_yaml_config(source.parent / parent)
    return _merge(parent_values, values)
