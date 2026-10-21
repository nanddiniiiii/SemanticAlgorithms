"""Strict real-image discovery: there is intentionally no synthetic fallback."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def require_torch_and_pillow() -> tuple[Any, Any]:
    try:
        import torch
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Real-image loading requires the documented Python 3.12 project environment.") from error
    return torch, Image


def discover_images(root: str | Path) -> list[Path]:
    directory = Path(root)
    if not directory.is_dir():
        raise FileNotFoundError(f"Required image directory does not exist: {directory}")
    images = sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No supported real images were found under: {directory}")
    return images


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_provenance(root: str | Path, limit: int | None = None) -> dict[str, object]:
    images = discover_images(root)
    selected = images if limit is None else images[:limit]
    listing = "\n".join(f"{path.relative_to(Path(root))}:{file_sha256(path)}" for path in selected)
    return {
        "root": str(Path(root)),
        "image_count": len(images),
        "hash_scope": "all" if limit is None else f"first_{limit}_lexicographic",
        "listing_sha256": hashlib.sha256(listing.encode("utf-8")).hexdigest(),
    }


def pil_to_tensor(image: Any) -> Any:
    torch, _ = require_torch_and_pillow()
    array = __import__("numpy").asarray(image, dtype="float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()
