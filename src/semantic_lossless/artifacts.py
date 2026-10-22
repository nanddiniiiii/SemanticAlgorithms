import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def export_manifest(run_dir: Path) -> dict:
    files = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "artifacts.json" or path.suffix in {".tmp", ".partial"}:
            continue
        if path.is_symlink():
            raise ValueError("Run artifacts cannot be symbolic links")
        files.append({"path": path.relative_to(run_dir).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    report = {"files": files, "file_count": len(files)}
    write_json(run_dir / "artifacts.json", report)
    return report
