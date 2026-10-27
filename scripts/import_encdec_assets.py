"""Copy retained codec assets without deleting sources or replacing unrelated files.

Uses only the standard library. Existing LFS pointers may be hydrated only when
their recorded identity matches the incoming file. Run from any working folder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".venv"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def lfs_identity(path: Path) -> str | None:
    if path.stat().st_size > 1024:
        return None
    content = path.read_bytes()
    if not content.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return None
    return next(line.split(b":", 1)[1].decode("ascii")
                for line in content.splitlines() if line.startswith(b"oid sha256:"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--supplemental-checkpoints", type=Path, required=True,
                        help="Models/trained_checkpoints directory with original production weights")
    parser.add_argument("--link-local-weights", action="store_true",
                        help="Hard-link immutable weight/cache files on the same volume to save disk space")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    supplemental = args.supplemental_checkpoints.resolve(strict=True)
    if source == ROOT or source in ROOT.parents:
        raise ValueError("Source must be a separate repository")
    records: list[dict] = []

    def copy_file(original: Path, relative: Path, group: str, source_relative: str) -> None:
        if original.is_symlink():
            raise ValueError(f"Refusing a symbolic-link source: {source_relative}")
        if lfs_identity(original):
            raise ValueError(f"Source is an unhydrated LFS pointer: {source_relative}")
        destination = ROOT / relative
        destination.resolve().relative_to(ROOT)
        sha = digest(original)
        disposition = "copied"
        pointer = None
        if destination.exists():
            pointer = lfs_identity(destination)
            if pointer is not None and pointer != sha:
                raise ValueError(f"LFS identity mismatch for {relative}")
            if pointer is None and digest(destination) != sha:
                raise FileExistsError(f"Refusing to replace different content: {relative}")
            disposition = "hydrated_lfs" if pointer else "already_identical"
        destination.parent.mkdir(parents=True, exist_ok=True)
        storage = "hardlink_immutable" if destination.exists() and destination.stat().st_nlink > 1 else "preexisting"
        if disposition != "already_identical":
            link = (args.link_local_weights and original.anchor.lower() == destination.anchor.lower()
                    and original.suffix.lower() in {".pt", ".pth", ".pkl", ".joblib"})
            if link:
                if destination.exists():
                    # The existing content has already passed its LFS identity check.
                    destination.unlink()
                os.link(original, destination)
                storage = "hardlink_immutable"
            else:
                temporary = destination.with_name(destination.name + ".importing")
                if temporary.exists():
                    raise FileExistsError(f"Prior partial transfer needs inspection: {temporary}")
                shutil.copy2(original, temporary)
                if digest(temporary) != sha:
                    raise IOError(f"Temporary copy hash mismatch for {relative}")
                temporary.replace(destination)
                storage = "independent_copy"
        if digest(destination) != sha:
            raise IOError(f"Post-copy hash mismatch for {relative}")
        records.append({"path": relative.as_posix(), "bytes": original.stat().st_size,
                        "sha256": sha, "source_group": group,
                        "source_relative_path": source_relative, "disposition": disposition,
                        "storage": storage})

    def copy_tree(relative: str, destination: str | None = None) -> None:
        directory = source / relative
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for original in sorted(directory.rglob("*")):
            child = original.relative_to(directory)
            if not original.is_file() or any(part in SKIP for part in child.parts):
                continue
            copy_file(original, Path(destination or relative) / child,
                      "scheduler_encdec", original.relative_to(source).as_posix())

    for relative in ["Archive/EncDec/__init__.py", "Archive/EncDec/Models/__init__.py", "Infrastructure/dependencies.py", "Infrastructure/errors.py"]:
        copy_file(source / relative, Path(relative), "scheduler_encdec", relative)
    for relative in ["Archive/EncDec/Research", "Archive/EncDec/Codebooks", "Archive/EncDec/Models/DeepJSCC", "Archive/EncDec/Models/SwinJSCC",
                     "Infrastructure/Artifacts", "Infrastructure/Interfaces", "Infrastructure/Registries", "Infrastructure/Channels"]:
        copy_tree(relative)
    for original, destination in [
        ("references/DeepJSCC_checkpoint_source", "Models/DeepJSCC/checkpoint_source"),
        ("references/NTSCC_JSAC22_official", "Models/NTSCC/official"),
        ("external/SwinJSCC", "Models/SwinJSCC/local_source"),
        ("models/SwinJSCC", "Models/trained_checkpoints/SwinJSCC/encdec_sa_ra"),
        ("models/Codebooks", "Models/trained_checkpoints/Codebooks/encdec"),
    ]:
        copy_tree(original, destination)
    for model in ["DeepJSCC", "NTSCC", "ADJSCC"]:
        directory = source / "models" / model
        for original in sorted(directory.rglob("*")):
            if original.is_file() and original.suffix.lower() in {".pt", ".pth", ".pkl", ".yaml", ".json"}:
                relative = original.relative_to(directory)
                if relative.name == "calibration_provenance.json":
                    relative = relative.with_name("encdec_calibration_provenance.json")
                copy_file(original, Path("Models/trained_checkpoints") / model / relative,
                          "scheduler_encdec", original.relative_to(source).as_posix())
    for relative in [
        "DeepJSCC/production/imagenet_10_0.33_200.00_32_19.pth",
        "NTSCC/production/ntscc_hyperprior_quality_1_psnr.pth",
    ]:
        copy_file(supplemental / relative, Path("Models/trained_checkpoints") / relative,
                  "supplemental_production_checkpoint", relative)

    # Importing the codec contracts must not import the scheduler's Core pipeline.
    init = ROOT / "Infrastructure/__init__.py"
    content = '"""Retained codec contracts and channels; no scheduling runtime."""\n'
    if init.exists() and init.read_text(encoding="utf-8") != content:
        raise FileExistsError(init)
    init.write_text(content, encoding="utf-8")
    source_commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "source_repository": "saltypal/SemanticSchedulerEnd2End",
               "source_commit": source_commit, "source_identity": "working files verified individually",
               "source_files_removed": False, "inference_validation": "not_run_by_importer",
               "file_count": len(records), "total_bytes": sum(r["bytes"] for r in records), "files": records}
    path = ROOT / "Docs/encdec_import_manifest.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ["file_count", "total_bytes", "source_commit"]}), flush=True)
    print(f"Manifest: {path}", flush=True)


if __name__ == "__main__":
    main()
