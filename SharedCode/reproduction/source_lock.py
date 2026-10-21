"""Fetch complete upstream snapshots; record commit, archive and file hashes.

This downloads source only. It does not import models, train, or certify that a
paper's experiment has been reproduced. Existing model folders are untouched.
"""
from __future__ import annotations

import concurrent.futures
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile
from datetime import datetime, timezone

import requests

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "Models/author_releases"
REPORTS = ROOT / "Results/reproduction_audit"
CATALOG = [
    ("ADJSCC", "alexxu1988/ADJSCC", "HEAD", "author_release"),
    ("NTSCC", "wsxtyrdd/NTSCC_JSAC22", "HEAD", "author_release"),
    ("NTSCC_plus", "wsxtyrdd/NTSCC_plus", "HEAD", "author_release_plus_only"),
    ("SwinJSCC", "semcomm/SwinJSCC", "HEAD", "author_release"),
    ("SQ-GAN", "frapez1/SQ-GAN", "0e5cdd7c0240b89749dad684645c53f783ac24cc", "author_release"),
    ("DiT-JSCC", "semcomm/DiTJSCC", "HEAD", "author_inference_release"),
    ("DeepJSCC-Q-candidate", "r0ymanesco/deepjscc_stream", "HEAD", "authorship_not_yet_verified"),
    ("DeepJSCC-third-party", "chunbaobao/Deep-JSCC-PyTorch", "HEAD", "third_party_reproduction"),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch(entry):
    name, repo, ref, provenance = entry
    response = requests.get(f"https://api.github.com/repos/{repo}/commits/{ref}", timeout=(20, 90))
    response.raise_for_status()
    commit = response.json()["sha"]
    destination = SOURCES / name / commit
    manifest_path = destination.parent / f"{commit}.manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for path, digest in manifest["files_sha256"].items():
            if sha((destination / path).read_bytes()) != digest:
                raise RuntimeError(f"Snapshot changed: {name}/{path}")
        return manifest
    archive_url = f"https://codeload.github.com/{repo}/zip/{commit}"
    response = requests.get(archive_url, timeout=(20, 180))
    response.raise_for_status()
    archive = response.content
    files = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        for item in zipped.infolist():
            if item.is_dir():
                continue
            path = PurePosixPath(item.filename)
            relative = PurePosixPath(*path.parts[1:])
            if not relative.parts or ".." in relative.parts or relative.is_absolute():
                raise ValueError(f"Unsafe archive path: {path}")
            output = destination.joinpath(*relative.parts)
            if not output.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Archive path escapes snapshot")
            content = zipped.read(item)
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists() and output.read_bytes() != content:
                raise RuntimeError(f"Refusing to overwrite different source: {output}")
            output.write_bytes(content)
            files[str(relative)] = sha(content)
    manifest = {"model": name, "repository": f"https://github.com/{repo}",
                "commit": commit, "provenance": provenance,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "archive_url": archive_url, "archive_sha256": sha(archive),
                "source_directory": str(destination.relative_to(ROOT)).replace("\\", "/"),
                "files_sha256": files, "file_count": len(files),
                "execution_verified": False, "paper_reproduction_verified": False}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-file", type=Path, default=ROOT / "sources.lock.json")
    parser.add_argument("--refresh", action="store_true", help="Explicitly resolve upstream HEAD; never used by benchmark runs")
    args = parser.parse_args()
    if args.refresh:
        catalog = CATALOG
    else:
        locked = json.loads(args.lock_file.read_text(encoding="utf-8"))
        entries = locked["sources"] if isinstance(locked, dict) else locked
        catalog = []
        for entry in entries:
            commit = entry["commit"]
            if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
                raise ValueError("Every upstream source must use an immutable 40-character commit")
            repo = entry["repository"].removeprefix("https://github.com/")
            catalog.append((entry["model"], repo, commit, entry["provenance"]))
    REPORTS.mkdir(parents=True, exist_ok=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        jobs = {executor.submit(fetch, entry): entry[0] for entry in catalog}
        for future in concurrent.futures.as_completed(jobs):
            name = jobs[future]
            try:
                manifest = future.result()
                results.append(manifest)
                print(name, manifest["commit"], manifest["file_count"], flush=True)
            except Exception as exc:
                results.append({"model": name, "error": str(exc)})
                print(name, "FETCH_FAILED", str(exc), flush=True)
    (REPORTS / "source_lock.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if any("error" in result for result in results):
        raise SystemExit("Some upstream sources could not be fetched; see source_lock.json")


if __name__ == "__main__":
    main()
