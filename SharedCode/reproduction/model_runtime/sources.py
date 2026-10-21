"""Verify fixed snapshot bytes before importing any upstream model code."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
RELEASES = {"DeepJSCC": ("DeepJSCC-third-party", "2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06"),
            "SwinJSCC": ("SwinJSCC", "a6d0e6da53548976acbe9317839a077ef31f190f"),
            "NTSCC_plus": ("NTSCC_plus", "765c6f6c11cf3bbe249a26b624dbaf47a4f0676d"),
            "ADJSCC": ("ADJSCC", "e5332e95faf592aab9f440992de96029162dc7dd"),
            "DeepJSCC_Q": ("DeepJSCC-Q-candidate", "da863de5e3acb4fbb307c0cf8dfe830303e99aa7"),
            "SQGAN": ("SQ-GAN", "0e5cdd7c0240b89749dad684645c53f783ac24cc")}


def verified_source(model, root=ROOT):
    release, commit = RELEASES[model]
    manifest_path = Path(root) / "Models/author_releases" / release / (commit + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["commit"] != commit:
        raise ValueError("Pinned manifest commit differs")
    source = Path(root) / "Models/author_releases" / release / commit
    for name, digest in manifest["files_sha256"].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != digest:
            raise ValueError("Changed upstream source: " + name)
    expected = {name for name in manifest["files_sha256"] if name.endswith(".py")}
    if {p.relative_to(source).as_posix() for p in source.rglob("*.py")} != expected:
        raise ValueError("Unexpected Python modules in source snapshot")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    return source, manifest
