"""Run an upstream entry point from a verified, separate execution copy.

This deliberately cannot turn a missing training release into an implementation.
Arguments are the upstream script's own arguments. Data symlinks and artifacts
live in the execution copy; the locked upstream snapshot is never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINTS = {
    "ADJSCC": ["adjscc_imagenet.py", "adjscc_cifar10.py"],
    "NTSCC": ["main.py"],
    "NTSCC_plus": ["main.py"],
    "SwinJSCC": ["main.py"],
    "SQ-GAN": ["train.py", "sample.py"],
}


def verify(folder, manifest):
    for name, digest in manifest["files_sha256"].items():
        path = folder / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Upstream source differs: {name}")
    expected_python = {name for name in manifest["files_sha256"] if name.endswith(".py")}
    extra = {p.relative_to(folder).as_posix() for p in folder.rglob("*.py")} - expected_python
    if extra:
        raise ValueError(f"Unexpected Python modules in upstream tree: {sorted(extra)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=ENTRYPOINTS)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--execution-dir", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--arguments-json", required=True, type=Path,
                        help="JSON list of arguments accepted by the original entry point")
    parser.add_argument("--dataset-links-json", type=Path,
                        help="Mapping of author-relative input paths to existing datasets")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.entrypoint not in ENTRYPOINTS[args.model]:
        raise ValueError("Entry point is not an upstream training/evaluation script")
    manifests = list((ROOT / "Models/author_releases" / args.model).glob("*.manifest.json"))
    if len(manifests) != 1:
        raise ValueError("Select one explicitly pinned source release before launching")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    source = ROOT / manifest["source_directory"]
    verify(source, manifest)
    destination = args.execution_dir.resolve()
    if destination == source.resolve() or destination.is_relative_to(source.resolve()):
        raise ValueError("Execution copy must be separate from locked source")
    if not destination.exists():
        shutil.copytree(source, destination)
    verify(destination, manifest)
    links = json.loads(args.dataset_links_json.read_text(encoding="utf-8")) if args.dataset_links_json else {}
    for relative, target_name in links.items():
        target = Path(target_name).resolve()
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Dataset link must be an author-relative path without traversal")
        link = destination / relative_path
        # Resolve the parent, not an existing final symlink into the dataset.
        if not link.parent.resolve(strict=False).is_relative_to(destination) or not target.exists():
            raise ValueError("Invalid dataset link or missing input")
        if link.exists() or link.is_symlink():
            if link.resolve() != target:
                raise ValueError(f"Existing input path does not match requested dataset: {relative}")
        else:
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(target, target_is_directory=target.is_dir())
    arguments = json.loads(args.arguments_json.read_text(encoding="utf-8"))
    if not isinstance(arguments, list) or not all(isinstance(v, str) for v in arguments):
        raise ValueError("arguments-json must contain a list of strings")
    command = [args.python, str(destination / args.entrypoint), *arguments]
    receipt = {"model": args.model, "repository": manifest["repository"], "commit": manifest["commit"],
               "source_files_verified": True, "model_source_edits": [],
               "command": command, "dataset_links": links,
               "command_prepared": True, "execution_started": False,
               "paper_reproduction_verified": False, "matched_comparison_verified": False,
               "note": "An unchanged entry point does not by itself certify datasets, dependencies, hyperparameters, channel equivalence, or measured completion."}
    receipt_path = destination.parent / (destination.name + ".launch.json")
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    if not args.execute:
        return
    env = dict(os.environ)
    # Exclude the old adaptation/sitecustomize folder from these launches.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONSTARTUP", None)
    env["WANDB_MODE"] = "offline"
    receipt["execution_started"] = True
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    result = subprocess.run(command, cwd=destination, env=env)
    verify(source, manifest)
    verify(destination, manifest)
    receipt["exit_code"] = result.returncode
    receipt["source_files_verified_after_execution"] = True
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
