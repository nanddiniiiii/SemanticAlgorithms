"""Stream a saved Kaggle SQ-GAN checkpoint, then inspect its real training state."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    owner, slug = args.kernel.split("/", 1)
    api = KaggleApi()
    api.authenticate()
    token = None
    target = None
    with api.build_kaggle_client() as client:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner
            request.kernel_slug = slug
            api._set_paging(request, 200, token)
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            if response.log:
                (args.output / "remote_run_log.json").write_text(response.log, encoding="utf-8")
            target = next((f for f in response.files or [] if f.file_name.endswith("/checkpoints/last.ckpt")), None)
            if target:
                break
            token = response.next_page_token
            if not token:
                raise RuntimeError("The Kaggle output manifest contains no last.ckpt")

    destination = args.output / target.file_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".ckpt.partial")
    started = time.monotonic()
    digest = hashlib.sha256()
    with requests.get(target.url, headers={"Range": "bytes=0-0"}, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        assert response.status_code == 206, "Server must support byte-range downloads"
        expected_size = int(response.headers["Content-Range"].split("/")[-1])
    pieces = destination.parent / "download_parts"
    pieces.mkdir(exist_ok=True)
    block_size = 16 * 1024 * 1024
    ranges = [(start, min(start + block_size, expected_size) - 1) for start in range(0, expected_size, block_size)]

    def download_piece(bounds):
        start, end = bounds
        piece = pieces / f"{start:012d}.part"
        size = end - start + 1
        if piece.exists() and piece.stat().st_size == size:
            return piece
        for attempt in range(40):
            try:
                already = piece.stat().st_size if piece.exists() else 0
                if already == size:
                    return piece
                if already > size:
                    raise ValueError("Unexpected range file size")
                offset = start + already
                # Persist partial bytes and resume them after connection truncation.
                with requests.get(target.url, headers={"Range": f"bytes={offset}-{end}"}, stream=True, timeout=(30, 45)) as response:
                    response.raise_for_status()
                    assert response.status_code == 206
                    assert response.headers["Content-Range"].startswith(f"bytes {offset}-")
                    with piece.open("ab") as stream:
                        for block in response.iter_content(256 * 1024):
                            stream.write(block)
                if piece.stat().st_size != size:
                    raise IOError("Incomplete range")
                return piece
            except (requests.RequestException, IOError):
                if attempt == 39:
                    raise
                time.sleep(min(attempt + 1, 4))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(download_piece, bounds) for bounds in ranges]
        completed_bytes = 0
        for job in as_completed(jobs):
            completed_bytes += job.result().stat().st_size
            print(f"Downloaded {completed_bytes / 2**20:.1f}/{expected_size / 2**20:.1f} MiB", flush=True)
    with partial.open("wb") as stream:
        for start, end in ranges:
            with (pieces / f"{start:012d}.part").open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    stream.write(block)
                    digest.update(block)
        stream.flush()
        os.fsync(stream.fileno())
    if partial.stat().st_size != expected_size:
        raise RuntimeError("Merged checkpoint size does not match the remote object")
    if args.expected_sha256 and digest.hexdigest() != args.expected_sha256:
        raise RuntimeError("Checkpoint checksum differs from the independently validated cloud copy")
    os.replace(partial, destination)

    import torch
    checkpoint = torch.load(destination, map_location="cpu", weights_only=False)
    optimizers = checkpoint.get("optimizer_states", [])
    schedulers = checkpoint.get("lr_schedulers", [])
    report = {
        "kernel": args.kernel,
        "file": str(destination),
        "remote_file": target.file_name,
        "bytes": destination.stat().st_size,
        "sha256": digest.hexdigest(),
        "saved_epoch_index": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "state_dict_tensors": len(checkpoint.get("state_dict", {})),
        "optimizer_states": len(optimizers),
        "lr_schedulers": len(schedulers),
        "optimizer_learning_rates": [[g.get("lr") for g in o.get("param_groups", [])] for o in optimizers],
        "loop_state_keys": list(checkpoint.get("loops", {})),
        "checkpoint_keys": list(checkpoint),
        "seconds": time.monotonic() - started,
    }
    assert report["state_dict_tensors"] > 0, "Missing model parameters"
    assert len(optimizers) == 2, "Expected the author's two optimizer states"
    assert schedulers, "Missing learning-rate scheduler state"
    (args.output / "checkpoint_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
