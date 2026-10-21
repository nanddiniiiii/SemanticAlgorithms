"""Continue successful Kaggle chunks until all six stages and evaluation finish.

Uses existing local Kaggle CLI authentication. No credentials enter a notebook.
Each private continuation notebook mounts the preceding successful notebook output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

ROOT = Path(__file__).resolve().parents[2]
PIN = "0e5cdd7c0240b89749dad684645c53f783ac24cc"
CANONICAL = ROOT / "Notebooks/project_notebooks/Training/sqgan_div2k_channel"
STATE_ROOT = ROOT / "Results/SQGAN/continuation"


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def get_outputs(api, kernel):
    owner, slug = kernel.split("/", 1)
    token = None
    files = []
    with api.build_kaggle_client() as client:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = owner, slug
            api._set_paging(request, 200, token)
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            files.extend(response.files or [])
            token = response.next_page_token
            if not token:
                return files


def complete_chunk(api, state):
    files = get_outputs(api, state["kernel"])
    item = next((f for f in files if f.file_name.endswith("/run_manifest.json")), None)
    if item is None:
        raise RuntimeError("Successful kernel has no run_manifest.json; refusing an unverified continuation")
    response = requests.get(item.url, timeout=(30, 90))
    response.raise_for_status()
    manifest = response.json()
    assert manifest["upstream_commit"] == PIN
    assert manifest["epochs_per_stage"] == 150
    assert set(manifest["channels"]) == {"awgn", "rayleigh"}
    expected = {(c, m) for c in ["awgn", "rayleigh"] for m in ["ssm", "img", "all"]}
    stages = manifest["stages"]
    assert {(s["channel"], s["mode"]) for s in stages}.issubset(expected)
    for stage in stages:
        assert stage["status"] in {"partial", "complete"}
        if stage["status"] == "complete":
            assert stage["completed_epochs"] >= 150
        if "optimizer_states" in stage:
            assert stage["optimizer_states"] == stage["lr_schedulers"] == 2
    folder = STATE_ROOT / f"chunk_{state['chunk']:02d}"
    folder.mkdir(exist_ok=True)
    save(folder / "run_manifest.json", manifest)
    save(folder / "remote_files.json", [{"path": f.file_name} for f in files])
    if manifest.get("training_complete") and manifest.get("comparison_ready"):
        assert len(stages) == 6 and all(s["completed_epochs"] >= 150 for s in stages)
        for f in files:
            if f.file_name.endswith((".csv", "kodak_summary.json", "final_checkpoint.json")):
                response = requests.get(f.url, timeout=(30, 90))
                response.raise_for_status()
                dest = folder / f.file_name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(response.content)
        state.update(status="COMPLETE", completed_at=now(), manifest=str(folder/"run_manifest.json"))
        save(STATE_ROOT / "state.json", state)
        print("ALL SIX 150-EPOCH STAGES AND KODAK/FLOPS EVALUATION COMPLETE", flush=True)
        return False

    next_chunk = state["chunk"] + 1
    if next_chunk > 32:
        raise RuntimeError("32-chunk safeguard reached; inspect progress before allocating more GPU time")
    next_id = f"hmmkk20/sqgan-div2k-150ep-cont{next_chunk:02d}-20261020"
    next_dir = ROOT / f"Notebooks/project_notebooks/Training/sqgan_chunks/continuation_{next_chunk:02d}"
    next_dir.mkdir(parents=True, exist_ok=True)
    source_name = "sqgan_div2k_awgn_rayleigh_training.ipynb"
    snapshot = STATE_ROOT/"source_snapshot_v2.ipynb"
    if not snapshot.exists():
        snapshot = STATE_ROOT/"source_snapshot.ipynb"
    shutil.copy2(snapshot, next_dir/source_name)
    metadata = json.loads((STATE_ROOT/"metadata_snapshot.json").read_text())
    metadata.update(id=next_id, title=f"SQGAN DIV2K 150ep cont{next_chunk:02d} 20261020", machine_shape="NvidiaTeslaT4", is_private=True, enable_gpu=True)
    metadata["kernel_sources"] = ["jagan028/div2k-dataset-generation-for-isr", state["kernel"]]
    save(next_dir/"kernel-metadata.json", metadata)
    # This operation is authorized training continuation; publish only to a private Kaggle notebook.
    result = api.kernels_push(str(next_dir), acc="NvidiaTeslaT4")
    reply = result.to_dict()
    if reply.get("error"):
        raise RuntimeError(f"Kaggle rejected the continuation: {reply['error']}")
    # Kaggle can normalize a newly created slug from its title. Follow its returned URL.
    from urllib.parse import urlparse
    if reply.get("url"):
        parts = urlparse(reply["url"]).path.strip("/").split("/")
        if len(parts) >= 3 and parts[0] == "code":
            next_id = "/".join(parts[1:3])
    state.update(kernel=next_id, expected_version=1, chunk=next_chunk, status="SUBMITTED", submitted_at=now(), previous_kernel=state["kernel"])
    save(STATE_ROOT/"state.json", state)
    print("CONTINUATION SUBMITTED", next_id, "with previous successful output attached", flush=True)
    return True


def main():
    # The Kaggle SDK omits a request timeout. Bound network waits in this local
    # monitor so a broken connection reaches the existing retry loop.
    original_send = requests.Session.send
    def bounded_send(session, request, **kwargs):
        kwargs.setdefault("timeout", (20, 90))
        return original_send(session, request, **kwargs)
    requests.Session.send = bounded_send
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Inspect one state; a running job is never replaced")
    args = parser.parse_args()
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    lock = STATE_ROOT / "watcher.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    try:
        if not (STATE_ROOT/"source_snapshot.ipynb").exists():
            shutil.copy2(CANONICAL/"sqgan_div2k_awgn_rayleigh_training.ipynb", STATE_ROOT/"source_snapshot.ipynb")
            shutil.copy2(CANONICAL/"kernel-metadata.json", STATE_ROOT/"metadata_snapshot.json")
        state_file = STATE_ROOT/"state.json"
        state = json.loads(state_file.read_text()) if state_file.exists() else {"kernel":"hmmkk20/sqgan-div2k-150ep-20261019", "expected_version":16, "chunk":1, "status":"STARTING", "source_sha256":hashlib.sha256((STATE_ROOT/"source_snapshot.ipynb").read_bytes()).hexdigest()}
        state["watcher_pid"] = os.getpid()
        api = KaggleApi()
        api.authenticate()
        errors = 0
        while True:
            if (STATE_ROOT/"STOP").exists():
                state.update(status="WATCHER_STOPPED", stopped_at=now())
                save(state_file,state)
                print("Watcher stopped; the existing Kaggle execution is left running", flush=True)
                return
            try:
                status = api.kernels_status(state["kernel"]).to_dict()["status"]
                owner, slug = state["kernel"].split("/",1)
                with api.build_kaggle_client() as client:
                    request=ApiGetKernelRequest(); request.user_name=owner; request.kernel_slug=slug
                    actual_version=client.kernels.kernels_api_client.get_kernel(request).metadata.to_dict()["currentVersionNumber"]
                assert actual_version == state["expected_version"], f"Unexpected notebook edit: version {actual_version}"
                if status != state.get("status"):
                    print(now(),state["kernel"],status,flush=True)
                state.update(status=status,checked_at=now())
                save(state_file,state)
                if status == "COMPLETE":
                    if not complete_chunk(api,state): return
                elif status not in {"RUNNING","QUEUED","PENDING","INITIALIZING"}:
                    raise RuntimeError(f"Kaggle ended with {status}; inspect logs before resubmitting")
                errors = 0
                for key in ("last_error", "error_at", "consecutive_errors"):
                    state.pop(key, None)
                save(state_file, state)
            except Exception as exc:
                errors += 1
                state.update(last_error=repr(exc),error_at=now(),consecutive_errors=errors)
                save(state_file,state)
                print("CONTINUATION CHECK ERROR",repr(exc),flush=True)
                if errors >= 5:
                    state["status"]="NEEDS_ATTENTION"; save(state_file,state)
                    return
            if args.once:
                return
            time.sleep(60)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
