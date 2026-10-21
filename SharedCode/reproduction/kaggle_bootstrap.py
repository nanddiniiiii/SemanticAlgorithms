"""Reset source by cloning a fresh run directory; preserve all saved artifacts.

This fetches the latest benchmark-repository commit once per notebook run.
Upstream model revisions remain separately pinned by the source manifests.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess


def checkout_latest(repository_url, branch="main", working_root="/kaggle/working"):
    if not re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?", repository_url):
        raise ValueError("Supply a public GitHub repository URL without credentials")
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", branch) or branch.startswith("-"):
        raise ValueError("Invalid branch")
    working_root = Path(working_root).resolve()
    runs = working_root / "semanticlossless_runs"
    runs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = runs / stamp / "code"
    destination.parent.mkdir(parents=True, exist_ok=False)
    subprocess.run(["git", "clone", "--depth", "1", "--branch", branch,
                    "--", repository_url, str(destination)], check=True)
    commit = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    receipt = {"repository": repository_url, "branch": branch, "commit": commit,
               "fetched_at": datetime.now(timezone.utc).isoformat(), "code": str(destination),
               "reset_scope": "fresh source checkout only", "previous_artifacts_deleted": False}
    (destination.parent / "checkout_receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def estimate_budget(step_seconds, batch_size, complete_epochs=150, source_images=800,
                    evaluation_seconds=0, setup_seconds=0, remaining_gpu_hours=30,
                    quota_multiplier=1):
    """Measured estimate only. Quota accounting multiplier is supplied, never guessed."""
    import math
    if min(step_seconds, batch_size, complete_epochs, source_images, quota_multiplier) <= 0:
        raise ValueError("Positive measured timing and workload values required")
    if source_images % batch_size:
        raise ValueError("Select a global batch dividing 800; no dropped source images")
    seconds = step_seconds * (source_images // batch_size) * complete_epochs + evaluation_seconds + setup_seconds
    hours = seconds * quota_multiplier / 3600
    return {"estimated_wall_seconds": seconds, "estimated_quota_hours": hours,
            "within_remaining_quota": hours <= remaining_gpu_hours,
            "epochs": complete_epochs, "images_per_epoch": source_images,
            "measurement_required": True, "guarantee": False}
