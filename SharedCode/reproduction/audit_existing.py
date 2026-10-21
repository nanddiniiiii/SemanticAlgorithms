"""Inventory source differences and experiment deviations without editing artifacts."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Results/reproduction_audit"
EXISTING = {
    "ADJSCC": "Models/ADJSCC/official",
    "NTSCC": "Models/NTSCC/official",
    "NTSCC_plus": "Models/NTSCC++/official",
    "SwinJSCC": "Models/SwinJSCC/official_working_copy",
    "SQ-GAN": "Models/SQ-GAN/official",
    "DiT-JSCC": "Models/DiT-JSCC/official_working_copy",
    "DeepJSCC-third-party": "Models/DeepJSCC/checkpoint_source",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reports = []
    for path in sorted((ROOT / "Models/author_releases").glob("*/*.manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        base = ROOT / manifest["source_directory"]
        changed_snapshot = [name for name, digest in manifest["files_sha256"].items()
                            if not (base / name).is_file()
                            or hashlib.sha256((base / name).read_bytes()).hexdigest() != digest]
        old = ROOT / EXISTING[manifest["model"]] if manifest["model"] in EXISTING else None
        same, missing, modified, line_endings = [], [], [], []
        if old is not None:
            for name, digest in manifest["files_sha256"].items():
                p = old / name
                if not p.is_file():
                    missing.append(name)
                elif hashlib.sha256(p.read_bytes()).hexdigest() == digest:
                    same.append(name)
                elif p.read_bytes().replace(b"\r\n", b"\n") == (base / name).read_bytes().replace(b"\r\n", b"\n"):
                    line_endings.append(name)
                else:
                    modified.append(name)
        reports.append({"model": manifest["model"], "commit": manifest["commit"],
                        "provenance": manifest["provenance"],
                        "snapshot_integrity": "PASS" if not changed_snapshot else "FAIL",
                        "snapshot_changed": changed_snapshot,
                        "previous_folder": str(old) if old else None,
                        "identical": same, "line_endings_only": line_endings,
                        "missing_from_previous_copy": missing, "different_from_current_upstream": modified,
                        "note": "A different file may reflect an older upstream revision. This comparison alone does not establish a local architecture edit. Byte-identical files do not exclude runtime monkey patches or altered training settings."})
    payload = {"checked_at": datetime.now(timezone.utc).isoformat(), "source_comparisons": reports,
               "certified_author_reproductions": [], "certified_matched_comparisons": [],
               "execution": "Source inspection only; no model training, inference, or numerical equivalence test performed."}
    (OUT / "existing_source_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for row in reports:
        print(row["model"], "snapshot=" + row["snapshot_integrity"],
              "missing_previous=" + str(len(row["missing_from_previous_copy"])),
              "different_previous=" + str(len(row["different_from_current_upstream"])))


if __name__ == "__main__":
    main()
