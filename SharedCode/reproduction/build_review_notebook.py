"""Create an honest consolidated reproduction status and native-launch notebook."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Results/reproduction_audit"
ISSUES = {
    "DeepJSCC": ("third_party", "Original-author 2019 code unverified; used repository is a third-party reproduction."),
    "ADJSCC": ("author_training_available", "Native ImageNet data/runtime and AWGN initialization/protocol must be reproduced; previous run is a DIV2K adaptation."),
    "NTSCC": ("author_training_available", "Separate 2022 model; required initialization, paths and 5000-epoch native-loop constraint unresolved."),
    "NTSCC_plus": ("author_training_available", "Restore original NATTEN-compatible environment and author training path; previous run used 128 images/epoch and added fading."),
    "NTSCC++": ("online_code_unreleased", "Author README marks online NTSCC++ inference as TODO; cannot rename NTSCC+ to NTSCC++."),
    "SwinJSCC": ("author_training_available", "Base w/o-SA-and-RA architecture matches source settings; training/initialization and common evaluation still require runtime validation."),
    "SQ-GAN": ("author_training_available", "Previous analog channel/pseudo-label adaptation differs from the published codec; native Cityscapes reproduction and external digital transport study are distinct."),
    "DeepJSCC-Q": ("candidate_source_found", "Dedicated trainer found in r0ymanesco/deepjscc_stream; authorship and journal-variant fidelity not established. Old simplified notebook is not certified."),
    "DiT-JSCC": ("author_inference_only", "Public README explicitly omits training workspace; original training code/weights required."),
    "DDM-SSCC": ("author_code_unverified", "No author release verified; exact lossless codec/transport cannot be certified from a substitute model."),
}


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [{"model": model, "release_status": status,
             "author_experiment_certified": False, "fair_comparison_eligible": False,
             "reason": reason} for model, (status, reason) in ISSUES.items()]
    report = {"updated_at": datetime.now(timezone.utc).isoformat(), "models": rows,
              "all_models_complete": False, "gpu_launches_by_this_correction": 0,
              "gpu_blocker": "Last confirmed Kaggle error: Maximum weekly GPU quota of 30.00 hours reached.",
              "old_checkpoints_preserved": True,
              "old_sqgan_coordinator_dispatch": "STOP guard installed",
              "legacy_sqgan_awgn_semantic_epochs": 150,
              "legacy_sqgan_run_is_author_exact": False,
              "proposed_common_training_geometry": [256, 512],
              "proposed_evaluation_protocol": "matched_tiled_v1",
              "protocol_runtime_validated": False}
    (OUT / "experiment_eligibility.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (OUT / "comparison_eligibility.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = "| Model | Release | Required correction |\n|---|---|---|\n" + "\n".join(
        f"| {r['model']} | {r['release_status']} | {r['reason']} |" for r in rows)
    cells = [cell("markdown", "# Author reproduction and matched comparison — correction workspace\n\n"
                  "**No all-model reproduction/comparison is certified yet.** This notebook records the audit and gives a launcher for original scripts. "
                  "It contains no fabricated model metrics and has not been executed on Kaggle. "
                  "Original author experiments and the controlled DIV2K/Kodak benchmark are reported separately. "
                  "Existing results/checkpoints remain historical artifacts.\n\n" + table),
             cell("markdown", "## 1. Locate the workspace\n\nOn Kaggle, attach a private dataset containing the reproduction scripts, locked source manifests/snapshots and audit JSON files. "
                  "Set SEMANTICLOSSLESS_ROOT to its mounted root. This notebook does not upload anything or allocate GPUs automatically."),
             cell("code", "import os, json, pathlib, subprocess, sys\n"
                  "ROOT = pathlib.Path(os.environ.get('SEMANTICLOSSLESS_ROOT', " + repr(str(ROOT)) + "))\n"
                  "assert (ROOT / 'SharedCode/reproduction').is_dir(), 'Attach the real workspace files before continuing'\n"
                  "status = json.loads((ROOT / 'Results/reproduction_audit/experiment_eligibility.json').read_text())\n"
                  "print(json.dumps(status, indent=2))\n"),
             cell("markdown", "## 2. Verify upstream source bytes\n\nThis checks saved source hashes, including dataset loaders. "
                  "It does not prove that a notebook used an unchanged runtime or that training completed."),
             cell("code", "import hashlib\n"
                  "locks = json.loads((ROOT / 'Results/reproduction_audit/source_lock.json').read_text())\n"
                  "for lock in locks:\n"
                  "    assert 'error' not in lock, lock\n"
                  "    source = ROOT / lock['source_directory']\n"
                  "    for name, digest in lock['files_sha256'].items():\n"
                  "        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, (lock['model'], name)\n"
                  "    print(lock['model'], lock['commit'], lock['file_count'], 'source bytes verified')\n"),
             cell("markdown", "## 3. Native author training entry points\n\n"
                  "Use the original scripts and their own arguments in their required environments. "
                  "Provide real datasets and any required author initialization checkpoints. "
                  "The example below is SQ-GAN's original Cityscapes semantic-stage command. It does not use the previous analog channel or pseudo-label wrapper. "
                  "The image stage and joint stage follow the original README and require their proper initialization/split components. "
                  "The example has not been run here; dependency/data availability is not assumed."),
             cell("code", "AUTHOR_PYTHON = os.environ.get('AUTHOR_PYTHON', sys.executable)\n"
                  "RUN_ROOT = pathlib.Path('/kaggle/working/author_reproduction') if pathlib.Path('/kaggle').exists() else ROOT / 'Results/author_reproduction'\n"
                  "RUN_ROOT.mkdir(parents=True, exist_ok=True)\n"
                  "arguments_path = RUN_ROOT / 'sqgan_ssm_args.json'\n"
                  "arguments_path.write_text(json.dumps(['--mode','ssm','--gpus','-1','--base','config/sqgan_cityscapes.yml','--max_epochs','150']))\n"
                  "CITYSCAPES_ROOT = os.environ.get('CITYSCAPES_ROOT')\n"
                  "assert CITYSCAPES_ROOT and pathlib.Path(CITYSCAPES_ROOT).is_dir(), 'Original Cityscapes RGB and gtFine train-ID masks are required'\n"
                  "links_path = RUN_ROOT / 'sqgan_dataset_links.json'\n"
                  "links_path.write_text(json.dumps({'path/to/cityscapes': CITYSCAPES_ROOT}))\n"
                  "command = [sys.executable, str(ROOT / 'SharedCode/reproduction/launch_upstream.py'), '--model','SQ-GAN', '--entrypoint','train.py', '--python',AUTHOR_PYTHON, '--execution-dir',str(RUN_ROOT / 'SQGAN_native_ssm'), '--arguments-json',str(arguments_path), '--dataset-links-json',str(links_path)]\n"
                  "print(command)\n"
                  "# Prepare and verify the execution copy; no training in this cell.\n"
                  "subprocess.run(command, check=True)\n"),
             cell("code", "# Run this cell only in the original compatible GPU environment with verified inputs.\n"
                  "# This executes the real original trainer; all errors propagate.\n"
                  "# subprocess.run(command + ['--execute'], check=True)\n"),
             cell("markdown", "## 4. Controlled DIV2K/Kodak evaluation\n\n"
                  "Use matched_evaluation.py after each author-code model callback and transport have been validated. "
                  "Every Kodak pixel is reconstructed; metrics use full original images. All padded tiles count toward FLOPs and channel uses. "
                  "Do not pass the old SQ-GAN analog vector adapter as the original discrete-index system. "
                  "Author-native fading assumptions differ, and unsupported native channels cannot be silently replaced. "
                  "The comparison validator rejects incomplete or incompatible run manifests. No callbacks/results are invented here."),
             cell("code", "# Empty until real measured, reviewed runs exist. No fallback to historical scores.\n"
                  "eligible_run_manifests = []\n"
                  "if len(eligible_run_manifests) < 2:\n"
                  "    print('No certified matched ranking available yet.')\n"
                  "else:\n"
                  "    sys.path.insert(0, str(ROOT / 'SharedCode/reproduction'))\n"
                  "    from matched_evaluation import validate_comparison\n"
                  "    print(validate_comparison(eligible_run_manifests))\n")]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    folder = ROOT / "Notebooks/AuthorReproduction"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "Author_Reproduction_and_Matched_Comparison.ipynb"
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
