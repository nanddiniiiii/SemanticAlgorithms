# Kaggle Stage 1 Run Map

Canonical operations record for Stage 1. Scope is existing-method evaluation and the Shared Semantic Codebook classifier prerequisite. Stage 2 and diffusion-model training are excluded. Do not infer quality results from a launched job or source bundle.

## Account access and quota check

Verified through `scripts/kaggle_account.py` on 2026-10-25:

| Account | Kaggle identity | CLI path | GPU quota | TPU quota | Refresh |
|---|---|---|---:|---:|---|
| A | `satyapaladugu` | Windows Kaggle CLI, identity wrapper | 30.00 h remaining | 20.00 h remaining | 2026-10-31 00:00 UTC |
| B | `satyapaladuguplus` | WSL CLI `/home/Satya/.venvs/semantic-kaggle/bin/kaggle`, isolated config `/home/Satya/.config/semantic-lossless/kaggle/account_b` | 30.00 h remaining | 20.00 h remaining | 2026-10-31 00:00 UTC |

No credentials or tokens are stored here. The verified command is `python scripts/kaggle_account.py A|B <kaggle-command>`; B export paths must be absolute `/mnt/d/...` paths. Account B was authenticated and had quota; the prior failure was not an auth, quota, or eligibility error.

## Prior failed runs and recovered evidence

| Account/version | Kernel | Last run | Status | Declared inputs | Observed environment | Failure and outputs |
|---|---|---|---|---|---|---|
| A v2 | [satyapaladugu/stage1-shared-codebook-classifier](https://www.kaggle.com/code/satyapaladugu/stage1-shared-codebook-classifier) | 2026-10-22 17:08 UTC | `KernelWorkerStatus.ERROR` | STL10 binary files; attempted working path `/kaggle/working/Semantic_Lossless/Datasets/STL10` | Python 3.13; torch 2.11.0+cu128; two Tesla T4 GPUs | `fetch_source()` called `project_root()` and raised `FileNotFoundError: Run from an editable repository checkout or set SEMANTIC_LOSSLESS_ROOT`. It failed before reading STL10 or starting the 2-step pilot. Retrieved source bundle and log only; no metrics, checkpoint, or executed notebook. |
| B v1 | [satyapaladuguplus/stage1-native-swin-evaluation](https://www.kaggle.com/code/satyapaladuguplus/stage1-native-swin-evaluation) | 2026-10-22 16:59 UTC | `KernelWorkerStatus.ERROR` | Kodak and private native Swin asset datasets; logged paths `/kaggle/input/datasets/sherylmehta/kodak-dataset` and `/kaggle/input/datasets/satyapaladuguplus/semantic-lossless-private-swin-assets/...C32.model` | Python 3.13; torch 2.11.0+cu128; CUDA 12.8; two Tesla T4 GPUs; runtime installed timm 0.4.12, pytorch-msssim 1.0.0 and lpips 0.1.4 | Same `project_root()` error while building fixed-C32. It failed before image/checkpoint loading or either evaluation. Retrieved source bundle, log and `environment.json`; no metrics, checkpoint, or executed notebook. |

Logs and the local bundle audit confirm the cause: the old archive omitted root `sources.lock.json`, required by `project_root()`. T4 hardware was allocated on both runs; these failures are not GPU-compatibility evidence. Dataset/checkpoint paths were declared, but actual mount accessibility was not reached or verified. Both were script kernels. No executed `.ipynb` file was returned; logs mention HTML conversion, but no HTML output was returned.

Exact-version status, quota and logs were accessible. `kernels files` returned an empty list for both old kernels, while `kernels output` retrieved output files into the local recovery folders. `kernels pull <kernel>/<version> -m` returned `403 Client Error: Forbidden` for both accounts, so original submitted metadata is unavailable. Retrieved files and SHA256 values are in [recovery_receipt.json](../Results/stage1_operations/recovery_receipt.json).

## Corrected, reviewed pre-launch snapshot

Both corrected jobs were generated from commit `8e23f699641d52fe98a4642cbf5ebe3aa8768cc4` with explicit working-tree edits (`working_tree_dirty=true`). The deterministic embedded ZIP includes `sources.lock.json`; its embedded digest was recomputed and verified locally.

| Job | Account / intended kernel | Local source | Kernel SHA256 | Source bundle SHA256 | Bundle bytes |
|---|---|---|---|---|---:|
| Shared Codebook classifier prerequisite | A / `satyapaladugu/stage1-shared-codebook-classifier` | `Kaggle/stage1_shared_training/kernel.py` | `b18e4e6eb0af65489178b117f06219fc24251fb168a2c1d60bd628178e8d4e3f` | `2275b3a76c1db3d4a416e15187a905cfcf2ea489fcfa889c64c6e32abb3418c1` | 163,713 |
| Native SwinJSCC evaluation | B / `satyapaladuguplus/stage1-native-swin-evaluation` | `Kaggle/stage1_swin/kernel.py` | `8f8f06107a4aa0944495cbcef1345366d058b8654ae9e5c1561bbded9c213e5d` | `2275b3a76c1db3d4a416e15187a905cfcf2ea489fcfa889c64c6e32abb3418c1` | 163,713 |

Generated metadata requests a Kaggle-managed script runtime, `NvidiaTeslaT4`, GPU and internet. Each new run must record its actual Python, package, CUDA and device report. The prior runtime report does not establish a future runtime. Expected source preflight output is `/kaggle/working/Semantic_Lossless/source_bundle.json`; stage stdout/stderr and status records go under `run_logs/` including `stages.jsonl`.

### Planned job contracts (not yet submitted)

| Account | Purpose and bounded workload | Dataset/checkpoint inputs | Expected outputs inside `/kaggle/working/Semantic_Lossless` |
|---|---|---|---|
| A | Real-data STL10 2-step classifier pilot; only on pilot success, run the 200-epoch classifier prerequisite with a 7,200-second cap. Overall timeout 9,000 seconds. This is not a trained shared-codebook codec. | `pratt3000/stl10-binary-files`; kernel locates `train_X.bin` and links its binary directory into `Datasets/STL10`. | `Results/stage1/shared_semantic_codebook/classifier_pilot/`, `classifier_training/`, `run_logs/`, `source_bundle.json`, environment/preflight output. |
| B | Native fixed-C32 and SA+RA SwinJSCC evaluation on Kodak's 24 images, including pixel/perceptual metrics and warmed encoder/decoder/full-forward profiles. Each variant timeout 3,600 seconds; overall timeout 7,200 seconds. | `sherylmehta/kodak-dataset`; `satyapaladuguplus/semantic-lossless-private-swin-assets` with both native checkpoints. | `Results/stage1/swinjscc/fixed_c32/`, `sa_ra/`, `run_logs/`, `source_bundle.json`, environment/preflight output. |

Launch the exact tested kernels above. Any changed snapshot must be rebuilt, re-reviewed and re-hashed first. Account A and B may run concurrently. Managed runtime and mounted-input availability are confirmed only by new remote logs; local CPU checks do not establish T4 compatibility.

## New run records

Append the actual version, launch timestamp, status, source hashes, actual runtime/hardware, observed input mounts and local output path for each submitted account. Update on status changes/completion; preserve the failed v2/v1 evidence and use unique version-specific download folders.

- Account A: `Results/stage1_operations/recovered/accountA/runs/v<N>/`
- Account B: `Results/stage1_operations/recovered/accountB/runs/v<N>/`, passed to WSL as an absolute `/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticLossless/Semantic_Lossless/Results/...` path.

Using the identity wrapper and CLI flags verified from local help, query `kernels status <owner/slug/version>`, `kernels logs <owner/slug/version>` and `kernels output <owner/slug/version> -p <unique-path>`. Save retrieved files, logs, hashes and terminal status under `Results/stage1_operations/`. Inspect for `source_bundle.json`, per-stage logs, metrics, figures, checkpoints and any notebook/HTML; record missing items. Verify representative file hashes and retrieval. `/kaggle/working` alone is not durable evidence.

## Local pre-launch checks

After `python scripts/build_stage1_jobs.py`, the existing suite passed 17/17 and compile checks passed:

```powershell
$env:PYTHONPATH='src;.runtime_deps'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
..\..\SemanticSchedulerEnd2End\.venv-models\Scripts\python.exe -m unittest discover -s tests/current -q
..\..\SemanticSchedulerEnd2End\.venv-models\Scripts\python.exe -m py_compile scripts\build_stage1_jobs.py Kaggle\stage1_swin\kernel.py Kaggle\stage1_shared_training\kernel.py
```

Receipts: [unittest-current.log](../Results/stage1_operations/unittest-current.log), [py_compile-generated.log](../Results/stage1_operations/py_compile-generated.log), and [recovery_receipt.json](../Results/stage1_operations/recovery_receipt.json). These validate local packaging/path behavior and syntax only, not remote GPU execution or scientific results.
