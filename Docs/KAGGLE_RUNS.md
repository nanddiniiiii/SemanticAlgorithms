# Kaggle Stage 1 Run Map

Canonical operations record for Stage 1. Scope is existing-method evaluation and the Shared Semantic Codebook classifier prerequisite. Stage 2 and diffusion-model training are excluded. Do not infer quality results from a launched job or source bundle.

## Account access and quota check

Verified through `scripts/kaggle_account.py` on 2026-10-25:

| Account | Kaggle identity | CLI path | GPU quota | TPU quota | Refresh |
|---|---|---|---:|---:|---|
| A | `satyapaladugu` | Windows Kaggle CLI, identity wrapper | 29.68 h remaining (latest check 2026-10-25) | 20.00 h remaining | 2026-10-31 00:00 UTC |
| B | `satyapaladuguplus` | WSL CLI `/home/Satya/.venvs/semantic-kaggle/bin/kaggle`, isolated config `/home/Satya/.config/semantic-lossless/kaggle/account_b` | 29.79 h remaining (latest check 2026-10-25) | 20.00 h remaining | 2026-10-31 00:00 UTC |

No credentials or tokens are stored here. The verified command is `python scripts/kaggle_account.py A|B <kaggle-command>`; B export paths must be absolute `/mnt/d/...` paths. Account B was authenticated and had quota; the prior failure was not an auth, quota, or eligibility error.

## Prior failed runs and recovered evidence

| Account/version | Kernel | Last run | Status | Declared inputs | Observed environment | Failure and outputs |
|---|---|---|---|---|---|---|
| A v2 | [satyapaladugu/stage1-shared-codebook-classifier](https://www.kaggle.com/code/satyapaladugu/stage1-shared-codebook-classifier) | 2026-10-22 17:08 UTC | `KernelWorkerStatus.ERROR` | STL10 binary files; attempted working path `/kaggle/working/Semantic_Lossless/Datasets/STL10` | Python 3.13; torch 2.11.0+cu128; two Tesla T4 GPUs | `fetch_source()` called `project_root()` and raised `FileNotFoundError: Run from an editable repository checkout or set SEMANTIC_LOSSLESS_ROOT`. It failed before reading STL10 or starting the 2-step pilot. Retrieved source bundle and log only; no metrics, checkpoint, or executed notebook. |
| B v1 | [satyapaladuguplus/stage1-native-swin-evaluation](https://www.kaggle.com/code/satyapaladuguplus/stage1-native-swin-evaluation) | 2026-10-22 16:59 UTC | `KernelWorkerStatus.ERROR` | Kodak and private native Swin asset datasets; logged paths `/kaggle/input/datasets/sherylmehta/kodak-dataset` and `/kaggle/input/datasets/satyapaladuguplus/semantic-lossless-private-swin-assets/...C32.model` | Python 3.13; torch 2.11.0+cu128; CUDA 12.8; two Tesla T4 GPUs; runtime installed timm 0.4.12, pytorch-msssim 1.0.0 and lpips 0.1.4 | Same `project_root()` error while building fixed-C32. It failed before image/checkpoint loading or either evaluation. Retrieved source bundle, log and `environment.json`; no metrics, checkpoint, or executed notebook. |

Logs and the local bundle audit confirm the cause: the old archive omitted root `sources.lock.json`, required by `project_root()`. T4 hardware was allocated on both runs; these failures are not GPU-compatibility evidence. Dataset/checkpoint paths were declared, but actual mount accessibility was not reached or verified. Both were script kernels. No executed `.ipynb` file was returned; logs mention HTML conversion, but no HTML output was returned.

Exact-version status, quota and logs were accessible. `kernels files` returned an empty list for both old kernels, while `kernels output` retrieved output files into the local recovery folders. `kernels pull <kernel>/<version> -m` returned `403 Client Error: Forbidden` for both accounts, so original submitted metadata is unavailable. Retrieved files and SHA256 values are in [recovery_receipt.json](../Results/stage1_operations/recovery_receipt.json).

## Corrected, reviewed pre-launch snapshot

The exact kernels were generated with embedded source metadata `git_revision=8e23f699641d52fe98a4642cbf5ebe3aa8768cc4` and `working_tree_dirty=true`; the reviewed source changes were then checkpointed at `b61f90649a2dca18a0ee488b3181d351752cf07c` before submission. The deterministic embedded ZIP includes `sources.lock.json`; its embedded digest was recomputed and verified locally. The bundle contents and kernel hashes below identify what was uploaded; the post-generation checkpoint itself does not change those embedded bytes.

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

### Submission 2026-10-25

Submitted the exact reviewed files concurrently from the paths in the table above, after checkpoint `b61f90649a2dca18a0ee488b3181d351752cf07c`:

| Account | Intended kernel | Uploaded kernel SHA256 | Embedded bundle SHA256 | Timeout | Kaggle version/status | Retrieval path |
|---|---|---|---|---:|---|---|
| A | `satyapaladugu/stage1-shared-codebook-classifier` | `b18e4e6eb0af65489178b117f06219fc24251fb168a2c1d60bd628178e8d4e3f` | `2275b3a76c1db3d4a416e15187a905cfcf2ea489fcfa889c64c6e32abb3418c1` | 9,000 s | v3 submitted; RUNNING at 2026-10-25 03:42:53 UTC | `Results/stage1_operations/recovered/accountA/runs/v3/` |
| B | `satyapaladuguplus/stage1-native-swin-evaluation` | `8f8f06107a4aa0944495cbcef1345366d058b8654ae9e5c1561bbded9c213e5d` | `2275b3a76c1db3d4a416e15187a905cfcf2ea489fcfa889c64c6e32abb3418c1` | 7,200 s | v2 COMPLETE; output retrieval succeeded at 2026-10-25 03:47 UTC | `Results/stage1_operations/recovered/accountB/runs/v2/` |
| B | `satyapaladuguplus/stage1-adjscc-kodak-profile` | `4cc46f1375dc71ffa920f53dba5f2d7d27678d5f8629c652ed6350ce48822b09` | `19c2fb9a2e677c115c24e9954ad2d92370478dcb38c2a73c6939c86a478fe3d7` | 3,600 s | v1 submitted; RUNNING at 2026-10-25 03:59:33 UTC; hardware/preflight pending | `Results/stage1_operations/recovered/accountB/runs/stage1-adjscc-kodak-profile/v1/` |

### Account B v2 result receipt

The job passed source preflight with bundle digest `2275b3a76c1db3d4a416e15187a905cfcf2ea489fcfa889c64c6e32abb3418c1`; actual runtime was Python 3.13.15, torch 2.11.0+cu128, CUDA 12.8, two Tesla T4 GPUs. It loaded both private checkpoints and Kodak and completed all 24 images per variant at SNR 10, seed 22. Mean metrics:

| Variant | PSNR dB | SSIM | MS-SSIM | LPIPS |
|---|---:|---:|---:|---:|
| fixed C32 | 29.3711 | 0.82334 | 0.95684 | 0.22124 |
| SA+RA | 21.3327 | 0.62473 | 0.78394 | 0.49045 |

Both variants produced full, encoder-only and decoder-only profiles for batch 1, float32, 512x768, warmup 3 and 10 repeats. Full profile: fixed-C32 28,162,000 parameters, 413.428 GFLOPs partial, p50/p95 0.26084/0.26187 s, incremental peak allocated 755,367,936 bytes; SA+RA 33,073,488 parameters, 428.977 GFLOPs partial, p50/p95 0.27457/0.27650 s, incremental peak allocated 764,809,728 bytes. FLOPs cover listed supported operators only; memory is whole model/process residency, not isolated component allocation. Rate remains main-stream CBR only, side-information/transport overhead is absent; no semantic labels exist for Kodak. Source logs, profiles, shapes, per-image CSVs, metrics, model metadata and reconstructions are hash-listed in `Results/stage1_operations/recovered/accountB/runs/v2/output_manifest.json`; summary is in `run_summary.json`. The CLI retrieved 131 Kaggle output files (1,608,561 bytes); the local manifest also includes its generated summary sidecar. No executed notebook or HTML was returned.

### ADJSCC C2 private asset and pre-launch record

The B-owned private asset dataset `satyapaladuguplus/semantic-lossless-private-adjscc-assets` is `ready`. Kaggle normalizes the checkpoint filename from `ADJSCC_C=2.pth.tar` to `ADJSCC_C2.pth.tar`; the ADJSCC kernel locates `.tar` files by their exact SHA256, so the normalized basename is accepted. Dataset file listing and a remote file download verified the expected 63,775,944-byte checkpoint hash `c29d7610eb7b2e475a5ff029d16f0031fc43ac89f316d461b154cef993753c54`. Receipt: `Results/stage1_operations/adjscc_dataset_receipt.json`.

The reviewed ADJSCC kernel hash is `4cc46f1375dc71ffa920f53dba5f2d7d27678d5f8629c652ed6350ce48822b09`; embedded 297,798-byte source ZIP hash is `19c2fb9a2e677c115c24e9954ad2d92370478dcb38c2a73c6939c86a478fe3d7`, includes `sources.lock.json`, and passed extracted fresh-process imports with `PYTHONPATH` and `SEMANTIC_LOSSLESS_ROOT` removed. Checkpoint, runner, builder, kernel and dataset IDs are recorded in `Results/stage1_operations/adjscc_preflight.json`.

| Account | Intended kernel | Workload and inputs | Timeout / accelerator | Expected output root | Submission status |
|---|---|---|---:|---|---|
| B | `satyapaladuguplus/stage1-adjscc-kodak-profile` | Native adapted DiffCom PyTorch ADJSCC C2 checkpoint on Kodak24 at SNR 10, seed 22; quality metrics plus encoder/decoder/full profiles. This is the named PyTorch C2 lineage, not the original TensorFlow reproduction. Checkpoint source and mounted dataset are hash-pinned. | 3,600 s / `NvidiaTeslaT4` | `/kaggle/working/Semantic_Lossless/Results/stage1/adjscc/diffcom_C2/` plus `run_logs/` and `source_bundle.json` | v1 submitted at 03:59 UTC; currently RUNNING, actual hardware and preflight pending. |

- Account A: `Results/stage1_operations/recovered/accountA/runs/v<N>/`
- Account B: `Results/stage1_operations/recovered/accountB/runs/v<N>/`, passed to WSL as an absolute `/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticLossless/Semantic_Lossless/Results/...` path.

Using the identity wrapper and CLI flags verified from local help, query `kernels status <owner/slug/version>`, `kernels logs <owner/slug/version>` and `kernels output <owner/slug/version> -p <unique-path>`. Save retrieved files, logs, hashes and terminal status under `Results/stage1_operations/`. Inspect for `source_bundle.json`, per-stage logs, metrics, figures, checkpoints and any notebook/HTML; record missing items. Verify representative file hashes and retrieval. `/kaggle/working` alone is not durable evidence.

### CPU source-codec and legacy DeepJSCC diagnostics (2026-10-25)

The local CPU source-codec run completed 11 Kodak24 variants (RAW, PNG, lossless JPEG2000, five JPEG qualities and three lossy JPEG2000 settings): `per_image.json` contains 264 image-variant rows. RAW, PNG and lossless JPEG2000 each reconstructed exact RGB pixels for 24/24 images. This measures complete file bytes except headerless RAW; it has no radio/channel cost. The runtime was Python 3.12.14, torch 2.10.0+cpu, Pillow 12.3.0 and OpenJPEG 2.5.4, with two CPU threads and one timing observation. Metrics and file hashes: `Results/stage1_operations/traditional_kodak24/`.

The retained legacy DeepJSCC diagnostic completed 24 unique Kodak images on local CPU: mean PSNR 16.42905 dB, SSIM 0.230969, MS-SSIM 0.476155. It uses the retained checkpoint's training SNR 200 dB and a legacy tiled real-AWGN evaluation at 10 dB. Each 128x128 tile emits 15,979 real values (odd count); complex uses and complete radio rate are null. This is not an original-paper reproduction or a matched complex-radio comparison. Metrics, shapes, per-image results and hashes: `Results/stage1_operations/deepjscc_legacy_kodak24/`. Combined validation receipt: `Results/stage1_operations/cpu_baseline_validation_receipt.json`.

### Authorized JCM and SQ-GAN submissions — pre-launch snapshot

Source snapshot is the generated Kaggle folders as present on 2026-10-25; embedded provenance records checkpoint `b61f90649a2dca18a0ee488b3181d351752cf07c` and `working_tree_dirty=true`. Before submission, py_compile passed for all three frozen candidate kernels (JCM, SQ-GAN, NTSCC). Fresh child processes with `PYTHONPATH` and `SEMANTIC_LOSSLESS_ROOT` cleared imported the extracted JCM, SQ-GAN and NTSCC source bundles; each has a root `sources.lock.json`. Full hashes and extraction paths are in `Results/stage1_operations/{jcm,sqgan,ntscc}_bundle_preflight/preflight.json` and `cpu_baseline_validation_receipt.json`.

| Candidate | Account / kernel | Exact kernel SHA256 | Embedded bundle SHA256 | Runtime request | Inputs observed before launch | Expected remote result root | Current state |
|---|---|---|---|---|---|---|---|
| JCM native profile | A / `satyapaladugu/stage1-jcm-native-profile` | `705f35db605edfa6ef0da66f3dfdeaa21db2f7fac5e75e5b52b608365fad8c8a` | `7c838474156c3ad645571085b42c3e10b602711ed1f32facf4e00bb93c10da1e` | `NvidiaTeslaT4`, 1,800 s | Private JCM dataset lists `CIFAR_SNR12_Trans128_16qam.pth.tar` (57,601,737 bytes); CIFAR10 is separately attached in metadata | `/kaggle/working/Semantic_Lossless/Results/stage1/jcm/` plus logs and source metadata | v1 COMPLETE, six profiles and metadata retrieved. Runtime log confirms Python 3.13.15, torch 2.11.0+cu128, 2x Tesla T4; source preflight passed. Batch 1/32 GPU component/full profiles only; no quality claim. |
| SQ-GAN native components | B / `satyapaladuguplus/sqgan-native-components-evaluation` | v10: `d88a7e129c6a7128e7fee2245b0dfe9f04c3dc68ad2677f134497406d3047529` (v9 failed SHA `1b207a20eac05a2f9eb8a45106213d323a550b1e8a11422b04e18e81b90e5aec`) | v10: `be5a696fa46cacbe0d1d5ab3929892213948ca3ef4eaece3f8246671a37da967` (v9 bundle `cfd8adc256a5abde9beb33df3bcb39385b429f11260554420c3ec3db7818d525`) | `NvidiaTeslaT4`, 3,600 s | Cityscapes test files visible; private asset dataset lists all 10 model/checkpoint files; pinned source dataset lists upstream source files | `/kaggle/working/Semantic_Lossless/Results/stage1/sqgan/` plus logs and source metadata | v9 ERROR, v10 COMPLETE; metrics, profile and metadata retrieved. v8 results remain separate. |

The NTSCC checkpoint upload is staged separately for a later account A run after JCM retrieval. Its private dataset `satyapaladugu/semantic-lossless-private-ntscc-assets` is ready; an exact remote representative download matched the 131,622,445-byte local checkpoint at SHA256 `7a8d1990dbd678ea5d3daa02c3aeaaf4459b347a96d6aa81a5c02eae6bfc7323`. Receipt: `Results/stage1_operations/ntscc_dataset_receipt.json`. Do not launch NTSCC until JCM has completed and been retrieved.

### JCM v1 completed; NTSCC launch record

JCM A v1 completed with source preflight passing (bundle `7c838474...`), Python 3.13.15, torch 2.11.0+cu128, two Tesla T4 GPUs and execution on `cuda:0`. It strict-loaded the native 16-QAM CIFAR10 checkpoint (`b7d3a1df...`), retained stochastic Gumbel softmax and batch-wide normalization, then profiled the first 32 official CIFAR10 test images at batches 1 and 32. All six full/encoder/decoder profile JSON files, environment, model, status, stage log and source provenance have been retrieved to `Results/stage1_operations/recovered/accountA/runs/stage1-jcm-native-profile/v1-artifacts/Semantic_Lossless/`. The status says “no new quality result or training.” There is no executed notebook or HTML in the retrieved selected artifacts; the remote log reports HTML conversion. JCM runtime artifacts are authoritative; unrelated transported source files from the full output pull are not needed.

JCM completion and retrieval released the approved NTSCC A job. Frozen kernel SHA256 `2c158196a5b9035d8504a517274e2d015761d7adb2b19c587dd03d5685a6ae5f`, embedded bundle SHA256 `9249bb497340eb87312ca8ac8dd9cba703a3f4d09ff1d7a84077652042683325` (198,198 bytes; root `sources.lock.json`). The bundle passed `py_compile` and fresh-child import with `PYTHONPATH`/`SEMANTIC_LOSSLESS_ROOT` removed. Inputs are Kodak24 and the hash-verified private NTSCC quality-1 checkpoint. Requested runtime: `NvidiaTeslaT4`, 3,600-second cap; workload is full Kodak24 evaluation and full-forward GPU profile only. Kernel slug `satyapaladugu/stage1-ntscc-kodak-profile`; v1 is COMPLETE. Output retrieval is verified at `Results/stage1_operations/recovered/accountA/runs/stage1-ntscc-kodak-profile/v1-artifacts/Semantic_Lossless/`; the output manifest and final receipt are `Results/stage1_operations/recovered/accountA/runs/stage1-ntscc-kodak-profile/v1-artifacts/output_manifest.json` and `Results/stage1_operations/stage1_final_gpu_transition_receipt.json`.

NTSCC v1 passed source preflight on Python 3.13.15 / torch 2.11.0+cu128 / CUDA 12.8 / T4x2 and strict-loaded the expected 131,622,445-byte checkpoint (all 426 keys, no missing/unexpected/mismatched keys; regenerated attention masks skipped) on `cuda:0`. All 24 Kodak images completed at SNR 10, seed 22: mean PSNR 29.79025 dB, SSIM 0.825330, MS-SSIM 0.957166, LPIPS 0.221387, mean main-stream CBR 0.02397537, exact RGB 0/24. This is a retained q1 evaluation with device/entropy compatibility adapter, not an author training reproduction. Main-stream rate excludes a fixed-width 6,144-bit/image index/mask representation as well as metadata, protection, headers and receiver rescaling; hyperprior refinement was disabled. Kodak has no task labels, so semantic metrics are null. Full retained-adapter profile: 31,358,531 parameters, 787.821 GFLOPs over partial supported operators, p50/p95 0.364812/0.367353 s, peak/incremental allocated 1,572,884,992/1,419,149,312 bytes, warmup 3/repeats 20. It includes auxiliary NTC synthesis/loss and JSCC synthesis, so it is not minimal transmitter cost. Profiles reset the RNG each invocation, reusing channel noise; repeats describe runtime variation only, not independent channel trials. Logs report HTML conversion but no `.ipynb` or HTML file was exposed by selected artifact retrieval.

SQ-GAN v10's full profile has 186,271,386 parameters, 2.3369 TFLOPs partial supported operators, p50/p95 0.645462/0.649613 s, peak/incremental allocated 1,364,836,864/575,687,680 bytes, warmup 3/repeats 20. The `forward_all` profile includes semantic and RGB branches, excludes packet coding and GAN training. The source packet result is an adapter format with synchronized checkpoints; no radio or channel stage is included. Its 500-image metrics/profile and supporting files are listed in `Results/stage1_operations/recovered/accountB/runs/sqgan-native-components-evaluation/v10-artifacts/output_manifest.json`. Logs report HTML conversion; neither executed notebook nor HTML was returned.

### Shared native codebook construction and codec prefix — authorized launch

The exact frozen kernel is `Kaggle/stage1_shared_native_prefix/kernel.py`, SHA256 `0a33e6a9ff052024fa6652584cb1fc3bbfae446540b5fc83686318314be1c7b5`; embedded 168,271-byte bundle SHA256 `48926eed0e70f1a66c0f439e49e66907d1dbdba16bb14431113cd83279a269be`, with root `sources.lock.json`. It passed py_compile and extracted fresh-child import with path environment variables removed; `native_construction_definitions` imported and `build_native_sac(device='cpu')` constructed the native WITT model. Full receipt: `Results/stage1_operations/shared_native_prefix_preflight/launch_preflight.json`. Runner asserts the classifier hash and 5,000-image STL10 train split; training requires an artifact whose payload reports native source epoch 290 and 2,900 construction batches. The two-step pilot raises if fewer than two steps complete, preventing the epoch prefix from launching. Stage bounds are 1,800 s construction, 600 s pilot, 3,600 s one-epoch prefix, with a 6,600 s overall kernel cap. This is a bounded released-code implementation check with known released construction defects retained; it does not establish paper convergence or held-out quality. Only A classifier dataset (`classifier.pt`, exact expected SHA) and public STL10 are mounted. Submit status, observed hardware, pilot gate, stage outcomes and recovered artifacts below. No retry is authorized automatically.

### SQ-GAN v9 bootstrap failure and bounded retry

Account B v9 ran on 2026-10-25 and returned `KernelWorkerStatus.ERROR`. Recovered log `Results/stage1_operations/recovered/accountB/runs/sqgan-native-components-evaluation/v9/sqgan-native-components-evaluation.log` shows `NameError: name 'json' is not defined` at `/kaggle/src/script.py:14` while writing `source_bundle.json`, at 0.8906 seconds. It failed before the environment/GPU report and before loading datasets or weights. The output contains the embedded source files and the log, but no `source_bundle.json`, metrics, executed notebook or HTML. Local folder: `Results/stage1_operations/recovered/accountB/runs/sqgan-native-components-evaluation/v9/`. The exact cause was fixed by importing `json` in the generated bootstrap. The standard local suite then passed 20/20, the regenerated kernel passed `py_compile`, and extracted-bundle imports passed in a fresh child with path variables removed. Evidence: `Results/stage1_operations/sqgan-fix-unittest-rerun.log`, `sqgan-v10-pycompile.log`, and `sqgan_v10_bundle_preflight/preflight.json`.

The one authorized B retry is frozen before submit as follows: kernel SHA256 `d88a7e129c6a7128e7fee2245b0dfe9f04c3dc68ad2677f134497406d3047529`; embedded 71,472-byte bundle SHA256 `be5a696fa46cacbe0d1d5ab3929892213948ca3ef4eaece3f8246671a37da967`, with root `sources.lock.json`; same Cityscapes and private SQ-GAN datasets, 3,600-second `NvidiaTeslaT4` cap, unique v10 retrieval path `Results/stage1_operations/recovered/accountB/runs/sqgan-native-components-evaluation/v10/`. v9 evidence remains unchanged.

SQ-GAN v10 completed on account B with source preflight passed and actual runtime Python 3.13.15 / torch 2.11.0+cu128 / CUDA 12.8 / Tesla T4x2. It evaluated 500 Cityscapes images; mean PNG-range PSNR 21.25659 dB, SSIM 0.660906, MS-SSIM 0.782834, LPIPS 0.276734. Native saved-range PSNR is separately reported (21.24084 fixed range; 21.00108 auto range). The evaluation is no-channel, with training not reproduced; InternImage mIoU and separate encoder/decoder profiles are unmeasured. The full-forward profile is recovered as `profile_full.json`. Output retrieval is to `Results/stage1_operations/recovered/accountB/runs/sqgan-native-components-evaluation/v10/`; logs report HTML conversion but no `.ipynb` or HTML file has been verified in the downloaded tree yet.

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

### Custom mixed-data ADJSCC adaptation — B launch record

Pre-launch validation used the frozen `Kaggle/custom_adjscc/kernel.py` SHA256 `6febaf6b6f133317518efffdc9877354bcc1c7973d78c2752a95e4e87199e30f`; embedded source ZIP SHA256 `48100ea6124b31d4c44197bc8216e04beac5486bd2b8fce4d71f0c7d9c1c6feb` (303,946 bytes, 170 entries; root `sources.lock.json`). Embedded provenance records revision `b61f90649a2dca18a0ee488b3181d351752cf07c` with dirty-tree flag; kernel and bundle hashes are execution authority. Local suite passed 21/21, and builder, resolver, freezer, trainer, dev evaluator, data adapter and kernel passed `py_compile`; logs: `Results/stage1_operations/custom_adjscc-final-unittest.log` and `custom_adjscc-final-pycompile.log`. A fresh `-I` child imported the extracted archive with `PYTHONPATH` and `SEMANTIC_LOSSLESS_ROOT` removed; `project_root()` found the archive marker, `build_adjscc` and `FrozenImages` imported, and the transported trainer asserts two CUDA devices and records encoder work on both GPUs. Receipt: `Results/stage1_operations/custom_adjscc_verified_preflight/preflight.json`. Local validation does not certify Kaggle GPU execution.

Account B authenticated as `satyapaladuguplus`; quota before launch was 29.79 GPU-hours remaining (20 TPU-hours). B file listings were readable for all five sources: `soumikrakshit/div2k-high-resolution-images`, `awsaf49/coco-2017-dataset`, `deeplumiere/textocr`, `sherylmehta/kodak-dataset`, and `satyapaladuguplus/semantic-lossless-private-adjscc-assets`. DIV2K listing confirms `DIV2K_train_HR/DIV2K_train_HR/*.png`; prior full listing found 800 train and 100 validation HR files, no LR subset. COCO listing exposed train annotations; `train2017`/`val2017` image roots remain runtime-resolver checks. TextOCR val JSON SHA256 is `0d04c1c9f5ba99e90be5b0df9ab34f6237b3fbc55443f263133d5c49b6435918`, containing 3,124 `set=val` images. Its first `file_name` is `train/a7ad2bcb93d48576.jpg`; that single remote image (164,718 bytes, SHA256 `b611039797f131fe56b927c4f255cd495cedd4f0752ce84c300e85caa724bdcf`) was retrieved from `textocr/textocr/train_images/`. No image corpus was bulk downloaded. The runtime resolver matches annotation basenames against one unique mounted image path. The private C=2 checkpoint listing shows 63,775,944 bytes; prior remote hash verification matches `c29d7610eb7b2e475a5ff029d16f0031fc43ac89f316d461b154cef993753c54`. Kaggle dataset license metadata is not treated as authoritative for underlying image terms.

| Account / kernel | Workload and split contract | Timeout / accelerator | Expected outputs | Version/status |
|---|---|---:|---|---|
| B / `satyapaladuguplus/custom-mixture-adjscc-adaptation` | Resolve official roots; freeze deterministic seed-22 per-image selection and cross-source pixel hashes. Expected train 3,800 (DIV2K 800 + COCO 2,000 + TextOCR 1,000), development 400 (100/200/100), and locked test 324 (COCO 200 + TextOCR 100 + Kodak 24). Freezer reads image bytes for membership/deduplication; model trainer and evaluator must not access locked test rows. Two-step pilot and before-development evaluation; <=2-hour ADJSCC fine-tune from pristine C=2 checkpoint; after-development evaluation only. DataParallel training must observe encoder work on `cuda:0` and `cuda:1`; development uses `cuda:0`. This is custom adaptation, not native ADJSCC training or matched paper reproduction. | 14,400 s / `NvidiaTeslaT4` | `/kaggle/working/Semantic_Lossless/Results/custom_adaptation/adjscc/mixture_v1/` (resolved sources, frozen manifest, pilot/status, before/after development metrics, checkpoints/config/environment/GPU work, logs and source bundle). Retrieve to a new version-scoped folder under `Results/stage1_operations/recovered/accountB/runs/custom-mixture-adjscc-adaptation/v<N>/`; actual persistence must be verified. | v1 submitted; RUNNING at first status query; logs not yet exposed. |

Freeze output and selected paths/counts/hashes do not exist before launch; record them from remote artifacts, distinguishing actual files from expected outputs. Stop on resolver/freeze failure, insufficient counts, pilot failure, missing two-GPU work evidence or OOM; preserve logs and do not retry automatically. Retrieve all partial outputs and logs. Executed notebook/HTML availability remains unknown until queried.