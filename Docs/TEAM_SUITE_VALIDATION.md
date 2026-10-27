# Three-team encoder/decoder implementation: validation record

Updated:27 October2026. This records the current implementation increment, not a
claim that twelve models are trained or paper reproductions are complete.

## Current per-model pilot status

| Model | Strongest current evidence | Remaining material gap |
|---|---|---|
| DeepJSCC | v14: synthetic train, reload, development inference and profile on two T4s | Third-party approximation; real-data run and full quality evaluation |
| ADJSCC | v13: same synthetic scope, random initialization | PyTorch port; author-checkpoint and real-data qualification |
| SwinJSCC | v18: fixed-C32 author weights, two real training updates, reload and one development crop/profile, two T4s | Full training and held-out Kodak evaluation |
| ISEC | v17: JSCC and denoiser updates, reload, one-step dev inference/profile | Full 50-iteration run and real-data qualification |
| NTSCC+ | Native source identified | Author NATTEN 0.14.6 build failed; model never constructed on Kaggle |
| HJSCC | v7: synthetic train, reload, dev inference and profile on two T4s | Real-data evaluation and convergence |
| JCM spatial | v15: author initializer, synthetic train, reload and dev inference/profile | Approved reconstruction variant; real-data evaluation, distinct from native classification |
| MDJCM | v9: synthetic train, reload, dev inference and profile on two T4s | Real-data quality and complete digital transport accounting |
| Masked SQ-GAN | v11: two adversarial updates, reload and synthetic dev inference/profile; two T4s visible | Real-data run, per-device workload evidence, and digital channel link |
| Shared semantic codebook | v6: synthetic prototype fit, codec train, reload, dual-T4 and dev inference/profile | Full real-data prototype/codec training; side index protection |
| DeepJSCC-WZ | Native-task runner and code available | Author paired stereo datasets, environment and end-to-end pilot |
| GBSED | Native-task runner and code available | 1043-syn graph/risk data, compatible mixed-framework stack and end-to-end pilot |

All nine passing models are *smoke-tested*, not fully trained. SwinJSCC has one
real-data smoke test; other models still need real-data qualification. The later detailed entries preserve failed runs and
supersede earlier training-only milestones when they are more recent.

## 2026-10-27 readiness checks

Kaggle dataset discovery now collapses nested aliases describing the same
recursive image set, rejects distinct candidate datasets, and skips roots supplied
explicitly. A regression test exercises nested Flickr2K folders and ambiguity.
Evaluation now rejects nonfinite reconstruction and reports failure when any
requested trial fails, while preserving successful rows and error records. The
generated launchers were rebuilt after the discovery change. All ten local
SQ-GAN author components match the published download ledger's SHA-256 values;
automatic asset discovery now requires these hashes rather than ten arbitrary
checkpoint files. This verifies file identity, not a complete SQ-GAN run.

Local checks: `python -m pytest tests/current -q`: 44 passed, two warnings.
These checks do not certify mounts, paper reproduction, GPU model quality,
notebook execution, or the training queue. The GPU pilot builder now requests
development inference and profiling after its training smoke test; see
KAGGLE_RUNS.md for actual remote evidence.

Real-data CPU preflight v1 encountered pixel-identical Flickr2K training files
`000291.png` and `000371.png` and stopped before creating a manifest. The
dataset code now retains and reports exact duplicates within training to preserve
official membership, and still rejects any duplicate involving validation or
test. The resulting nonuniform training exposure must be reported. CPU preflight
v2 passed: 3,450 train entries, 100 validation and 24 test, with 59 repeated
training entries (3,391 distinct decoded RGB images) and no duplicate involving
protected splits. Its recovered manifest identity is
`add977eb3277909e5ad1447770abdb368458445ab42b2f48a91c24f4c31f7fa0`,
independently rehashed locally; launchers check this identity. The manifest is
under `Results/stage1_operations/team_dataset_preflight/v2/retrieved/`.

The v5 two-model Kaggle pilot finished but both paths failed. HJSCC's
DataParallel reconstruction hook selected a different replica's tensor. A
thread-scoped capture fix was implemented and rechecked in v7 on two T4s.
MDJCM failed while importing timm after HJSCC in the same pilot process; this
does not invalidate its earlier isolated training-only pass. Isolated v9 then
passed two updates, strict reload, dual-T4 work and development inference/profile
on a synthetic image. Team notebooks invoke each model in a separate process.
The saved v5 log is `Results/team_suite_validation/kaggle/v5/kaggle_kernel_v5.log`.

Account B v6 passed the shared semantic-aware codebook pilot: pseudo-label
prototype construction on generated fixtures, two codec updates (four generated
images), strict reload, work observed on both T4 devices, and a development
inference/profile. This is only execution evidence, not convergence, source-rate
validity, or quality on real data. Evidence is under
`Results/team_suite_validation/kaggle/v6/` and the job map.

Account B v7 passed isolated HJSCC training, strict checkpoint reload and one
development inference/profile on synthetic images; encoder work was observed on
both T4s. This does not establish real-image performance or paper reproduction.
The failed v5 evidence remains preserved. An independent CPU concurrency check
also exercised the hook's thread isolation; the later local device assertion was
not included in the v7 source snapshot.

SQ-GAN v8 failed inside its released image loss: its Cityscapes YAML has 20
weights including null, while `semantic[:, :-1]` supplies 19 non-null channels.
The local adapter now drops only the unused null weight, retaining the 19
specified weights; a unit test checks the mapping. This is a documented fix to
the released configuration interaction, not a successful training result. The
v8 log is `Results/team_suite_validation/kaggle/v8/kaggle_kernel_v8.log`.
The v10 retry completed one adversarial step, then exhausted the primary T4
during its next forward. A targeted local change suppresses generator autograd
on the discriminator-only pass, whose native loss detaches real/fake inputs,
and clears optimizer gradients/outputs between passes. This does not alter the
native crop or loss. Account B v11 then passed two synthetic adversarial steps,
strict reload and one development inference/profile on a two-T4 runtime. Device
visibility and DataParallel use were recorded, but a per-device work receipt was
not collected for this pilot. This does not establish convergence, real-data
quality, or transmission over AWGN. The v11 log is
`Results/team_suite_validation/kaggle/v11/kaggle_kernel_v11.log`.

Account B v12 passed native fixed-C32 SwinJSCC: two synthetic training updates,
strict reload, encoder work on both T4s and one development inference/profile.
The exact log is `Results/team_suite_validation/kaggle/v12/kaggle_kernel_v12.log`;
this is not a real-image benchmark or a reproduction of convergence.
The first real-data attempt, v16, failed because the frozen absolute paths from
the CPU preflight were not mounted at the same locations in the GPU runtime.
After runtime root discovery and per-file hash validation, v18 passed two actual
DIV2K/Flickr2K training updates, strict reload, work on both T4s, and one DIV2K
development crop inference/profile. Kodak was not evaluated. This is not a
converged quality comparison; see `Results/team_suite_validation/kaggle/v18/`.

Account B v13 passed the DiffCom PyTorch ADJSCC path with random initialization:
two updates, strict reload, work on both T4s and one synthetic development
inference/profile. It is not verification of the original TensorFlow model or
of an author checkpoint. Evidence: `Results/team_suite_validation/kaggle/v13/`.

Account B v14 passed the third-party DeepJSCC approximation and v15 passed the
approved spatial reconstruction JCM adaptation. Each completed two synthetic
updates, strict reload, dual-T4 encoder work and one development inference/profile.
These are not paper-exact reproductions or real-data quality results. Evidence:
`Results/team_suite_validation/kaggle/v14/` and `v15/`.

Account B v17 passed an ISEC two-stage synthetic pilot: two JSCC updates, two
denoiser updates, strict reload, work on both T4s, and one development inference
and profile with one iterative step. It is not the full 50-step evaluation or a
quality benchmark. See `Results/team_suite_validation/kaggle/v17/`.

## Delivered source structure

`src/semantic_lossless/team_suite/` contains dataset freezing/loading, pinned
architecture factories, budgeted training, evaluation/profiling, author asset
retrieval, native task runners, SQ-GAN adversarial fine-tuning, shared-codebook
construction, logging/export and verified result collection. Native model
definitions remain in fixed-revision upstreams fetched by `sources.py`.

`Notebooks/TeamSuite/{you,teammate_1,teammate_2}.ipynb` provide initialization,
pilot/training and evaluation/export cells for each assigned model. Failures
are recorded and do not masquerade as successful model results. Source remains
local until review/publication; a fresh remote clone cannot retrieve unpublished
files. The operating guide is `configs/team_suite/README.md`.

`scripts/download_you.py`, `download_teammate_1.py` and
`download_teammate_2.py` poll explicitly versioned Kaggle jobs, retrieve saved
outputs, verify checksums and update a locked/idempotent central comparison CSV.
They do not submit/retry jobs or stream unsaved runtime files.

## Local checks actually executed

- Full current CPU suite:39 passed, two existing dependency/profiler warnings.
- Thirteen team-suite behavioral/integration tests cover frozen membership,
  duplicate/tamper rejection, dimensions, three notebook assignments, safe
  exports, verified result collection, native-data gates and a real DeepJSCC
  two-update training/checkpoint/development-inference path on synthetic inputs.
- Syntax compilation of generated notebooks and new Python modules passed.
- `.venv-team` creation and actual Torch import were checked, using inherited
  Windows Python3.14/Torch2.11.0+cu126; CUDA unavailable. Model-specific dependency
  environments are not all installed or validated by that bootstrap check.

Synthetic CPU model pilots (no image-quality claim):

| Model | Tested scope | Outcome |
|---|---|---|
| DeepJSCC third-party implementation |128 RGB; finite forward/backward/update, strict state reload, evaluation | Passed |
| ADJSCC DiffCom port |128 RGB; author checkpoint strict load, finite update/reload/evaluation | Passed |
| JCM approved spatial variant |32 RGB numerical path; additional256 RGB forward/backward/update | Passed |
| ISEC author architecture |JSCC update, frozen encoder/decoder denoiser update, one latent-refinement iteration | Passed |
| Native SwinJSCC fixed-C32 |Strict author checkpoint construction/load | Passed loading; forward CUDA-only on this laptop |

Receipts: `Results/team_suite_validation/cpu_pilots.json`,
`jcm_256_cpu.json` and `logs/`. Some early numerical reload checks were in-memory;
the integration test and GPU pilots separately exercised serialized reloads.

## Kaggle pilot v1: actual hardware and retrieved evidence

Kernel: `satyapaladuguplus/team-suite-dual-t4-contract-pilot`, version1.
Source bundle SHA256:`9667fb16155d9a2cdeed0cd062674f8bb951ceefafa0f455bbfaab3b5e930925`.
Notebook SHA256:`89106e4c95280caa8205c9804cf54998b6bf0e5d4dd209d8e2598ac459d2f6da`.

Actual runtime:Python3.13.15, Torch2.11.0+cu128, CUDA12.8, two TeslaT4 GPUs.
The notebook reached terminal COMPLETE while recording individual model failures;
that kernel status alone did not establish every model passed.

| Model | GPU result | Coverage |
|---|---|---|
| DeepJSCC third-party implementation | Passed |2 updates;4 synthetic image presentations; strict reload; dual-device probe |
| ADJSCC | Failed before model execution |Pilot bypassed prepare and lacked DiffCom source; no GPU-model evidence from this attempt |
| Native SwinJSCC fixed-C32 | Passed |2 updates; strict reload; observed encoder work on both GPUs |
| ISEC | Passed |JSCC and denoiser stages;4 total updates; strict reload; both GPUs |
| JCM spatial adaptation | Passed |2 updates; verified author initializer; strict reload; both GPUs |

Retrieved301 files,594,394,956 bytes. All26 files listed in the per-model
artifact manifests matched recorded sizes and SHA256. The large retrieval
included transported sources; later retrieval should filter run artifacts.
Executed `.ipynb`/HTML were mentioned by Kaggle conversion logs but were not
exposed in retrieved files. Missing notebook capture is not silently certified.

Recovery root: `Results/team_suite_validation/kaggle/v1/`. See
`retrieval_receipt.json`, `remote_logs.txt`, and the authoritative version/run
entries in `Docs/KAGGLE_RUNS.md`. The initialization defect was corrected by
fetching DiffCom source in the team ADJSCC factory before construction. A
targeted ADJSCC-only version2 recheck completed successfully:2 updates,
4 synthetic image presentations, strict serialized checkpoint reload, observed
encoder work on both T4s,10.59 seconds in the training loop. Its initialization
was random DiffCom-PyTorch C=2 architecture, not pretrained-weight qualification.
All6 model-manifest files matched size/SHA after filtered retrieval of19 files
(192,574,107 bytes). Version2's original `all_passed` reporting field is false
because the pilot builder overwrote `status=passed` with `pilot_completed` before
aggregation. Individual stage/checkpoint/device receipts establish the pilot
outcome. The builder is corrected for future runs; original receipts stay intact.
No extra rerun is justified solely to change this reporting field.

## Review findings corrected

- Pilot gates require two optimizer updates per prerequisite stage, including
  crossing epochs when a tiny dataset supplies only one batch per epoch.
- ISEC cannot be admitted as trained if its denoiser prerequisite never ran.
- HJSCC/MDJCM receive the runtime device expected by native constructors.
- HJSCC's native total-batch channel-use count is divided per image.
- MDJCM explicitly enables modulation in nested encoder/decoder settings and
  dispatches the demonstrated faulty parent-method call correctly.
- WZ Hydra additions use `+` for absent strategy/fast-dev keys.
- Artifact hashing follows finalized stage logs; the still-active export-cell
  logfile is explicitly excluded. Actual CLI export/checksum roundtrip passed.
- Collection excludes unverified/in-progress downloads, waits briefly on its
  exclusive lock and preserves distinct source/run identities.
- Shared Codebook integer side-bit outputs are converted to tensors for
  DataParallel gathering; actual multi-GPU qualification remains outstanding.

## Outstanding acceptance gates

| Model | Remaining gate |
|---|---|
| DeepJSCC |Common-data full training/evaluation; known third-party kernel differences prevent paper-exact claim |
| ADJSCC |v2 dual-T4 pilot passed; custom-data full experiment remains |
| SwinJSCC |Real-data full training/evaluation and full-image metric/profile checks |
| ISEC |Full two-stage custom-data training;50-iteration final profiling/quality |
| NTSCC+ |Pinned NATTEN0.14.6 dependency build and native GPU pilot |
| HJSCC |v3 dual-T4 training pilot passed; fresh-process reconstruction hook/PSNR assertion and full evaluation remain |
| JCM |Custom-data full experiment; variant remains explicitly different from native CIFAR/classification JCM |
| MDJCM |v4 dual-T4 training pilot passed; full digital-rate evaluation remains |
| Masked SQ-GAN |Full native-objective GPU pilot, teacher/source payload validation; digital AWGN delivery choice pending |
| Shared Codebook |Mounted frozen classifier, pseudo-label/prototype qualification, dual-GPU pilot and complete codec training |
| DeepJSCC-WZ |Native paired dataset, dependency qualification, native DDP training/evaluation and normalized profiling |
| GBSED |1043-syn native graphs/labels, author environment, end-to-end graph/radio/task pilot and profiling |

The common3,450-train/100-development/24-test membership is a protocol in code,
not a claim that those images are already mounted/frozen on all three accounts.
Four maximum-length training runs require16 hours per teammate before setup,
pilots and evaluation. Use an explicit model subset or lower budget when the
actual session limit is shorter; automatic cross-session resume is not present.

Source-only SQ-GAN cannot be labelled AWGN-trained. GBSED's native graph task
and WZ's receiver-side information cannot be reduced to unconditional Kodak
PSNR. No final model quality, novelty, convergence or twelve-model readiness
claim follows from the pilot evidence above.

## Subsequent v3/v4 checks

Version3 verified HJSCC on both T4s:two updates, strict reload,4 synthetic image
presentations,15.45s training-loop time. Seven retrieved non-checkpoint files
passed manifest hashes. Its1.582GB random smoke checkpoint was intentionally
not retrieved; this is a disclosed persistence omission, not recovered weights.
MDJCM failed on the renamed CompressAI `ste_round` import before construction.

The factory now restores exactly the original
[CompressAI1.2.0 straight-through rounding function](https://github.com/InterDigitalInc/CompressAI/blob/v1.2.0/compressai/ops/ops.py)
when newer CompressAI omits that symbol. Its rounded values and identity
gradient have an independent test. The temporary symbol is removed after
import; author source files remain unchanged. Other dependency compatibility
remains subject to the actual forward/backward pilot, not inferred from this fix.

Version4 tested the diagnosed MDJCM fix and pinned NATTEN0.14.6 for NTSCC+.
No newer neighborhood-attention kernel is substituted. The build has two
workers, T4's architecture target, and a600s limit; training is capped at120s
per model. The saved summary establishes MDJCM passed:two updates,4 image
presentations, observed encoder work on both GPUs and strict checkpoint reload,
6.09s training-loop time. NTSCC+ failed before model construction because the
legacy NATTEN wheel build exited nonzero. Captured pip output states no detailed
build output was available; the compiler root cause is not established. Do not
claim it was an OOM or a specific Torch API error without further diagnostics.
Version-specific retrieval checks/omissions are in KAGGLE_RUNS.md and
`Results/team_suite_validation/kaggle/v4/`. Seven model paths now have bounded
dual-T4 training evidence, not seven finished custom-data experiments.

No blind NATTEN retry or silent newer-kernel substitution is authorized by that
failure. The smallest useful next check is a verbose build in a compatible
legacy environment, with compiler output persisted and bounded build resources.

Further focused fixes: fresh-process checkpoint loading regenerates only known
registered resolution masks and rejects unexpected mask/learned-weight keys;
WZ evaluation receives explicit paired-data and run-owned output paths;
GBSED training/evaluation use separate runtime config copies and cannot write
over original configs/checkpoints. Native WZ OmegaConf checkpoint compatibility
with modern `torch.load` is still unverified. ISEC now uses its denoiser's native
2e-4 learning rate and stage schedulers; budget-limited custom data remains an
adaptation rather than reproduction of the native full training schedule.
SQ-GAN's native VQ EMA observations are now aggregated across DataParallel
replicas before one native EMA update, avoiding discarded secondary-GPU updates.
A test compares this aggregation against the author's combined-batch update.
This is CPU numerical evidence; the full adversarial dual-GPU path remains
unqualified until its own pilot passes.
