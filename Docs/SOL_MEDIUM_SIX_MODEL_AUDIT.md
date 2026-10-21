# Six-model implementation and regression audit

20 October 2026. Scope: DeepJSCC, ADJSCC, released NTSCC+, SwinJSCC,
SQ-GAN and DeepJSCC-Q; DiT-JSCC and DDM-SSCC are audit-only.

This report distinguishes code inspection, synthetic CPU regression execution,
and model validation. No GPU training, authentication, publication or submission
was performed by this audit. CPU helper tests are not model reproduction results.

## Findings requiring explicit handling

| Priority | Finding | Evidence and required gate |
|---|---|---|
| High | Native/full-image contract conflicts with the original comparison gate. | `SharedCode/reproduction/experiment_contract.json:25` specifies native geometry; the inspected `matched_evaluation.py:207-208` accepted only `matched_tiled_v1`. Regression `test_native_comparison_can_pass_without_silent_tiling` reproduces rejection. Native and tiled receipts must remain distinct and internally consistent. |
| High | Repeated upstream launch preparation rejects its own external dataset symlink. | Inspected `launch_upstream.py:68-73` resolved the symlink before lexical containment checking. Regression `test_launch_preparation_is_repeatable_with_external_dataset_symlink` reproduces the second-call failure without launching a subprocess or model. |
| High | Two-GPU operation requires model-specific orchestration. | `experiment_contract.json:12` requests DataParallel/MirroredStrategy; inspected author entrypoints have no such setup. Swin `net/network.py:67,86` returns Python float/int metadata alongside tensors, requiring output/gather adaptation. Its `net/channel.py:44-46` normalizes over the replica-local batch. Verify both device participation, loss weighting, global batch coverage and channel semantics. |
| High | Native NTSCC+ resume is incomplete. | Pinned `NTSCC_plus/.../utils.py:101,103` comments out main optimizer/scheduler restoration; `main.py:411` constructs a new scheduler after checkpoint loading. Do not count restored weights as an exact continuation. Restore every optimizer, scheduler, scaler, RNG and data position under the same contract. |
| High | NTSCC+ side-information evaluation can collide or count stale output. | Pinned `main.py:176-177` shells out to FLIF using fixed paths without checking status. Require unique paths, checked execution, actual serialized bytes and declared transport overhead. |
| High | Swin CPU core compatibility is narrower than GPU readiness. | Saved `Results/reproduction_audit/swin_cpu_runtime_probe/report.json` says `PASS_CPU_CORE_ONLY`, native channel false, full Kodak geometry false and zero real epochs. The synthetic optimizer step is not benchmark progress. |
| Medium | Older documents disagree on universal versus model-specific geometry. | `REPRODUCTION_CORRECTION_20261020.md:53-58` proposes universal tiles; `RERUN_DECISION_20261020.md:22-27` prefers native full images. Freeze one explicit evaluation track per receipt before running a comparison. |

## Model admission gates

- **DeepJSCC:** pinned source is explicitly third-party. Resolve original-author
  provenance or retain a clearly named third-party track. Old channel SNR and
  fading conventions cannot be certified merely by preserving their labels.
- **ADJSCC:** retain useful weights, but strictly load them in the actual TFC/GPU
  runtime, validate both Kodak orientations, and profile applied gradients.
  DIV2K adaptation is not the original ImageNet experiment.
- **NTSCC+:** pin compatible NATTEN v0.14.6, validate both optimizers and original
  loss, expose every one of the 800 distinct images per complete epoch, and
  measure side information. NTSCC++ online inference remains unreleased in the
  pinned README; NTSCC+ training does not reproduce NTSCC++.
- **SwinJSCC:** C=96 existing weights are fixed base w/o-SA-and-RA at nominal
  CBR 1/16. A C=256 or adaptive configuration requires corresponding weights.
  Test native channel and resolution transitions, then two-GPU reduction.
- **SQ-GAN:** preserve the actual released discrete-index system, native semantic
  geometry and required stage losses. DIV2K labels/digital transport are explicit
  adapters. Previous analog VQ-vector transport cannot be relabeled exact native
  SQ-GAN. Stage receipts must state which optimizers and complete stages ran.
- **DeepJSCC-Q:** resolve candidate authorship and journal variant, then retain
  FeatureEncoderAF/FeatureDecoderAF and learned-modem annealing. The prior small
  ConvJSCCQ wrapper is a distinct paper-based adaptation.
- **DiT-JSCC/DDM-SSCC:** audit-only; no substitute network or fabricated training
  result. Missing training release/provenance remains an external blocker.

## Regression execution

Initial helper suite command, from the SemanticLossless root:

```powershell
$env:PYTHONPATH = "$PWD\SharedCode\reproduction\.cpu_dependencies"
python -m pytest SharedCode/reproduction/tests -q -rs
```

Initial measured outcome: **15 passed, 2 failed in 9.29 seconds**. Both failures
are the native comparison and repeat-symlink regressions above. No metric tests
were skipped in that run. Tests use temporary source fixtures and synthetic
images, never author model execution or GPU computation. Updated final results
are recorded below after the shared-core and runtime changes are reviewed.

Final executed assigned CPU suite after shared-core fixes: **36 passed in 7.23
seconds, zero skips**. The original native-protocol and repeat-symlink defects
are fixed and regression-covered. Additional checks cover traversal rejection,
valid/invalid plateau receipts, epoch bounds and integer counts, conservative
FLOP coverage even when an unsupported-operator list is empty, calibrated AWGN
complex SNR, finite gradients, repeatable Rayleigh samples, RNG replay, exact
800-source coverage and interrupted atomic save preserving prior bytes.

The first expanded run also exposed acceptance of 151/fractional epochs and
`complete_declared` FLOP coverage from an empty unsupported list. Shared-core
fixes resolved those three failures before the final run. These are helper and
synthetic transport checks, not completed training or GPU compatibility tests.

## Review of new runtime modules

`model_runtime/adapters.py` returns tensor-only metadata and weighted loss sums,
which addresses the identified ordinary DataParallel gather hazard. Its explicit
matched transport preserves per-image normalization and counts symbols; the
synthetic CPU calibration verifies that helper's AWGN convention. It does not
certify the source architectures on two GPUs.

`model_runtime/engine.py` saves the model, optimizer, scheduler, early-stop state,
all RNG states, intra-epoch order/cursor/seen sources and partial objective sum.
Its update/save order supports rollback to the last atomic checkpoint, and it
counts an epoch only after complete coverage and validation. Resume identity
checks prevent silently continuing a different contract. An actual interrupted
author-model GPU continuation remains unverified.

`model_runtime/evaluation.py` labels profiler totals as partial supported-operator
counts and leaves training FLOPs unmeasured. This is honest disclosure, not full
complexity measurement. An empty unsupported-operator list cannot establish
coverage, and different precision/framework profiler totals cannot be treated as
complete comparative costs.

The inspected candidate DeepJSCC-Q quantizer increments annealing state even in
evaluation mode, and twice during a training call. Engine validation restores
buffers/RNG. Evaluation warm-up/profile calls must also preserve the selected
checkpoint buffers before independent trials; this issue was sent to the runtime
owner for correction. Native source behavior and benchmark adapter changes must
remain disclosed.

At this review point, `required_assets()` still explicitly blocks complete
ADJSCC, NTSCC+ and SQ-GAN training pending their actual runtime/transport/stage
implementations and assets. The six-target launcher must report these blockers;
it cannot honestly be described as six verified runnable model pipelines.

## Runtime and reporting acceptance

Before training: strict source/weight identity, finite real forward/backward,
all required optimizer updates, actual two-device operation, and a measured
warmed-up budget. Before resume: atomically saved complete state and matching
contract/data/source identities. Before scores: all 24 original Kodak images,
all configured trials, actual stream counts, one metric implementation, and
explicit matched/native channel definitions. Stop-policy state and completed
source coverage must determine progress; a notebook push or synthetic probe
cannot establish completed training.

Browser upload corruption/overlap was not independently confirmed from the
inspected audit artifacts. Future browser uploads should be serialized and
verified against the intended bytes/version before execution. This is a
recommended acceptance gate, not a claim that an incident was reproduced.

## Subsequent independent delivery review

Latest additional regression file: `tests/test_delivery_evidence.py`. Executed
independently: **1 passed, 7 failed in 3.99 seconds**. The well-formed 1200-row
synthetic grid exports as unranked evidence; the inspected consolidator also
incorrectly admits seven deliberately malformed cases: negative or Boolean
channel-use counts, negative timing, SSIM above one, missing MS-SSIM, missing
LPIPS, and PSNR inconsistent with MSE. These are real validation defects exposed
by synthetic fixtures, not measured model outputs. They were sent to the root
owner for correction. The earlier 36-pass result applies before these additional
tests; the expanded suite must be rerun after delivery fixes.

Further inspection found that the master notebook had no reproducible per-model
dependency provisioning and did not pass SQ-GAN's stage to the CLI. The root
owner is implementing provisioning and stage selection. These additions need
fresh review and execution; source construction alone is not runtime validation.

The inspected runtime CLI's `PREFLIGHT_PASSED_NOT_TRAINED` status followed source
verification, dataset checks and architecture construction, without actual
forward/backward/channel/optimizer execution. CPU mode could also reach that
status. Rename it as preparation-only or execute explicit diagnostic gates before
claiming runtime preflight passed. AWGN metadata should disclose no fading and
its receiver-CSI assumptions rather than automatically repeating the Rayleigh
perfect-CSI/fading scope fields.

The Kaggle wrapper inspected here checks explicit expected ownership and the
authenticated CLI username before push. Interactive authentication occurs only
through its explicit Auth action. No credential exposure or automatic login was
observed; this audit did not execute authentication, push or download commands.
