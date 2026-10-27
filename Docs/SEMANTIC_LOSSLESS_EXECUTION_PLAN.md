# Semantic Lossless: current execution plan

## Current increment — twelve models and three source-driven notebooks

26 October 2026: the user selected twelve models, three notebooks and a common
DIV2K800 + Flickr2K2650 training pool, DIV2K100 development and Kodak24 test.
The user explicitly preserved native graph data for GBSED and paired-view data
for DeepJSCC-WZ. This supersedes the eighteen-model/mixed-dataset design below
for this implementation increment; historical results and originals stay intact.

- You: DeepJSCC, ADJSCC, SwinJSCC, ISEC.
- Teammate1: NTSCC+, HJSCC, JCM, MDJCM.
- Teammate2: Masked SQ-GAN, Shared Semantic-Aware Codebook, DeepJSCC-WZ, GBSED.

Reusable implementation lives in `src/semantic_lossless/team_suite/`; three
generated notebooks live in `Notebooks/TeamSuite/`. Source fetching is pinned;
three downloader entrypoints retrieve saved-version outputs and merge only
verified artifacts. The operating guide is `configs/team_suite/README.md`.

Current evidence: eight targeted CPU behavior/integration tests pass. Synthetic
numerical checks passed DeepJSCC, ADJSCC, approved spatial JCM and both ISEC
training stages. Swin strictly loaded but native forward calls CUDA on this
CPU-only laptop. This is not twelve GPU-validated models or a dataset benchmark.
The project-local environment was created and Torch CPU import verified.
Subsequent bounded Kaggle pilots verified DeepJSCC, ADJSCC, native SwinJSCC,
both ISEC stages and spatial JCM on actual two-T4 hardware, including two-step
updates, strict checkpoint reload and observed encoder work on both devices.
These used synthetic images; no convergence or dataset-quality result follows.
HJSCC and MDJCM subsequently passed bounded dual-T4 training pilots as well.
NTSCC+ was blocked by the legacy NATTEN wheel build before model construction.
Versioned evidence and reporting caveats are in `TEAM_SUITE_VALIDATION.md`.

Acceptance still outstanding: native-data full pilots for each model; dual-T4
training/memory verification; NATTEN and other dependency compatibility; frozen
common-data materialization on Kaggle; native graph/paired assets; SQ-GAN radio
link definition (currently source-only); WZ/GBSED common profiling/ingestion;
published source revision and verified saved-version recovery. No new full
training jobs have been launched or results claimed for this suite.

## Prior decision — eighteen non-diffusion targets, three GPU lanes

26October2026: user has confirmed two teammates each have an independent Kaggle
T4x2 session, in addition to accountB. Latest instruction reopens training, with
at most four hours per algorithm and deadline Wednesday28October07:00IST.
The pretrained comparison remains the baseline evidence. Actual teammate
account identities/quota balances are not yet inspected; no credentials belong
in this document. Three independent sessions are not six independent model
slots: each paired-T4 job uses its own GPUs via compatible DataParallel.

### Data decision

Use `CUSTOM_DATASET_PROTOCOL_V3.json`: target4,000train/500dev/500lockedtest,
eight source families, preserved native model dimensions. This is a finalized
selection design, not an already materialized or frozen image manifest.
The source-aware dev counts and locked-test counts must be verified before
training. Preserve official splits, parent groups and cross-source exclusions.
No source image is replaced by an upscale. ImageNet is a class-stratified
subset, not full1,000-class coverage. Kodak remains a separate historical test.
Native task datasets remain necessary for JCM, SharedCodebook, GBSED and
DeepJSCC-WZ; SQ-GAN custom training requires the already approved pseudo-labels.

### Channel decision

The common first-round channel is calibrated complex AWGN. Fixed-SNR models
use10dB; inherently adaptive models preserve their native-supported multi-SNR
training mechanism/configuration (Swin grid1/4/7/10/13where supported).
Evaluate common supported points and label extrapolation stress points.
Do not force native OFDM/multipath or graph/task pipelines into an unrelated
AWGN replacement. Those have native channel results in separate tables.
Rayleigh is a later robustness evaluation, with gain distribution, coherence,
CSI and equalizer recorded; no blanket Rayleigh retraining in this round.

### Eighteen method targets and action

| Lane | Order | Method | Action / gate |
|---|---:|---|---|
| A | 1 | SwinJSCC | Recover native fine-tuned checkpoint reload, then fine-tune; do not continue the flawed historical EncDec port |
| A | 2 | Improved NTSCC, releasedNTSCC+ | Retrieve author weights; pinNATTEN; fine-tune after native forward/backward/reload gate |
| A | 3 | ADJSCC, DiffComPyTorch lineage | Fine-tune pristine releasedC2 weights; keep originalTF results separate |
| A | 4 | DeepJSCC | Train corrected declared reproduction after complex packing/power gate; retained legacy weights not author-faithful baseline |
| A | 5 | Original NTSCC | Fine-tune author weights only after accounting/refinement contract is explicit |
| A | 6 | DeepJSCC-L++ | Author source exists; nativeCIFAR32 training first; no silent large-image conversion |
| B | 1 | Shared Semantic-Aware Codebook | Continue our released nativeSAC ablation with full compatible resume state; otherwise explicit weights-only continuation |
| B | 2 | MDJCM | Author checkpoint access and staged training prerequisites first; no pretend scratch convergence |
| B | 3 | Masked SQ-GAN | Native required stages/trainer gate; fine-tune author components only when verified; pseudo-label adaptation declared |
| B | 4 | JCM | Fine-tune native supplied16-QAM checkpoint onCIFAR; modified spatial variant separate |
| B | 5 | SSC spatial/residualVQ | High-risk native staged training prefix only; reconcile config/paper and resource budget |
| B | 6 | DeepJSCC-Q | BLOCKED until known quantizer/objective defects and paper variant are resolved |
| C | 1 | GBSED | Native graph/task training and data gate; not RGB reconstruction |
| C | 2 | OFDM-guided Deep JSCC | Native channel/estimation/equalization training; input/data readiness gate |
| C | 3 | ISEC | Load author base and denoiser; train/fine-tune denoiser if actual training desired; iterative inference itself is not training |
| C | 4 | DeepJSCC-WZ | Author source/weights plus native correlated stereo data; fine-tune after receiver-side-information gate |
| C | 5 | MambaJSCC | Author weights first; native CUDA/dependency gate; no further debugging beyond slot preflight budget |
| C | 6 | HSC semantic-complementary | BLOCKED: no verified full author release; projection module alone is insufficient; paper-derived implementation must be labelled |

This is an18-target programme, not18already runnable trainers or a universal
top18quality ranking. Four main triplets cover continuous, entropy/rate,
digital modulation and codebook/generative mechanisms; six specialists add
waveform, refinement, state-space, correlated side information, graph and hybrid
coverage. A method can belong to multiple families. The hybrid family remains
unfulfilled until its gate is passed; do not hide that gap behind a substitute.

### Slot accounting and stop rules

18*4=72paired-T4-sessionhours. With three independent sessions the nominal
training/evaluation-slot wall clock is24hours. Reserve roughly3hours for shared
data/preflight and6hours before the deadline for aggregation/reporting; actual
readiness and quota determine executable scope. Each slot includes20minutes
preflight, up to3hours bounded training, and40minutes evaluation/retrieval.
Setup/trainer development is real work and must be counted; missing prereqs
can prevent a method from reaching training. Skip a blocked slot rather than
silently substituting a different method. Advanced high-risk pipelines are
not guaranteed meaningful convergence within the cap.

Each teammate owns a separate six-method queue and new run directories, pinned
to one source snapshot and shared frozen dataset version. Keep logs/checkpoints
independent; one local collector merges results idempotently. Select by dev
only; final holdout once after settings freeze. Preserve all pretrained weights.
No experiment was submitted while writing this plan. Prior pretrained-only
scope and20kdata targets below are historical and superseded for new training.

## Previous immediate scope — pretrained comparison

Latest user direction,26October2026: suspend retraining for now. Compare existing
author checkpoints, our existingSwin checkpoint and SharedCodebook artifacts;
MambaJSCC andDiffJSCC use published results only. EvaluateDiffCom where its
existing runtime permits. The current deliverable and inference-only gap tests
are in [PRETRAINED_MODEL_COMPARISON.md](PRETRAINED_MODEL_COMPARISON.md).
The longer training/data plan below is retained and deferred; it does not
authorize automatically advancing training queues during this comparison.

## Current plan — 26 October 2026

This section supersedes the historical specification below. The immediate priority
is a usable, frozen diverse dataset and one verified coupled ns-3/Sionna pilot.
Then complete the existing-method inventory and run controlled data experiments.
New work uses account B (`satyapaladuguplus`, authenticated through WSL).
Previously measured results and author checkpoints remain immutable references.

### Verified starting point and correction

The coupled pilot v1 has reached Kaggle COMPLETE, but its own report is
`partial_or_failed`: zero of eleven recorded dataset checks passed and radio
execution never started. KaggleHub rejected dataset attachment inside a
non-interactive run with no declared dataset inputs. The environment diagnostic
then raised `AttributeError: module 'sionna' has no attribute '__version__'`.
Two T4s and dependency installation were observed; coupled radio execution was
not. Outputs were retrieved into
`Results/stage1_operations/recovered/accountB/runs/ns-3-sionna-and-custom-dataset-pilot/v1/`.
This is evidence of two launcher faults, not evidence against the datasets or
the ns3sionna method. Preserve v1 and use new version/run identities for fixes.

The expanded v2 dataset remains a proposal. The existing frozen v1 dataset and
Stage-1 results remain available. Neither sample accessibility nor a Kaggle
worker state establishes dataset ingestion, training convergence or quality.

### Work package 1: load and freeze the actual selected dataset

1. Pin the Kaggle dataset version and upstream partition for each source.
   Declare required dataset inputs in batch-job metadata before submission.
   First test one declared source in a CPU job, including image decode and
   artifact retrieval; then extend that exact path to the remaining sources.
   Do not rely on runtime KaggleHub auto-attachment in a batch session.
2. Use CPU data-preparation runs, with source batches if attachment/resource
   limits require them. Read mounted source data and materialize the selected
   images into a compact dataset artifact. Full upstream corpora need not be
   copied. Count accepted/rejected images and bytes; inspect actual dimensions.
3. Use the following target mixture, already proposed in
   `CUSTOM_DATASET_PROTOCOL_V2_PROPOSED.json`. Counts are targets until verified:

| Source | Train | Development | Purpose |
|---|---:|---:|---|
| DIV2K | 800 | 100 | High-resolution reconstruction anchor |
| Flickr2K | 2,000 | 200 | Detailed natural photographs |
| LSDIR | 6,000 | 400 | Broad high-resolution photographic variation |
| ImageNet original resolution | 5,000 | 500 | Object/class diversity, with class balance tracked |
| COCO | 2,000 | 250 | Multi-object scenes and everyday context |
| TextOCR | 2,000 | 250 | Text-heavy natural images |
| DocLayNet | 1,200 | 200 | Document text, layout and fine structure |
| FFHQ | 1,000 | 100 | Facial details and identity-sensitive content |
| **Target** | **20,000** | **2,000** | |

4. Reserve a proposed fresh 1,000-parent-image mixed holdout, with counts
   conditional on verified eligible groups: ImageNet 200, COCO 200, TextOCR 150,
   DocLayNet 150, Flickr2K 100, LSDIR 100, FFHQ 100. Protect official evaluation
   partitions and exclude all selected holdouts from training. Kodak remains a
   historical evaluation reference that has already informed development.
5. Partition at original photograph/document level before cropping. Preserve
   official splits where available; record any within-source group split.
   Exclude v1 development/test images from new training. Check exact-file and
   decoded-RGB duplicates across sources/splits; review or group perceptual
   matches. Hash heuristics alone do not prove independence.
6. Preserve original pixels and metadata. Do not upscale small images and call
   them high resolution. Tag resolution cohorts; record rejections and quota
   shortfalls instead of silently substituting sources. Review stratified
   contact sheets for corruptions, duplicates and domain coverage.
7. Freeze train/dev/test manifests with source versions, parent IDs, partition,
   dimensions, hashes, quality flags and split reason. Verify every selected
   image decodes, selected counts reconcile, and a retrieved artifact can be
   reopened. Store a versioned private Kaggle dataset plus a local manifest and
   recovery copy. Keep image licensing/provenance with the dataset.

Acceptance: actual selected images are available to a fresh run, split/hash
checks pass, source counts and bytes reconcile, and the manifest is immutable.
One sample per source is a preflight only. If source access fails, retain the
partial selection and explicitly revise its quota before freezing.

### Work package 2: one independent ns-3/Sionna integration pilot

Keep radio validation independent of dataset downloads and training. Preserve
the pinned ns-3 3.40/ns3sionna/SionnaRT 1.2.1 combination. Fix version reporting
with distribution metadata; import RT first, verify the actual polarized CUDA
variant, then run the upstream static-room example without an image model.

Acceptance requires the ns-3 example to exit successfully, the RT server to
initialize and finish cleanly, at least one finite SNR/channel trace, recorded
scenario/units/seeds, and retrieved logs/configuration/trace. A GPU listing alone
does not pass this gate. Record cold setup separately from simulation runtime.
Allow one evidence-led corrective retry after a concrete failure; do not keep
submitting variations without a discriminating check. Data preparation can
proceed independently if radio integration remains unresolved.

### Work package 3: close Stage 1 with existing weights and native contracts

The twelve-paper coverage remains the programme in `MODEL_SELECTION.md`:

| Track | Methods | Immediate treatment |
|---|---|---|
| Continuous CNN/adaptation | DeepJSCC, ADJSCC | Retain weights; close channel, rate and profiling gaps |
| Transformer | SwinJSCC | Retain author variants; diagnose strict-reload/adaptive anomalies |
| Entropy/rate adaptive | Improved NTSCC (released NTSCC+ branch) | Dependency and author-checkpoint gate before evaluation |
| Digital neural modulation | DeepJSCC-Q, MDJCM | Verify quantizer/objective or staged modulation; pilot before training |
| VQ/generative semantics | Masked SQ-GAN, SSC | SQ-GAN native baseline/trainer gate; SSC remains conditional on fidelity and cost |
| Receiver refinement | ISEC | Strict-load author JSCC/denoiser; charge all receiver iterations |
| Physical-layer structure | OFDM-guided Deep JSCC | Native multipath/estimation/equalization pilot |
| Graph/task communication | GBSED | Separate native graph/task evaluation; no invented RGB score |
| Complementary hybrid | HSC | Explicit paper-derived prototype; charge basis, coefficients and protection |

JCM and Shared Semantic-Aware Codebook remain preserved auxiliary studies,
including the approved spatial JCM variant and classifier pseudo-label option.
They do not replace rows in the current twelve-paper coverage. Diffusion/Mamba
scratch training stays deferred. No promise that all twelve can be faithfully
trained within a fixed few-hour allocation is made.

For each method record: exact variant, source/checkpoint hash, retained versus
author versus newly trained weights, native dataset/geometry/objective/channel,
strict-load result, pilot status, missing dependencies and next blocking gate.
Evaluate working weights first. Train missing weights only after a complete
forward/backward, actual-rate, save/reload and artifact-retrieval pilot passes.
No substitute architecture may be labelled as a paper reproduction.

### Work package 4: controlled custom-data training

Start with the verified ADJSCC path as the end-to-end dataset/training reference,
then native SwinJSCC once its reload failure is resolved, then DeepJSCC once
channel accounting is calibrated. Advance digital/VQ methods in readiness order,
not an assumed four-hour duration: MDJCM, faithful DeepJSCC-Q, Masked SQ-GAN,
Improved NTSCC, then other feasible candidates. Readiness can change this order.

For each executable model, compare native-data training with v2 training from
the same initialization and matched optimizer-update budget. Record image and
pixel exposure because equal steps alone do not equal equal data/compute.
Evaluate both on native evaluation data and the frozen mixed holdout. Retain
author pretrained results separately when their training exposure cannot be
matched. Use development data for configuration and quality checks; open final
holdouts only after selection is fixed.

Preserve native input/loss/model contracts. Common parent images may yield
different native patches; record crop/padding/rate overhead. The approved larger
JCM reconstruction variant remains explicitly modified. SQ-GAN may use the
approved frozen segmentation pseudo-labels, with confidence/domain coverage and
teacher cost. Shared Codebook may use the approved classifier pseudo-labels,
with coverage reported. Task models require their own valid labels/graphs.

Use PyTorch DataParallel where compatible and verify work/gradients on both T4s.
Do not silently alter a native objective or custom forward API to force it.
Measure a short real-data pilot's memory and throughput before assigning a
wall-clock cap. Record completed full passes, steps and checkpoint state;
150 epochs is not a universal convergence guarantee. Retrieve each completed
run before advancing the GPU queue. Observe current quota before budgeting;
the original twenty-hour deadline is historical, not a current resource fact.

### Work package 5: common evaluation and environment expansion

First establish a transparent AWGN baseline (10 dB training where appropriate;
evaluation at 0, 5, 10, 15 and 20 dB), using native-supported rates and recorded
SNR/power definitions. Preserve native channel results in their own tables.
Then add controlled block-Rayleigh and a validated OFDM multipath chain; couple
ns-3/Sionna traces only after their power/noise/CSI conventions are reconciled
with the model interfaces. Do not replace all native training channels at once.

Report resolution/range, PSNR/SSIM/MS-SSIM, LPIPS, domain-specific semantic
metrics with valid labels, actual payload bits/bpp, complex uses/CBR, metadata,
side streams, FEC/pilots/CP and retransmissions. Compare continuous/digital links
at charged resource budgets. A one-shot result means one declared scheduled
burst with no ARQ; state its resource limit and empirical failure rate.
Include RAW/PNG/JPEG/JPEG2000 controls with the same defined transport when
making channel comparisons. Distinguish source-coding quality from radio cost.

Profile encoder/decoder/full inference, parameters, supported FLOPs and missing
operators, synchronized latency, precision, actual hardware and peak memory.
Charge all ISEC iterations, segmentation and secondary streams. Never replace
missing FLOPs with latency or rank incompatible resolutions as compute wins.

### Delivery and immediate sequence

1. Record/reconcile the failed v1 pilot and preserve retrieved evidence.
2. Fix and preflight the two independent data and radio launch paths.
3. Load/freeze the actual v2 selection and complete the radio smoke.
4. Refresh Stage-1 inventory/results from receipts; close executable baselines.
5. Run one controlled ADJSCC data comparison and verify the complete collection
   and reporting path before expanding to the remaining models.
6. Expand by readiness, then evaluate shared channel upgrades and hypotheses.

Maintain this file as the plan, `KAGGLE_RUNS.md` as the run/artifact map, and
`STAGE_ONE_REPORT.md` as measured evidence. Keep per-image metrics and run
provenance beneath the existing Results tree; one local collector updates the
comparison CSV idempotently. Unknown metrics remain missing. New mechanism or
novelty claims require controlled evidence after this baseline is stable.

---

# Historical specification — retained for provenance

The specification below predates the current plan and does not govern new
account selection, model coverage, data loading, or scheduling decisions.

Date: 21 October 2026 (Asia/Calcutta).

Status: notebook/training specification. See `RETAINED_ENCDEC.md` for the completed
local codec transfer and CPU checks, and `MODEL_SELECTION.md` for the latest
model recommendations including DiffJSCC. No GPU pilot or new training is claimed.

## Repository and preserved work

- Target: https://github.com/nanddiniiiii/Semantic_Lossless.
- Live authenticated repository inspection: public; current collaborator has
  push permission but not administrator permission. No visibility change is needed.
- Inspected remote HEAD: `6e019b65b1a165ada4e857771982e8e2c3e00e8c`.
- The canonical, fully checked-out clone is `Semantic_Lossless/`. Existing
  workspace research documents, teammate artifacts, and scheduler code remain intact.
- The repository already contains `SharedCode/reproduction/`, a source lock,
  runtime adapters, notebooks, and historical results. Reuse suitable components
  after inspection; do not replace them with an unrelated second runtime.
- Some existing documentation calls the project SemanticAlgorithms. The explicit
  target above governs this work; reconcile stale naming during integration.

## Fifteen-model registry

| Model ID | Current action | Queue | Benchmark/default |
|---|---|---|---|
| deepjscc | Reuse checkpoint; verify channel convention, rate, evaluation and inference profiling | Reference evaluation | DIV2K/Kodak with recorded pretrained provenance |
| adjscc | Fresh AWGN training requested; preserve old artifacts separately | A1 | DIV2K train/validation; Kodak test; label data adaptation |
| swinjscc | Reuse existing fixed and adaptive checkpoints under distinct variant names | Reference evaluation | DIV2K/Kodak |
| ntscc | Original NTSCC only; reuse supplied checkpoint | Reference evaluation | Native compatible high-resolution evaluation |
| deepjscc_q | Verify public trainer against chosen paper, then bounded training | B4 | CIFAR-10 initially |
| jcm | Load published checkpoints and evaluate before modification | B1 | Native CIFAR-10 geometry |
| mdjcm | Retrieve linked MDJCM-A checkpoint; validate dependencies and inference | B2 | Released general-image configuration; Kodak where supported |
| shared_semantic_codebook | Construct native codebook/classifier and train | A3 | STL-10 native labeled setup |
| masked_vqvae | Research and source audit only until a working image-reconstruction/channel pipeline exists | Not queued | Public demo is classification-oriented; inspected VQ/AWGN calls are disabled |
| sqgan | Load author weights, establish native baseline, then bounded fine-tuning | A2 | Cityscapes images and class-ID masks |
| diffcom | Pretrained inference and sampling/guidance experiment; no diffusion scratch training | A4 | Released ImageNet/FFHQ setup |
| dit_jscc | Research only until actual JSCC checkpoints are released | Not queued | Checkpoint interface currently documents future release |
| hda_deepsc | Research only; no verified author executable release identified | Not queued | Paper-specific protocol |
| hybrid_semantic_complementary | Implement a clearly labeled paper-inspired projection/complementary-information prototype | A5 | Frozen existing codec; DIV2K development and Kodak test |
| mambajscc | Build CUDA extensions, retrieve checkpoint and validate native inference | B3 | Released DIV2K setup; record checkpoint channel, including Rayleigh when applicable |

MOC-RVQ stays in the literature notes. Its public repository currently announces
future code and weights; it is not a sixteenth executable implementation.
NTSCC+ and NTSCC++ are outside this implementation scope.

There are nine active experiment slots, five on A and four on B, plus three
reused reference architectures and three research-only model entries. A registry
entry or directory is not proof of an implemented or validated model.

## Source layout and interfaces

- Put model adapters under `src/semantic_lossless/algorithms/<model_id>/`.
- Each executable algorithm owns `train.py`, its model/configuration integration,
  and evaluation hooks. Keep its upstream revision and checkpoint provenance.
- Common components own datasets, channels, evaluation, inference profiling,
  checkpoint handling, event logging, artifact export and Kaggle orchestration.
- Keep dependency environments separate for incompatible PyTorch, TensorFlow and
  native-extension implementations. Notebooks are launchers, not copies of training code.
- Provide explicit `prepare`, `pilot`, `train`, `evaluate`, `profile`, and `export`
  stages. Inference-only and unavailable models report their status explicitly;
  they do not silently execute a substitute model.
- Preserve pinned upstream source and license provenance through the existing
  source-lock mechanism. Launch by immutable repository commit.

## Two notebooks and three cells

Preferred default, pending the user's notebook-shape preference: two reusable
account notebooks, with one algorithm per execution and an automatic sequential
queue for each account. This permits retrieval after each algorithm finishes.

The three cells for an algorithm are:

1. Prepare: fetch the pinned code revision, select the isolated environment,
   verify data/weights, create a unique run directory, print the exact run plan.
2. Pilot/train: run the pilot gates and the requested training or inference
   experiment; print flushed stage and step/epoch progress and save checkpoints.
3. Evaluate/export: run held-out inference, rate/quality evaluation, inference
   FLOPs and latency profiling, then finalize a hashed artifact manifest.

If multiple algorithms must execute in one long notebook, retain these three
cells per algorithm and add explicit intermediate artifact publication. Do not
assume a local mount or a completed cell automatically exposes remote files.
Validate intermediate publication in a pilot before depending on it.

Initialization can clean only the current run's disposable cache/scratch paths.
It must not delete existing checkpoints, historical results or another run's
files. Resume from a checkpoint is an explicit operation with a compatible config.
Different algorithms run sequentially within an account; accounts run concurrently.

## Artifact and CSV contract

Remote root: `/kaggle/working/results/<model_id>/<run_id>/`.
Local root: `results/<model_id>/<run_id>/` beneath the execution checkout.
Separate baseline and modified variants in run metadata and directory identities.

Keep best/last checkpoints, optimizer/scheduler/scaler/RNG resume state where
supported, configuration, upstream commit, environment/device metadata, logs,
per-step and per-epoch observations, per-image/channel-trial metrics, reconstructed
examples, timing/FLOP coverage and final checksums. Do not save the full training
dataset or every intermediate tensor into output storage.

Each remote run writes append-only structured event records. One local collector
owns the canonical `results/metrics.csv` and appends newly downloaded observations
from both accounts. Remote accounts do not concurrently edit a shared CSV file.

Use a long-format CSV: record ID, timestamp, account, algorithm, variant, run ID,
stage, dataset/split, sample or batch IDs, epoch/step/trial, seed, channel/SNR,
actual channel usage, metric name/value/unit, status and artifact reference.
Record IDs prevent duplicate rows when synchronization is repeated. Failed and
partial runs are recorded with status; unavailable metrics are not zero.

Kaggle does not write directly into the WSL-mounted Windows folder. The collector
polls account/run status, retrieves available artifacts, verifies complete files
before promotion, appends CSV rows once, and then advances that account's queue.
Local CSV freshness is bounded by artifact publication and synchronization.
Live stage output belongs in the notebook logs; guaranteed cross-platform live
export requires a separately verified publication path.

## GPU pilot gates before full training

Perform local syntax/config/interface checks first. GPU-dependent memory and
extension validation must run on Kaggle's actual T4 hardware before long jobs;
a CPU check cannot certify two-T4 compatibility.

For each executable model, require:

- Correct dataset/mask pairing, split separation, tensor shapes and ranges.
- Strict checkpoint load, or documented scratch initialization.
- A finite forward pass and several complete optimizer steps, including the
  discriminator when present; validate the channel convention and actual rate.
- Validation inference at the intended dimensions and batch size.
- Save/reload/resume validation plus a small artifact export/download receipt.
- Measured peak memory after optimizer state allocation, not only a forward pass.
- For two-GPU training, verify both ranks/devices receive work and effective batch
  size/gradient reduction is correct. Use model-compatible DDP or TensorFlow strategy.

T4 memory is per device, not a pooled 32 GB. Use supported FP16 mixed precision,
gradient accumulation and measured batch sizing with approximately 15-20 percent
memory headroom. Do not maximize RAM usage as an objective or claim zero OOM risk.
Fallbacks must preserve algorithm behavior and be recorded. An OOM restarts from
the last valid checkpoint with explicit batch/accumulation changes.

Only inference FLOPs are required. Report encoder and decoder separately and,
for diffusion, the cost of the complete sampling trajectory. Include semantic
teacher/segmentation overhead in a separate full-system scope. Unsupported
operators make the estimate partial; latency does not replace FLOPs.

## Learning and research comparison

For each model, record input/tensor flow, transmitted representation, losses,
decoder operation, native metrics, limitations and one testable improvement idea.
Keep published paper results, our reproduced baseline and our modification in
separate columns. Only compare improvements directly when data, rate, channel,
training budget and evaluation protocol match.

Native datasets are allowed. Keep CIFAR-10, STL-10, Cityscapes and general-image
benchmark tables separate. Kodak is a held-out evaluation dataset, not a training
or codebook-fitting split. Start new channel-dependent training with AWGN; preserve
and label native checkpoint channel conditions rather than relabeling them.

Treat proposed improvements as hypotheses until tested and checked against prior
work. Give continued-training controls the same additional optimizer-step budget.
Run equal-cost ablations and repeat the strongest findings with multiple seeds.

## Run budget and release gates

Use a measured pilot to estimate training duration. Four hours per slot is an
upper allocation, not a guaranteed convergence time. Target 150 complete epochs
where requested but record actual completed epochs/image exposure at a wall-clock
stop. Inference-only methods do not need artificial training epochs.

Allow at most 45 minutes for a setup blocker before moving to another executable
slot. Recheck both accounts' live quotas and device availability before scheduling.
Reserve at least six hours before the presentation for final evaluation, plots,
learning consolidation and PPT verification.

Before full jobs: CPU/config checks, actual GPU pilots, artifact round-trip and
CSV deduplication checks must pass. Record each model as research-only, prepared,
pilot-passed, running, partial, failed or completed. No uploaded notebook or model
quality result is claimed until the corresponding receipt exists.

## 2026-10-27 structure migration checkpoint

The active package is flattened under `src/`; notebooks and local launch scripts are numbered 1–3, channel infrastructure is under `Infrastructure/`, and retained EncDec is under `Archive/EncDec/`. Existing data, checkpoints and results were preserved. The original active source was copied to the sibling cleanup archive before migration.

DeepJSCC supports independent training and evaluation AWGN or flat Rayleigh perfect-CSI conditions. Other model adapters do not yet support independently differing conditions and reject them. This changes the DeepJSCC channel adapter and is not a paper-exact reproduction claim. The source revision in the generated notebooks is intentionally unset; local scripts reject submission until a reviewed published revision is pinned. No Kaggle run was submitted for the migration.

Local verification: `python -m pytest tests/current -q` passed 52 tests; editable installation, notebook generation and the `semcom.py` CLI help/list were checked. Previous GPU pilots apply to prior source snapshots, not this changed channel adapter or the flattened package. Remaining: validate and integrate train/test conditions adapter by adapter, run an approved real GPU preflight on the migrated snapshot, then publish/pin a reviewed revision if authorized. Archived/historical scripts have not all been executed under the new paths.

## 2026-10-27 channel and delivery increment

Shared AWGN/perfect-CSI Rayleigh profiles now integrate with notebook preflight. DeepJSCC/ADJSCC and explicitly adapted fixed-C32 Swin support independent conditions; native digital chains remain protected by capability checks. User selected saved notebook outputs only; no private cloud outbox is authorized. A completion ledger is exported, and local collectors poll at 30 minutes with persistent recovery/publication queues. Result-only GitHub API commits preserve other teammates' events without modifying the local source checkout. Missing permissions retain pending events. Exact training resume and real-time live-file retrieval remain unsupported. See `Docs/RESULT_COLLECTION.md` for the canonical operating procedure. No new remote training was launched.

## Revised twelve-model assignment

Person 1: DeepJSCC, ADJSCC, ISEC, NTSCC+. Person 2: masked SQ-GAN, MDJCM, SharedSemanticCodebook, SSC. Person 3: HJSCC, OFDM-guided JSCC, DeepJSCC-MIMO, DeepJSCC-WZ. SwinJSCC is retained evaluation-only; no duplicate training slots. Existing historical folders/results remain intact.

Assignments are not twelve implemented runners: SSC requires synchronized EMA updates and native channel accounting; OFDM requires audited legacy FFT conversion and runner integration; MIMO requires source pinning and integration. These three slots fail explicitly before training. NTSCC+ requires NATTEN CUDA qualification. WZ preserves paired data/native DDP and separate reporting. No new training was launched. The authoritative machine-readable readiness gates are configs/models.json.
