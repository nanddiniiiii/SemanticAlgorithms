# Semantic Lossless: model registry and two-account execution specification

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
