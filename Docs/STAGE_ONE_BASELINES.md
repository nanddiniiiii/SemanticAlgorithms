# Stage 1: existing-method baselines before our dataset adaptation

Active scope established 22 October 2026. Stage 2 is not launched by this work.
Use the existing methodology and native input/output contract for each selected
paper. Preserve previously trained weights; prefer verified author checkpoints;
train missing models only after their native method and training path are checked.
Downloading weights, matching architecture, running inference, completing training
and measuring all metrics are separate milestones.

## Recovery checkpoint 25 October 2026

Both earlier Stage1 submissions obtained two T4 GPUs but failed before model
execution because the source bundle omitted sources.lock.json, required by the
project-root contract. No classifier training or Swin quality result came from
those runs. The bundle now includes that file and a fresh-process preflight;
the regression test executes the transported checkout rather than the local one.
Subprocess progress/errors are persisted by named stage with elapsed time and
snapshot identity. A bounded Medium assessment accepted the startup correction.
Its two reporting findings were also addressed: native checkpoint variants are
distinct in the canonical CSV, and memory measurements separate warmed residency
from forward allocation growth, exclude FLOPs tracing and state that component
measurements retain the whole model. The canonical remote artifact map is
KAGGLE_RUNS.md; only retrieved files establish durable output recovery.

## Classification

25 October execution: account B native Swin version2 completed Kodak24 for both
checkpoints, with encoder/decoder/full T4 profiles recovered locally. Fixed-C32
PSNR29.3711dB; SA+RA21.3327dB at requested10dB/C32. The adaptive result remains
flagged pending explanation; it does not establish an architecture ranking.
Account A classifier version3 continues independently. The report is
STAGE_ONE_REPORT.md and output/pdf/SemanticLossless_Stage1_Report.pdf.

NTSCC accounting correction: the configured retained decoder has hyperprior
refinement disabled (`use_side_info=False`). Its known missing transport is
the rate-index/mask information supplied directly to the decoder, not a
demonstrated mandatory hyperprior bitstream. Sixteen choices require four
fixed-width bits per spatial index; Kodak has32x48=1536 indices, or6144 uncoded
bits (768bytes,0.015625bpp), before protection/headers. This is a simple explicit
representation, not an entropy lower bound. Encoder-derived power restoration
also assumes receiver availability inside the simulator. Main-stream CBR
therefore remains incomplete even after this metadata estimate is recorded.

The categories below are action priorities, not mutually exclusive facts.
SwinJSCC, for example, has both retained weights and downloaded author weights.
Implementation fidelity is a separate column rather than a third weight category.

| Algorithm | Retained model | Author pretrained availability | Code / implementation fidelity | Stage 1 action |
|---|---|---|---|---|
| DeepJSCC | Yes, original production and other historical assets | Existing retained author-lineage checkpoint usable for audit | Checkpoint-compatible port; legacy channel and packing need verification; production training setting 200 dB | Retain immutable; verify operating point; evaluate/profile; no automatic retraining |
| NTSCC original | Yes, quality-1 production checkpoint | Existing retained original weights | Forward evaluation works with Windows entropy compatibility; complete side-stream rate not verified | Retain; extend evaluation/profile/rate ledger; do not relabel as NTSCC+ |
| SwinJSCC | Yes, previously trained port | Native fixed-C32 and SA+RA author checkpoints downloaded | Retained port has power mismatch and is audit-only; native base architecture strictly loads | Preserve retained model; use separate native author runs for valid comparisons |
| ADJSCC | Historical artifact retained separately | DiffCom authors' PyTorch C2 downloaded | Adapted PyTorch lineage; GPU optimizer pilot passed; not an exact original TensorFlow reproduction | Evaluate/profile C2; no need to retrain just to obtain a model |
| JCM | Not required | Native digital and analog-initialization checkpoints downloaded | Native forward/loss/gradient equivalence and full CIFAR evaluation passed; modern environment differences declared | Evaluate/profile native32x32 baseline; no Stage1 scratch retraining |
| SQ-GAN | Not previously trained by this execution | Ten native inference components downloaded | Native component inference/preprocessing checked; native three-stage trainer not integrated | Evaluate/profile native Cityscapes inference; Stage1 does not require retraining available weights |
| VQ-DeepSC | None verified | Author code/checkpoint release not verified in 25 October 2026 check | Existing survey entry; 2022 preprint / 2023 journal; no local implementation | Added as multiscale CNN/VQ/GAN digital candidate; verify executable source before implementation/training |
| VQ-DeepISC | None verified | Author code/checkpoint release not verified in 25 October 2026 check | Existing survey entry; 2025 preprint; no local implementation | Added as Swin/VQ/channel-adaptive QPSK-OFDM candidate; verify executable source before implementation/training |
| SC-LMKB | None verified | Code/checkpoint availability not verified | [Semantic Communication with an LLM-enabled Knowledge Base](https://arxiv.org/abs/2604.05504), 7 April 2026 preprint; earlier ICC2025 conference version noted by authors | User-selected knowledge-base reference; cross-modal retrieval task, not a direct image reconstruction baseline; no implementation/training launched |
| MambaJSCC | No verified local training | Native AWGN no-CSI C16 and Rayleigh CSI-ReST C32 pairs downloaded | Native CUDA build fails before inference | Resolve compatibility and evaluate downloaded weights; no scratch training while runtime is broken |
| Shared Semantic-Aware Codebook | No verified trained codec for the requested native method | No checkpoint found in checked author tree; saved_model directories contain instructions only | Native backbone/construction components integrated with adaptations; published SAC without the proposed information-theoretic loss branch only; full proposed-loss branch not certified | Missing-weight training candidate: classifier prerequisite, codebook construction, selected native codec branch, then evaluation |
| DeepJSCC-Q | No verified trained model | No official author checkpoint source verified | Candidate public trainer needs paper-variant and quantizer fixes verified | Missing-weight candidate, but blocked from long training until method/parameters pass audit |
| MDJCM | No verified trained model | MDJCM-A author-linked Baidu checkpoint inaccessible in current unauthenticated checks | Pinned author code; backend not yet verified | Prefer resolving author weight access; otherwise audit native training before scratch run |
| MOC-RVQ | None verified | Author-associated repository says code and weights forthcoming | Current checked release has no executable method | Deferred; cannot train an absent implementation as a paper reproduction |
| Masked VQ-VAE | None verified for requested path | None verified | Inspected classification path disables VQ/AWGN; target communication algorithm unresolved | Deferred until specific paper/code path verified |
| HDA-DeepSC | None verified | None verified | Verified author executable release not identified | Deferred; do not substitute our prototype |
| Our complementary hybrid | Experimental reference code | Reuses ADJSCC; no native-paper weights | Our ADJSCC+DCT reference, not verified author reproduction | Keep outside Stage1 published-method ranking; candidate Stage2/reference work |
| DiffCom | Downloaded prior/base and completed pilot | Downloaded | Runtime pilot passed with adaptations | Parked under current diffusion exclusion; keep existing results |
| DiffJSCC | Downloaded checkpoint | Downloaded | Integration pending | Parked under current diffusion exclusion |
| DiT-JSCC | None verified | Required JSCC weights unavailable in inspected release | No verified executable backend | Parked under current diffusion exclusion |

Shared Codebook is not yet the complete paper method: a classifier checkpoint
is only a prerequisite, and construction alone is not a trained encoder/decoder.
No completed model training is claimed by launching that prerequisite.
The checked availability is recorded with primary links in PRETRAINED_DOWNLOADS.md.

### VQ additions checked 25 October 2026

The user's "Vector Quantized Deep Semantic communication" is interpreted as
VQ-DeepSC, officially titled [Vector Quantized Semantic Communication System](https://arxiv.org/abs/2209.11519).
It and [VQ-DeepISC](https://arxiv.org/abs/2508.03740) were already described in
the parent workspace's research/survey.md (R30/R31); VQ-DeepISC also appears as
paper36 of its top50 review. They were missing from this executable-capability
catalog and Stage1 inventory. These additions close that tracking gap without
claiming runnable code, downloaded weights or completed results. A public paper
is not evidence of an executable release. Targeted paper/code searches did not
verify author source or checkpoints; this is not proof that no release exists.
Do not substitute a generic VQ-VAE or another DeepSC repository under their names.

SC-LMKB was added at the user's request on 25 October 2026. It augments source
and channel knowledge using LLM generation, with hallucination filtering and
cross-domain fusion. Its reported retrieval gains must not be copied into our
image PSNR, lossless recovery or compression results. Potential transfer to our
data/knowledge-base methodology is a hypothesis, separate from native baselines.

## Training and evaluation queue

1. Shared Codebook classifier: run a real STL10 two-step GPU pilot with strict
   save/reload, then its native 200-epoch classifier schedule under a bounded
   wall-time budget. Export actual coverage; a budget stop is not convergence.
2. Shared Codebook construction and codec: verify selected published branch,
   construction semantics, classifier dependencies, objective and channel before
   launching the next stage. Keep any source fixes explicitly named.
3. DeepJSCC-Q: resolve candidate quantizer evaluation and annealing behavior;
   verify paper variant and rate before permitting a long run.
4. MDJCM: author weight access first; native training only after verification.

In parallel, existing weights require no training to measure inference FLOPs.
Native Swin fixed-C32 and SA+RA Kodak evaluation/profiling is packaged as a
separate job. JCM, ADJSCC, NTSCC and SQ-GAN need additional profiling and metric
coverage. Training every model for an arbitrary 150 epochs is not this stage.

## Stage 1 results contract

One record identifies a model variant, checkpoint, dataset/split, preprocessing,
rate, channel, SNR, precision, hardware and random-trial protocol. Do not merge
different operating points into a single unexplained model score.

| Required group | Fields / handling |
|---|---|
| Identity and dimensions | Paper variant, source commit, checkpoint hash, RGB/semantic input shapes, latent/mask shapes, reconstructed shape, crop/pad/resize/tile contract |
| Pixel fidelity | MSE, RGB PSNR, SSIM/MS-SSIM, maximum error, exact-pixel/image equality where meaningful |
| Perceptual fidelity | LPIPS with network/version and preprocessing; mark missing dependencies as unmeasured |
| Semantic fidelity | Classification, OCR, detection/count or segmentation only where task labels/evaluator exist; original-image evaluator baseline; no invented generic semantic score |
| Transmission | Actual complex uses/CBR or encoded bits/bpp, side streams, masks and index costs; identify incomplete rate and oracle assistance |
| Compute | Encoder/decoder/whole forward parameters and inference FLOPs, operation convention and unsupported-op coverage; never label a partial count total FLOPs |
| Runtime | Batch-one synchronized latency distribution, preprocessing/CPU work separately, hardware/precision, peak allocated/reserved memory |
| Reliability | Successful/failed inputs, nonfinite/OOM/decoder failures, trial seeds, repeated-noise variability; source exactness distinct from channel delivery |
| Training provenance | Native vs adapted objective/optimizer/data, epochs/steps/images covered, stopped/complete status and available training-state history |

An unavailable or unsupported metric remains null with a reason. A runtime test
with two optimizer steps is not a quality result. A model working at a resized
input is not a full-resolution benchmark. Native dataset comparisons and common
dataset comparisons remain separate tables.

Stage 1 exit requires all selected runnable models to have attributable evaluation
and profiling records, with unresolved fields explicit. Deferred models remain
listed with blockers; their existence does not imply Stage1 validation. No Stage2
improvement claim is permitted until the applicable baseline passes its checks.

## Stage 2 change ledger (design only)

Combine approved training sources and partition before fitting. Never mix held-out
test images into optimization or codebook construction. Group source duplicates,
paired scenes, crops and corruptions into one split. Preserve a final untouched
test set separate from development cases used to discover weaknesses.

For each future modification record: original method and checkpoint; exact change
to input/output, preprocessing, labels, architecture, loss or training data; reason;
compatibility with pretrained weights; parameter/FLOPs/payload delta; fixed training
budget; measured gain/loss per content category; and remaining failure cases.
Compare data-only adaptation against architecture changes to separate their effects.
Keep diffusion models outside this phase under the current user instruction.

## Execution locations

- Source jobs: scripts/build_stage1_jobs.py and scripts/train_shared_native_classifier.py.
- Native Swin runner: scripts/run_native_swin_evaluation.py; separate factory
  src/semantic_lossless/swin_native.py (does not overwrite the retained adapter).
- Kaggle account A job: satyapaladugu/stage1-shared-codebook-classifier.
- Kaggle account B job: satyapaladuguplus/stage1-native-swin-evaluation.
- Remote artifact root: Results/stage1/<algorithm>/<run>/ inside the Kaggle export.
- Existing results and local paths: ALL_MODELS_RESULTS.md.

Submission alone does not establish actual GPU allocation, completed pilot or
training. Job statuses must be checked and outputs downloaded before reporting
results. The two account credentials remain separate; none belongs in Git.

## Continuation checkpoint

## 26 October 2026: native-contract comparison and WSL-only training

Current user authorization: new training uses account B (`satyapaladuguplus`)
through its isolated WSL credentials only, with DataParallel after a real
two-device pilot. No new MambaJSCC, NTSCC or diffusion training is queued.
Existing artifacts remain preserved. Global batch size, normalization scope,
codebook EMA and BatchNorm behavior must be checked: DataParallel alone does not
establish native mathematical equivalence.

The model target roster is DeepJSCC, ADJSCC, SwinJSCC, JCM, masked SQ-GAN,
Shared Semantic-Aware Codebook, DeepJSCC-Q, MDJCM, VQ-DeepISC and AMP-SC.
These are ten targets, not ten verified trainable implementations. AMP-SC's
author source is pinned at063d8098633cf31acd236982b8510d969d225d7c; its pretrained
feature-codec prerequisite remains unresolved. VQ-DeepISC still lacks a verified
author code/checkpoint release. DeepJSCC-Q quantizer fidelity and MDJCM's digital
configuration/staged training remain gates. No generic VQ model will be relabeled
as either paper. BPG/learned entropy coding plus LDPC/QAM are separate digital
reference systems and do not inflate the semantic-model count.

Budget estimate: setup/pilots for already executable methods may take2-4hours;
a first bounded multi-model comparison round may take12-24hours of elapsed
queue time, conditional on pilot throughput and available GPU quota. Neither
range includes guaranteed paper-level convergence or implementation of models
without a complete verified release. Measured initial ADJSCC throughput was
4181steps in7201.7seconds withB8 andtwoT4s; do not transfer that throughput to
GANs, transformer branches or staged digital training. Every run has an explicit
wall-time cap and actual completed update counts, not an assumed150epochs.

The user-approved JCM spatial variant now has a separate implementation at
src/semantic_lossless/jcm_spatial.py and training entry point
scripts/train_custom_jcm_spatial.py. CPU64x96B2 forward/backward passed, with
768complex uses/image and3072potentialamplitude-label bits. Reused native
encoder/decoder weights remain immutable. The new spatial probability head,
removed classification objective and per-image normalization are explicit
adaptations. Its sample-scaled finite-amplitude simulation does not implement
bit serialization/demapping or scale signalling, so raw label bits are not a
measured digital packet size. A separate dualGPU pilot gates bounded training.

Native encoder/decoder input dimensions, ranges, preprocessing, loss and channel
contract are preserved for each native branch. The explicitly approved JCM
larger-image reconstruction-only variant is separate from native32x32/CIFAR10
joint reconstruction/classification JCM; modifications to dimensions, classifier
head and objective must be recorded. These two variants are not two independent
paper methods.

Approved custom input preparation, without silently removing native inputs:

- SQ-GAN: use a frozen Cityscapes19-class segmenter for custom RGB images,
  preserve its native20-channel map (19 classes plus ignore), paired transforms,
  masked VQ branches and256x512 contract. Preserve native Cityscapes GT training
  separately. Pin model/processor revisions, record pseudo-label provenance and
  measure segmenter compute/latency separately and in the complete transmitter.
- Shared Codebook: use the frozen native10-class classifier to produce custom
  prototype-construction pseudo-labels; record confidence, class coverage and
  source-domain distribution. Keep true-label STL10 construction/training as
  the native branch. Pseudo-labels are not ground truth or a semantic-accuracy
  benchmark for unseen classes.
- RGB-only methods: use their own native crop/resize/range contract on the same
  frozen parent-image memberships. Do not force one256crop pipeline onto every
  method. Compare quality only at compatible reconstruction resolutions/tasks.

For each compatible model, evaluate two checkpoints (native-trained and
custom-trained) on two domains (native heldout and custom heldout). First retain
author/legacy checkpoint results as provenance-labeled references; for causal
data comparisons, train paired native-data/custom-data branches from the same
initialization with matched update budgets, optimizer, rate and channel. Save
actual images seen and wall time; equal epochs across different dataset sizes
are not equal compute. Keep budget-limited fine-tuning distinct from full paper
convergence. Do not train another native branch merely to recreate already
available weights without a defined comparison purpose.

The frozen first mixture has3800training images (800DIV2K,2000COCO,1000TextOCR),
400development images (100DIV2K,200COCO,100TextOCR), and324test-role images
(200COCO,100TextOCR,24Kodak). Counts are a bounded starting design, not an
optimized or representative population split. Kodak has already been inspected
in baseline experiments and is an external development-exposed benchmark,
not a pristine new final holdout. The other300test-role images remain reserved.
Report per-source metrics plus macro averages so the larger COCO subset cannot
hide text or high-quality-image failures. Only official training partitions feed
optimization. Parent-image memberships, hashes and dedup decisions stay fixed;
all derived crops and pseudo-labels inherit those memberships.

Recovered custom ADJSCC v1 is a completed two-hour adaptation experiment:
4181optimizer steps,9epochs entered with the last incomplete, actual DataParallel
work on both T4s. Paired400-image development PSNR26.80484474->26.81924662dB;
SSIM0.78881273->0.79078064. The changes are small development effects, not a
confirmed generalization improvement. This run used the explicitly documented
common ADJSCC adaptation pipeline; it is not a native-paper training reproduction.
Recovered: native Swin fixed/adaptive Kodak24 profiles, ADJSCC Kodak24 profiles, Shared auxiliary classifier200epochs, traditional11x24 source-codec results, legacyDeepJSCC diagnostic24. All imported into Results/metrics.csv and Docs/validation/stage1_summary.csv. Report: output/pdf/SemanticLossless_Stage1_Report.pdf (6pages).
Updated execution: JCM A v1, SQ-GAN B v10 and NTSCC A v1 completed and were recovered with hashes. SQ v9 failure remains preserved. Latest JCM and NTSCC measurements are imported by scripts/import_stage1_gpu_extension.py (repeat import appended zero rows); SQ v10 is imported separately. Shared native construction/prefix implementation is prepared with checkpoint and pilot gates; a launch does not establish training completion.

Custom-data extension authorized: DIV2K plus diverse RGB sources; freeze actual
image identities and splits before training. The initial design is
CUSTOM_DATASET_PROTOCOL.json. Custom ADJSCC adaptation uses separate modified
runs and pristine author initialization; it must not replace Phase1 native
checkpoints. Kodak stays evaluation-only. No dataset selection or successful
training is claimed by the protocol file alone.
