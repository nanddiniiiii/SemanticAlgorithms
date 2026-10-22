# Current implementation and execution status

Updated 22 October 2026. This document describes the new `src/semantic_lossless`
runtime. Historical notebooks and `SharedCode/reproduction` remain separate.

### Latest verified update

Paper fidelity now takes precedence over the generic 150-epoch plan. See
`REPRODUCTION_PROTOCOL.md` and `JCM_REPRODUCTION_AUDIT.md`. Existing generic
training examples are adapted experiments, not certified paper reproductions.

Kaggle dual-T4 pilot version 3 completed: ADJSCC and JCM each performed two
real-CIFAR optimizer steps, validation on a limited subset and strict checkpoint
reload; hooks observed work on both devices. ADJSCC also passed one synthetic
256x256 AMP step. JCM is float32 after an AMP overflow in version 2. These checks
establish neither convergence nor published quality. A native JCM baseline uses
single-GPU batch semantics rather than this DataParallel compatibility pilot.

Independent JCM/native CPU comparison passed with exactly equal reconstruction,
class logits, objective and all parameter gradients under identical RNG state.
The native pretrained evaluation completed on account A on all 10,000 CIFAR-10
test images: native PSNR 23.409674 dB and classification accuracy 0.8764.
Account B was reauthenticated in WSL Firefox; its private-dataset query and asset
uploads succeeded. Earlier GPU requests received no GPU hardware, including
with explicit accelerator selection and a working CUDA image. Submission retry
version 6 then allocated two real Tesla T4s and SQ-GAN proceeded with CUDA.
The provisioning failure's cause is unconfirmed. SQ-GAN version 8 completed all 500 validation images; see the update below.

Shared-codebook construction and codec have synthetic CPU checks, not native
dataset training results. The DCT complementary hybrid is our proposed reference,
not a reproduction of a named hybrid paper. DiffCom's prior strict load and a
32x32 native UNet forward passed; complete posterior reconstruction is pending.
MambaJSCC and DiffJSCC checkpoints are downloaded and hashed, not inference-tested.

DiffCom full posterior pilot version 4 subsequently passed on one real Kodak
center crop (256x256): 235 denoiser evaluations, 4096 complex channel uses plus
32 assumed-correct side bits, 284.10 seconds and 8,703,745,536 peak allocated GPU
bytes on a T4. This is a complete runtime check, not a matched paper benchmark.
Its lightweight AWGN operator and identity interleaver remain declared adaptations.
The native UNet gradient flags are required by its attention checkpoint backward;
there are no optimizer steps. The catalog now enables this inference backend.

Swin resolution-dependent checkpoint reload passed with exactly equal synthetic
128x128 reconstruction under identical RNG, then finite 256x256 inference. Only
regenerated attention-mask buffers are replaced; all learned tensors load strictly.

Original NTSCC quality-1 retained checkpoint completed native-resolution Kodak24
evaluation at 10 dB AWGN, one trial: 29.790670 dB mean RGB PSNR, 0.825372 mean
SSIM and 0.02397565 mean main-stream CBR. Side information is omitted, so this
is not a complete transmission-budget comparison. Windows entropy compatibility
and metric conventions remain declared adaptations. Small aggregate receipt:
`Docs/validation/ntscc_kodak24_metrics.json`.

SQ-GAN version 6 completed a 500-image adapted-preprocessing evaluation; the
subsequent native loader check exposed PIL/tensor nearest-label differences.
The loader was corrected and an independent boundary-pattern test now matches
native semantic maps exactly. Version 8 completed the native-preprocessing and saved-image evaluation.
The first result is preserved separately and is not a native paper baseline.

DeepJSCC-Q candidate quantizer checks found nonfinite fixed-constellation eval
and mutating learned-constellation eval state. See its reproduction audit before
enabling training. Mamba's adaptive CUDA extension compiled, but its standard extension failed
with native C++ source errors (an extra argument and trailing comma); neither its AWGN no-CSI nor Rayleigh CSI-ReST pair is yet runtime-certified.

The earlier table below is an implementation inventory; this update supersedes
its older statements about absent GPU checks and downloaded assets.

## What runs now

| Model | Actual implementation | Local verification | Training |
|---|---|---|---|
| DeepJSCC | Retained checkpoint-compatible CNN and legacy channel | Strict load and complete synthetic evaluation/export path | Disabled pending calibrated channel/packing decision |
| Original NTSCC | Retained original author model with Windows entropy compatibility | Prior strict-load/finite CPU forward receipt | Disabled; original, not NTSCC+ |
| SwinJSCC | Retained adaptive SA+RA encoder/decoder and per-image complex AWGN | CPU batch-two forward, backward, optimizer step | Fine-tuning into a new run enabled |
| JCM | Pinned author network, finite constellation, native classifier/reconstruction objective | Author checkpoint strict-load; CPU forward/backward/optimizer; runner pilot, save and strict reload | Native CIFAR-10 runner enabled |
| ADJSCC | DiffCom authors' released PyTorch attention/GDN implementation | Author C=2 checkpoint strict-load; CPU batch-two forward/backward/optimizer | Enabled; separate lineage from original TensorFlow implementation |
| SQ-GAN | Ten pinned author components; image and semantic branches; independent packet decoder | Ten strict loads; CPU native-forward comparison; source packet round trip | Inference only; GAN/three-stage trainer pending |

All checks above used synthetic inputs unless stated otherwise. They are runtime
checks, not published/reproduced quality results. No T4 certification or full
150-epoch training is claimed by these receipts. Tests and measurements are under
`Results/implementation_checks/`, excluded from the canonical experiment CSV.

Other registered algorithms deliberately fail or skip rather than substitute a
small unrelated model: Shared Semantic-Aware Codebook, MDJCM, MambaJSCC, DiffCom,
DiffJSCC, DeepJSCC-Q, the complementary hybrid prototype, DiT-JSCC, HDA-DeepSC,
and Masked VQ-VAE. Pinned source retrieval exists for eight new author codebases.
Source availability does not mean their dependencies/checkpoints/pilots passed.

## Small structure

- `semcom.py`: root launcher; use the installed console command after editable install.
- `src/semantic_lossless/`: configs, data, model interfaces, training, metrics,
  events, source retrieval, source packet serialization, CLI.
- `src/semantic_lossless/algorithms/<id>/train.py`: per-model training entry points;
  capability checks remain authoritative for unsupported trainers.
- `configs/current/`: small JSON examples. Replace uppercase dataset placeholders.
- `Notebooks/Current/account_a.ipynb`, `account_b.ipynb`: thin three-cell-per-model
  launchers, not duplicated architectures. Configure model paths in the setup cell.
- `Kaggle/account_a`, `account_b`: private job metadata requesting T4 x2.
- `Models/pretrained_downloads/`: downloaded immutable artifacts; never Git blobs.
- `Results/experiments/<algorithm>/<UTC-run-id>/`: isolated results and checkpoints.

The Kaggle launchers require a checkout containing this new code and access to
its required assets. The old remote HEAD does not magically contain uncommitted
files. They have been generated and their code cells syntax-checked; they have
not yet been run end-to-end on Kaggle.

## Commands

```text
python semcom.py list
python semcom.py prepare --config configs/current/jcm.json --fetch
python semcom.py pilot --config configs/current/jcm.json
python semcom.py train --config configs/current/jcm.json
python semcom.py evaluate --config path/to/evaluation_config.json
python semcom.py profile --config path/to/evaluation_config.json
python semcom.py export --run-dir Results/experiments/jcm/RUN_ID
python semcom.py collect
```

Use `pilot --inference-only` to retain an existing model without a training pilot.
After training, evaluate `best.pt` explicitly, with `from_scratch=false`. The
notebooks construct that evaluation config. Passing a training config unchanged
to evaluate can otherwise load its original pretrained initialization.

`pip install -e .` installs the small launcher dependencies. Framework environments
are explicit: the retained Swin manifest currently requires PyTorch 2.10.x and
timm 0.4.12; add the matching CUDA wheel on Kaggle. SQ-GAN inference additionally
uses PyYAML and einops. SSIM and LPIPS dependencies are optional, and absent metrics
are null rather than made-up values. Inference currently runs float32; AMP is a
training option and must pass a GPU pilot.

## Data and comparison protocol

JCM: official CIFAR-10 training set split 45,000/5,000 by a recorded seed; official
10,000 test images remain test-only. This validation protocol is an adaptation
from the native trainer and must be reported. It is not DIV2K training.

Image folders: train/validation/test are explicit paths. Training uses random
crops without hidden resize; evaluation uses native dimensions, with model padding
recorded/charged. Byte-identical images across folder splits are rejected.

SQ-GAN: paired Cityscapes `leftImg8bit/{train,val}` and `gtFine/{train,val}` with
0..18/255 trainIds. The loader explicitly resizes RGB to 256x512 with bilinear
interpolation and labels with nearest-neighbor. A training-only 90/10 holdout is
used for development; official val is the final evaluation split. Raw labelIds
must be converted before use; filenames alone do not establish correct labels.
This adapter does not reproduce the author's training augmentation pipeline.

ADJSCC: architecture comes from the pinned DiffCom release. The original TF
ADJSCC and old teammate artifacts remain different variants. The transport
adapter uses per-image normalization, whereas the DiffCom channel implementation
normalizes over a batch. At batch one these normalization scopes agree. The
shared runner's scheduler is an explicit training adaptation. Any measured
comparison must identify this implementation and protocol.

## Rates, exactness, and source packets

CBR is complex latent uses divided by original RGB scalar count, `k/(3HW)`.
Padding uses belong in k. Frame headers, pilots, CSI, protection, and side streams
are not implemented by a continuous latent alone. Every current codec therefore
reports `rate_complete=false`.

DeepJSCC's imported legacy latent packing is not certified: its complex uses/CBR
are null. NTSCC reports main-latent uses only. Swin and JCM report their actual
main-stream complex counts. Missing costs never become zero overhead.

SQ-GAN's experimental `SQ01` packet contains two 512-position bitmaps, 10-bit
indices for each 1024-entry codebook, image dimensions/counts, and CRC32. Positions
are serialized in ascending order with their corresponding indices. The decoder
receives only these bytes and the pre-shared model; it does not receive the
original image or original semantic map. This is our packet format, not a claim
to reproduce the paper's entropy coder. The fixed synchronized checkpoint is
pre-agreed, not transmitted anew with every image. A clean packet is a source
coding test; no AWGN, modulation or channel code is attached yet. CRC detects
corruption and cannot correct errors or guarantee delivery.

A synthetic 256x512 check at retained fractions 0.25/0.25 produced 464 source bytes.
Against the pinned author's `forward_all`, the maximum float difference was
1.6391277313232422e-6. This validates index serialization/lookup, not reconstruction
quality, dataset compression performance, losslessness, or wireless reliability.

Metrics separate float reconstruction from rounded uint8 pixel equality. Pixel
identity after RGB conversion is not byte-for-byte identity of the input file
(metadata, alpha, and higher bit depths are outside this RGB pipeline).

## Logging and limits

Events print and flush after each stage/optimizer step/image trial. Each event has
an ID; one controller appends deduplicated long-format rows to `Results/metrics.csv`.
The collector uses a lock so two account downloads cannot write concurrently.
Checkpoints, environment/model metadata, config and artifact hashes remain local
to each run. Unsupported profiler operations are explicit: inference FLOPs are a
partial supported-operator count, not a complete FLOP certificate. No training
FLOPs are collected.

Training uses Adam, optional AMP, accumulation, explicit dual-GPU DataParallel,
finite checks, validation and new best/last checkpoints. Time/step limits report
`budget_stopped`, with actual epoch coverage and validation count. A partial
validation cannot replace the full-validation best checkpoint. Checkpoint restart
from saved weights is supported; automatic optimizer/mid-epoch resume is not yet
implemented and must not be claimed. OOM/nonfinite failures preserve the failed
run status; no automatic batch-size change silently alters the experiment.

Kaggle notebook stdout updates per cell. Saved run artifacts are created on disk
as models finish; downloading partial notebook output while a job is still running
is a separate controller/platform capability, not implemented by these notebooks.

## Twenty-hour execution order

1. Establish local runtime checks and native checkpoint loading; generate launchers.
2. Run a short actual T4 x2 pilot before committing long GPU slots. Verify AMP,
   per-GPU memory, both-device work, optimizer step, strict save/reload and export.
3. Start ADJSCC and JCM training on separate accounts after real-data pilots.
4. Evaluate retained DeepJSCC/NTSCC/Swin and SQ-GAN weights while those jobs train.
5. Integrate shared-codebook construction/classifier and SQ-GAN native fine-tuning;
   reserve CUDA-extension/legacy-dependency work for measured compatibility gates.
6. Spend remaining time on matched comparisons and one defensible ablation.

Each training run has a four-hour cap, not a promise of 150 completed epochs.
Unavailable releases and slow native pipelines are recorded as gaps rather than
claiming all registered methods have been trained in a fixed wall-clock window.

## Latest completed verification

SQ-GAN version 8 passed a real-image native preprocessing check and evaluated
all 500 Cityscapes validation images through the native forward function.
At branch fractions 0.25/0.25, conventional rounded RGB PSNR is 21.256595 dB;
native saved-image fixed-range PSNR is 21.240841 dB. The auto-range formula
yields 21.001077 dB; the original unpinned torchmetrics version is not certified.
The codec has no radio channel, so these are not AWGN results. LPIPS, MS-SSIM
and segmentation mIoU remain unmeasured. Per-image results were collected locally.

The retained Swin adapter is quarantined after its active-symbol power audit.
Both downloaded native base checkpoints (fixed C32 and SA+RA) strictly load
against the pinned authors' configuration with 28,162,000 and 33,073,488
parameters respectively. This verifies architecture compatibility; native CUDA
forward and quality evaluation remain pending.
