# Author-code reproduction correction — 20 October 2026

## Current claim

No existing run has been certified as both an exact author experiment and a
matched all-model comparison. Existing training/checkpoints/results are retained
as historical experiments. Their measured values must not be relabeled as new
author-code results. This correction supersedes earlier blanket "official" or
"completed comparison" descriptions.

The old SQ-GAN stage coordinator has a STOP guard. It must not launch the analog
adaptation automatically under the new author-fidelity requirement. No running
Kaggle job or checkpoint was deleted. The published adaptation's AWGN semantic
stage reached 150 epochs / 120000 optimizer steps; this is useful historical work,
but not certification of the original SQ-GAN communication system.

## Evidence and required correction

| Model | Release evidence | Existing deviation / next requirement |
|---|---|---|
| DeepJSCC | [Paper](https://arxiv.org/abs/1809.01733); [used third-party source](https://github.com/chunbaobao/Deep-JSCC-PyTorch) | The used repository explicitly calls itself a reproduction. Original-author code for the 2019 method has not been verified. Do not substitute the authors' distinct DeepJSCC-f feedback model or certify this source as official. |
| ADJSCC | [Author source](https://github.com/alexxu1988/ADJSCC) | Core attention modules were reused. DIV2K loader, fixed-SNR training, variable image inputs, high-resolution fading path and runtime were adapted. The original ImageNet entry point hardcodes AWGN even though it accepts a channel flag. A channel flag alone is not evidence of high-resolution Rayleigh training. |
| NTSCC (2022) | [Author source](https://github.com/wsxtyrdd/NTSCC_JSAC22) | Separate model from NTSCC+. The native script requires an initialization checkpoint and hardcodes 5000 epochs in its training branch, regardless of its parsed epochs flag. Original source channel only implements AWGN/noiseless. No new 150-epoch DIV2K run has been certified here. |
| NTSCC+ | [Author source](https://github.com/wsxtyrdd/NTSCC_plus) | Prior run samples 128/800 images per epoch, uses AMP/new-NATTEN constructor shim and an added Rayleigh channel. Use the released loss and optimizer steps and original compatible NATTEN environment, with 800 images per benchmark epoch. The README warns that newer NATTEN can give incorrect results. |
| NTSCC++ | Same author's README | Online NTSCC++ inference/editing is explicitly TODO. NTSCC+ must not be relabeled NTSCC++. Full exact implementation remains blocked by unreleased code. |
| SwinJSCC | [Author source](https://github.com/semcomm/SwinJSCC) | Saved encoder/decoder widths, depths and heads match the official base w/o-SA-and-RA variant. That does not make it the full adaptive variant. Old experiment adds augmentation, AMP and custom orchestration. Author entry point also unconditionally loads a checkpoint; initialization must be explicit and checked. |
| SQ-GAN | [Author source](https://github.com/frapez1/SQ-GAN); [paper](https://arxiv.org/abs/2502.09520) | Official codec has Cityscapes semantics, native transforms, three training stages and discrete index communication. Our pseudo-labeled DIV2K + analog VQ-vector channel changes the system. The 150-epoch adaptation checkpoint cannot simply become a paper-exact checkpoint. Native reproduction requires Cityscapes inputs/environment; a DIV2K study must identify its dataset adapter and use a documented external digital transport, not the old analog wrapper. |
| DeepJSCC-Q | [Paper](https://arxiv.org/abs/2206.08100); [new candidate](https://github.com/r0ymanesco/deepjscc_stream) | Old ConvJSCCQ notebook is paper-based, not verified author code. Candidate has a dedicated trainer, FeatureEncoderAF/FeatureDecoderAF and learned-modem annealing, unlike that simplified notebook. Candidate authorship and fidelity to the journal variant still need primary-source confirmation. Earlier "no implementation found" was an incomplete search. |
| DiT-JSCC | [Author source](https://github.com/semcomm/DiTJSCC) | Public release is inference-oriented and explicitly omits training code/private workspace. Do not invent a small replacement and call it the paper model. Author weights and missing training pipeline are required. |
| DDM-SSCC | [Paper](https://arxiv.org/abs/2606.06273) | No author training repository was verified in the current search. It is lossless SSCC. Exact-recovery probability and serialized source/channel bits are required; an arbitrary diffusion codec is not an exact reproduction. |

The new snapshots in `Models/author_releases/` include loader source files that
were absent from some older copied folders. Per-file hashes and immutable commit
identifiers are recorded in `Results/reproduction_audit/source_lock.json` and
adjacent source manifests. The byte audit proves snapshot integrity, not runtime
or numerical equivalence. New snapshots were not imported or GPU-tested.

## Reproduction and comparison requirements

1. Native-paper reproduction uses each paper's actual model variant, data,
   normalization, training loss, schedule, pretrained initialization and channel
   assumptions. Different native paper datasets cannot become one controlled
   DIV2K experiment merely by putting scores in a table.
2. The common DIV2K/Kodak experiment preserves released model/loss code and
   explicitly records only necessary dataset/transport/orchestration adapters.
   It is identified as a matched benchmark, not as identical original paper
   experiments. No unavailable model is replaced silently.
3. Every benchmark epoch exposes all 800 distinct training images exactly once;
   validation uses the separate 100 DIV2K validation images. Kodak is final test
   only. Per-stage epochs, image exposures, crop geometry, processed pixels,
   optimizer updates and initialization must be recorded. Equal epochs alone do
   not establish equal optimization or pixel budgets.
4. Proposed common geometry is 256x512 training/evaluation tiles, to preserve the
   released SQ-GAN ratio and token geometry. Original Kodak images are covered
   fully using non-overlapping edge-padded tiles. Reconstruction is stitched,
   padding removed, then PSNR/SSIM/MS-SSIM computed against original RGB. This is
   a separate evaluation protocol, not claimed to be the author default. Each
   model needs a validated callable before any new score is generated.
5. All tiled computation and transmitted symbols, including padded regions,
   both SQ-GAN streams and transmitted control/FEC information, are counted.
   CBR = actual complex channel uses / (3 * original H * original W).
   Analog floats are not serialized bits. Unknown rates are not fabricated.
6. Channel name and SNR alone are insufficient: the Swin release uses independent
   Rayleigh amplitudes over symbols, whereas our NTSCC extension used a single
   complex coefficient per image plus equalization. Author-native results keep
   these definitions visible. A matched fading comparison needs an explicit
   common physical transport and CSI contract and is an extension wherever the
   released method lacks that transport.
7. Use the shared metric implementation, 10 documented channel trials per test
   image, full image coverage, the same image manifest, SNR grid and measured
   rate budgets. Report uncertainty and quality-versus-actual-CBR curves. Do not
   rank differing rates as though they were identical operating points.
8. Forward FLOPs use all tiles required for each original Kodak image. Training
   FLOPs include forward and backward for every optimizer's actual training
   path; optimizer-update arithmetic is labeled separately. Multiply-add = two
   FLOPs. A profiler that omits fused attention/backward kernels must report
   partial supported-operator FLOPs, never a complete cross-model FLOP ranking.
9. Preserve checkpoints. Resume is exact only if architecture, preprocessing,
   transport, optimizer/scheduler states and training contract match. Otherwise
   record an explicit fine-tuning experiment or start a distinct run; old epochs
   must not be counted under a newly changed training contract.

## Implemented tools and actual execution status

- `source_lock.py`: downloads complete pinned upstream archives and file hashes.
- `audit_existing.py`: compares copied source with upstream; notes missing files
  and distinguishes text changes from line-ending changes.
- `launch_upstream.py`: prepares an isolated, verified source copy and executes
  only released entry points on explicit request. It removes the old adaptation
  PYTHONPATH. Native data/dependencies/weights remain real prerequisites.
- `matched_evaluation.py`: full-image coverage, pad/tile/stitch, uniform metrics,
  communication accounting, dataset manifests/leakage checks and comparison
  contract rejection. Model callbacks have not been validated on GPU.

No GPU training was launched by this correction. Last confirmed Kaggle resource
error is the account's 30-hour weekly GPU limit. Complete source availability,
native data/weights and runtime checks are separate blockers for individual
models. No claim of all-model completion or architecture equivalence is made.
