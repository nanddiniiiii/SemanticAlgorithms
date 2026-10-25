# Stage 1 experimental report

Snapshot: 2026-10-25T05:28:36+00:00
Source checkpoint: `b61f90649a2dca18a0ee488b3181d351752cf07c`

This report consolidates retained models, author checkpoints, executed evaluations and missing-weight training. It is a measured progress report, not a claim that every requested paper has been reproduced or that Stage 1 is complete.

Prepared 2026-10-25T05:28:36+00:00. Source checkpoint b61f90649a2dca18a0ee488b3181d351752cf07c. Generated receipts and the report may include uncommitted work after this checkpoint; each remote source bundle has its own hash in KAGGLE_RUNS.md.

## Decisions and evidence

Retain existing checkpoints unchanged. Prefer verified author weights to unnecessary retraining. Train absent weights only when the paper variant, implementation, dimensions, data split and channel are defensible. Stage 2 dataset adaptation and diffusion training remain outside this execution.

Two Kaggle accounts were used concurrently. Completed and recovered: the 200-epoch STL10 classifier prerequisite, two native SwinJSCC Kodak evaluations, ADJSCC C2 quality/profiles, JCM GPU profiles, NTSCC GPU quality/profiles and SQ-GAN metric extension. Shared codec construction/training and custom-data adaptation are separate next runs; current versions and status are recorded in KAGGLE_RUNS.md. A classifier alone does not establish a trained semantic codec.

Complete metric coverage is still pending. Missing values mean unmeasured or unsupported, never zero. No model in this report establishes guaranteed lossless one-shot delivery.

## Reproducible comparison contract

| Axis | Recorded contract |
| --- | --- |
| Image | RGB layout, tensor dimensions, native preprocessing, dataset and split; no training on Kodak |
| Channel | AWGN or native digital chain, SNR definition, noise seed/trials and normalization; ideal source codecs kept separate |
| Rate | Actual complex uses, CBR = k/(3HW), or payload bits and bpp; headers, side streams, pilots and protection distinguished |
| Quality | PSNR, SSIM, MS-SSIM, LPIPS and exact-pixel checks where available; semantic accuracy only with task evidence |
| Compute | Inference only; encoder, decoder and full scope; parameters, counted operators and omissions |
| Runtime | Actual device, precision, batch, dimensions, warmup/repeats, synchronized latency and memory scope |

## Measured quality results

| Model | Native workload | PSNR dB | SSIM | MS-SSIM | LPIPS |
| --- | --- | --- | --- | --- | --- |
| Swin fixed C32 | Kodak24; 10 dB AWGN | 29.3711 | 0.82334 | 0.95684 | 0.22124 |
| Swin SA+RA C32 | Kodak24; 10 dB AWGN | 21.3327 | 0.62473 | 0.78394 | 0.49045 |
| ADJSCC C2 | Kodak24; 10 dB AWGN; declared lineage | 27.6456 | 0.78966 | 0.94078 | 0.29997 |
| NTSCC q1 | Kodak24; 10 dB AWGN; incomplete side rate | 29.7902 | 0.82533 | 0.95717 | 0.22139 |
| JCM 16-QAM | CIFAR10 test10k; 12 dB | 23.3636 | 0.67811 | Unmeasured | Unmeasured |
| SQ-GAN | Cityscapes val500; ideal codec | 21.2408 | Unmeasured | 0.78283 | 0.27673 |

These are separate native evaluations, not a matched leaderboard. JCM SSIM preserves the author convention, including its unusual range setting; it cannot be pooled with standard RGB SSIM. SQ-GAN PSNR uses native saved-image postprocessing. Its conventional rounded-image PSNR is 21.2566 dB.

Measured semantic fidelity: JCM classification accuracy is 87.64% on the official 10,000-image CIFAR10 test split. SQ-GAN decoded semantic-branch mIoU is 0.740114 against supplied paired labels; this is not segmentation of reconstructed RGB or the paper InternImage metric. Kodak has no independent task annotations, so semantic metrics remain unmeasured.

Swin fixed-C32 reaches 29.3711 dB, while the adaptive SA+RA checkpoint reaches only 21.3327 dB at the same requested C32 and 10 dB. This large gap is an unresolved checkpoint/operating-point result, not proof that adaptation is generally worse. Strict state loading is necessary but does not certify expected published performance. Keep the adaptive result flagged until checkpoint configuration and rate/SNR sweeps explain it.

## Rate accounting and exactness

| Model | Main-stream budget | Limit |
| --- | --- | --- |
| Swin C32 / ADJSCC C2 | k = 24,576 complex uses at 512 x 768; CBR 1/48 = 0.0208333 | Transport, pilot and side overhead unspecified |
| NTSCC q1 | Mean CBR 0.02397565; range 0.02043321-0.03062439 | Rate-index/mask transport unmodeled; total rate uncertified |
| JCM | 128 complex uses / 32 x 32 RGB image; CBR 1/24 | Native 16-QAM; 12 dB; cannot compare directly with Kodak |
| SQ-GAN | 3,584 paper payload bits; 0.02734375 bpp; SQ01 packet 3,712 bits | No noisy channel/FEC; bit budget is not complex-use budget |

CBR is complex channel uses per RGB scalar sample in these JSCC records. It is not file compression ratio, bpp, bandwidth in hertz or a packet count. Converting digital bits to radio resources needs modulation order, code rate and overhead. Semantic similarity and perceptual quality do not establish exact pixels or exact files.

NTSCC correction: decoder hyperprior refinement is disabled in this retained configuration. The decoder is given spatial rate indices and a scatter mask without modeling their transport. With 16 choices and 32 x 48 Kodak latent positions, a fixed-width index representation costs 6,144 uncoded bits (768 bytes, 0.015625 bpp). This is not a minimum rate or a radio-use count. Encoder-derived receive rescaling also assumes an unmodeled available power value.

## Traditional source-codec reference

All 24 Kodak images are decoded as native RGB. These measurements include complete encoded file bytes (except headerless RAW), with no noisy channel, modulation or FEC. They establish a source-compression reference; they cannot be ranked against neural radio CBR without a common digital transmission chain.

| Codec setting | Mean bpp | RGB PSNR dB | SSIM | Exact images |
| --- | --- | --- | --- | --- |
| RAW_RGB | 24.0000 | Infinite (exact) | 1.00000 | 24/24 |
| PNG | 13.4311 | Infinite (exact) | 1.00000 | 24/24 |
| JPEG2000_lossless | 13.4546 | Infinite (exact) | 1.00000 | 24/24 |
| JPEG_q10 | 0.3266 | 26.6718 | 0.74491 | 0/24 |
| JPEG_q30 | 0.6598 | 30.4915 | 0.86076 | 0/24 |
| JPEG_q50 | 0.9055 | 32.1738 | 0.89572 | 0/24 |
| JPEG_q70 | 1.2388 | 33.9167 | 0.92274 | 0/24 |
| JPEG_q90 | 2.3463 | 38.0340 | 0.96194 | 0/24 |
| JPEG2000_ratio12 | 1.9984 | 35.2680 | 0.90465 | 0/24 |
| JPEG2000_ratio24 | 0.9989 | 31.6713 | 0.83280 | 0/24 |
| JPEG2000_ratio48 | 0.4991 | 28.9842 | 0.75636 | 0/24 |

PNG and reversible JPEG2000 recover every RGB sample in this test. They do not reproduce the original PNG container bytes. Lossy JPEG uses Pillow quality settings with 4:2:0 chroma subsampling; lossy JPEG2000 uses OpenJPEG rate targets and an irreversible transform. Actual encoded bytes, not requested targets, determine bpp.

On this Kodak set, the lossless file rates are about 13.4 bpp versus 24 bpp RAW, approximately 1.8 times smaller. This is an empirical result for these images and settings, not a universal compression limit. CPU encode/decode observations are saved, but one timing observation per image is not a stable latency distribution or a GPU comparison.

## Retained DeepJSCC diagnostic

The historical retained checkpoint gives 16.4290 dB PSNR, SSIM 0.23097 and MS-SSIM 0.47616 on Kodak24 using 128 x 128 tiles and its legacy real AWGN channel at 10 dB. It was trained at 200 dB. Each tile produces 19 x 29 x 29 = 15,979 real values; a full Kodak image produces 383,496. No explicit complex packing/radio protocol is implemented. This is a diagnostic of this checkpoint and port, not an estimate of the original DeepJSCC paper's performance. It remains outside the matched-radio comparison.

## Measured GPU inference profiles

| Component | M params | Partial GFLOPs | p50 ms | p95 ms | Extra MiB |
| --- | --- | --- | --- | --- | --- |
| Swin fixed C32 / encoder | 14.083 | 206.71 | 128.27 | 129.04 | 720.0 |
| Swin fixed C32 / decoder | 14.079 | 206.71 | 133.89 | 134.14 | 720.0 |
| Swin fixed C32 / full | 28.162 | 413.43 | 260.84 | 261.87 | 720.4 |
| Swin SA+RA / encoder | 19.005 | 217.09 | 136.07 | 138.53 | 720.0 |
| Swin SA+RA / decoder | 14.069 | 211.88 | 140.30 | 142.63 | 725.3 |
| Swin SA+RA / full | 33.073 | 428.98 | 274.57 | 276.50 | 729.4 |
| ADJSCC C2 / encoder | 5.288 | 277.04 | 103.96 | 104.76 | 288.3 |
| ADJSCC C2 / decoder | 10.647 | 103.97 | 521.84 | 533.22 | 1248.3 |
| ADJSCC C2 / full | 15.935 | 381.02 | 583.42 | 595.96 | 1248.8 |
| NTSCC q1 / full | 31.359 | 787.82 | 364.81 | 367.35 | 1353.4 |
| JCM32x32 / encoder | 9.028 | 0.63 | 2.44 | 2.62 | 7.1 |
| JCM32x32 / decoder | 5.341 | 0.45 | 2.40 | 2.56 | 2.1 |
| JCM32x32 / full | 14.369 | 1.08 | 4.85 | 6.52 | 7.1 |
| SQ-GAN / full | 186.271 | 2336.92 | 645.46 | 649.61 | 549.0 |

Swin profiles use one T4, FP32, batch 1, 512 x 768 RGB, three warmups and ten repeats. Two T4s were allocated; this native inference path uses CUDA0. Parameter counts are component-specific; whole-model memory remains resident for component profiles. Extra MiB is peak allocation above the warmed baseline, not deployment memory. LPIPS and MS-SSIM evaluation are outside timed forwards; Swin native full forward still includes its own distortion calculations.

Additional profile contracts: ADJSCC and NTSCC use 512 x 768, three warmups and twenty repeats. NTSCC full scope includes auxiliary NTC reconstruction and JSCC; repeats reuse seeded channel noise. JCM rows use 32 x 32, ten warmups and one hundred repeats, including both reconstruction and classification; batch32 profiles are separate. SQ-GAN uses paired RGB/semantic 256 x 512, three warmups and twenty repeats, with both branches included. These workload differences prevent a uniform latency ranking.

The profiler counts only supported operations and is not a total FLOPs certification. A separate registered-kernel counter adds coverage for transposed convolutions and emits observed uncounted operator types. Its reference tests check a 1,728-FLOP transposed convolution and a 48-FLOP linear layer. Never add the two counters together or compare incompatible scopes as speedups.

ADJSCC illustrates the coverage problem: the standard profiler reports 381.02 GFLOPs, while the registered-kernel counter reports 1666.58 GFLOPs for the same full forward. The latter includes transposed convolution; unregistered elementwise work remains omitted. Ranking this model against Swin using only the smaller profiler number would be misleading.

Earlier CPU profiles exist for DeepJSCC, NTSCC, ADJSCC, JCM and SQ-GAN. They are computational diagnostics, not T4 latency comparisons. Training FLOPs have not been measured. Runtime reliability here means observed completion and finite outputs, not a certified radio BLER or a large repeated-noise trial.

## Training status and scientific boundaries

Shared Codebook: native STL10 classifier prerequisite uses 5,000 labelled training images, 96 x 96 input, batch 64, FP32 and the source learning-rate schedule, capped at 200 epochs or 7,200 seconds. A two-step real-data CUDA pilot and strict save/reload gate precede training. The classifier must then feed audited codebook construction and codec training; these stages cannot be skipped.

Completed classifier evidence: 200 epochs, 15,800 optimizer steps, 1,000,000 training-image presentations in 971.6 seconds. Strict checkpoint reload passed. Final training accuracy is 100.00% and loss 0.000978; these are training-set metrics, not generalization or semantic-codec results.

DeepJSCC-Q remains blocked: the public candidate mixes paper variants and has verified quantizer evaluation and regularizer defects. No long training or faithful trained baseline is claimed.

## All requested methods: executable status

| Algorithm | Stage 1 category | Implementation boundary |
| --- | --- | --- |
| DeepJSCC | retain existing | Legacy channel; complex-use mapping not certified |
| SwinJSCC | retain existing | Retained port audit-only for power mismatch; separate native fixed/adaptive Kodak24 evaluations and T4 profiles completed; adaptive quality anomaly unresolved |
| NTSCC | retain existing | Original NTSCC; main-latent rate excludes side information |
| JCM | use downloaded author weights | Native 32x32; reconstruction plus classification objective |
| ADJSCC | use downloaded author weights | DiffCom PyTorch architecture; per-image AWGN adaptation; original TF lineage stays separate |
| Shared Semantic-Aware Codebook | code only candidate | SAC without proposed information-theoretic loss term; lossy reconstruction; construct classifier/codebook first |
| SQ-GAN | use downloaded author weights | Ten-part strict checkpoint inference; native trainer pending |
| VQ-DeepSC | unverified executable release | VQ-DeepSC: 2022 preprint, 2023 journal; already in survey; author code/checkpoints not verified; no local implementation |
| VQ-DeepISC | unverified executable release | VQ-DeepISC: 2025; already in survey; author code/checkpoints not verified; no local implementation |
| SC-LMKB | reference only non reconstruction | SC-LMKB: arXiv2604.05504, 2026; source/channel knowledge generation and hallucination mitigation; cross-modal retrieval reference, not a pixel-reconstruction baseline; code/weights unverified |
| MDJCM | author weights inaccessible | Checkpoint and native runtime pilot required |
| MambaJSCC | use downloaded author weights | Author AWGN/Rayleigh pairs downloaded; native CUDA extension build failed; inference unverified |
| DiffCom | excluded from current stage | Complete T4 posterior pilot passed; AWGN adapter differences remain explicit; no diffusion training |
| DiffJSCC | excluded from current stage | Author Hugging Face checkpoint; inference integration pending |
| DeepJSCC-Q | code only candidate | Candidate mixes paper variants; quantizer evaluation/regularizer defects verified; no faithful trained model yet |
| Our complementary hybrid | excluded from current stage | Our frozen ADJSCC plus fixed DCT complement reference; not an author reproduction |
| DiT-JSCC | excluded from current stage | JSCC weights reserved for future release |
| HDA-DeepSC | unverified executable release | Verified author executable release not identified |
| Masked VQ-VAE | unverified executable release | Released classification demo disables VQ/AWGN in inspected path |
| MOC-RVQ | unverified executable release | Checked author-associated repository has code/weights forthcoming; not executable |

## Failure recovery and controls

Earlier A/B jobs allocated T4 hardware but failed before model execution because their source archive omitted sources.lock.json. The corrected bundle is extracted and imported in a fresh process during preflight. Stage logs now retain start/end, elapsed time, traceback and source hash. Failed runs remain archived; they produce no quality or training claim.

The retained Swin port had a masked-power normalization error that made nominal and effective SNR disagree. Its quality results remain quarantined. Native author checkpoints are evaluated separately. NTSCC side-rate incompleteness, ADJSCC implementation lineage and SQ-GAN label dependence remain explicit limitations.

Retrieved files, not a running-cell display or /kaggle/working alone, establish saved evidence. Script-kernel logs, JSON/CSV files and reconstructions were recovered. Executed notebook/HTML exports were not exposed for the completed Swin run; this capture limitation is recorded rather than concealed.

## Completion gates and next decisions

Stage 1 closes per algorithm only after checkpoint provenance, native input/output contract, finite inference, quality/rate records, inference profiles and representative artifact recovery are checked, or after an explicit unavailable/blocked disposition. It is not honest to mark missing native implementations as completed models.

Remaining Phase1 work: resolve the adaptive Swin anomaly; progress Shared Codebook through construction, pilot and bounded native-branch training; validate Mamba CUDA compatibility; keep absent releases and DeepJSCC-Q fidelity blockers explicit. Custom-data adaptation is now authorized separately, with frozen train/development/test membership and untouched native checkpoints. Dataset or architecture changes must not be presented as native-paper reproduction.

## Evidence and reproduction map

Canonical plan: Docs/STAGE_ONE_BASELINES.md. Remote run identities, versions and retrieval: Docs/KAGGLE_RUNS.md. Machine-readable inventory: Docs/validation/stage1_results_inventory.json. Per-run events/checkpoints: Results/experiments/<algorithm>/<run-id>/. Canonical appended metric records: Results/metrics.csv. Author checkpoint sources: Docs/PRETRAINED_DOWNLOADS.md. All paths are relative to this repository.

Rebuild inventory: python scripts/build_stage1_results.py. Import a recovered run: python scripts/import_stage1_outputs.py --input <version-specific-folder>. Rebuild this report with the managed Python runtime: python scripts/build_stage1_report.py. Fresh remote runs require their own recorded source revision, bundle hash, inputs and actual hardware check.

