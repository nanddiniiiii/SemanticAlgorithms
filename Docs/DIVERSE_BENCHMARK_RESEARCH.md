# A diverse benchmark for finding image semantic-communication failures

Research date: 22 October 2026. Status: literature-backed design, not a completed
image dataset or a new experimental result. No training was launched for this
design. Existing model implementations, checkpoints and baselines remain intact.

## Recommendation

Build a versioned benchmark collection with a representative natural-image set,
separate content-specific stress sets, and a separately generated corruption
extension. Keep training data independent. Start by evaluating frozen models;
otherwise it becomes difficult to tell whether a failure comes from the model,
the data, the channel, the input transformation or a software defect.

Use LSDIR as the principal additional high-quality training-data candidate.
For testing, prioritize SCLIC, TextOCR, COCO, SCID, LIU4K-v2 and ACDC alongside a
pinned CLIC release. Retain Kodak as a small historical reference, not the sole
evidence of generalization. DOTA and broader Open Images subsets are valuable
extensions after the first benchmark is functioning.

The immediate question is: at the same declared rate, channel and computational
budget, which content is lost, and does the model know that it has lost it?
Do not assert that all existing models fail before measuring them. Do not select
a test set only by searching for images where our future model wins.

## What was reviewed

The local 50-paper review and research survey were used to recover the project's
source/pixel/perceptual/task distinctions and rate-accounting requirements.
New primary-source checks focused on compression robustness, localized artifacts,
semantic miscompression and the original dataset releases. A literature survey
is a map of claims; local source audits and experiment receipts remain the
authority for what our implementations actually do.

Four especially relevant publications/resources:

| Work | Evidence relevant to this benchmark | Consequence for our design |
|---|---|---|
| Lieberman et al., NeurIPS 2023, Neural Image Compression: Generalization, Robustness, and Spectral Biases | Introduces Kodak-C/CLIC-C with 15 corruptions and spectral inspection of compression error | Separate natural distribution shifts from channel noise; inspect spatial frequencies, not only average PSNR |
| Hofer and Böhme, ACM Multimedia 2025, Challenging Cases of Neural Image Compression | SCLIC supplies human-annotated semantic miscompressions and 1,563 source images associated with six codecs/two quality settings | Re-encode source images using our own models; old reconstruction annotations are evidence about those old codecs, not ground truth about new outputs |
| Tsereh et al., JPEG AI Image Compression Visual Artifacts, 2024 preprint and updated 2026 resource | Localizes texture/boundary, color and text artifacts; compares neural and traditional reconstructions at matched bitrates | Evaluate local regions; an attractive whole image can conceal serious local damage |
| Dogaroglu et al., 2024, Adapting Learned Image Codecs to Screen Content via Adjustable Transformations | Reports screen-content adaptation using small additional transforms | Screen content is a relevant stress domain; dataset mixtures/adaptation alone are established ideas, so novelty needs a stronger measured contribution |

These are compression studies, not direct experiments proving our JSCC models
have identical weaknesses. Transfer of their findings to wireless reconstruction
is a research hypothesis to test.

## Dataset candidates and their jobs

| Dataset / release | Primary-source facts | Recommended role | Hypothesis to test, not a measured finding here |
|---|---|---|---|
| LSDIR | Authors describe 84,991 high-quality HR training images with diverse content; academic-research terms; release availability differs between announced and downloadable splits | Expand training beyond DIV2K using a deterministic, capped subset; use HR originals, not supplied SR downscales | Whether broader content improves held-out reconstruction and reduces rare-feature/codebook errors |
| CLIC 2024 image validation | 30 PNG images with long edge resized to 2048 | Small general compression control set, pinned explicitly to 2024 validation | Broad natural-image rate/quality performance |
| SCLIC v1.0.0 | Public source-image archive and codec-specific semantic-change annotations; CC BY 4.0 | Dedicated semantic-failure stress set | Whether plausible outputs change text, object identity, counts or other annotated content |
| TextOCR | 28,134 natural images and 903,069 word annotations | Text-rich evaluation using an accessible, annotated official split | Whether small words/numbers survive even when global similarity stays high |
| COCO 2017 | Detection/segmentation/caption annotations and 80 object categories | Multi-object/counting/small-object evaluation, preferably validation | Whether small or secondary objects disappear at low rates |
| SCID | 40 clean reference screen images at 1280x720, with separate distorted versions; includes text, graphics and mixed content | Use the 40 clean references as a screen-content stress set | Whether sharp edges, small fonts and flat colors expose photo-trained codec weaknesses |
| LIU4K-v2 | 1,600 training and 400 validation images; at least 3K, mainly 4K–6K; primary page states CC0 origins | High-resolution detail, geometry, memory and scaling tests | Whether texture loss, tiling seams and computational growth become limiting |
| ACDC | 4,006 adverse-condition images spanning fog/night/rain/snow, annotations in 19 Cityscapes classes, corresponding normal scenes | Natural lighting/weather shift and especially SQ-GAN-compatible semantic tests | Whether low visibility and domain shift cause semantic deletion or wrong generation |
| DOTA v1.0, optional | High-resolution aerial imagery with oriented object boxes; academic-use terms | Small/dense-object and unfamiliar-domain extension | Whether tiny distant objects survive under fixed resource budgets |
| Open Images V7, optional | Rich labels, boxes, relationships and instance masks across broad content | Additional stratified diversity once the compact suite works | Whether gains generalize beyond curated hard cases |
| JPEG AI dataset, later option | Official page lists 5,264/350 train/validation PNGs, 256x256 to 8K; access instructions involve an SFTP password request | Codec-oriented alternative if access is already available | Broad resolution/content robustness; not a dependency for immediate work |

LSDIR's site describes 1,000-image validation/test sets while also documenting a
250-image validation download and unreleased data. Verify actual files before
claiming a particular split is available. LIU4K licensing claims also differ in
third-party summaries: use the original release terms and preserve their version.
Downloading a Kaggle mirror does not change the original split or license.

Natural-image task datasets may contain already JPEG-compressed sources. Record
source format/history; converting a JPEG into PNG does not undo its artifacts.
The benchmark's exact target is the declared decoded source raster, not an
imagined uncompressed camera original. Use available clean reference originals,
not existing distorted reconstructions, as source inputs.

## Start with a proposed 270-image pilot

This is a requested sampling design, not a populated manifest. Actual IDs and
hashes must be selected before evaluation. The target is small enough to debug
but broad enough to reveal different types of failure.

| Stratum | Target count | Selection made without looking at our model outputs |
|---|---:|---|
| CLIC2024 natural-image control | 30 | Entire official image validation set |
| SCLIC semantic-risk cases | 40 | Fixed-seed spread of ROI size, original-codec family and content; unique sources outside DIV2K and selected CLIC controls |
| TextOCR | 40 | Label/geometry-based spread of text sizes, counts and word/number content |
| COCO | 40 | Spread of object sizes, categories and scene crowding from validation |
| SCID clean screen references | 40 | All clean references; exclude distorted derivatives |
| LIU4K-v2 | 40 | Fixed selection from validation covering resolution/texture/aspect ratio |
| ACDC | 40 | Ten per adverse condition, from accessible annotated held-out material |
| Total | 270 | Dataset/stratum identity stays visible in every result |

First execute a 35-image software/metric pilot, five from each stratum. It is
only a readiness check. Expand to 270 after all adapters and metric paths pass.
Use all Kodak24 separately, and the existing native datasets for paper checks.
These pilot counts cannot establish rare-event reliability or a large-scale
semantic failure rate. Expand the held-out sample when the pipeline is stable.

Retain an ordinary representative evaluation set alongside hard cases. SCLIC
and artifact collections were selected around particular codec errors, so their
failure frequency is not an estimate for everyday photographs. Report each
stratum separately, plus a clearly defined macro-average where appropriate.

## Prevent leakage and preserve the baseline comparison

Maintain three distinct tracks:

1. Native reproduction: original datasets, transforms, losses and channel/rate
   definitions. Existing JCM CIFAR results and SQ-GAN Cityscapes results belong here.
2. Frozen-model cross-dataset evaluation: keep checkpoints fixed and run the
   same declared inputs. This measures checkpoint transfer, not architecture alone.
3. Controlled adaptation: retrain/fine-tune eligible models on the same training
   sources, then compare on an untouched held-out set. Match updates/data exposure
   and report both training compute and inference compute.

Do not train on the hard test cases after finding them and still call the same
set an unseen test. Separate development and final holdout by original source
image and, when possible, photographer/scene. Group all crops, corruptions and
normal/adverse paired captures with their parent split. Deduplicate exact pixels
and near duplicates across all source collections before locking the manifest.
Record known pretraining overlaps; absence of visible overlap is not proof that
a pretrained model has never seen the scene.

The downloaded SCLIC metadata exposes an immediate overlap hazard: 603 unique
source filenames originate in DIV2K-train, 62 in DIV2K-validation, and 27 in
CLIC2024-validation. Thus, using all SCLIC sources as an independent unseen test
after training on DIV2K would be incorrect. Its RAISE_PNG-origin subset has 382
source filenames and is a useful candidate for the 40-image stress selection,
subject to pixel/near-duplicate checks and known checkpoint-training history.
This overlap does not make SCLIC unusable; it changes which claims its subsets
can support. Its CSV contains ROI boxes and annotator identifiers, not labels
classifying every error as text/count/identity. Such categories need an additional
annotation step if we want to report them. See the local metadata audit.

Candidate future training pool: DIV2K-train plus a fixed 5,000-image LSDIR training
subset, selected before training and deduplicated against tests. This is a
proposed manageable starting budget, not a literature-standard optimal size.
If adding task-domain training data, use only that dataset's training split and
keep the evaluation stratum independent. Keep Kodak evaluation-only.

## Some models cannot share one unrestricted leaderboard

JCM's verified release is a 32x32 CIFAR joint reconstruction/classification
system. Shrinking scene-text or 4K photographs to 32x32 would remove the very
information we intend to measure. Keep it in a native small-image track unless
we explicitly implement and train a changed architecture.

SQ-GAN takes paired Cityscapes-vocabulary labels at its encoder. ACDC is an
especially useful candidate because its class vocabulary matches. Verify ID
conversion/ignore masks and alignment rather than assuming the files plug in.
On general photos, frozen-segmenter labels would be an explicit adaptation with
additional encoder compute. COCO masks are not interchangeable with Cityscapes
labels. Human-provided label input is oracle assistance and must be identified;
do not compare it silently against an RGB-only encoder.

Fixed-size diffusion models need their own declared crop/resize/tile protocol.
Test images unavailable at a model's supported resolution should be marked
unsupported or adapted, not silently squeezed into a favorable result.

## Two spatial protocols

Use an input manifest that fixes identical decoded RGB targets and transformations
for each compatible comparison group.

- Controlled-patch protocol: common deterministic patches at a supported size;
  annotation geometry transformed exactly. Compare each reconstruction against
  the same patch. Exclude or flag partially cropped OCR/object annotations.
- Full-image protocol: original resolution with declared padding, or a pinned
  tile size/overlap/blending scheme where needed. Count every padded/overlapping
  symbol, side field and tile header. Report seams, total image latency and memory.

A 256x256 patch score does not certify 4K performance. Never downsample all
images and then claim that the system preserved small text in the originals.
Use a separate fixed-pixel-budget evaluation if studying absolute one-burst
constraints rather than a rate proportional to image area.

## Failure hypotheses and measurements

| Content | Candidate mechanism | Required measurements |
|---|---|---|
| Fine text and numbers | Lossy latent/VQ replacement or generative alteration of local structure | OCR character/word error rate against labels; exact word/number match; local pixel error; original OCR baseline |
| Tiny/dense objects | Background-dominated distortion objective or masking removes low-area content | Object detection AP/recall at fixed evaluator; small-object subset; count error and missed-object examples |
| Fine texture and repeated geometry | Rate-limited bottleneck loses high-frequency structure | ROI PSNR, frequency-binned residual energy, perceptual distance, geometry crops |
| Screens and diagrams | Photo training distribution poorly matches sharp edges and flat colors | Edge/line continuity, OCR, color error and local reconstruction inspection |
| Night/fog/rain/snow | Distribution/visibility shift changes feature and entropy statistics | Per-condition pixel/perceptual metrics; supported-label mIoU; invalid-region handling |
| Large images | Attention/feature memory, padding and tiling cost | Encoder/decoder/whole-image FLOPs, latency, peak memory, OOM and unsupported cases |
| Unfamiliar visual concepts | Codebook/prior coverage insufficient | Code-use distribution, quantization distance, reconstruction/semantic errors, manually checked instances |

Quantization distance or codebook perplexity is a diagnostic, not an automatic
measure of semantic truth. A nearest codeword always exists; distance from the
codebook alone does not prove an image is unrepresentable. Global CLIP-like
similarity must not replace text, count or small-object checks. Tiny-sample FID
is not a suitable principal metric for this pilot.

Frozen OCR/detection/segmentation networks are imperfect measuring instruments.
Report their original-image scores and their reconstruction scores, together
with paired changes. Use real labels where present and manual review of sampled
errors. Pseudo-label consistency is not human-ground-truth semantic accuracy.

## Common resource and quality protocol

Record PSNR/SSIM with fixed RGB/range conventions, MS-SSIM and LPIPS where
implemented, plus the annotation-specific metrics above. For near-lossless
analysis add maximum absolute sample error, fraction of exact samples, and
whole-image pixel equality. A perceptually strong result is not exact recovery.

For continuous JSCC, report actual complex uses k, source dimensions and
CBR=k/(3HW); include a second denominator for padded input when relevant.
For discrete codecs, report actual encoded bits including required side streams.
Source-codec bpp and continuous channel-use CBR are different quantities.

Compare AWGN models initially at 0,5,10,15,20 dB with fixed trial seeds and at
supported resource operating points. Compare rate/quality curves, not one model's
best unrelated setting against another's. Do not extrapolate outside overlapping
operating ranges. Matched full radio comparisons of digital codecs require a
declared FEC/modulation/framing chain; a source-codec result must remain in its
own table until that chain exists. Record digital failures/erasures, rather than
silently deleting them before computing image quality.

Use JPEG, JPEG2000, JPEG XL (lossy and lossless), PNG and raw RGB references as
available. State codec library/version, quality/effort, subsampling and metadata.
PNG/raw are lossless resource anchors, not equivalent low-bitrate competitors.
Decoded source pixels are the equality target; exact original-file preservation
is a separate experiment. File size ratios do not establish radio resource savings.

For robustness extensions, keep source corruption distinct from channel noise.
When a noisy/blurred image is the source, reconstruction should be compared to
that same source for compression fidelity. Comparing to its clean parent instead
asks a joint restoration question. Keep both targets explicitly separate if used.

## Inference FLOPs and performance: required for every eligible model

Report encoder, decoder and complete receiver pipeline separately. Include any
segmentation used at the encoder, entropy/hyperprior modules, maskers/demaskers,
residual path and all repeated generative operations. A neural-only count must
be labelled as such when entropy coding, CPU work or unsupported kernels remain.

Adopt a declared convention: one multiply-accumulate is two FLOPs. As a sanity
check, a convolution has approximately 2*Hout*Wout*Cout*(Cin/groups)*Kh*Kw FLOPs,
excluding bias and other operations. Transformer counts include attention matrix
products and projections; custom scans and non-matmul operations require separate
coverage checks. Padding, tile overlap, batch size, rate, steps and resolution
are part of the reported operating point.

Report profiler-counted FLOPs and unsupported operations explicitly. An incomplete
profiler count must not be labelled total model FLOPs. Provide analytic or
validated custom counts where possible; otherwise report the gap as unmeasured.
GFLOPs per image and per megapixel should both be shown. For iterative diffusion,
count every denoiser call and required gradient computation; multiplying a UNet
forward count by step count can miss posterior input-gradient cost.

Measure batch-one latency on the same T4/precision/environment with CUDA
synchronization, warmup and repeated trials. Separate GPU compute timing from
end-to-end preprocessing/encoding/transmission-simulation/decoding time. For
expensive diffusion runs use a bounded repetition budget and state the sample
count; do not invent reliable p95 estimates from one run. Log peak allocated and
reserved memory, model parameters, OOM, CPU fallback and actual hardware.

Report semantic metrics at each rate/SNR, per stratum, with paired uncertainty
over images. Use image/scene-level bootstrap units; crops from the same scene
are not independent. Include worst-decile quality, task failures and unsuccessful
inputs alongside means. A single composite score would hide these tradeoffs.

## From failure finding to a testable improvement

First reproduce a failure on untouched baselines. Then distinguish implementation
bugs, preprocessing/rate mismatches, domain mismatch and intrinsic low-rate loss.
Fixing a software bug is necessary, but does not establish a novel architecture.

Candidate interventions after measurement include text/detail-aware rate
allocation, uncertainty-triggered residual transmission, task-balanced masking,
content-aware codebook updates with explicit synchronization, or efficient
receiver refinement. Choose based on the observed failing stratum. Any new
branch, codebook update, label extraction and signalling must enter the resource
and FLOPs ledger. Evaluate with ablations against data-only retraining, equal
compute and equal transmitted resources. Test whether gains harm ordinary images.

Mixing existing datasets alone is benchmark engineering, not automatically a
new dataset contribution. A publishable benchmark would need reproducible sample
selection, documented coverage, independent annotation where necessary, strong
baseline comparisons and an untouched test protocol.

## Primary sources checked

- [LSDIR authors' project](https://ofsoundof.github.io/lsdir-data/)
- [CLIC 2024 tasks and image data](https://archive.compression.cc/2024/tasks/)
- [SCLIC v1 source release](https://zenodo.org/records/16780952)
- [SCLIC authors' code](https://github.com/NoraH2004/SCLIC)
- [TextOCR authors' dataset](https://textvqa.org/textocr/)
- [COCO official dataset](https://cocodataset.org/)
- [SCID authors' page](https://eezkni.github.io/publications/ESIM.html)
- [LIU4K-v2 authors' page](https://structpku.github.io/LIU4K_Dataset/LIU4K_v2.html)
- [ACDC official release](https://acdc.vision.ee.ethz.ch/)
- [ACDC original ICCV paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Sakaridis_ACDC_The_Adverse_Conditions_Dataset_With_Correspondences_for_Semantic_Driving_ICCV_2021_paper.pdf)
- [DOTA official data and annotation formats](https://captain-whu.github.io/DOTA/dataset.html)
- [Open Images official release](https://storage.googleapis.com/openimages/web/index.html)
- [JPEG AI official dataset](https://jpeg.org/jpegai/dataset.html)
- [Compression robustness paper, NeurIPS 2023](https://arxiv.org/abs/2307.08657)
- [Compression robustness authors' code](https://github.com/klieberman/ood_nic)
- [JPEG AI artifact authors' resource, updated 2026](https://videoprocessing.ai/datasets/jai-artifacts.html)
- [JPEG AI artifact 2024 preprint](https://arxiv.org/abs/2411.06810)
- [Screen-content adaptation paper, 2024](https://arxiv.org/abs/2402.17544)
- [Perception-distortion theory](https://arxiv.org/abs/1711.06077)

The artifact project's original preprint and current release report different
counts as the collection expanded; the current page lists 50,370 artifact
instances, not 50,370 independent source images. Only a public portion is
downloadable. Similarly, SCLIC annotations are codec/output-specific. Preserve
release versions and avoid pooling duplicate source images as independent cases.
