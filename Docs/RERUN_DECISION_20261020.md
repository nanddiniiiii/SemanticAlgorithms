# Rerun decision from saved artifacts and released code

Date: 20 October 2026. This is a decision audit, not evidence of new training.

## Decision scope

The requested common DIV2K/Kodak experiment and an exact reproduction of every
paper's original experiment are different claims. Some papers use ImageNet,
Flickr/CLIC or Cityscapes and different channel definitions. Keeping a released
network unchanged while using DIV2K is an author-code-based benchmark with a
documented data adapter. It is not an identical original paper experiment.

No single unconditional rerun count can cover arbitrary future rate and channel
choices. The following concrete recommendation minimizes unnecessary replacement
of trained official-model weights while fixing the existing comparison:

- Primary nominal rate: CBR 1/6, measured as complex uses / (3 * original H * W).
  This retains ADJSCC's existing rate. Also retain SwinJSCC's 1/16 measurements as
  a distinct operating point, without ranking 1/16 and 1/6 as equal bandwidth.
- Training budget: 800 unique DIV2K training source images per epoch, at least
  150 complete epochs per channel/stage. Report crop area and optimizer steps;
  equal source-image exposures do not equal equal pixel or compute exposure.
- Test on all original Kodak-24 pixels. Prefer native full-image inference when
  supported. Fixed-geometry methods may require documented padding/tiling and
  stitching; count all overhead and discard only padding before scoring. The
  earlier universal-tiling proposal is not required merely for fairness and has
  not been adopted as a validated all-model protocol.
- AWGN: normalized complex symbol power 1, complex noise variance 10^(-SNR/10).
- Proposed matched fading: one CN(0,1) coefficient per image/codeword,
  independent between images, complex AWGN, perfect receiver CSI/equalization.
  This matches the intent of ADJSCC's existing slow_fading_eq path. Different
  fading scope or CSI implies a separate experiment. This proposed transport
  is an extension for repositories that do not release that channel; it cannot
  be represented as unchanged native author channel code.
- Use a documented SNR grid and at least 10 channel realizations per image with
  fixed trial seeds. Quality-versus-actual-CBR curves are required for adaptive
  rate methods. A nominal target is not proof of matched realized channel usage.

## Five completed experiments: exact rerun recommendation for this protocol

| Model | AWGN | Matched slow Rayleigh | Why / what stays useful |
|---|---|---|---|
| DeepJSCC | New corrected training; official provenance unresolved | New corrected training; official provenance unresolved | The used source is third-party, was cloned without a recorded immutable revision in the original notebook, and cannot be certified as original-author code. The inspected reference channel also has incompatible complex-noise/fading conventions. Preserve the old checkpoints as third-party historical results. Resolve source fidelity before spending GPU time. |
| ADJSCC | Keep weights; rerun evaluation and profiling | Keep weights for the documented slow_fading_eq experiment; rerun evaluation and profiling | Author core modules and source commit match; 150 full 800-image epochs are supported by loader plus completed logs, and CBR is already 1/6. Published AWGN/Rayleigh .weights.h5 files exist. Strict loading and runtime reconstruction checks must still pass before certification. This is a DIV2K author-code adaptation, not original ImageNet paper training. |
| NTSCC+ | Fresh full-data training required | Fresh full-data training with an explicitly labeled transport extension | Both old runs used 128 images/epoch, totaling 19200 source-image exposures rather than 120000. New NATTEN shim/AMP and added fading are additional departures. Use the released training/loss path and original-compatible NATTEN. Target/tune rate on validation and report actual rate; do not assume it equals 1/6. |
| SwinJSCC base w/o SA/RA | Train an additional C=256 configuration for nominal CBR 1/6 | Additional C=256 configuration with the common transport | Existing official base C=96 checkpoints are valid candidates for the 1/16 author-model operating point. C=96 to C=256 changes trained projection dimensions; passing a different rate to this fixed-rate model is not a substitute for training. Native Swin fading is per-symbol Rayleigh amplitude, unlike the proposed slow complex fading with equalization. Preserve both old C=96 checkpoints. |
| DeepJSCC-Q | Replace simplified implementation and train | Replace simplified implementation and train | The old notebook wraps a small third-party CNN in custom nearest-constellation/straight-through code. It is not a certified author implementation. The newly found candidate uses FeatureEncoderAF/FeatureDecoderAF plus learned-modem annealing. Verify authorship and the correct journal variant before adopting it. Old weights cannot become that different model by renaming/re-evaluation. |

Thus this recommended primary-rate/common-transport benchmark requires **new
training work for four of the five models**, and **ADJSCC needs evaluation and
profiling reruns, not an automatic training restart**. DeepJSCC and DeepJSCC-Q
must first clear provenance/implementation blockers. The existing Swin runs are
retained; their additional training is needed for the new rate/transport, not
because an architecture defect was found in the old base model.

If the primary rate is changed to 1/16 instead, Swin AWGN could be retained and
ADJSCC would require a different bottleneck configuration. A full identical
training-geometry/precision protocol would require additional training too.
Those are different experiments, not evidence that all previous work is invalid.

## Evidence inspected in this audit

1. The saved five-model table has mean CBR 1/6 for DeepJSCC, ADJSCC and custom
   DeepJSCC-Q; SwinJSCC has 1/16, and old NTSCC+ averages about 0.047–0.049.
   The fixed-rate difference is 8/3 (about 2.67) in channel uses per source sample.
2. Old NTSCC+ AWGN and Rayleigh epoch CSVs each contain 150 rows, all with
   images_in_epoch=128. Swin CSVs contain 150 rows/channel with samples_seen=800.
3. CPU checkpoint inspection found epoch=150 and optimizer state in DeepJSCC,
   NTSCC+, SwinJSCC and custom DeepJSCC-Q AWGN files. These files do not contain
   a complete scheduler/RNG/AMP-scaler resume state. This does not invalidate
   evaluation weights; it prevents claiming a bit-identical continuation from
   only the inspected checkpoint contents. Some runs used no scheduler.
4. Both Swin checkpoints have encoder block counts [2,2,6,2], decoder counts
   [2,6,2,2], and C=96 projection shapes [96,320] / [320,96]. These agree with
   the released base w/o-SA-and-RA model settings. This was a state/schema check,
   not a fresh numerical inference equivalence test.
5. Live ADJSCC output listing confirmed both latest.weights.h5 files. Its final
   epoch logs record epoch 150 for both channels; summary source commit is
   e5332e95faf592aab9f440992de96029162dc7dd, matching the pinned author snapshot.
   Weight bytes were not downloaded or loaded by this audit.
6. The inspected third-party DeepJSCC channel applies independent real gains to
   I/Q halves shared across a batch. This is not multiplication by a common
   complex h per image. It also sets real noise variance to mean(real_latent^2)
   / (2*SNR_linear). Under the benchmark's paired-complex interpretation, that
   would yield a 3.01 dB noise-convention difference. This is a source-derived
   consequence, not a newly measured channel result or proof of the unrecorded
   historical source revision. It is another reason not to certify old SNR labels.
7. The Swin author channel draws a separate Rayleigh amplitude for each complex
   symbol. ADJSCC uses supplied complex per-example fading plus equalization.
   Old NTSCC+ and custom DeepJSCC-Q use added per-example complex fading paths.
   Equal SNR text therefore does not establish equal physical experiments.

## Required evaluation and complexity reruns: all five

- Use one RGB [0,1] PSNR/SSIM/MS-SSIM implementation with fixed windows,
  constants, scales, image averaging, clipping and trial aggregation.
- Reconstruct all 24 full original Kodak images, preserving portrait orientation
  and accounting for any permitted preprocessing. Do not compare cropped SQ-GAN
  regions against other methods' complete-image reconstructions.
- Count actual transmitted complex symbols, all communicated streams, required
  side information/control, padding and digital framing/FEC where applicable.
  Distinguish measured serialization from theoretical index-bit bounds.
- Profile the same complete original image workload. Existing 128x128/256x256
  inference counts cannot be labeled full-Kodak costs.
- Training profiling must include the real forward/backward paths for all
  optimizers; report actual batch size and amortized per-image cost, total run
  cost, precision and initialization. Separate optimizer-update arithmetic.
- Apply a common multiply-add=2 convention and audit operator coverage. PyTorch
  supported-operator FLOPs and TensorFlow graph FLOPs are not automatically
  comparable totals. Fused attention and unsupported backward operations must
  be accounted for or reported as missing; do not guess a multiplier.
- Do not use Kodak for selecting rate parameters or checkpoints. Use DIV2K
  validation for tuning and retain uncertainty across channel trials/seeds.

## Remaining methods and strict author-experiment limit

- Original NTSCC (2022) is a separate model and has no certified new matched
  150-epoch training result here; it requires its own run.
- NTSCC++ online inference is still marked unreleased in the author README.
  Training NTSCC+ again cannot satisfy full NTSCC++ reproduction.
- Current SQ-GAN adaptation is not the original discrete-index communication
  system. Its completed semantic-stage weights are preserved as an adaptation;
  they cannot count as exact original-system training. Native Cityscapes and a
  controlled DIV2K/digital-transport study must be labeled separately.
- DiT-JSCC public release omits training code; DDM-SSCC author code remains
  unverified. They cannot be filled with reduced models and certified as official.
- If "reproduce authors" means identical complete original paper experiments,
  **none of the five is currently certified**, including ADJSCC and SwinJSCC,
  because the original data/protocol and runtime equivalence have not all been
  reproduced. Rerunning alone cannot resolve missing author code/assets.

No training was started, no checkpoint was overwritten and no Git push was made
by this analysis. The shared evaluation helper is preparatory code and has not
been validated against all model runtimes. Complete fairness remains unverified
until the declared benchmark is executed and its artifacts pass the stated checks.

## Primary references

- https://github.com/alexxu1988/ADJSCC
- https://github.com/semcomm/SwinJSCC
- https://github.com/wsxtyrdd/NTSCC_plus
- https://github.com/chunbaobao/Deep-JSCC-PyTorch
- https://github.com/r0ymanesco/deepjscc_stream
- https://github.com/frapez1/SQ-GAN
- https://github.com/semcomm/DiTJSCC
