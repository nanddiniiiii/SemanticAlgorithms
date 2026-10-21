# DeepJSCC/NTSCC integration and interpretation

## What the experiment measures

The comparison asks: **at the same source image, resolution, AWGN Es/N0 and
channel-use definition, how much reconstruction quality does each scheme obtain
for the radio resource it consumes?**  It is a rate--distortion/perception
experiment, not by itself proof of task-level semantic understanding.

The repository-wide definition is

`CBR = number of complex channel uses / (3 * image height * image width)`.

Do not compare that CBR with compressed-file bits divided by raw-image bits.
Those are different physical quantities.

## Signal paths

### DeepJSCC

`RGB image [0,1] -> CNN encoder -> dense real latent -> per-image I/Q pairing
-> power normalization -> complex AWGN -> real latent -> CNN decoder -> RGB`

The historical checkpoint uses `c=19`.  For the verified 128x128 sanity input,
the encoder output is `(1, 19, 29, 29)`: 15,979 real values.  Pairing two real
values as I and Q requires `ceil(15979/2) = 7,990` complex channel uses.  The
old notebook's `0.325094` value was the real-value ratio
`15979/(3*128*128)`; the repository CBR is `7990/(3*128*128) = 0.162557`.

These are continuous floating-point channel symbols, not quantized payload
bits.  `15979*32` is only the memory needed to store the tensor as FP32 and is
not an over-the-air bit count.

The checkpoint was previously instantiated with a 200 dB channel setting.  A
25.6476 dB reconstruction at 200 dB is a near-noiseless sanity check; it must
not be placed in a 10 dB comparison table.  The runner now performs repeated,
seeded trials at the requested test SNR and reports mean and standard deviation.

### NTSCC

`RGB -> nonlinear analysis transform g_a -> latent y -> hyperprior/entropy
model -> content-dependent rate allocation -> JSCC encoder -> complex AWGN ->
JSCC decoder (+ hyperprior refinement) -> synthesis transform g_s -> RGB`

NTSCC differs from fixed-rate DeepJSCC because it estimates the information
content of local latent patches and selects a channel dimension for each patch.
It sends more symbols for difficult/informative regions and fewer for simple
regions.

The official source computes
`cbr_y = channel_usage / (3*H*W)`, where `channel_usage` is already the number
of complex symbols after I/Q pairing.  Therefore, for the old 256x256 result
`cbr_y=0.031779`, the correct count is approximately
`0.031779 * 3 * 256 * 256 = 6,248` complex uses.  Multiplying by only `H*W`
undercounts by three; dividing by two again is also incorrect.

`bpp_y` and `bpp_z` are entropy-model diagnostics for the nonlinear-transform
latents.  `(bpp_y+bpp_z)*H*W` is an estimated entropy-coded representation size,
not the number of bits sent by the analog NTSCC channel path.  The official
forward pass also gives the decoder the per-patch rate indexes directly.  That
side information is not included in `cbr_y`; the WSL runner now reports its
fixed-width size separately.

## Metrics

- `MSE = mean((x-x_hat)^2)` on RGB values in `[0,1]`.
- `PSNR = 10*log10(1/MSE)` dB. Higher is better, but it rewards pixel fidelity.
- `SSIM` measures local luminance/contrast/structure similarity in `[0,1]`.
- `CBR` is complex channel uses per RGB source value. Lower is better only at
  comparable quality.
- `frame_success_rate` is mandatory for the digital baseline. PSNR/SSIM are
  reported only for CRC-valid frames; a failed frame is not assigned a fake
  reconstruction score.
- `bpp_y`, `bpp_z`, latent shape and parameter counts are diagnostics, not
  substitutes for channel uses.

LPIPS, CLIP similarity and downstream task accuracy may be added as perceptual
or semantic metrics. They should be computed on the same reconstruction and
must not replace communication accounting.

## Two-environment run order

From the NTSCC Python 3.8 WSL environment:

```bash
cd "/mnt/d/<local OneDrive>/Desktop/image_semcom/SemanticSchedulerEnd2End"
python scripts/run_ntscc_benchmark.py \
  --output-dir "/mnt/d/<local OneDrive>/Desktop/image_semcom/SemanticSchedulerEnd2End/results/runs" \
  --snr-db 10 --trials 5 --device cpu
```

Then, from the Windows environment that can load DeepJSCC and Sionna:

```powershell
$env:SEMCOM_ROOT = "D:\<local OneDrive>\Desktop\image_semcom"
cd "$env:SEMCOM_ROOT\SemanticSchedulerEnd2End"
python scripts\run_communication_benchmark.py `
  --snr-db 10 --trials 5 `
  --ntscc-result "results\runs\ntscc_kodim08_10db.json"
```

The aggregator rejects mismatched image sizes and SNRs.  If a 256x256 test is
required, create one explicit reference image and pass that exact file to both
runners; never resize only one reconstruction during metric calculation.

## Codebook integration

A latent codebook changes either JSCC model into a hybrid learned-source-code /
digital-channel-code experiment:

1. Collect calibration latents without using test images.
2. Convert `B,C,H,W` to `B,(H*W),C` token vectors.
3. Fit a shared K=256 codebook. Each fixed-width index costs 8 bits.
4. Encode a packet containing indices plus rate mask, versioned metadata and
   CRC. Keep shared codebook storage separate from per-image packet cost.
5. Send the actual packet through the repository 5G-LDPC/QPSK/AWGN baseline.
6. On CRC success, replace indexes with code vectors, restore the latent shape,
   and use the existing decoder. On failure, report a failed frame.
7. Report codebook usage (used/dead entries, perplexity, largest usage share),
   quantization-only distortion, and digital over-air performance separately.

For K=256 and N tokens, the index payload is `8*N` bits before overhead.  The
shared codebook costs `256*C*precision_bits` bits once; it is not charged to
every image when both endpoints already possess the same version.  NTSCC should
generally use one codebook per rate class or a product/residual quantizer because
its allocated channel dimension varies by patch.  The rate-map indexes must also
be transmitted.

Post-hoc quantization will usually reduce rate but add distortion because the
pretrained decoder was not trained for code vectors.  The serious experiment is
quantization-aware fine-tuning with a straight-through estimator or VQ loss.

A separate semantic prototype codebook can cluster CLIP embeddings and send a
prototype ID for scheduling/KG tasks.  That is useful for task-oriented
communication, but it is not the same object as a reconstruction-latent
codebook and should be evaluated with downstream task accuracy.

## What conclusions are valid

DeepJSCC supplies a fixed dense representation and typically degrades smoothly
as noise increases. NTSCC adds nonlinear-transform priors and content-adaptive
rate allocation, so it may obtain better quality at low CBR. Digital JPEG/LDPC
can be very efficient when channel decoding succeeds but has a cliff: a CRC
failure yields no valid image.

One Kodak image is a debugging example only. A defensible result uses all 24
Kodak images (or another declared held-out set), multiple channel seeds, an SNR
sweep, confidence intervals, matched CBR operating points, and all side
information. Parameter count, latency, peak memory and downstream task accuracy
should be added if the claim concerns deployment or semantics.

## Corrected issues

- Removed unsupported `channel_type` and `snr` arguments passed to the adapter.
- Removed erroneous image `*255` / reconstruction `/255` scaling.
- Reused the repository AWGN implementation for per-image I/Q pairing, power
  normalization and Es/N0 semantics.
- Standardized every CBR to complex uses divided by `3*H*W`.
- Corrected NTSCC channel uses from `cbr*H*W` to `cbr*3*H*W`.
- Removed hard-coded/fallback model scores and silent reference-image resizing.
- Added repeated seeded trials, JSON schemas, checkpoint/parameter reporting and
  strict image-size/SNR validation.
- Kept NTSCC entropy BPP and untransmitted rate-map overhead visibly separate
  from native radio channel use.
