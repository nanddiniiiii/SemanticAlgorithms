# NTSCC-aware codebook experiment

## Design inspected and selected

Official NTSCC computes one `rate_index` per spatial token. The index selects an
active prefix length from:

`[16, 32, 48, 64, 80, 96, 102, 118, 134, 160, 186, 192, 208, 224, 240, 256]`.

Only this prefix is selected for the analog channel, and the official decoder
uses the same rate index when mapping the received prefix back to the 256-D
analysis representation.

Exact K=256 codebooks per rate index were rejected as the smallest redesign
because real CIFAR-10 calibration latents did not cover all valid test-time
rates. With 64 training images at 128x128, rate-index counts were `{0: 180,
1: 3140, 2: 774, 3: 2}`; the test image used indexes through 7. Training a
test-only/fallback rate table would violate the calibration constraint.

The implemented redesign is therefore **active-prefix product quantization**:

- split each officially active prefix into 16-value blocks;
- fit one shared K=256 codebook on real, full active blocks only;
- never include inactive zero dimensions in fitting or distance;
- encode one eight-bit centroid index per active block;
- for non-multiple prefix lengths, compare/reconstruct only the real remaining
  coordinates of the final block;
- transmit the unchanged four-bit official rate index per spatial token;
- use the rate map to determine how many block indices belong to every token;
- rebuild the zero-filled `[1,256,32,48]` tensor and call the frozen official
  NTSCC decoder.

This is the smallest calibration-covered, rate-aware change found. It is more
precisely active-dimensionality-aware than a separate exact-rate table, while
remaining K=256 and keeping the official model unchanged.

The wrapper directly reuses the team `CodebookPacketHeader`, index packing,
128-bit metadata, packet CRC verification, outer length/CRC framing, Sionna 5G
LDPC, QPSK and AWGN implementations. No official or team file was modified.

## Calibration

- Dataset: CIFAR-10 training split, indices 0–63.
- Resize: 128x128.
- Test image used for fitting: no.
- Active training blocks: 8,790.
- Codebook: 256 centers x 16 values.
- Shared 32-bit center storage: 131,072 bits.
- Test codebook utilization: 256/256, no dead entries, perplexity 234.97.

## Noiseless gate

The predefined gate was PSNR at least 1 dB above the generic K=256 baseline and
higher SSIM. It passed, so noisy transport was allowed.

| Method | PSNR (dB) | SSIM | LPIPS | CLIP cosine |
|---|---:|---:|---:|---:|
| Native NTSCC | 25.7058 | 0.8207 | 0.1695 | 0.9674 |
| Generic NTSCC K=256 | 12.6616 | 0.2536 | 0.7829 | 0.5696 |
| Active-prefix PQ16 K=256, noiseless | 14.9065 | 0.3685 | 0.5518 | 0.5881 |

The redesign improved the generic codebook by 2.2449 dB PSNR and 0.1149 SSIM,
but remains 10.7993 dB and 0.4522 SSIM below native NTSCC.

Active-prefix quantization MSE was 2.121066, computed only across 72,252
officially active values. Native NTSCC has no codebook quantization MSE. The
generic baseline's previously reported 0.559943 averages the entire zero-padded
256-D tensor, so it is not numerically comparable to the active-only MSE.

## Noisy LDPC/QPSK/AWGN result

All five seeds 42–46 passed both CRC layers at 10 dB and recovered the packet
exactly, so reconstruction metrics matched the noiseless ablation.

| Method | Packet bits | Complex uses | CBR | PSNR (dB) | SSIM | LPIPS | CLIP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native NTSCC | N/A | 36,126 | 0.030624 | 25.7058 | 0.8207 | 0.1695 | 0.9674 |
| Generic K=256 + PHY | 18,592 | 19,456 | 0.016493 | 12.6616 | 0.2536 | 0.7829 | 0.5696 |
| Active-prefix PQ16 K=256 + PHY | 42,440 | 43,008 | 0.036458 | 14.9065 | 0.3685 | 0.5518 | 0.5881 |

The quality improvement is real, but it costs more resources: 43,008 complex
uses are 19.05% above native NTSCC and 121.05% above the generic codebook. The
result establishes that respecting active dimensions helps reconstruction, but
this particular 16-D product packet is not yet a favorable rate-quality tradeoff.

## Scope

Only new NTSCC experiment wrappers and outputs were added. Native NTSCC,
DeepJSCC, RAW/JPEG experiments, existing validated output files, official/team
repositories and GitHub were not modified or rerun.
