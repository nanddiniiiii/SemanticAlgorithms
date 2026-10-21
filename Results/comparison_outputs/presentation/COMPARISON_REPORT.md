# Presentation comparison: DeepJSCC, NTSCC, and teammate SwinJSCC

## Evidence boundaries

This comparison uses three deliberately separate evidence classes:

1. **Your measured results:** `ImageFeatures_DeepJSCC_NTSCC_FINAL_REPAIRED.ipynb`, evaluated on `kodim08` (768×512), AWGN 10 dB, channel seeds 42–46.
2. **Teammate measured results:** the executed SwinJSCC notebook inherited by the `nandini-integration` branch, evaluated on all 24 Kodak images. The current remote branch head was verified as `c36333dd82dc0eeb3587e7e9f75d5adceeb728f8`.
3. **Paper-reported results:** claims and aggregate values from the original DeepJSCC, NTSCC, and SwinJSCC papers. They are context only and are not mixed into the measured bar charts.

Numbers from different evidence classes are **not directly ranked** because the datasets, aggregation, training budgets, model checkpoints, and sometimes rate definitions differ.

Metric names are also kept exact: your notebook reports single-scale **SSIM**, whereas the NTSCC and SwinJSCC papers mainly plot **MS-SSIM** (often transformed to dB). Those are not interchangeable, so paper MS-SSIM curves are not placed on the SSIM chart. LPIPS/CLIP are included only for your rows because the teammate notebook did not save them; none is inferred from PSNR.

## Exact teammate model

The teammate implemented the official high-resolution **SwinJSCC Base model with both SNR Adaptation and Rate Adaptation**, named `SwinJSCC_w/_SAandRA` in the official code.

- Backbone: hierarchical Swin Transformer.
- Encoder depths: `[2, 2, 6, 2]`; decoder depths: `[2, 6, 2, 2]`.
- Embedding dimensions: `[128, 192, 256, 320]`; attention heads: `[4, 6, 8, 10]`; window size: 8.
- Channel: AWGN.
- Training: DIV2K, random 256×256 crops.
- Evaluation: all 24 Kodak images, full images cropped to multiples of 128.
- Two stages: fixed `SwinJSCC_w/o_SAandRA` pretraining at SNR 10 dB and C=96 for 10 epochs, followed by `SwinJSCC_w/_SAandRA` for 10 epochs.
- Adaptive grid: SNR `{1,4,7,10,13}` dB and active channels `C={32,64,96,128,192}`.
- Official high-resolution CBR: `C/1536`, giving `{0.020833,0.041667,0.062500,0.083333,0.125000}`.

The notebook wraps the official encoder, decoder, and channel to make the output safe for two-GPU `DataParallel`; it does not replace the SwinJSCC mathematics. Its short 10+10 epoch schedule is explicitly marked configurable and is not a paper-exact convergence run.

### Teammate measured PSNR grid

All entries are Kodak24 mean PSNR in dB. The notebook saved no SSIM, LPIPS, CLIP, payload-bit, or coded-bit values.

| C | CBR | 1 dB | 4 dB | 7 dB | 10 dB | 13 dB |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.020833 | 21.172 | 21.883 | 22.449 | 22.783 | 22.981 |
| 64 | 0.041667 | 22.480 | 23.106 | 23.548 | 23.819 | 23.960 |
| 96 | 0.062500 | 22.686 | 23.164 | 23.465 | 23.648 | 23.747 |
| 128 | 0.083333 | 22.990 | 23.393 | 23.628 | 23.777 | 23.854 |
| 192 | 0.125000 | 23.221 | 23.585 | 23.743 | 23.840 | 23.889 |

The rate curve is slightly non-monotonic: at 10 dB, C=64 is better than C=96 and C=128 despite using less CBR. This is a measured notebook result and is retained rather than smoothed.

## Your measured results at 10 dB

The four JPEG rows below are the actual JPEG packets passed through the saved LDPC/QPSK/AWGN path. All had frame-success rate 1.0, so their reconstruction metrics equal the corresponding codec-only reconstructions in this particular 10 dB run.

| Method | PSNR dB | SSIM | LPIPS ↓ | CLIP cosine ↑ | CBR | Complex uses | Payload / coded bits |
|---|---:|---:|---:|---:|---:|---:|---:|
| JPEG Q25 + LDPC/QPSK | 26.841 | 0.861 | 0.0879 | 0.9513 | 0.31076 | 366,592 | 366,488 / 733,184 |
| JPEG Q50 + LDPC/QPSK | 29.426 | 0.910 | 0.0436 | 0.9724 | 0.46962 | 553,984 | 552,960 / 1,107,968 |
| JPEG Q75 + LDPC/QPSK | 32.156 | 0.944 | 0.0189 | 0.9855 | 0.69010 | 814,080 | 813,912 / 1,628,160 |
| JPEG Q90 + LDPC/QPSK | 36.231 | 0.973 | 0.0068 | 0.9871 | 1.12326 | 1,325,056 | 1,324,216 / 2,650,112 |
| Native DeepJSCC | 15.983 | 0.411 | 0.7023 | 0.7777 | 0.16255 | 191,748 | N/A: continuous analog symbols |
| Native NTSCC | 25.706 | 0.821 | 0.1695 | 0.9674 | 0.03062 | 36,126 | N/A: continuous analog symbols |
| DeepJSCC K=256 | 17.215 | 0.394 | 0.6090 | 0.6113 | 0.13715 | 161,792 | 161,632 / 323,584 |
| Generic NTSCC K=256 | 12.662 | 0.254 | 0.7829 | 0.5696 | 0.01649 | 19,456 | 18,592 / 38,912 |
| NTSCC active-prefix PQ16 K=256 | 14.907 | 0.368 | 0.5518 | 0.5881 | 0.03646 | 43,008 | 42,440 / 86,016 |

Important accounting qualifications:

- Native DeepJSCC, native NTSCC, and SwinJSCC transmit continuous channel symbols; a conventional payload bitrate is therefore **not defined**. Their comparable resource measure is complex channel uses/CBR.
- SwinJSCC's FP32 latent-memory size is likewise not an over-air bitrate. Only its saved CBR/rate level is used.
- NTSCC's entropy estimate (`bpp_y+bpp_z`) is a diagnostic, not transmitted radio bits.
- The native NTSCC CBR excludes the notebook's untransmitted 6,144-bit rate map, which must be included in a deployable digital signaling design.
- The DeepJSCC checkpoint was trained at 200 dB but evaluated at 10 dB. Its low score is a real notebook measurement, but not a fair reproduction of the DeepJSCC paper's matched-training experiments.

## Paper-reported evidence

### DeepJSCC

The original paper trains a CNN JSCC encoder/decoder end-to-end and defines bandwidth compression as complex channel symbols divided by source dimension. Its Kodak protocol trains on ImageNet 128×128 crops, evaluates all 24 768×512 Kodak images, and averages 100 channel realizations. The paper reports that DeepJSCC is strongest against JPEG/JPEG2000 separation at low SNR and constrained bandwidth and exhibits graceful degradation rather than a digital cliff. These paper curves are not inserted into the plots because no paper table supplies points under your exact checkpoint/image protocol. [Original DeepJSCC paper](https://arxiv.org/html/1809.01733).

### NTSCC

The NTSCC paper uses nonlinear transforms, an entropy hyperprior, content-adaptive rate allocation, and hyperprior-aided decoding. On high-resolution images it reports NTSCC above standard DeepJSCC across the tested CBR range and at least 1 dB better in its matched-SNR sweep at CBR 1/16. On Kodak, relative to its BPG+LDPC baseline, it reports **17.96% BD-CBR saving or +0.81 dB BD-PSNR**. These are paper aggregates, not your measured numbers. [Original NTSCC paper](https://arxiv.org/html/2112.10961).

### SwinJSCC

The teammate's architecture matches the paper's Base SA+RA variant. The paper trains on DIV2K and tests on Kodak/CLIC across SNR 1–13 dB and CBR 0.0208–0.125. It reports SwinJSCC above CNN DeepJSCC and comparable or superior to BPG+LDPC. Its Kodak BD-rate table reports **−25.10% for SA+RA** and **−29.71% for fixed SwinJSCC** versus BPG+LDPC. The paper explicitly says the fixed model gives the best quality, while SA+RA trades some quality for one-model SNR/rate adaptability. It does not compare against NTSCC, so the papers do not establish an NTSCC-versus-SwinJSCC winner. [Original SwinJSCC paper](https://arxiv.org/html/2308.09361) and [official implementation](https://github.com/semcomm/SwinJSCC).

## Directly comparable conclusions

### Within your `kodim08`, 10 dB experiment

- **Highest reconstruction quality:** JPEG Q90, at 36.231 dB PSNR and 0.973 SSIM, but it also has the largest CBR (1.12326).
- **Highest-quality native neural JSCC:** NTSCC, at 25.706 dB PSNR, 0.821 SSIM, and CBR 0.03062.
- **Lowest channel usage:** generic NTSCC K=256, at CBR 0.01649 and 19,456 complex uses, but its quality falls to 12.662 dB/0.254 SSIM.
- **Strongest balanced quality–resource point:** native NTSCC. It retains 25.706 dB and 0.821 SSIM using only 36,126 channel uses. It is the clear knee of the measured Pareto curve; all DeepJSCC and active-prefix codebook points are dominated by it. JPEG provides higher quality only by spending at least about 10.1× more channel uses (Q25).

### Within the teammate's Kodak24, 10 dB experiment

- **Highest PSNR:** C=192, CBR 0.125, PSNR 23.840 dB.
- **Lowest channel usage:** C=32, CBR 0.020833, PSNR 22.783 dB.
- **Strongest practical quality–resource knee:** C=64, CBR 0.041667, PSNR 23.819 dB—only 0.021 dB below C=192 while using one third of its CBR. C=96 and C=128 are directly dominated by C=64 in the saved 10 dB results.

There is **no direct winner between your NTSCC and the teammate's SwinJSCC** from these notebooks: your value is one image averaged over noise seeds, whereas the teammate value is an image-set mean after a short training run.

## Files

- `01_psnr_bar.png` — separate PSNR panels for your and teammate protocols.
- `02_ssim_bar.png` — your measured SSIM values; teammate SSIM explicitly unavailable.
- `03_channel_usage_cbr_bar.png` — measured CBR and your absolute channel uses.
- `04_rate_vs_quality.png` — PSNR/SSIM versus measured rate, with cross-source caution.
- `nandini_measured_values.csv` and `teammate_swinjscc_measured_values.csv` — plotted source data.
- `generate_presentation_plots.py` — reproducible plot generation; reads saved results only.
