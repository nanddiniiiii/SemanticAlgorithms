# DeepJSCC, NTSCC, and teammate SwinJSCC comparison

## Bottom line

On the saved 10 dB AWGN measurements, **native NTSCC is the strongest low-rate point**: 25.706 dB at CBR 0.03062. The teammate SwinJSCC curve is 22.783 dB at CBR 0.02083 and 23.819 dB at CBR 0.04167; linear interpolation at the NTSCC CBR is 23.270 dB, so the observed gap is about **+2.436 dB for NTSCC**. This is descriptive, not a controlled significance result, because NTSCC was tested on only `kodim08` with five channel seeds while SwinJSCC is an average over all 24 Kodak images.

The saved DeepJSCC result is last among the native neural models: 15.983 dB at CBR 0.16255. This does **not** establish that DeepJSCC is intrinsically poor. Its checkpoint was trained at 200 dB and evaluated at 10 dB, which is a severe train/test channel mismatch; the original paper trains models for the channel conditions under study.

The original papers support **NTSCC and SwinJSCC above classical CNN DeepJSCC for high-resolution images**, but they do not support a clean NTSCC-versus-SwinJSCC total order: the SwinJSCC paper does not include NTSCC in its comparisons, and the papers use different training and BD-rate protocols.

## What was actually compared

| Evidence | Data/protocol | What it contributes |
|---|---|---|
| `Nandini_DeepJSCC_NTSCC_Integration.ipynb` on `origin/nandini-integration` | Orchestration notebook and accounting rules | No independent final score; it explains how DeepJSCC, NTSCC, and the team PHY should be run and prevents invalid CBR comparisons. |
| `ImageFeatures_DeepJSCC_NTSCC_FINAL_REPAIRED.ipynb` | One 768×512 image (`kodim08`), AWGN 10 dB, five seeds, strict official checkpoint loading | Native DeepJSCC/NTSCC, JPEG+LDPC/QPSK, generic codebooks, and the NTSCC active-prefix codebook follow-up. |
| Teammate `notebookbf0f6f4129.ipynb` | DIV2K training, all 24 Kodak test images, AWGN, five SNRs and five CBRs | SwinJSCC SA+RA curve. It trained only 10 fixed-rate + 10 adaptive epochs, so it is a teammate experiment, not a paper-exact reproduction. |
| Original papers | Published experimental protocols and aggregate claims | External plausibility/order checks; paper figures are not mixed into the measured local curves. |

The linked branch was audited at commit `c36333dd82dc0eeb3587e7e9f75d5adceeb728f8` (`nandini-integration`). The branch adds the Nandini DeepJSCC/NTSCC integration to the teammate repository; the executed SwinJSCC notebook comes from the repository history inherited by that branch.

## Saved-results ranking

### 1. Native neural JSCC, reconstruction quality per channel use

At 10 dB:

| Order | Method | CBR | PSNR | SSIM | Important qualification |
|---:|---|---:|---:|---:|---|
| 1 | Nandini native NTSCC | 0.03062 | 25.706 dB | 0.821 | One Kodak image, five noise seeds; native CBR excludes a 6,144-bit untransmitted rate map. |
| 2 | Teammate SwinJSCC SA+RA | 0.02083–0.12500 | 22.783–23.840 dB | Not saved | Kodak24 mean; substantially undertrained relative to the paper. |
| 3 | Nandini native DeepJSCC | 0.16255 | 15.983 dB | 0.411 | Checkpoint trained at 200 dB but tested at 10 dB; not a fair DeepJSCC-paper reproduction. |

NTSCC also dominates the teammate curve in the saved experiment: it has higher PSNR than every teammate point while using less CBR than four of the five SwinJSCC operating points. Do not present this as “NTSCC universally beats SwinJSCC”; the test sets and training budgets differ.

The teammate curve is nearly flat after CBR 0.04167 and is slightly non-monotonic (`C=64` outperforms `C=96` at 10 dB). Together with the 20 total training epochs, this is evidence of undertraining or unstable adaptive-rate training, not evidence against the published SwinJSCC architecture.

### 2. Codebook variants

| Method | CBR | PSNR | SSIM | Interpretation |
|---|---:|---:|---:|---|
| DeepJSCC K=256 | 0.13715 | 17.215 dB | 0.394 | Slightly improves the mismatched native DeepJSCC PSNR, but remains poor. |
| NTSCC active-prefix PQ16 K=256 | 0.03646 | 14.907 dB | 0.368 | Better than the generic NTSCC codebook, but uses more resources than native NTSCC and is still 10.80 dB worse. |
| Generic NTSCC K=256 | 0.01649 | 12.662 dB | 0.254 | Lowest transmitted rate, but the reconstruction penalty is too large. |

For NTSCC, **native transmission strictly dominates the active-prefix codebook** in this experiment: lower CBR (0.03062 versus 0.03646) and much higher PSNR (25.706 versus 14.907). The generic codebook is useful only if minimizing rate is more important than recognizable reconstruction quality. A serious codebook result needs quantization-aware joint fine-tuning, not frozen post-hoc clustering.

### 3. Conventional JPEG + LDPC/QPSK

The valid digital points range from JPEG Q25 at CBR 0.31076 / 26.841 dB to JPEG Q90 at CBR 1.12326 / 36.231 dB. They win in absolute PSNR when enough channel use is allowed, but consume roughly 10–37 times NTSCC's CBR. All saved frames succeeded at 10 dB, so this run does not expose the digital cliff effect; an SNR sweep with CRC failures is required for that claim.

## What the original papers say

- **DeepJSCC (2019):** CNN JSCC beats JPEG/JPEG2000 plus idealized channel coding mainly at low SNR and constrained bandwidth, and degrades gracefully rather than showing a digital cliff. Its Kodak experiment trains on ImageNet patches, evaluates all 24 Kodak images, and averages 100 channel realizations. See the [original DeepJSCC paper](https://arxiv.org/html/1809.01733), especially its Kodak protocol and Figures 7–11.
- **NTSCC (2022):** on high-resolution images it reports NTSCC above standard DeepJSCC across CBRs, with at least 1 dB gain in the matched SNR sweep. On Kodak, relative to BPG+LDPC, the paper reports **17.96% average bandwidth saving or +0.81 dB BD-PSNR**. See the [original NTSCC paper](https://arxiv.org/html/2112.10961).
- **SwinJSCC (2024 version):** it trains on DIV2K 256×256 crops and evaluates Kodak/CLIC, with SNR 1–13 dB and CBR 0.0208–0.125. It reports SwinJSCC above CNN DeepJSCC and comparable/superior to BPG+LDPC. Its Kodak table reports BD-rate values of **−25.10% for SA+RA** and **−29.71% for fixed SwinJSCC** versus BPG+LDPC. The paper itself says the fixed model is best in quality, while SA+RA sacrifices some quality for one-model rate/SNR adaptability. See the [original SwinJSCC paper](https://arxiv.org/html/2308.09361) and [official implementation](https://github.com/semcomm/SwinJSCC).

Those paper-reported savings are shown separately because “NTSCC BD-CBR” and “SwinJSCC BD-rate” were not generated by one shared evaluator. They are context, not a controlled leaderboard.

## Defensible order

There are three different answers depending on the question:

1. **Your saved 10 dB rate–PSNR measurements:** NTSCC first, teammate SwinJSCC second, DeepJSCC third. This ordering is provisional because the protocols differ.
2. **Your frozen codebook experiments:** native NTSCC first; none of the codebook variants beats its native rate–quality point. Generic NTSCC codebook gives the lowest CBR but the worst quality.
3. **Published high-resolution evidence:** NTSCC and SwinJSCC are both ahead of classical DeepJSCC. A strict NTSCC-versus-SwinJSCC winner is not established by the cited papers. If forced to choose by deployment flexibility, SwinJSCC SA+RA wins because one model spans rates/SNRs; if forced to choose by the saved local low-rate PSNR, NTSCC wins.

## What is needed for a publication-grade “who is better” claim

Run all three native models on the same 24 Kodak images, identical full-resolution preprocessing, the same AWGN convention, the same five or more seeds, and matched CBR points (for example 1/48, 1/32, 1/16, 1/12, 1/8). Report mean and 95% confidence interval for PSNR, SSIM/MS-SSIM, LPIPS, and CLIP; include every side-information bit/channel use. Retrain or select DeepJSCC for the tested SNR, and train the teammate SwinJSCC to convergence or use the official checkpoint. Only then calculate paired per-image differences and BD-rate.

## Generated artifacts

- `measured_comparison.png`: measured CBR–PSNR, CBR–SSIM, and SNR–PSNR plots.
- `paper_reported_savings.png`: separate paper-reported aggregate savings plot.
- `comparison_data.csv`: combined extracted local/team data.
- `comparison_summary.json`: key interpolation and paper-context numbers.
- `generate_comparison.py`: reproducible extractor/plot generator.

