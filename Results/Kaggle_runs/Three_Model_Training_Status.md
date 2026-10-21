# Three-model Kaggle training and comparison status

All three models completed **150 epochs on AWGN and 150 epochs on Rayleigh at 10 dB**. Each run evaluated all 24 Kodak images. The NTSCC++-named experiment uses the author-linked `NTSCC_plus` implementation; the Kaggle T4 x2 run completed in 7,581 seconds.

## Kodak comparison

| Model | Channel | PSNR (dB) | SSIM | MS-SSIM | Complex uses/image | CBR |
|---|---|---:|---:|---:|---:|---:|
| DeepJSCC | AWGN | 23.368 | 0.6451 | 0.8728 | 196,608.0 | 0.16667 |
| DeepJSCC | Rayleigh | 20.332 | 0.5324 | 0.7446 | 196,608.0 | 0.16667 |
| ADJSCC | AWGN | 32.226 | 0.9201 | 0.9748 | 196,608.0 | 0.16667 |
| ADJSCC | Rayleigh | 29.809 | 0.8685 | 0.9543 | 196,608.0 | 0.16667 |
| NTSCC++ (NTSCC_plus release) | AWGN | 21.127 | 0.5454 | 0.7897 | 55,442.3 | 0.04700 |
| NTSCC++ (NTSCC_plus release) | Rayleigh | 21.003 | 0.5701 | 0.7956 | 57,522.3 | 0.04876 |

## FLOPs and training setup

| Model | Channel | Train FLOPs/sample | Training profile input | Inference FLOPs/image | Inference profile input | Images sampled/epoch |
|---|---|---:|---|---:|---|---:|
| DeepJSCC | AWGN | 167,346,181 | one 128x128 RGB sample (forward + backward) | 167,182,338 | one 128x128 RGB sample | 800/800 |
| DeepJSCC | Rayleigh | 167,362,567 | one 128x128 RGB sample (forward + backward) | 167,182,340 | one 128x128 RGB sample | 800/800 |
| ADJSCC | AWGN | 70,469,104,100 | one 128x128 RGB sample (forward + backward; Adam update excluded) | 23,490,280,974 | one 128x128 RGB sample | 800/800 |
| ADJSCC | Rayleigh | 70,469,112,292 | one 128x128 RGB sample (forward + backward; Adam update excluded) | 23,490,289,166 | one 128x128 RGB sample | 800/800 |
| NTSCC++ (NTSCC_plus release) | AWGN | 225,518,008,470 | one 256x256 RGB crop (PyTorch profiler; fused NATTEN may be under-counted) | 481,300,316,170 | one full-resolution 768x512 Kodak image (PyTorch profiler; fused NATTEN may be under-counted) | 128/800 |
| NTSCC++ (NTSCC_plus release) | Rayleigh | 225,518,090,394 | one 256x256 RGB crop (PyTorch profiler; fused NATTEN may be under-counted) | 481,300,316,174 | one full-resolution 768x512 Kodak image (PyTorch profiler; fused NATTEN may be under-counted) | 128/800 |

FLOP counts are profiler estimates per sample/image, not total-epoch FLOPs. DeepJSCC and ADJSCC were profiled on 128×128 crops; NTSCC++ training was profiled on a 256×256 crop and inference on a full 768×512 Kodak image. ADJSCC TensorFlow profiler coverage may omit custom fused operations; PyTorch may under-count fused NATTEN arithmetic. Training FLOPs include forward and backward passes; optimizer updates are excluded.

## Training protocol and comparability

- DeepJSCC and ADJSCC each sampled all 800 DIV2K images per epoch (150 complete passes), using 128×128 crops. Batch sizes were 32 and 16 respectively.
- NTSCC++ sampled 128 distinct images per epoch from the full 800-image DIV2K pool, using 256×256 crops and batch size 8. That is 19,200 sampled image-crops over 150 epochs, about 24 image exposures on average; it is not 150 full DIV2K passes.
- DeepJSCC and ADJSCC use nominal CBR 1/6. Rate-conditioned NTSCC++ averaged about 0.047–0.049 CBR, so quality comparisons are not at matched rates.
- The NTSCC_plus release supports AWGN. Rayleigh is an extension using per-image quasi-static complex fading and perfect CSI/equalization.
- FLOP tools and profile dimensions differ, so interpret them as measured estimates rather than an apples-to-apples compute ranking.

## Completed private Kaggle runs

- [DeepJSCC — 150 epochs, AWGN + Rayleigh](https://www.kaggle.com/code/hmmkk20/deepjscc-div2k-kodak-150ep)
- [ADJSCC — 150 epochs, AWGN + Rayleigh](https://www.kaggle.com/code/hmmkk20/adjscc-div2k-kodak-150ep-20261016)
- [NTSCC++ / NTSCC_plus — completed run log](https://www.kaggle.com/code/hmmkk20/ntscc-plus-div2k-kodak-150ep-20261016/log)
- [DeepJSCC FLOP profiler helper](https://www.kaggle.com/code/hmmkk20/deepjscc-flops-profile-20261017)

## Completion and artifact availability

All requested training and evaluation runs are complete: 150 epochs per channel for each model, AWGN and Rayleigh, with all 24 Kodak images evaluated. Kaggle CLI status was rechecked for DeepJSCC, ADJSCC, NTSCC_plus, and the DeepJSCC FLOP profiler; all four report `COMPLETE`. ADJSCC's output page was inspected directly: it contains `kodak_awgn.csv`, `kodak_rayleigh.csv`, `summary.csv`, and both channel epoch folders. Its two Kodak CSV previews each show 24 rows. The ADJSCC aggregate and FLOP values are included in the local comparison; the per-image ADJSCC CSVs have not been downloaded locally.

The local notebook is a consolidated results/report notebook. Training remains in three separate Kaggle kernels because the official implementations have different TensorFlow/PyTorch dependencies. No GitHub push was made.
