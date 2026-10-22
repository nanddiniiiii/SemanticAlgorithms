# JCM reproduction audit

Paper: [Joint Coding-Modulation for Digital Semantic Communications via
Variational Autoencoder](https://arxiv.org/html/2310.06690v2), section IV-A,
Tables I and II. Author code pinned at
[`9a53ef0`](https://github.com/SJTU-mxtao/Joint-Coding-Modulation-for-Digital-Semantic-Communication/tree/9a53ef0a3df2f41ecae6cd531968f70a13d53088).

## Paper versus released defaults

The paper specifies CIFAR-10, batch 32, Adam, 300 epochs and initial learning
rate 0.0005 decaying to 0.000001. Its Table II selects lambda 250 at 12 dB for
QAM. In contrast, released `main.py` defaults to learning rate 0.0001,
one epoch and lambda 200; `train.py` halves the probability-layer learning rate
and uses cosine warm restarts with period `train_iters + 1`. These are recorded
as a paper/code discrepancy; neither source is silently treated as the other.

## Native source contract

- `main.py`: full official 50,000 training images, 10,000 test images; horizontal
  flip for training; tensors normalized from [0,1] to [-1,1].
- `network.py`: Gumbel-softmax temperature 1.5, including stochastic evaluation;
  16-QAM has four probability categories for each of 256 real coordinates.
- Tensor route at n=128: Bx3x32x32 -> Bx256x4x4 -> Bx1024 ->
  Bx256x4 category logits -> Bx256 real coordinates -> two receiver branches,
  producing Bx10 class logits and Bx3x32x32 reconstruction.
- `modules.py`: batch-wide real-coordinate mean-square normalization; real
  noise standard deviation `10**(-snr_db/20)` independently per coordinate.
- `train.py`: CE + lambda*MSE in [-1,1]; Adam with a half-rate probability group.
  Native model selection uses test classification accuracy in the final ten
  epochs. This test-based selection must be disclosed if reproduced; a held-out
  validation alternative belongs to a separate adapted experiment.
- `main.py`: native digital training loads compatible analog encoder/decoder
  weights first. Fine-tuning a finished digital checkpoint is a different run.

## Metric differences

`utils.PSNR` averages three per-channel PSNR values per image, then averages
images. This differs from PSNR of one RGB MSE. `utils.SSIM` operates on [-1,1]
arrays but supplies `data_range=1.0`; that source behavior must be preserved in
an author-code metric column, alongside a clearly named standard SSIM column.
Do not silently repair it and compare the changed number with the paper.

## Current evidence and remaining work

The modular adapter uses the pinned native network and objective. The script
`scripts/verify_jcm_native.py` independently constructs a native instance and
compares reconstruction, logits, loss and every parameter gradient with identical
weights/input/RNG. Its receipt is a synthetic implementation check only.

Current generic training config is adapted: 45k/5k split, 150-epoch request,
possible digital-checkpoint initialization, and DataParallel replica-local
BatchNorm/power normalization. Two GPUs with local batch 16 are not equivalent
to native single-GPU batch 32. A faithful baseline should use one GPU per native
JCM run; parallelize separate experiments instead of changing its batch semantics.

Long paper-baseline training remains pending resolution of paper/code settings,
analog initialization and the native metric path. Pretrained evaluation can run
separately without claiming to reproduce training.

## Completed pretrained evaluation: 22 October 2026

Kaggle job `satyapaladugu/jcm-author-checkpoint-evaluation`, version 2 completed
on all 10,000 official test images at 12 dB, 16-QAM, n=128, batch 32 on one T4.
Checkpoint SHA-256: `b7d3a1df66512421b057eabf4f7faa937a3dc22d6fe83632d3f0bca0f0c3b897`.

| Metric | Measured | Author repository README |
|---|---:|---:|
| Native per-channel averaged PSNR, dB | 23.409674 | 23.4145 |
| Classification accuracy | 0.8764 | 0.8765 |
| Native SSIM | 0.678113 | Not tabulated |
| Standard per-image RGB PSNR, dB | 23.363642 | Different metric convention |

Native [0,1] MSE was 0.0051999874. Both PSNR values use identical outputs;
do not interchange their definitions. This was one stochastic evaluation trial,
not a significance test. Loop duration was 16.13 seconds including CPU metrics,
not isolated GPU inference latency. Torch 2.11.0+cu128 and torchvision 0.26.0
differ from the author environment. This is pretrained evaluation, not training
reproduction.

Receipts: `Results/author_evaluation_download/jcm_v2/Semantic_Lossless/Results/author_evaluation/jcm_16qam_12db/`.
