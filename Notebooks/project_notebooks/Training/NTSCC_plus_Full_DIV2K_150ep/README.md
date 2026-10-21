# NTSCC_plus full-DIV2K replacement run

**Version 1 failed; version 2 submitted with an AMP optimizer-isolation fix. Training stability is pending verification.** The existing subset
notebook and its results are preserved. SQ-GAN continues independently.

## Training contract

| Setting | Value |
|---|---|
| Training pool | All 800 DIV2K training images |
| Images visited per epoch | 800, each exactly once |
| Global batch size | 8, split across two T4 GPUs |
| Batches per epoch | 100, with no dropped final batch |
| Epochs | 150 per channel |
| Image exposures | 120,000 per channel |
| Channels | AWGN; explicit quasi-static Rayleigh/perfect-CSI extension |
| Training SNR | 10 dB |
| Crop | 256 x 256 |
| Precision | FP16 autocast with GradScaler |
| Author commit | 765c6f6c11cf3bbe249a26b624dbaf47a4f0676d |
| Initialization | Fresh training; no subset-run checkpoint |

The author network, rate-distortion loss, eta=0.2, lambda=0.18, and existing
optimizer learning rates are retained. The zero attention-dropout argument is
adapted to current NATTEN at import time without editing author source files.
Rayleigh and the DIV2K/Kodak experiment remain declared adaptations, not a claim
of exact reproduction of the paper's full training protocol.

Each epoch asserts that all indices 0 through 799 were processed once. Logs
record sample count, unique sample count, batch count, actual successful AMP
optimizer updates, losses and measured epoch duration. Profiler FLOPs describe
FP32 forward/backward and native-resolution Kodak inference; actual training
uses FP16. Fused NATTEN operations may be under-counted.

## Launch after a GPU slot and quota are available

Run in PowerShell:

```powershell
Set-Location -LiteralPath '.\Notebooks\project_notebooks\Training\NTSCC_plus_Full_DIV2K_150ep'
py -m kaggle kernels push -p . --accelerator NvidiaTeslaT4
py -m kaggle kernels status hmmkk20/ntscc-plus-full-div2k-150ep-20261020
```

The metadata requests a private notebook with T4 x2. Dependencies still need to
install against the Kaggle container at launch; the NATTEN wheel is pinned to the
same Torch 2.11/CUDA 12.8 environment as the earlier successful run. The notebook
fails on installation or source integrity errors. No new training or measured
result is claimed before that GPU run executes.

## Session continuation

The notebook normally stops at an epoch boundary before its 9.5-hour session
budget expires. Each epoch atomically saves `latest.pt` with model weights,
both optimizers, GradScaler, random states, progress and history. It does not
restart at epoch 1 when a compatible previous output is attached.

1. Wait for the chunk to finish successfully and publish its output.
2. Copy this notebook folder to a new continuation folder. Give its metadata a
   new matching `id` and `title`, and append the preceding successful notebook's
   `owner/slug` to `kernel_sources` alongside the DIV2K dataset notebook.
3. Submit the continuation with the same code and protocol. Attach only one
   preceding full-data output. Do not attach the old 128-image subset output.
4. Verify `RESUME VERIFIED` and subsequent `TRAIN PROGRESS` logs before counting
   the continuation as active training.

This notebook has not been added to the SQ-GAN automatic continuation monitor.
It will not consume a GPU until explicitly submitted.

## Evaluation and completion

Each completed channel produces a `model.pth`, `completed.json`, and 24-image
Kodak evaluation with PSNR, SSIM, MS-SSIM, complex channel uses and CBR. Aggregate
results go into `summary.csv`. `run_manifest.json` reports completion only when
both channels have trained and been evaluated. Output root:
`/kaggle/working/ntscc_plus_full_div2k_results/`.

The full dataset exposure fixes the subset-training issue. It does not make the
other models' CBR, crop sizes or profiling methods identical. Preserve the old
results as preliminary and keep this fresh run separate until new metrics exist.


## Version 2 numerical fix

Version 1 reached the first training batch, then failed on non-finite entropy auxiliary loss on the second batch. The AMP wrapper scaled the main loss while sharing quantile gradient handling and clipping across the separate main/auxiliary optimizers. Version 2 isolates quantile parameters from the scaled main backward pass, clips only unscaled main-optimizer gradients, clears auxiliary gradients before the FP32 auxiliary backward, and checks auxiliary gradients for finiteness. The author model source, objective and architecture are unchanged. Existing finite-loss assertions remain. This is a diagnosis from code inspection; successful sustained GPU training must still confirm the fix. SQ-GAN code, checkpoint files and session were not modified.
