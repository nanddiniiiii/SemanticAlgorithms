# SQ-GAN speed optimization: staged measurements

## State

The production SQ-GAN session and its continuation source snapshots are unchanged.
The source architecture, losses, per-GPU batch size 1, DDP across two T4 GPUs,
256x512 geometry, 800-image training pool, 100-image validation pool, six stages,
and 150 full epochs per stage remain the contract. Completed training checkpoints
are never overwritten by benchmark checkpoints. No Git push was performed.

The requested GPU benchmark submission was rejected by Kaggle with:
`Maximum batch GPU session count of 2 reached.` Both SQ-GAN and NTSCC_plus are
running. No session was stopped to create capacity.

## Active CPU data experiment

https://www.kaggle.com/code/hmmkk20/sqgan-cpu-data-benchmark-20261020/log

This private CPU notebook runs separately, with GPU disabled. It compares:

1. Original image decoding/resizing and zero loader workers.
2. Cached resized RGB/mask PNGs and zero workers.
3. The same cache and two workers.

Every measured pass checks 800 unique filenames. Resized images are losslessly
stored before random crop/flip, and 20 fixed-seed examples must produce exactly
equal image, label and semantic tensors. Cache building time is recorded separately.
Its data-loading timing is NOT GPU training throughput or model quality evidence.

## GPU experiment, waiting for capacity

Notebook folder: `Notebooks/project_notebooks/Training/SQGAN_speed_benchmark/`.

`SharedCode/sqgan/dispatch_speed_benchmark.py` waits locally for a GPU slot for up
to 24 hours. It never cancels SQ-GAN or NTSCC_plus. On successful submission it
monitors the benchmark and downloads its JSON report into
`Results/SQGAN/speed_benchmark/gpu_output/`. Its status and logs are in the same
results folder. The local computer must remain awake and connected for dispatch.
Creating `Results/SQGAN/speed_benchmark/STOP` stops this dispatcher; it does not
cancel any Kaggle run.

Five candidates: baseline FP32; pinned-memory FP32; cached/pinned FP32;
two-worker cached/pinned FP32; cached/pinned FP16. Each begins from the same
immutable successful 98-epoch checkpoint, restores both optimizers and schedulers,
and runs 40 batches with ten warmup batches excluded. Both GPU ranks produce
compute/input-transfer timing and peak allocated-memory reports. nvidia-smi is
sampled for GPU utilization and memory usage. Loss and final-parameter finiteness
are checked. The original checkpoint SHA256 must remain unchanged.

The benchmark does not access the live training filesystem or count partial
benchmark epochs as training. It does not reproduce full validation or checkpoint
serialization overhead; its speedup is a measured steady-state batch estimate.

## Adoption requirements

No optimization has been applied to production yet and no speedup is claimed.

- The FP32 baseline must pass and candidates must use the same full dataset,
  geometry and batch size.
- Consider caching first if tensor equality and measured throughput support it.
- Workers must pass an entire 800-image DDP epoch, coverage and checkpoint/resume
  checks before adoption. Worker scheduling can change augmentation RNG sequences.
- FP16 needs a full-epoch stability check, finite gradients, scaler-step checks,
  and reconstruction/validation comparison. A short finite probe is insufficient
  evidence of quality equivalence. Never claim bitwise-identical numerics.
- Apply accepted changes only to a future continuation using the latest successful
  production output, preserving model, optimizer and scheduler state. Record the
  precise epoch of transition and settings. Keep a fallback FP32 snapshot.

These experiments are adaptations around the author source. They do not turn
DIV2K/pseudo-label/noisy-channel training into an exact Cityscapes paper reproduction,
and matching image exposure does not itself match CBR across other models.


## Executed CPU results

All three passes processed 800 unique images. Twenty fixed-seed image/label/semantic comparisons were exactly equal. Original workers0: 32.9679 seconds; cached workers0: 32.2135 seconds; cached workers2: 21.6722 seconds. Cache construction cost 37.0083 seconds. Two workers plus caching achieved 1.52x CPU data-pipeline throughput in this single run, not 1.52x end-to-end GPU training speed. No production settings have changed; full DDP and GPU measurements remain pending available capacity.
