# Inference guide

This guide documents the stable public inference path. The repository contains
only the minimal checkpoint-compatible model core; the full research workspace,
training scripts, and manuscript sources are intentionally kept separate.

## Hardware and software

- Python 3.10 or newer
- PyTorch 2.1 or newer
- NVIDIA CUDA GPU recommended
- `fp16` is the default CUDA precision; use `bf16` on supported accelerators or `fp32` for debugging

The diffusion transformer and DINOv2 encoder make CPU inference possible in principle but impractically slow for normal use.

## Required model files

| File | Purpose | Default path |
|---|---|---|
| DiT-JSCC 256 | 256 × 256 transmitter and conditional DiT | `checkpoints/dit_jscc_256.pth` |
| DiT-JSCC 512 | 512 × 512 transmitter and conditional DiT | `checkpoints/dit_jscc_512.pth` |
| Stable Diffusion VAE | Decode the sampled latent | `checkpoints/autoencoder_kl.pth` |
| DINOv2 ViT-B/14 | Frozen semantic feature extractor | Downloaded by `torch.hub` |

Custom checkpoint paths can be passed through the CLI or Python constructor.

## Image preprocessing

Inputs are EXIF-corrected, converted to RGB, then center-cropped and resized with Lanczos resampling to the selected square resolution. The tensor is mapped from `[0, 255]` to `[-1, 1]`, matching the original evaluation pipeline.

## Channel controls

`channel_type` selects AWGN or Rayleigh fading simulation. `snr_db` controls
the channel signal-to-noise ratio; lower values add stronger channel noise.

The paper trains separate checkpoints for each channel/SNR setting. Changing
`snr_db` in the public wrapper changes the simulated test channel, not the
training condition encoded in the loaded checkpoint. Use the checkpoint that
matches the paper setting when reproducing reported numbers.

The two rates select active real-valued feature dimensions per token:

| Resolution | Semantic rate | Detail rate |
|---|---|---|
| 256 | 4 to 28, step 2 | 0 to 24, step 2 |
| 512 | 16 to 112, step 8 | 0 to 96, step 8 |

Both branches use a 16 × 16 token grid and pairs of real values form complex symbols. Therefore:

```text
complex symbols = 16 × 16 × (semantic_rate + detail_rate) / 2
CBR = complex symbols / (3 × resolution²)
```

Examples:

- 256 px with rates 8 + 8: CBR = 1/96
- 256 px with rates 8 + 24: CBR = 1/48
- 512 px with rates 32 + 32: CBR = 1/96

## Samplers

- `dpm_solver`: recommended interactive default; 20 steps is a useful starting point.
- `euler_ode`: deterministic probability-flow ODE for a fixed initial seed and channel draw.
- `euler_sde`: stochastic reverse SDE used by the original paper scripts; the paper configuration uses up to 200 steps.

All samplers retain stochastic channel simulation. Fix the seed to reproduce the complete run, including initial latent and channel noise.

## KC-inspired adaptive allocation

`dit_jscc.bandwidth` implements the paper's textual-complexity score and
bandwidth equation without forcing BLIP-2, NLTK, or spaCy into the base
inference environment. Supply a caption and min-max statistics measured on the
evaluation set:

```python
from dit_jscc import MinMaxStats, allocate_bandwidth, analyze_caption, kc_score
from dit_jscc.pipeline import SUPPORTED_RATES

features = analyze_caption(blip2_caption)
stats = MinMaxStats(
    word_count=(wc_min, wc_max),
    lexical_diversity=(ld_min, ld_max),
    syntactic_complexity=(sc_min, sc_max),
)
score = kc_score(features, stats, weights=(delta_wc, delta_ld, delta_sc))
rates = allocate_bandwidth(
    score,
    mean_score=mean_kc_score,
    baseline_semantic_rate=8,
    total_rate=16,
    eta=eta,
    semantic_candidates=SUPPORTED_RATES[256]["semantic"],
    detail_candidates=SUPPORTED_RATES[256]["detail"],
)
```

Min-max normalization is dataset-level in the paper. The web demo therefore
does not invent single-image normalization statistics; it exposes the final
semantic and detail rates directly.

## CLI examples

Single image:

```bash
python infer.py --input image.png --output outputs/run1 --resolution 256
```

Batch directory with the paper-style sampler:

```bash
python infer.py \
  --input images/ \
  --output outputs/euler_sde \
  --resolution 512 \
  --channel awgn \
  --snr 0 \
  --semantic-rate 32 \
  --detail-rate 32 \
  --sampler euler_sde \
  --steps 200 \
  --guidance-scale 2 \
  --seed 1000
```

Use explicit model paths:

```bash
python infer.py \
  --input image.png \
  --checkpoint /models/dit_jscc_256.pth \
  --autoencoder /models/autoencoder_kl.pth
```

## Offline DINOv2

The default loader uses the official `facebookresearch/dinov2` torch hub repository. In an offline environment:

```bash
git clone https://github.com/facebookresearch/dinov2.git /models/dinov2
export DINOV2_REPO=/models/dinov2
```

The pretrained DINOv2 weights must already be present in the local torch cache if the machine has no network access.

## Troubleshooting

### A checkpoint is missing

Run `python infer.py --check --resolution 256`. The command prints every resolved path without loading weights.

### Checkpoint/model mismatch

The public loader rejects a checkpoint when fewer than 80% of model keys match. Confirm that the selected resolution and released checkpoint agree, and avoid using a training optimizer checkpoint in place of `nnet.pth` unless it contains a recognized `state_dict`, `model`, `nnet`, or `ema` entry.

### CUDA out of memory

- close other GPU workloads;
- use the 256 px model;
- keep `precision=fp16`;
- run one request at a time (the web demo already enforces this);
- restart the process after switching large checkpoints if memory remains fragmented.

### DINOv2 cannot download

Check network access or use the `DINOV2_REPO` offline setup above. This project does not silently replace DINOv2 with a different encoder because doing so would invalidate checkpoint behavior.

### Results differ from paper figures

Verify the checkpoint, channel type/SNR, exact semantic and detail rates, sampler, step count, CFG scale, preprocessing, and seed. Paper tables use a fixed ImageNet subset and model checkpoints trained for the stated channel setting.
