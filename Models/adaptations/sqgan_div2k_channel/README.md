# SQ-GAN DIV2K adaptation modules

These modules are separate from the author repository at `references/SQ-GAN`.
They are scaffolding for a paper-based DIV2K/noisy-channel adaptation and do
not constitute trained results.

- `generate_pseudo_labels.py` uses the Cityscapes-finetuned NVIDIA SegFormer B0
  to create 19-class pseudo labels for DIV2K and Kodak. It records the resolved
  model revision in metadata. [Model card](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024).
- `dataset.py` reads image/mask pairs, shares geometric transforms, maps labels
  to the 20-channel SQ-GAN convention, and returns the dictionary expected by
  the author Lightning module.
- `channel.py` adds differentiable normalized AWGN or flat Rayleigh noise to
  256-D quantized latent vectors (128 complex uses per selected token).
- `quantizer.py` wraps the author VQ module and inserts that channel after the
  VQ straight-through operation.

Before training, integrate these components into an isolated Kaggle notebook,
fix/verify Python 3.13 compatibility for the author's PyTorch Lightning 1.5.6
trainer, check all three author training modes, and profile one epoch on T4 x2.
The official repository expects three sequential phases (`ssm`, `img`, `all`),
so 150 epochs per phase and per channel means 900 phase-epochs. No SQ-GAN run
has been launched yet.
