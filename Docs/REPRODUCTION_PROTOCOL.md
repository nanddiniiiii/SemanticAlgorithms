# Paper reproduction requirements

Author fidelity takes priority over the earlier blanket 150-epoch/four-hour plan.
No current adapter is certified as a complete paper reproduction. A successful
strict checkpoint load or GPU pilot establishes narrower runtime evidence only.
The `baseline` variant means the reference for our experiment, not automatically
an exact reproduction. Every config records `reproduction_status` explicitly;
the default is `unverified`.

Before long training, record the paper experiment and pinned author source for:

1. Dataset version, split, sample coverage, augmentation, resizing/cropping,
   input range and color space.
2. Input/output and intermediate tensor shapes, architecture variant, latent
   dimensions, complex packing, normalization, and operating-point definition.
3. AWGN/fading equations, variance convention, CSI assumptions, SNR distribution,
   constellation and rate, including all side information.
4. Loss terms and coefficients, optimizers and parameter groups, scheduler,
   effective batch size, precision, initialization, stages and training duration.
5. Evaluation dataset, SNR/rate grid, stochastic trials, clipping/quantization,
   PSNR/SSIM/LPIPS definitions and averaging convention.
6. Exact source locations supporting each setting; unresolved paper/code
   disagreements remain visible instead of being silently resolved.

Compare an instrumented forward against the pinned native forward on identical
inputs, weights and noise where feasible. Record tolerances and tensor shapes.
Check optimizer parameter coverage and one native objective step before training.
Then validate the real dataset loader and the target GPU environment.

## Known deviations requiring resolution

| Integration | Current difference from native paper protocol | Treatment |
|---|---|---|
| JCM | Additional 45k/5k holdout; shared training scheduler/limits require native audit | Adapted until checked; checkpoint evaluation remains separate |
| ADJSCC | DiffCom PyTorch lineage rather than original TensorFlow; per-image normalization; shared scheduler | Adapted implementation, not original-paper reproduction |
| SwinJSCC | Retained patched source and strict checkpoint environment; shared fine-tuning runner | Retained inference; training protocol must be audited |
| Shared semantic-aware codebook | Corrected nearest-prototype loop; shared schedule; omitted native periodic stage | Adapted experimental implementation |
| SQ-GAN | Custom SQ01 packet, fixed mask operating point; training augmentation/trainer incomplete | Inference integration only; packet is our extension |
| Complementary hybrid | Our DCT residual reference | Proposed method, never an author reproduction |
| DiffCom | Native prior loaded; full posterior pipeline still needs end-to-end verification | Unverified integration |

Do not launch a shortened/adapted job as a paper reproduction. Time-limited runs
must state completed steps, images and epochs and remain separate from published
numbers. Pretrained inference does not reproduce training. Improvements require
a matched verified baseline and must not be inferred from unmatched datasets,
rates, SNRs, compute budgets or metric implementations.
