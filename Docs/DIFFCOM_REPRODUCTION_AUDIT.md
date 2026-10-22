# DiffCom integration audit

Source: `wsxtyrdd/diffcom@a8cc4d63304f5bb514d69576e689ff2412510161`.
Variant: HiFi-DiffCom, ImageNet 256x256 unconditional prior, ADJSCC C=2,
10 dB AWGN, native adaptive start and nominal 1000-step schedule.

## Native reconstruction route

For a 1x3x256x256 image, the released ADJSCC encoder produces 1x2x64x64:
8192 real values, paired into 4096 complex channel uses. Unit complex energy
normalization divides real coordinates by `sqrt(2*mean(latent**2))`.
Independent real AWGN coordinates have variance `0.5*10**(-SNR_dB/10)`.
The ADJSCC decoder produces an initial MSE-oriented reconstruction.

The received measurement then constrains reverse diffusion. Native conditioning
compares predicted transmitted latents with the received latents and, for the
HiFi variant, compares the re-encoded/re-decoded image with the initial image.
The denoiser's input gradient steers the posterior sample; its weights are not
updated. At 10 dB, the native adaptive schedule in this configuration ran 235
denoiser evaluations. The output follows `main_diffcom.py`: the final posterior
state, not the intermediate clean-image estimate.

The native attention checkpoint backward explicitly differentiates parameter
inputs. Turning off all UNet gradient flags breaks that implementation even
though sampling is inference. Our adapter retains those native flags and uses
no optimizer. CPU prior loading alone did not reveal this; the full GPU pilot did.

## Measured runtime evidence

Job: `satyapaladugu/diffcom-full-posterior-pilot`, version 4.
One Kodak image was explicitly center-cropped to 256x256 for this runtime check.
It completed on T4 with finite correctly shaped output, 235 denoiser evaluations,
4096 complex uses, 284.10 seconds and 8,703,745,536 peak allocated GPU bytes.
See `Docs/validation/diffcom_gpu_pilot.json`. No quality comparison or lossless
claim follows from this one-image check.

## Declared differences and limits

- Lightweight AWGN operator replaces the larger native channel wrapper.
- Identity interleaving replaces the native permutation. IID AWGN is invariant
  under a consistent coordinate permutation, but exact same-RNG noise realization
  equivalence has not been checked against the complete native operator.
- Schedule arrays are ported from the native `NoiseSchedule`; controller, losses,
  diffusion helpers and UNet come from pinned source.
- Receiver normalization power is assumed known; 32 side bits are counted
  separately. Protection, pilots, framing and synchronization are not implemented.
- The ImageNet prior is an author-supported alternative to the YAML's active
  FFHQ default. This crop is not the paper's complete ImageNet evaluation set.
- Modern Torch 2.11+cu128 differs from the author's environment.

The main-stream CBR for this crop is 4096/(3*256*256)=1/48. Side information and
radio overhead must be added before claiming a complete transmission budget.
The shared prior is about 2.21 GB on disk and is not retransmitted per image;
its distribution/synchronization cost is a separate deployment assumption.

Default CLI profiling uses one complete diffusion execution with no warmup,
reports it as a single observation and leaves p95 unset. Full inference FLOPs
remain unmeasured: denoiser-only MACs would omit the posterior input-gradient
work and cannot stand in for the full algorithm's cost.
