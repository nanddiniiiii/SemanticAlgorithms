# DeepJSCC-Q candidate-code audit

Paper: https://arxiv.org/abs/2206.08100.
Pinned candidate: `r0ymanesco/deepjscc_stream@da863de5e3acb4fbb307c0cf8dfe830303e99aa7`.
The repository describes a video-streaming project with a DeepJSCC-Q trainer
retained from prior work. Its architecture/protocol must still be matched to
the particular paper experiment before it is called a reproduction.

## Verified candidate behavior

`scripts/check_deepjscc_q_quantizers.py` executes the original quantizer class
definitions, without modifying source and without importing unrelated modules.
On a small synthetic four-point constellation:

- `SoftHardQuantize.eval()` produces nonfinite output. It sets sigma to infinity,
  then computes softmax of `-sigma * distance`; this is not a stable numerical
  implementation of the hard nearest-neighbor limit.
- `LearnedQuantize.eval()` produces finite output but increments its annealing
  counter once. The unconditional `_anneal()` call overrides its eval branch.
- `LearnedQuantize.train()` increments the counter twice per forward.

These are observations about the pinned candidate code, not a claim that the
published algorithm necessarily used this buggy path. A corrected hard argmin
or frozen evaluation schedule must be named as a documented implementation fix
and compared against the intended paper equations.

## Parameters requiring a paper match

The supplied YAML uses CIFAR-10, SSIM loss, 600 epochs, batch 6, learning rate
0.0001, 64-point learned modulation and fixed 20 dB. It also sets `c_out=256`.
For its reduced encoder, a 32x32 image becomes a 256x8x8 real latent: 8192
complex values, giving CBR 8192/(3*32*32)=2.6667. This configuration is not a
low-bandwidth operating point and must not be blindly used in our comparison.

FeatureEncoderAF also uses CompressAI residual/attention blocks and SNR attention;
the modem can choose fixed QAM or learned constellations. Matching only the
trainer's name is insufficient. Dependency setup, correct rate selection,
training/evaluation annealing and a complete native GPU pilot remain pending.
