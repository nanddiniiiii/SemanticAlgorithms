# Model runtime

Run from the repository root with `SharedCode/reproduction` on `PYTHONPATH`:

```powershell
$env:PYTHONPATH = 'SharedCode/reproduction'
python -m model_runtime --model DeepJSCC --stage main --mode preflight --contract SharedCode/reproduction/experiment_contract.json --train-root <DIV2K_train_HR> --val-root <DIV2K_valid_HR> --kodak-root <Kodak24> --output <new-run> --channel awgn
```

`--mode train` starts fresh weights, or continues this runtime's full checkpoint
with `--resume <run>/latest.pt`. `--chunk-seconds 1800` saves and returns without
counting a partial epoch. `--mode evaluate --resume <run>/best.pt` requires a
completed latest checkpoint alongside the selected checkpoint. TensorFlow uses
`latest.pkl` / `best.pkl`. Never load untrusted pickle/checkpoint files.

The source lock is checked byte for byte before import. The runtime disables
upstream bytecode writes. One model runs per interpreter because upstream
releases reuse absolute package names. Existing historical checkpoints cannot
contribute epochs. The project sources are not modified.

## Implemented paths and current evidence

| Model | Architecture and loss | Execution boundary |
|---|---|---|
| DeepJSCC | Audited third-party `2665e0dc`, actual `DeepJSCC(c=8)`, author 255-scale MSE | Real PyTorch trainer and native-image evaluator; third-party provenance must remain visible |
| SwinJSCC | Official `a6d0e6da`, base depths `[2,2,6,2]`, `SwinJSCC_w/o_SAandRA`, C256, author Distortion(MSE) | Real PyTorch trainer and native-image evaluator |
| DeepJSCC_Q | Candidate `da863de5`, full FeatureEncoderAF/Decoder C256, released learned 64-symbol constellation, author SSIM loss | Real PyTorch trainer and evaluator; candidate provenance is not a verified paper reproduction |
| ADJSCC | Official `e5332e95`, unchanged TensorFlow Attention_Encoder/Decoder, 16 real latent channels, 255-scale MSE | Isolated TF2.1 trainer/evaluator; native compiled dependencies and GPU execution remain unverified |
| NTSCC_plus | Official checkerboard2, unchanged author loss and main/auxiliary optimizers | See `ntscc.py`; strict native natten0.14.6 and actual FLIF/digital receiver path |
| SQGAN | Original three stages and complete GAN objectives required | Explicit required-assets gate; no miniature substitute is supplied |

The runtime is implementation, not measured benchmark completion. Local CPU
checks do not establish two-GPU execution. Preflight admits training only after
an actual two-GPU forward, finite backward, applied optimizer step, checkpoint
roundtrip, and observed participation of both devices. Synthetic diagnostic
weights are saved separately and discarded before fresh training. A CPU-only
preparation returns `PREPARATION_PASSED_RUNTIME_UNVERIFIED`, never GPU readiness.

ADJSCC's bundled tensorflow_compression explicitly requires TF2.1. Its packaged
custom operations may be platform specific. `adjscc_tf21_environment.yml` is an
**unexecuted dependency recipe**, not a solved/validated environment. It does not
install modern `tensorflow-compression` as a replacement. Import or native-op
failures stop execution; author convolution/GDN layers are never replaced.

## Protocol adaptations

The common channel is explicitly a benchmark transport extension. Every image
uses unit mean complex symbol energy and independent circular complex noise
with variance `10**(-snr_db/10)`. Rayleigh draws one CN(0,1) gain per whole image,
uses perfect receiver CSI and exact zero forcing, and never clips deep fades.
Features are scaled back to decoder units after transport. AWGN has no receiver
CSI requirement. Both coordinate count and actual complex uses are measured.

DeepJSCC preserves the released ImageNet preprocessing (tensor resize to128),
Adam lr1e-4/weight decay5e-4 and ReduceLROnPlateau factor0.5/patience15 from
`train.py`; DIV2K replaces ImageNet. Swin preserves random256 crops, Adam1e-4,
and its released MSE. Candidate Q uses its full ImageNet architecture and
random256 crops, Adam1e-4 and multiplicative0.8 schedule on the adapted
half-patience validation interval. Candidate Q's upstream quantizer advances its
annealing buffer twice per training call and even during eval; author behavior
is preserved during training, while validation/evaluation restore buffers to
prevent validation or trial order changing future state.

ADJSCC uses deterministic random128 patches as a documented replacement for
the author's already extracted ImageNet128 records, original Adam1e-4 and no
invented scheduler. TF noise is stateless, keyed by saved global step and replica;
validation uses independent fixed per-image keys. All Adam variables/iterations,
model variables, Python/NumPy RNG, order, cursor, history and early stop state
are atomically persisted. Its current FLOP measurement remains unavailable.

Training uses all800 distinct sources once per completed epoch and all100
validation sources; batch16 is global across two devices. Training objectives
are weighted by replica image counts before reduction. Minimum50, maximum150,
patience20 and relative delta0.001 govern complete validation epochs. Kodak is
never used for checkpoint selection. Checkpoints include every optimizer,
scheduler, RNG and sampler cursor and are saved at safe step boundaries at
most300 seconds apart plus chunk/epoch boundaries (a single longer step cannot
be interrupted safely). Incomplete epochs resume; they never count as complete.

Kodak evaluation uses the complete original RGB image, model alignment padding
only, then removes padding for all metrics. It never tiles or resizes Kodak.
Every one of24 images is evaluated at0/5/10/15/20dB for10 independent trials.
CBR divides all actual complex uses by `3*original_height*original_width`.
LPIPS uses real pretrained AlexNet assets; missing packages/assets fail execution.
Rows record timings, geometry, source revision and full stream ledgers.

`torch.profiler` measures actual forward/backward/optimizer paths with FMA=2;
unaccounted operations are listed and counts are explicitly **partial**. GPU
worker thread coverage can also be partial. No estimated backward multiplier is
presented as measured FLOPs. Partial coverage prevents total-compute ranking.

Outputs are `preflight.json`, `progress.json`, `latest.pt`, `best.pt`, optional
`training_profile.json`, `profiling.json`, `measured_results.jsonl`, and
`measurement_receipt.json`. Receipts expose incomplete provenance/coverage;
delivery must derive eligibility from actual coverage and validation evidence.
