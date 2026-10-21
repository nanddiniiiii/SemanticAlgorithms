# NTSCC+ runtime

The adapter targets the unchanged `wsxtyrdd/NTSCC_plus` release pinned at
`765c6f6c11cf3bbe249a26b624dbaf47a4f0676d`. It imports the released
`net.ntscc.NTSCC_plus` checkerboard2 architecture. No network or layer source is
rewritten. It subclasses only `feature_pass_channel` to apply the documented
common transport. This is an explicit benchmark adaptation, not a claim of an
identical original-paper experiment.

## Environment and execution boundary

The original compiled **natten 0.14.6** CUDA extension is mandatory. A modern
NATTEN API shim, fallback attention implementation, CPU model replacement, or
reduced architecture is prohibited. Preflight checks the installed version and
compiled extension origin. Import/ABI/kernel failures remain real failures;
runtime diagnostics must execute the original model forward, backward, both
optimizer updates and checkpoint restore on the target GPUs before claiming
GPU compatibility. The common engine requires two CUDA GPUs for training.

`compressai`, `timm`, `PyYAML`, and the pinned author's compatible PyTorch/CUDA
stack must be installed. Local CPU checks exercise transport, serialization
framing, loss extraction and optimizer state primitives. They do not validate
the full NTSCC+ architecture or prove GPU runtime compatibility. No local CUDA
training, benchmark reconstruction, FLOP total or quality result is claimed.

## Objective and optimization

`released_loss` extracts and compiles the actual `RateDistortionLoss` class from
the pinned `main.py` AST. It avoids executing the author's command-line and W&B
initialization. The released `config/ntscc.yaml` selects lambda **0.72**, eta
**0.2**, and gradient clipping norm **1.0**. The rate choices and optimizer
learning rates come from that source's parser defaults rather than a shortened
replacement configuration.

The main objective is the released
`lambda * 255^2 * (MSE_JSCC + MSE_NTC) + bpp_y + bpp_z`.
Main Adam updates all trainable parameters except names ending `.quantiles`,
with learning rate `1e-4`. Auxiliary Adam updates the entropy quantiles with
learning rate `1e-3`. Each step zeros both optimizers, backpropagates the main
objective, clips the native model gradients to norm 1, updates main Adam,
zeros auxiliary gradients, backpropagates `core.ntc.aux_loss()`, and updates
auxiliary Adam. The released main `MultiStepLR([600000], gamma=0.1)` advances
once per batch. A shorter benchmark does not reinterpret its milestones as
epochs.

The generic engine saves both optimizer states, scheduler state, full adapter
state, RNG state, shuffled source order, incomplete-epoch cursor/coverage and
early-stopping state. Only complete coverage of all **800** training sources
counts as an epoch; **100** validation sources are separate. The approved
global batch is **16**, training SNR is **10 dB**, and stopping is minimum
**50**, maximum **150**, patience **20**, relative delta **0.001**. Those are
benchmark settings, not original-paper training claims.

The author's loader directly reads prepared image files. Deterministic
256-by-256 crops of full DIV2K images are therefore a disclosed external
dataset adaptation. Validation uses fixed center crops and fixed RNG seeds.
During optimization the decoder receives perfect rate indexes. That is a
training assumption; evaluation below sends actual bytes through a noisy PHY.

Each replica computes the native scalar loss before DataParallel gathers it.
The gathered dictionary contains tensor loss sums and batch counts. The
global objective weights replicas by their actual source counts; observed
channel uses are **summed**, never averaged as a scalar rate estimate.

## Actual channel coordinates

The native rate-adaption encoder produces the mask. The adapter transmits
only values selected by this actual mask, not `rate * H * W` or dense masked
zero arrays. The author places anchors for every image first and nonanchors
second along the batch dimension. `CheckerboardTransport` joins rows `i` and
`i+B` into one original-image codeword before power normalization and fading.
One complex fade is shared by both checkerboards and the rate-index stream.

Only active real coordinates are packed into I/Q values. If their count is
odd, one explicit zero real coordinate is transmitted and counted. The observed
complex-use total is therefore `(active_real_coordinates + padding)/2` per
image. Inactive coordinates remain zero at the decoder and do not influence
normalization or active-symbol gradients. A native empty or nonbinary mask
fails explicitly rather than inventing a rate or changing the native mask.

The common channel uses per-image unit mean complex energy, circular AWGN,
and one independent complex Rayleigh coefficient per complete image/codeword,
with perfect receiver CSI and zero forcing. Deep fades are not clipped.
Decoder scale is restored after transmission. The extension differs from the
released author's dense/batch-normalized AWGN channel; results must be labelled
as the matched transport experiment.

## Rate-index serialization and receiver

The released JSCC `decode(s_hat, indexes)` requires **rate indexes**. It does
not consume hyperprior `z`. In this architecture `z` supports transmitter rate
estimation and the auxiliary NTC branch. Counting a fictional transmitted
hyperprior would misdescribe this receiver.

Evaluation requires a real encoder/decoder executable configured as
`model_assets.NTSCC_plus.flif_executable` in the experiment contract, or a real
`flif` executable on PATH. `FLIFRateIndexCodec` writes an actual grayscale PNG
of the native rate indexes and invokes:

```text
flif --overwrite -e <unique-directory>/indexes.png <unique-directory>/indexes.flif
flif --overwrite -d <another-unique-directory>/received.flif <another-unique-directory>/received.png
```

Subprocess failures, timeouts, missing/empty outputs, different decoded indexes,
or wrong geometry fail explicitly. Every invocation uses temporary unique file
paths and checked return codes. No shared `output.png`, stale `.flif`, PNG
fallback, estimated entropy size or ideal-capacity conversion is used. The
actual executable SHA256 is recorded. The encoder verifies a local lossless
roundtrip before transmission; this validation cost is included in encoder
time.

The FLIF bytes are framed with magic/version, rows, columns, payload length and
CRC32. The actual packet bytes enter `digital_transport.transmit_bytes`, which
adds its own length/CRC32 framing, repetition-3 coding and unit-energy QPSK.
All header, CRC, coding and modulation padding uses are counted. This is an
explicit benchmark PHY, **not** the paper's commented 2/3 LDPC plus 16QAM
example and not a capacity-achieving code. The digital stream reuses the exact
analog image fade and uses independent noise.

Only decoded receiver packet indexes reach `core.decode`. A PHY CRC failure or
invalid received FLIF packet yields a documented **constant RGB 0.5 erasure
reconstruction**. Every failed trial remains in the metric denominator and
retains its complete transmitted-use count. The encoder's original indexes
are never used to recover a failed packet. The trial ledger reports CRC,
side-information outage, exact analog/digital uses, serialized bits, padding,
codec identity, shared fade and the receiver error policy.

The complete CBR is `(observed_analog_uses + actual_digital_uses) /
(3 * original_height * original_width)`, including model padding and sideinfo.
Native full-image Kodak24 evaluation is 5 SNR points `[0,5,10,15,20]` and 10
independent trials per image. It requires a completed fresh run and the
validation-selected checkpoint. Source integrity, real GPU execution, actual
FLIF availability and measured outputs remain admission requirements.

## Validation status

`tests/model_runtime/test_ntscc.py` exercises exact native loss extraction,
released operating-point parsing, observed-mask packing, joined checkerboard
identity, odd-coordinate padding, differentiability, RNG replay, separate
optimizer state, FLIF subprocess path/framing/error behavior, real digital
transport integration, received-index use, shared fading and explicit erasure.
The FLIF subprocess tests are mocked and are labelled as such; they are not a
claim that native FLIF has executed. Full architecture/CUDA validation and a
real FLIF roundtrip must run on the compatible target environment.
