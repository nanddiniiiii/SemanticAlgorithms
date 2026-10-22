# SQ-GAN reproduction audit

Pinned source: `frapez1/SQ-GAN@0e5cdd7c0240b89749dad684645c53f783ac24cc`.
The ten released components are strict-loaded. Native forward equivalence and
independent packet decoding passed synthetic CPU checks; these are not quality
results or training reproduction.

## Input and evaluation contract

`data/cityscapes.py` uses paired ground-truth Cityscapes trainIds as semantic
encoder input. RGB is 256x512 with ImageNet normalization. The semantic tensor is
20x256x512: 19 evaluated classes and an ignored class. Raw labelIds are not
trainIds: use explicit `model_options.label_format=label_ids` to convert them via
the official Cityscapes mapping before one-hot encoding. Never use color or
instance maps. The official mapping is documented at
https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py.

`sample.py` runs InternImage on reconstructed RGB for semantic-quality evaluation;
it does not replace the paired semantic encoder input in that native sampling
path. InternImage mIoU remains unimplemented in our evaluation, so no semantic
retention comparison is certified yet.

## Rate accounting

The native sampling script counts each retained codebook index as 10 bits and
each branch mask as 512 bits. For retained fractions rx and rs, source payload is
`10*floor(512*rx) + 10*floor(512*rs) + 1024` bits, divided by 256*512 for bpp.
At rx=rs=0.25 this is 3584 bits, or 0.02734375 bpp.

Our independent SQ01 packet additionally has a 12-byte header and 4-byte CRC:
464 bytes at this operating point, or 0.0283203125 bpp. Those extra bits are our
transport extension. Neither source count includes modulation, channel coding,
radio metadata or pilots. Do not convert these bits into AWGN channel uses
without specifying that missing transmission chain.

## Training requirements

The author README specifies semantic-branch, image-branch and final joint stages,
each with a 150-epoch command. One generic 150-epoch MSE run is not equivalent.
Native training also includes semantic augmentation, traffic-light/sign region
enhancement, LPIPS, VQ commitment, adversarial losses and separate optimizers.
The current generic Cityscapes loader does not reproduce that augmentation.
Training remains disabled until those native stages and dependencies are verified.
The YAML batch-size comment and supplied inference setting must not be assumed
to establish the paper training batch size.

## Adapted evaluation and preprocessing correction

Version 6 evaluated all 500 validation images with the independent packet
adapter, measuring 21.2663926 dB under conventional RGB clipping and rounding.
On its first input, the packet decoder differed from native forward by at most
1.4305115e-6, when supplied identical tensors. That did not establish native
dataset preprocessing agreement.

The subsequent loader cross-check found 5040 unequal one-hot elements out of
2,621,440: our PIL nearest-neighbor resizing differed from the author's tensor
nearest-neighbor label resizing. RGB preprocessing agreed within tolerance.
The shared loader now follows the native label sequence: trainId conversion,
ToTensor scaling, tensor nearest resize, multiplication by 255, integer cast,
then ignore-class mapping. Native comparison is required before new evaluation.
Version 6 remains an adapted-preprocessing result, not a paper baseline.

The author's saved-image code also uses a normalized-space clamp with rounded
channel-specific bounds, inverse normalization and NumPy uint8 truncation.
This differs from conventional [0,1] clipping and rounding. New evaluation
records both postprocessing conventions instead of conflating their PSNR values.

## Completed corrected evaluation (version 8)

The real-image preprocessing check passed before the full 500-image validation.
Native forward and native normalized-space clamp/uint8 truncation were used.
Mean fixed-range saved-image PSNR: 21.240841 dB; conventional clip/round PSNR:
21.256595 dB. Auto-range PSNR formula: 21.001077 dB, with original library-version
equivalence unverified. Runtime was 1137.56 seconds for the entire evaluation,
including preprocessing and metrics; this is not isolated model latency.
Results do not establish training reproduction, channel robustness, LPIPS or mIoU.
Aggregate and preprocessing receipts are stored in Docs/validation.
