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
