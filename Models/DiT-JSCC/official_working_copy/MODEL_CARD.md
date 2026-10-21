---
model_name: DiT-JSCC
pipeline_tag: image-to-image
library_name: pytorch
license: apache-2.0
---

# DiT-JSCC model card

## Model description

DiT-JSCC is a generative joint source-channel coding system for robust image transmission under low bandwidth and low signal-to-noise ratio. It combines a DINOv2-based semantic branch, a SwinJSCC detail branch, a differentiable wireless channel, and a coarse-to-fine conditional diffusion transformer receiver.

## Intended use

- Research on generative JSCC and semantic communications
- Controlled evaluation of image reconstruction under AWGN or separately trained Rayleigh channel settings
- Analysis of semantic/detail bandwidth allocation
- Comparison of perceptual, semantic, and realism metrics in extreme channel regimes

The released interface is a research prototype, not a production communication codec.

## Training data

The paper reports training on the ImageNet training set at 256 × 256 and 512 × 512. Users are responsible for obtaining ImageNet under its applicable terms. DINOv2 and Stable Diffusion autoencoder weights remain governed by their respective licenses.

## Evaluation

The paper evaluates a held-out subset of 100 ImageNet validation images using LPIPS, DISTS, CLIP, DreamSim, DINOv2 similarity, and FID. Results cover CBRs from 1/384 to 1/48 and low-SNR AWGN/Rayleigh channels. Consult the manuscript for complete baseline and protocol details.

## Limitations

- Generative reconstruction is not bit-exact and may synthesize plausible details absent from the source.
- Performance outside the trained channel, SNR, resolution, rate, or data distribution is not guaranteed.
- Semantic metrics can conceal errors in small text, faces, counting, and fine spatial relations.
- The diffusion decoder is computationally expensive compared with conventional codecs.
- ImageNet-trained checkpoints may inherit dataset and foundation-model biases.

## Responsible use

Do not use reconstructed imagery as evidentiary or diagnostic ground truth. For safety-critical communication, pair generative outputs with integrity checks and task-specific validation.
