# SQ-GAN DIV2K / Kodak channel adaptation

Status: design grounded in the cloned author repository; no SQ-GAN training has been launched yet.

## What the author implementation expects

- Author repository: `references/SQ-GAN` (public clone, source unchanged).
- `config/sqgan_cityscapes.yml` hardcodes 256px input, 20-channel one-hot semantic labels (19 classes plus null), Cityscapes loaders, batch size 1, 19-class label-conditioned maskers, and 1024-entry vector quantizers.
- `README.md` prescribes three distinct phases: semantic path (`ssm`), image path (`img`), then joint (`all`), each shown with `--max_epochs 150`. Its evaluation relies on InternImage-generated semantic maps and Cityscapes-trained weights.
- Therefore a DIV2K/Kodak/AWGN/Rayleigh experiment is an adaptation of the official method, not the authors' original protocol. Calling it an exact paper reproduction would be inaccurate.

## Candidate DIV2K adaptation

1. Keep the author network, 256px input, vector-quantizer size, masked-token mechanism, and three-phase schedule. Implement a separate DIV2K image dataset and produce a 19-class Cityscapes-style pseudo-label map for each image using one fixed, publicly pretrained Cityscapes segmenter; reserve class 19 as null. Save labels once and record segmenter name/revision/checksum. This avoids running segmentation inference in every training epoch.
2. Add a configurable channel layer immediately after VQ lookup and before demasking/decoding. The candidate experiment sends the quantized feature vectors as real-valued baseband payload and adds power-normalized AWGN or Rayleigh noise at the target SNR, with the same stochastic channel draw used in the matching training and validation pass. This preserves gradients through the VQ straight-through estimator.
3. Report channel uses as `selected_tokens * codebook_dim / 2` complex uses per image when two real latent coordinates form one complex use. Report the resulting rate as complex uses / (3*H*W), and separately report the source codec's 10-bit code-index rate plus mask/position side information if measuring serialized digital payload. Do not conflate the analog latent-channel rate with the original SQ-GAN digital-code rate.
4. Train/evaluate AWGN and Rayleigh separately, use the 800 DIV2K training pool and all 24 Kodak images, record 150 epochs per prescribed phase, per-epoch progress, optimizer state, checkpoints, SSIM/PSNR/MS-SSIM, channel uses, and train/inference FLOPs.

## Feasibility gate before a full run

- Verify pseudo-label shape/range and image-label geometric alignment on several DIV2K samples.
- Verify a forward/backward batch for all three modes and both channels, including AWGN/Rayleigh power normalization and finite gradients.
- Measure one full epoch on Kaggle T4 x2 before committing to the full multi-stage run. The official protocol implies three 150-epoch phases per channel (900 phase-epochs total); do not silently relabel that as 150 total epochs.
- The candidate latent-channel insertion changes the original digital transmission path. If a faithful digital bitstream/channel implementation is required instead, index bits, token positions, mask metadata, modulation, and channel coding must be serialized and trained/evaluated explicitly; clean SQ-GAN alone does not provide that noisy-link path.

## Decision record

The analog VQ-feature channel above is a proposed compatibility adaptation, not yet implemented or measured. No model result should be published until the user-facing notebook clearly labels this boundary and a Kaggle run produces verifiable outputs.
