# Seven-model semantic image transmission study: paper and code audit

Audit date: 2026-10-18. “Author-linked” means the paper or authors' project page points to the repository. Code presence does not prove an exact paper reproduction.

| Model | Paper | Official / author code | Training and comparison boundary |
|---|---|---|---|
| DeepJSCC | Bourtsoulatze et al., IEEE TCCN 2019. https://arxiv.org/abs/1809.01733 | https://github.com/chunbaobao/Deep-JSCC-PyTorch — author-linked PyTorch reproduction; its README allows paper/code differences. | Prior 150 epochs/channel are a DIV2K/Kodak adaptation, not the exact original paper protocol. |
| ADJSCC | Xu et al., IEEE TCSVT 2022. https://arxiv.org/abs/2012.00533 | https://github.com/alexxu1988/ADJSCC — author code (TensorFlow; CIFAR/ImageNet pipelines). | Prior run used official attention encoder/decoder/channel with Kaggle compatibility and Rayleigh extension; dataset/crop/batch/epochs are adapted. |
| NTSCC_plus (prior benchmark label “NTSCC++”) | Wang et al., IEEE JSTSP 2023. https://arxiv.org/abs/2303.14637 | https://github.com/wsxtyrdd/NTSCC_plus — author-linked. Public README says NTSCC++ online inference/editing is TODO. | Prior run is the released NTSCC_plus model, not full NTSCC++ online editing. 150 epochs/channel; 128/800 DIV2K images sampled each epoch. Rayleigh is an extension. |
| SwinJSCC | Yang et al., IEEE TCCN 2024. https://arxiv.org/abs/2308.09361 | https://github.com/semcomm/swinjscc — official PyTorch implementation with training code. | New training required. Kaggle adaptation must replace workstation paths, set 150 epochs, and document initialization. |
| SQ-GAN | Pezone et al., IEEE TCCN 2025. https://arxiv.org/abs/2502.09520 | https://github.com/frapez1/SQ-GAN — author code. The paper is a source codec (not neural JSCC); official code trains on Cityscapes with 20-class segmentation maps in three stages. | Cannot be called a DIV2K/AWGN/Rayleigh reproduction unchanged. A DIV2K pseudo-label and digital-link adaptation must be labeled, measuring serialized bits and channel uses. |
| DDM-SSCC (user-confirmed) | Ren et al., “Adapting Diffusion Language Models for Lossless Pixel-Level Image Transmission,” arXiv:2606.06273 (2026). https://arxiv.org/abs/2606.06273 | No author code link found on the paper page or in the audit searches. Paper-based implementation required unless a repository is found later. | Separate source/channel coding for exact recovery, not distortion-oriented semantic JSCC. Report exact-recovery probability, BPP, coded bits, modulation symbols/channel uses, PSNR/SSIM/MS-SSIM. CBR is not directly comparable with analog JSCC. |
| DiT-JSCC | Tan et al., “DiT-JSCC: Rethinking Deep JSCC with Diffusion Transformers and Semantic Representations,” arXiv:2601.03112 (2026). https://arxiv.org/abs/2601.03112 | https://github.com/semcomm/DiTJSCC — author-org inference toolkit. README says full training workspace/scripts are omitted and no model weights are included. | Paper-grade retraining is blocked by missing training code and required pretrained weights/large-scale initialization. A reduced paper-based implementation would be new code, not exact reproduction. |
| DeepJSCC-Q | Tung et al., IEEE JSAIT 2022. https://arxiv.org/abs/2206.08100 | No author-linked standalone public repository found in this audit. | Paper-based reproduction required: train the image autoencoder with finite constellation-constrained channel inputs and differentiable/straight-through quantization; test AWGN/Rayleigh. |

## Existing local artifacts and notebooks checked

The prior training kernels and consolidated report are separate Kaggle notebooks. The standalone DeepJSCC notebook was the first end-to-end runnable training path. ADJSCC requires TensorFlow Compression and a Python 3.11 compatibility environment; NTSCC_plus requires a separate PyTorch/NATTEN stack and compatibility shim. The consolidated notebook imports recorded results and produces tables/plots; it does not retrain. A separate helper notebook profiles DeepJSCC FLOPs.

The Kaggle kernels currently report `COMPLETE`: DeepJSCC, ADJSCC, NTSCC_plus, DeepJSCC FLOPs profile, and the three-model report. No GitHub push was made.

## Local checkpoint distinction

The fresh prior DeepJSCC Kaggle training produced `latest.pt` for AWGN and Rayleigh under `results/jscc_three_model/DeepJSCC/`. The filename `model.pth` is not present for those fresh checkpoints. Other `.pth`/`.pt` files in the project are pre-existing model/calibration weights; do not attribute them to this recent training run. Downloading ADJSCC/NTSCC Kaggle checkpoints was attempted, but the output endpoint stalled and returned an SSL EOF, so local presence of their Kaggle checkpoints is not confirmed.

## Comparison rules

- Evaluate all 24 Kodak images when the method supports them; record resize/crop details.
- Compute PSNR, SSIM, and MS-SSIM with shared implementations and data range 1.0.
- Report units: analog JSCC uses complex channel symbols; digital codecs report serialized bits and modulated symbols after channel coding. These are not interchangeable.
- Measure inference FLOPs/image and training FLOPs/sample with documented profiler, dimensions, forward/backward inclusion, and unsupported fused ops.
- Record AWGN/Rayleigh definitions, SNR, seeds, batch/crop size, DIV2K image exposures, and all 150 completed epoch rows.
- Call a Kaggle job complete only after kernel status, outputs, epoch records, test metrics, and checkpoint files are checked. Never substitute paper results or estimates for missing run output.
