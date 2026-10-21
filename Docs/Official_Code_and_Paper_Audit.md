# Official paper and code audit

Checked 2026-10-16. “Official” means linked from an author/project page or released by the named paper authors; popularity or a compatible checkpoint is not enough.

| Name | Published paper | Official/author code status | Local target-folder status |
|---|---|---|---|
| DeepJSCC | Bourtsoulatze, Kurka, Gündüz, “Deep Joint Source-Channel Coding for Wireless Image Transmission,” IEEE TCCN 2019. [Paper](https://arxiv.org/abs/1809.01733) | Yes. The commonly linked PyTorch implementation is [chunbaobao/Deep-JSCC-PyTorch](https://github.com/chunbaobao/Deep-JSCC-PyTorch); its own README describes it as an implementation/reproduction and notes possible differences. | Local checkpoint-compatible source exists under `references/DeepJSCC_checkpoint_source`; this is usable for architecture provenance but not evidence that training was run here. |
| NTSCC++ | Wang et al., “Improved Nonlinear Transform Source-Channel Coding to Catalyze Semantic Communications,” IEEE JSTSP (2023). [Paper](https://arxiv.org/abs/2303.14637), [author project page](https://semcomm.github.io/ntscc_plus/) | The author-linked repository is [wsxtyrdd/NTSCC_plus](https://github.com/wsxtyrdd/NTSCC_plus). Its current README documents an NTSCC+ training/evaluation codebase and explicitly lists NTSCC++ online inference as TODO. Therefore the released repository does not provide a complete exact implementation of the online latent-editing NTSCC++ method. | The local `references/NTSCC_JSAC22_official` is the **2022 NTSCC** repository. I separately cloned the author-linked NTSCC_plus source for audit; this does not make its released NTSCC+ training path equivalent to full NTSCC++. |
| ADJSCC | Xu et al., “Wireless Image Transmission Using Deep Source Channel Coding With Attention Modules,” IEEE TCSVT 2022. [Paper](https://arxiv.org/abs/2012.00533) | Yes. Author-linked repository: [alexxu1988/ADJSCC](https://github.com/alexxu1988/ADJSCC). It provides TensorFlow code for CIFAR-10 and ImageNet variants. | A local `.pt` file and its README are present, but the exact compatible architecture is absent from this target. The TensorFlow source is not locally vendored. The PyTorch file is not a substitute for the official release. |

## Scope and reproducibility caveats

- Official code existing does not mean the code is paper-exact, environment-compatible, or already trained on DIV2K. The source repositories specify their own datasets and protocols.
- For NTSCC++, the available repository's own TODO says online inference for NTSCC++ is unreleased. A trained result from the exposed checkboard/NTSCC+ training code must be named NTSCC+ (released training core), with the missing NTSCC++ online editing step stated alongside it.
- A Kaggle x2 GPU session is different from a TPU session. The cited official code is primarily CUDA/PyTorch or TensorFlow; TPU support would require porting and is not “exactly the same project.”
- DIV2K training and Kodak testing is a requested cross-dataset protocol, not necessarily the paper's original protocol. Report it as a new matched protocol, with source commit and configuration.
- “MMSIM” is interpreted as multi-scale SSIM (MSSSIM); the notebook names the field `msssim`. If another metric is intended, change that definition before interpreting results.
- DeepJSCC and ADJSCC are analog JSCC models: channel uses are the meaningful transmission count. Do not turn FP32 activation size into over-the-air bits. For NTSCC++, count any digital side information only if actual serialized/channel-coded bits are represented.
- Do not claim that results exist until the Kaggle job completes and its checkpoints and metric outputs are retrieved.
