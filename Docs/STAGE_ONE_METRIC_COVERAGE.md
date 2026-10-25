# Stage 1 metric coverage

Generated from local receipts by scripts/build_stage1_results.py.
Missing values are unmeasured, not zero. Different native datasets/protocols cannot be ranked directly.

| Model | Action group | Evaluation data | RGB PSNR dB | SSIM | LPIPS | Profile input BxCxHxW | Counted forward GFLOPs (partial) |
|---|---|---|---:|---:|---:|---|---:|
| deepjscc | retain_existing | — | — | — | — | 1x3x128x128 | 0.142504 |
| swinjscc | retain_existing | — | — | — | — | — | — |
| ntscc | retain_existing | Kodak24 | 29.790670 | 0.825372 | — | 1x3x512x768 | 787.821029 |
| jcm | use_downloaded_author_weights | CIFAR10 test10000 | 23.363642 | 0.678113 (author convention) | — | 1x3x32x32 | 1.080805 |
| adjscc | use_downloaded_author_weights | — | — | — | — | 1x3x256x256 | 63.503600 |
| shared_semantic_codebook | code_only_candidate | — | — | — | — | — | — |
| sqgan | use_downloaded_author_weights | Cityscapes val500 | 21.240841 | — | — | 1x3x256x512 | 2336.917424 |
| vq_deepsc | unverified_executable_release | — | — | — | — | — | — |
| vq_deepisc | unverified_executable_release | — | — | — | — | — | — |
| mdjcm | author_weights_inaccessible | — | — | — | — | — | — |
| mambajscc | use_downloaded_author_weights | — | — | — | — | — | — |
| diffcom | excluded_from_current_stage | — | — | — | — | — | — |
| diffjscc | excluded_from_current_stage | — | — | — | — | — | — |
| deepjscc_q | code_only_candidate | — | — | — | — | — | — |
| hybrid_semantic_complementary | excluded_from_current_stage | — | — | — | — | — | — |
| dit_jscc | excluded_from_current_stage | — | — | — | — | — | — |
| hda_deepsc | unverified_executable_release | — | — | — | — | — | — |
| masked_vqvae | unverified_executable_release | — | — | — | — | — | — |
| moc_rvq | unverified_executable_release | — | — | — | — | — | — |

JCM RGB PSNR is the standard RGB receipt value; its separately reported author PSNR is23.409674dB.
SQ-GAN PSNR uses native saved-image postprocessing and has no simulated radio channel.
See STAGE_ONE_BASELINES.md for fidelity, blockers, semantic/rate/runtime fields and the training queue.
The JSON inventory preserves metric groups and reasons. GPU job submission does not establish completion.
