# Semantic model expansion: execution status

Updated: 2026-10-18 (Asia/Kolkata)

## New Kaggle jobs

| Model | Kaggle notebook | Status checked | Artifact status |
|---|---|---|---|
| SwinJSCC | https://www.kaggle.com/code/hmmkk20/swinjscc-div2k-kodak-150ep-20261018 | RUNNING | No downloadable output exposed yet |
| DeepJSCC-Q | https://www.kaggle.com/code/hmmkk20/deepjscc-q-div2k-kodak-150ep-20261018 | QUEUED | Not started; behind SwinJSCC |

Both kernels are private, select GPU T4 x2, and attach DIV2K and Kodak inputs. A queued/running kernel is not evidence of completed epochs or results. Do not add its metrics to the consolidated comparison until its output has epoch 150 for AWGN and Rayleigh, per-image Kodak metrics, and model checkpoints.

## Paper-based boundaries

- SwinJSCC uses the official repository architecture `SwinJSCC_w/o_SAandRA`, fixed-rate C=96, and fresh weights. The notebook adapts the repository's hardcoded paths and epoch loop. Its current implementation uses GPU 0 even though Kaggle allocates T4 x2.
- DeepJSCC-Q uses the author-linked DeepJSCC PyTorch encoder/decoder plus a learned finite constellation and straight-through quantization. This is a paper-based adaptation, not an author-released DeepJSCC-Q repository.
- DDM-SSCC is the confirmed model name. The paper is a separate lossless source/channel coding pipeline, so coded bits and digital channel uses must be reported separately from analog JSCC CBR.
- SQ-GAN is a semantic source codec trained on Cityscapes with segmentation maps and three training phases in its official repository. DIV2K plus AWGN/Rayleigh would be a new dataset/channel adaptation.
- The DiT-JSCC author repository provides inference software without the needed weights or training workspace. A scratch implementation would be a paper-based adaptation, not the official training implementation.

## Local artifacts

- Paper/repository audit: `docs/semantic_models_expansion/PAPER_CODE_AUDIT.md`
- SwinJSCC Kaggle source: `Notebooks/Training/SwinJSCC_DIV2K_Kodak_150ep/`
- DeepJSCC-Q Kaggle source: `Notebooks/Training/DeepJSCC_Q_DIV2K_Kodak_150ep/`
- Existing three-model Kaggle report source was pulled to `Notebooks/Training/JSCC_Comparison_Kaggle/consolidated_existing/` for the later seven-model update.

No GitHub push was made.

## 2026-10-18 live status update

- SwinJSCC Kaggle version 1 remains `RUNNING`; the Kaggle logs page still shows a loading spinner and `Output 0 B`. Runtime visible in the page is about 74 minutes. No epoch count can be verified.
- DeepJSCC-Q reports `CANCEL_ACKNOWLEDGED`; it has no run artifacts and did not train.
- An instrumented SwinJSCC version 2 is prepared locally as `swinjscc_div2k_kodak_150ep_v2.ipynb`. It prints batch progress, uses mixed precision, and wraps the fixed-size model in `DataParallel` when two GPUs are present.
- Kaggle CLI rejected pushing version 2 with `Maximum batch GPU session count of 2 reached`. The remote kernel is unchanged. Retry only after Kaggle frees a batch GPU session; do not claim any training metrics until output is available.

## Corrective rerun: Kaggle version 4

- The original opaque run was cancelled through Kaggle's session API after the user requested a fix. Its progress was not recoverable because no output artifacts had been published.
- The first instrumented rerun exposed a real configuration failure at model construction: the decoder does not accept the encoder-only `patch_size` argument. That run failed before training.
- Version 4 separates encoder and decoder kwargs, removes unsupported decoder fields, logs GPU state and batch progress, uses AMP, and wraps the fixed-size training model in DataParallel when two GPUs are available.
- Version 4 was successfully pushed and currently reports `RUNNING`. Its live-log stream endpoint returned HTTP 500 during this check, and no run artifacts are available yet. Training start and epoch progress remain unverified until fresh logs or outputs arrive.

## 2026-10-19 DeepJSCC-Q v3 live check (2026-10-19 08:24:09 +05:30)

- Kaggle kernel: https://www.kaggle.com/code/hmmkk20/deepjscc-q-div2k-kodak-150ep-20261018
- Status: RUNNING on GPU T4 x2; verified live log output through AWGN epoch 60/150, with all 25 batches/epoch and 800 DIV2K training samples/epoch. Rayleigh segment has not started yet.
- FLOPs profile completed: 168,034,406 training FLOPs/sample; 167,878,658 inference FLOPs/image at 128x128. Complex-symbol CBR is 1/6; constellation size 16.
- The run cloned author-linked DeepJSCC base commit 2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06; this remains a paper-based DeepJSCC-Q adaptation.
- No run outputs are downloadable while the kernel is live (Kaggle Output remains 0 B). Epoch checkpoints are written by the notebook to /kaggle/working/seven_model_results/DeepJSCC_Q/<channel>/model.pth; download and verify after completion.
- Next: finish 150 AWGN epochs, evaluate all 24 Kodak images, repeat 150 epochs for Rayleigh, evaluate Kodak-24, download/verify epoch tables, metrics, and checkpoints, then refresh the interim comparison. No GitHub push.

## 2026-10-19 DeepJSCC-Q completed and verified (08:32 Asia/Kolkata)

- Kaggle status: `COMPLETE`; 2x T4; reported runtime 416.4 seconds.
- The output bundle was downloaded to `outputs/DeepJSCC_Q_DIV2K_Kodak_150ep_Kaggle/`. AWGN and Rayleigh each have exactly 150 epoch rows, ending at epoch 150; each epoch records 800 DIV2K samples and 25 batches.
- Both channel checkpoints exist locally (`model.pth`, 1,910,411 bytes each), as do epoch histories, Kodak per-image metrics, and summary CSVs.
- Measured Kodak-24 at 10 dB: AWGN PSNR 23.5104 dB, SSIM 0.636530, MS-SSIM 0.850744; Rayleigh PSNR 22.9092 dB, SSIM 0.578737, MS-SSIM 0.795830. CBR=1/6, 196,608 complex channel uses/image, constellation size 16.
- Measured profiler counts per 128x128 RGB sample: AWGN train 168,034,406 FLOPs / inference 167,878,658; Rayleigh train 168,116,335 / inference 167,895,051.
- Runtime is plausible: 25 batches per epoch, about 1.15-1.21 seconds/epoch after startup; 300 epochs totals roughly six minutes, with setup and Kodak evaluation accounting for the rest. Kernel elapsed time is not being used as sole evidence; the saved epoch CSVs and checkpoints were checked.
- A local five-model interim comparison now includes the prior four completed models plus this measured DeepJSCC-Q paper-based adaptation: `Notebooks/Training/JSCC_Comparison_Kaggle/five_model_interim/`. It has 10 summary rows, 1,200 epoch rows, and 192 Kodak per-image rows. This is a local notebook snapshot and has not been uploaded to Kaggle.
- Kaggle CLI reports both DeepJSCC-Q and SwinJSCC kernels `COMPLETE`. No GitHub push was made.

## Next model readiness

- Next code-bearing target is SQ-GAN. Its author-linked repository has been cloned to `references/SQ-GAN`. The official pipeline is Cityscapes-specific and has three sequential 150-epoch stages (semantic subnetwork, image subnetwork, joint model), with segmentation labels and InternImage evaluation dependencies. This does not directly match DIV2K/Kodak or an AWGN/Rayleigh channel. Before claiming a comparable SQ-GAN result, implement and label the DIV2K pseudo-segmentation/channel adaptation, then report serialized coded bits and modulation channel uses separately from analog CBR.
- DDM-SSCC and DiT-JSCC remain paper-based tasks; prior audit found no complete author training code for the exact requested pipeline. Keep them unmeasured until their adaptation is implemented and an executed Kaggle output is verified.

## SwinJSCC runtime audit (2026-10-19)

- Downloaded v5 outputs confirm 150 AWGN + 150 Rayleigh epochs, 800 DIV2K samples/epoch, batch 16, 256x256 crops, and complete Kodak-24 results/checkpoints.
- Epoch CSVs report mean 32.70 s/epoch AWGN and 32.72 s/epoch Rayleigh (about 2 h 43 min cumulative epoch time). The final log's elapsed time is 10,045 s (~2 h 47 min), including setup, FLOPs profiling, and Kodak evaluation.
- Its measured profile is approximately 206.5 billion training FLOPs/sample and 68.9 billion inference FLOPs per 256x256 image. DeepJSCC-Q uses 128x128 crops and reports ~168 million training FLOPs/sample and ~168 million inference FLOPs/image. The input sizes and architectures differ; these profile counts are indicative and not a strict normalized speed comparison.
- The user's earlier long no-epoch wait belonged to the failed/opaque first Swin attempt, not the completed v5 run. The compatibility issue was an unsupported `patch_size` argument passed to the decoder; later instrumented reruns split encoder/decoder arguments, added batch progress, and verified epoch outputs. The first live run also did not expose useful logs/artifacts promptly.

## SQ-GAN adaptation scaffolding (2026-10-19)

- Cloned the official author repository under `references/SQ-GAN`; source checkout remains untouched.
- Implemented preliminary separate adaptation modules under `adaptations/sqgan_div2k_channel/`: Cityscapes-ID pseudo-label generation, paired DIV2K image/mask dataset, differentiable AWGN/Rayleigh layer over 256-D VQ features, and a wrapper inserting the channel after the author's VQ straight-through operation.
- These modules are scaffolding only; no SQ-GAN training has started and the Kaggle notebook is not ready. Next gates: integrate author three-stage config and Kodak evaluator, pin/log segmentation model revision and checksums, resolve Python 3.13 vs upstream Lightning 1.5.6 trainer incompatibility, validate forward/backward on T4 x2, then run a measured one-epoch timing before full training.
- Proposed channel insertion treats quantized feature vectors as continuous symbols. It is a clearly labeled adaptation; original SQ-GAN is a masked VQ source codec. Report its 10-bit code-index rate separately from the noisy latent channel-use rate.

## SQ-GAN adapter integration and Kaggle compatibility preflight (2026-10-19)

- Local Kaggle notebook: `Notebooks/Training/sqgan_div2k_channel/sqgan_div2k_awgn_rayleigh_training.ipynb`; private kernel: https://www.kaggle.com/code/hmmkk20/sqgan-div2k-150ep-20261019 . No GitHub push was made.
- The author checkout remains untouched at `references/SQ-GAN`, pinned to commit `0e5cdd7c0240b89749dad684645c53f783ac24cc`.
- Kaggle version 5 completed the compatibility smoke test on Python 3.13.15, PyTorch 2.11.0+cu128, PyTorch Lightning 1.9.5 and 2 Tesla T4 GPUs. It instantiated the channelized all-mode model, produced image output `[1,3,256,256]` and semantic output `[1,20,256,256]`, and completed both author optimizer loss/backward paths on a synthetic smoke batch.
- The smoke pair took 10.63 seconds and used 6,691,968,512 peak CUDA bytes; profiler-reported count was 2,514,462,196,226 FLOPs for the two optimizer forward/backward paths. These are compatibility-profile measurements only, not per-epoch training results or Kodak metrics.
- Integration fixes: adapter config uses square 256x256 crops (`ratio_dim=1`); the released image-loss config’s 20 weights are sliced to the 19 non-null classes consumed by `semantic[:, :-1]`; trained stage checkpoints are split from Lightning `state_dict`s so trained components are carried into final `all` mode; the channel wraps author VQ after the straight-through output.
- The full 3-stage-per-channel training loop is integrated but has not been launched. It comprises 900 stage-epochs at 150 each (ssm, img, all × AWGN/Rayleigh). Current notebook defaults to `RUN_MODE="preflight"` to avoid accidentally starting this long run. DIV2K pseudo-label generation, epoch training, Kodak PSNR/SSIM/MS-SSIM, final checkpoints and full-run FLOPs therefore remain pending.
- Full-run evaluation counts analog latent uses as 128 complex uses per 256-D selected VQ token, reports the image and semantic streams separately, and records 10-bit VQ index lower bounds separately; mask/index side information is not included in analog CBR.
- Evidence file: `outputs/SQGAN_preflight_20261019/preflight_report.json`.

## SQ-GAN full training launch (2026-10-19)

- Switched the SQ-GAN Kaggle notebook to `RUN_MODE="full"` and pushed Kaggle version 6 to `hmmkk20/sqgan-div2k-150ep-20261019`.
- Kaggle CLI confirmed the new version is `RUNNING` on the configured GPU T4 x2 kernel. It will run three 150-epoch stages per AWGN and Rayleigh (900 stage-epochs total), then Kodak evaluation and FLOPs profiling.
- Added a fail-fast check that counts DIV2K train/validation and Kodak images and confirms pseudo-mask pairs before entering any training stage.
- At launch, Kaggle had not emitted a log payload yet, so no epoch progress or training results are confirmed. Check Kaggle status/logs before reporting progress. No GitHub push was made.

## SQ-GAN launch retry (2026-10-19)

- Kaggle version 6 passed dataset preparation (800 DIV2K train, 100 validation, 24 Kodak images; all paired with generated pseudo-label masks) but failed before epoch 1 when the Lightning 1.9 early-stopping callback referenced `np.Inf`, removed in Kaggle's NumPy 2 runtime. No training epochs completed in v6.
- Added a `sitecustomize.py` alias shim only to the child author-training process; this addresses the NumPy API rename without changing the author model or training loss. Pushed version 7.
- Kaggle CLI now reports the kernel `RUNNING`; at the latest check it has not published any v7 logs yet. Epoch progress and the NumPy fix are not yet confirmed by runtime logs. No GitHub push.

## SQ-GAN training confirmed live (2026-10-19)

- Kaggle version 10, run `359915382`, is running on GPU T4 x2 in full mode. The live Kaggle log page shows AWGN / ssm training with 450 steps/epoch.
- Compatibility repairs in the cloned author checkout: NumPy 2 `np.Inf` alias for the child trainer; Lightning 1.9 `DDPPlugin` replaced by equivalent `DDPStrategy(find_unused_parameters=True)`; removed callback hook renamed to `on_fit_start`; TensorBoard-only integer label visualization cast to float. These are compatibility/logging fixes, not model/loss changes. Official author source in the local clone remains untouched.
- Verified a real epoch completed: AWGN / ssm epoch 0, global step 800, validation `val_rec_loss=2.62176`; Lightning saved `epoch=0-val_rec_loss=2.6218.ckpt`. The live log then advanced to epoch 1 (75/450 batches at latest check). No Kodak evaluation or final metrics yet.
- Measured first epoch duration is about 4.3 minutes including its 50-batch validation pass. The 900 stage-epochs would take roughly 64 hours at this rate, while Kaggle documents a 12-hour maximum per GPU notebook session. The current all-in-one run therefore needs a checkpointed multi-session continuation plan before claiming all six 150-epoch stages can finish.
- No GitHub push was made.

## SQ-GAN live runtime diagnosis (2026-10-19)

- Kaggle version 10 run `359915382` is genuinely training; the web Logs panel is stuck on its loading spinner, but `kaggle kernels logs -f` yields live tqdm output.
- Latest live evidence: AWGN/ssm epoch 58 completed all 450/450 steps in 4m17s. Logged epoch-58 validation values include `val_rec_loss=0.501`, `val_aeloss=0.735`; training loss fields include `train_rec_loss=0.374`, `train_aeloss=0.665`. This is epoch progress, not Kodak performance.
- At that pace, one 150-epoch stage is about 10h43m; the six-stage plan (150 x ssm/img/all x AWGN/Rayleigh = 900 stage-epochs) projects about 64h across sessions. Kaggle's official notebook limit is 12 hours per GPU session, so the monolithic run cannot finish all six phases in one run. Kaggle `kernels output` currently returns no files while this run is still active.
- Fix required: split execution into one stage per Kaggle run, with explicit checkpoint outputs/inputs between dependent stages, then run Kodak evaluation after the `all` stage for each channel. Do not claim complete training/results until those stage artifacts and final metrics are downloaded and verified.
- Follow-up CLI tail check: epoch 59 reached 421/450 steps (94%) at approximately 4m14s/epoch; the Kaggle run remained `RUNNING`. The browser log spinner is a presentation problem, not evidence that the trainer is idle.
- Progress follow-up on 2026-10-19: Kaggle CLI still reports `RUNNING`. Live tail reached AWGN/ssm epoch 81 at 292/450 steps (65%); approximately 81 epochs are complete and epoch 82 is in progress. Current epoch log shows `val_rec_loss_epoch=0.392`, `val_aeloss_epoch=0.607`, `train_rec_loss_epoch=0.285`, `train_aeloss_epoch=0.531` (interim training/validation losses only). Output files remain unavailable while the run is active.
