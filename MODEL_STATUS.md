# Model phases and current status

This is the quick index for the encoder/decoder work. The phases reflect the
order the model groups entered this project; they do not certify that the
experiments followed one common protocol. The repository preserves older work
so it can be checked and reused where valid.

## At a glance

| Phase | Models | What exists | Current conclusion |
|---|---|---|---|
| 1 — initial three-model batch | DeepJSCC, ADJSCC, NTSCC+ | Training notebooks, historical metrics and/or checkpoints | All three have prior run artifacts. None is certified as a fair, author-exact result under the current shared comparison contract. |
| 2 — next model additions | SwinJSCC, DeepJSCC-Q | Separate training notebooks; Swin has local AWGN and Rayleigh checkpoints and Kodak outputs | Swin's base architecture is the official no-SA/RA variant, but its existing 1/16 results do not match the planned 1/6 primary rate. DeepJSCC-Q's old network is not yet verified as the author implementation. |
| 3 — added codec | SQ-GAN | Three-stage-related notebooks, logs and historical checkpoint(s) | The saved DIV2K/analog-channel run is an adaptation, not the author's native discrete-index communication experiment. |
| 4 — paper/code audit only | DiT-JSCC, DDM-SSCC | Source/paper audit material | No trained benchmark results are claimed. |

**So: yes, there are five earlier model tracks with training artifacts** — the
three in Phase 1 plus SwinJSCC and DeepJSCC-Q in Phase 2. “Working” means that
notebooks/runs/artifacts were produced; it does not mean every implementation
is author-exact, locally checkpoint-verified, or ready for a fair ranking.
SQ-GAN is an additional historical adaptation. The original NTSCC (2022) is a
separate model from NTSCC+ and is not counted among those five completed tracks.

## Phase-by-phase detail

### Phase 1 — DeepJSCC, ADJSCC, NTSCC+

| Model | Evidence kept here | Main issue to resolve before using it in the final table | Next action |
|---|---|---|---|
| DeepJSCC | [Training notebook](Notebooks/JSCC_DeepJSCC_NTSCCpp_ADJSCC_Kaggle_Training.ipynb), [Kaggle run artifacts](Results/Kaggle_runs/), AWGN/Rayleigh `latest.pt` files under `Results/Kaggle_runs/consolidated-kaggle-run/jscc_results/` | Historical code came from a third-party reproduction, not verified original-author code. Its channel convention and source revision are not certified against the shared contract. | Resolve/provenance-lock the implementation first; then treat a corrected run as a new experiment. Preserve the old checkpoints as historical. |
| ADJSCC | [Training notebook](Notebooks/project_notebooks/ADJSCC_DIV2K_Kodak_150ep.ipynb), [saved remote training evidence](Results/reproduction_audit/adjscc_remote_evidence/) | Author attention core was used, but DIV2K/high-resolution/channel handling was adapted. The saved local archive has logs/summary; the Kaggle weight files were not locally reloaded and verified. | Retrieve or access the exact weights, strict-load them, then rerun matched Kodak evaluation and full-image profiling. Retraining is not the first step unless loading or protocol checks fail. |
| NTSCC+ | [Training notebook](Notebooks/project_notebooks/NTSCC_plus_DIV2K_Kodak_150ep.ipynb), [run logs and Kodak CSVs](Results/Kaggle_runs/NTSCC_plus_Training_Results/) | Historical epochs sampled 128 of 800 DIV2K images. Added runtime/channel changes also need validation. This released model is **NTSCC+**, not complete NTSCC++ online inference. | Fresh full-coverage training with compatible author dependencies and declared channel extension; keep the old run separate. |

### Phase 2 — SwinJSCC and DeepJSCC-Q

| Model | Evidence kept here | Main issue to resolve before using it in the final table | Next action |
|---|---|---|---|
| SwinJSCC | [Training notebooks](Notebooks/project_notebooks/Training/SwinJSCC_DIV2K_Kodak_150ep/), [AWGN/Rayleigh results and checkpoints](Results/seven_model_results/SwinJSCC/) | Checkpoint tensor shapes match the official base model without SA/RA, and logs show 150 epochs with 800 images/epoch. It is an existing 1/16 operating point, while the current primary comparison target is 1/6; full Kodak evaluation and all metric/FLOP checks still need certification. | Preserve the existing C=96 checkpoints; validate their load/evaluation. Train a separate C=256 operating point only if 1/6 remains the required primary rate. |
| DeepJSCC-Q | [Training notebooks](Notebooks/project_notebooks/Training/DeepJSCC_Q_DIV2K_Kodak_150ep/), [interim comparison archive](Notebooks/project_notebooks/Training/JSCC_Comparison_Kaggle/five_model_interim/) | The earlier ConvJSCC-Q network is a simplified paper-inspired implementation, not yet verified against the author trainer and learned-modem/annealing implementation. | Confirm the paper variant and author-code provenance; build/validate that implementation before any new training. Do not relabel old checkpoints. |

### Phase 3 — SQ-GAN

- Relevant work is under [`Notebooks/project_notebooks/Training/`](Notebooks/project_notebooks/Training/)
  (`sqgan_*`, `SQGAN_*`, and `sqgan_chunks/`) and [`Results/SQGAN/`](Results/SQGAN/).
- Historical DIV2K/pseudo-label plus analog VQ-vector channel runs are adaptations.
  The paper's codec uses semantic labels, multi-stage training, and discrete
  index communication; those are different end-to-end systems.
- **Next:** decide whether to reproduce the native Cityscapes experiment or
  clearly define a separate DIV2K semantic-label/digital-transport adaptation.
  Validate the original three-stage trainer before calling a run author-faithful.

### Phase 4 — audit-only models

- **DiT-JSCC:** the public release is inference-oriented and does not include
  author training code. No trained result is claimed. See
  [`Docs/Official_Code_and_Paper_Audit.md`](Docs/Official_Code_and_Paper_Audit.md).
- **DDM-SSCC:** paper/code discovery only; no author training implementation was
  verified and no trained result is claimed.
- **NTSCC (2022):** a distinct model from NTSCC+. Existing exploratory codebook
  experiments are not a completed matched training run.

## What “fixed and comparison-ready” means

For every model, confirm its pinned source and architecture, load the intended
checkpoint strictly, record training images/epoch and channel implementation,
evaluate all 24 original Kodak images, count actual channel uses, and profile
encoder/decoder inference and training FLOPs with coverage disclosed. Then run
the shared PSNR/SSIM/MS-SSIM, timing, rate, and channel-trial protocol. Until
those checks pass, historical scores remain useful diagnostics, not a final
cross-model ranking.

## Source notes

Use [`Docs/RERUN_DECISION_20261020.md`](Docs/RERUN_DECISION_20261020.md) for the
detailed five-track rerun recommendation and evidence inspected, and
[`Docs/REPRODUCTION_CORRECTION_20261020.md`](Docs/REPRODUCTION_CORRECTION_20261020.md)
for corrections to earlier author-fidelity claims. `Results/` and notebook
folders intentionally retain historical layouts; this index is the navigation
map and does not move or rewrite those experimental artifacts.
