# SQ-GAN experiment artifacts

The experiment is still training. No completed AWGN/Rayleigh Kodak comparison is claimed yet.

- `cloud_recovery_validation/sqgan_div2k_channel_results/checkpoint_validation.json`: independently measured state of the recovered checkpoint, including its size, SHA256, epoch counters, both optimizers and schedulers.
- Successful cloud checkpoint backup: https://www.kaggle.com/code/hmmkk20/sqgan-checkpoint-recovery-20261020/output
- `validated_recovery_v15/`: resumable local transfer of that same checkpoint. Only a final `last.ckpt` accompanied by a successful `checkpoint_validation.json` is validated. `.partial` files and `download_parts/` are transfer intermediates.
- `local_resume_contract/` and `local_resume_contract_with_deadline/`: tiny CPU-only software tests. Their small probe checkpoints are synthetic test artifacts, not SQ-GAN training results.
- `continuation/state.json`: current Kaggle job, watcher PID, chunk number and most recent status check.
- `continuation/watcher.log` and `watcher.err`: automatic continuation diagnostics.
- `continuation/chunk_XX/`: saved manifests and, after complete evaluation, downloaded metric tables for successful training chunks.
- Earlier `recovery_v15/` and `checkpoint_recovery_v15/` directories contain incomplete CLI transfer attempts. A zero-byte file there must not be used as a model checkpoint.

The local watcher requires this computer to remain awake and connected. It launches a new private Kaggle continuation after each verified successful chunk, with the previous chunk's output as input. It stops for inspection on failed/cancelled runs or invalid manifests. The currently submitted Kaggle execution can continue when the computer is asleep, but the next chunk cannot be submitted until the local watcher runs again.

The training source, runtime helpers and checkpoint recovery utilities are under `SharedCode/sqgan/` and `Notebooks/project_notebooks/Training/`. The audit and conditional runtime estimate are documented in `Docs/SQGAN_checkpoint_recovery_20261020.md`.
