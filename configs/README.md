# Experiment setup

The numbered notebooks pin the published implementation revision. Review the teammate handoff and run a fresh GPU pilot before long training.

The assignments are in `configs/models.json`. Set `MODELS_TO_RUN` in the setup cell to choose their execution order. Local entry points are `scripts/1.py`, `scripts/2.py`, and `scripts/3.py`.

See `Datasets/README.md` for the Kaggle inputs and the separate shared-codebook classifier asset. Current local outputs use `Results/` and `Models/`.

Channel settings are `train_channel`, `train_snr_db`, `test_channel`, and `test_snr_db`. The DeepJSCC adapter supports AWGN and flat Rayleigh with perfect CSI; other neural adapters currently reject non-default or independently differing channel conditions rather than silently applying them. Rayleigh results assume perfect channel knowledge and zero-forcing equalization. SQ-GAN is source coding without a simulated radio channel.

The older guide is preserved in `Archive/configuration_guide_before_migration.md`.

## Saving and sharing

Local retrieval writes `Results/team_downloads/<number>/<run-id>/` and refreshes `Results/comparison.csv`. Completed bundles are imported into `Models/teammate_<number>_models/<Model>/<run-id>/`. Zip the whole teammate folder for Drive/OneDrive; retain all bundle files. Failed/partial runs retain their checkpoints in raw results. Source commits are manual; result-only publication is handled by the durable queue described below.

The CSV is one typed ledger: filter `record_type` before aggregating metrics. Training steps, stage status, failures and successful evaluation trials are separate rows. Remote output retrieval occurs when the saved Kaggle run terminates; this is not live local filesystem synchronization.

## Shared radio environment and collector

Use `configs/environment.json` and notebook `CHANNEL_PROFILE`; see `Infrastructure/README.md` for adapter coverage. The numbered local scripts default to 30-minute polling and recover their local run receipts. GitHub result publication is now automatic when authenticated write access is available; missing access leaves pending events on disk. Source commits and model weights remain outside this automated publication. See `Docs/RESULT_COLLECTION.md` for commands, saved-output limits and recovery.
