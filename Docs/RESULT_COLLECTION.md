# Channels, saved-output collection and result publication

Person1 exception: the corrected `src/person1` runner adds genuine training
continuation and150-complete-epoch gates. It creates private checkpoint Dataset
inputs only after locally verifying saved outputs, and preserves the existing
result-only GitHub queue. See `PERSON1_IMPLEMENTATION.md`. The legacy bounded
trainer and collector-only recovery limitations below still apply to other runners.

## Configuration

`configs/environment.json` is shared by all three notebooks. Default: AWGN at 10 dB for training and evaluation. Profiles also define symbol-iid Rayleigh, one-block-per-image Rayleigh, and AWGN training / Rayleigh evaluation. Set `CHANNEL_PROFILE` in notebook setup. Unsupported model/profile combinations fail before expensive setup. See `Infrastructure/README.md` for supported adapters, normalization, perfect-CSI assumptions and exclusions.

The three notebooks use the active Kaggle kernel for initial setup and separate model subprocess environments for incompatible author dependencies. Local validation used `.venv-team`, Python 3.14.0, Torch 2.11.0+cu126, CPU (CUDA unavailable). This does not certify Kaggle's managed runtime or T4 performance. Source revision gates remain in place until the reorganized source is published and pinned.

## Start and recover

From the repository root, using the Python environment containing the Kaggle CLI:

```powershell
python scripts/1.py --username YOUR_KAGGLE_USERNAME
python scripts/1.py --username YOUR_KAGGLE_USERNAME --resume
python scripts/1.py --username YOUR_KAGGLE_USERNAME --resume --once
python scripts/1.py --publish-only
```

Use `2.py` or `3.py` for the other teammates. Default polling is 1800 seconds (30 minutes). `--interval` changes it; `--max-hours` bounds one monitor invocation (default 24). Running the same command recovers an incomplete saved local run instead of submitting training again. After a completed collection, use `--new-run` to explicitly submit another run. `--no-publish` retains results locally. Credentials stay in the platform's normal credential storage, not the notebook or Git.

The scripts submit saved notebook versions only when no recoverable local run exists. They are not an attach-to-any-interactive-session service. If submission was interrupted before version acknowledgement, the script refuses to guess or submit again: inspect Kaggle and the recorded submission first. Current-version and baked-in submission-identity checks protect retrieval; repeated executions of the same saved version remain a platform identity limitation.

## Persistence and limits

- Notebook `completed_models.json` records completed per-model stages. It is a file in the saved notebook outputs, not an external cloud queue.
- With the selected **saved notebook outputs only** approach, local collection happens after Kaggle publishes files, normally when the saved run terminates. Polling cannot expose live `/kaggle/working` files. No private Kaggle Dataset or secret-based cloud upload was added.
- A laptop interruption does not itself stop an independent saved Kaggle run. Local submission/version receipts, collector state and publication events survive process restart.
- A Kaggle runtime reset can lose unpublished runtime files. Exact training resume is not implemented; collector recovery is different from training recovery.
- OS-owned locks release after process death. Verified bundle imports stage files before atomic rename. Interrupted staging folders can remain hidden, but do not block subsequent imports.
- Network/download failures remain retryable; six consecutive collector failures stop that invocation with state retained. Identity/integrity violations require investigation.

## Results and GitHub

Raw retrieval: `Results/team_downloads/<number>/<owner__slug>/<run-id>/`.
Local master ledger: `Results/comparison.csv`.
Models: `Models/teammate_<number>_models/<Model>/<run-id>/`.
Pending publication: `Results/publication_queue/` (ignored by Git).
Remote model results: `Results/published/<event-id>/` plus merged `Results/comparison.csv`.

The publisher uses authenticated GitHub API requests through a locally configured `GH_TOKEN`/`GITHUB_TOKEN`, or `gh auth login` when GitHub CLI is installed. Do not paste tokens into chat or notebook cells. The account must have repository Contents write access; branch protection can also prevent publication. Missing permission/authentication/network leaves events pending. The monitor retries pending events on subsequent polls; `--publish-only` retries once independently of Kaggle.

Only manifest-verified, allowlisted JSON/CSV result files and compact protocol metadata are published. Weights, raw images, absolute constructor paths, credentials and source changes are excluded. Completed runs produce one immutable event. The publisher creates a Git tree/commit based on the latest remote default branch and advances its reference without force; it does not stage, reset, merge or push the user's working checkout. Concurrent teammate changes cause a safe retry against the newer head. A matching remote event receipt prevents duplicate commits after an interrupted acknowledgement. GitHub's master CSV merges prior remote events; the local CSV contains the runs actually retrieved on that laptop.

Reference mechanisms: [Kaggle CLI notebook commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md), [GitHub CLI API authentication](https://cli.github.com/manual/gh_api), [GitHub non-force reference update](https://docs.github.com/en/rest/git/refs#update-a-reference).

## Validation scope

Local regression tests cover channel equations, active-symbol normalization, train/test selection, profile validation, normalized ADJSCC decoder inputs, ordered notebook dispatch, model save/reload, interrupted collector retry, locks, publication queue retry, result-only atomic commit requests and remote CSV merging. GitHub/Kaggle responses in delivery tests are fixtures. No live GitHub write, new Kaggle training run or T4 pilot was performed for these changes. Windows GitHub CLI is currently unavailable. Successful local checks do not establish scientific validity or model convergence.
