# Three-person training workflow audit

Date: 2026-10-27. Scope: current local, uncommitted source snapshot. No Kaggle submission or new training was performed for this audit.

## Outcome

The shared workflow is implemented locally, with the corrections below. It is **not yet a published, GPU-certified handoff**. The notebooks intentionally reject an empty source revision; the reorganized code must be reviewed, committed, published and pinned before remote use.

| Requirement | Local implementation / evidence | Limitation |
|---|---|---|
| Three independent people | `Notebooks/1.ipynb`, `2.ipynb`, `3.ipynb`; corresponding `scripts/1.py`, `2.py`, `3.py` | Separate Kaggle authentication and GPU eligibility are required. |
| Four slots each, ordered selection | `configs/models.json`; notebook slot dispatch follows `MODELS_TO_RUN` order | Twelve slots are nine distinct models, with repeated baselines. |
| Shared code | Modules directly in `src/`; source clone pinned by revision | Revision remains blank pending publication; do not remove the gate. |
| Three cells per model | Initialize/pilot, train, evaluate/package/export | Setup and final session summary are additional cells. |
| Common inputs | Frozen DIV2K/Flickr2K training, DIV2K development, Kodak test manifest | Model-specific objectives/assets and adaptations remain explicitly distinct. |
| Two GPUs | CUDA/two-device setup guard and DataParallel training checks | Local checks do not establish T4 performance or freedom from OOM. |
| Four-hour training budget | Per-model soft time budget and periodic checkpoints | Setup, pilots, evaluation, export and in-flight steps add time; not a four-hour total guarantee. |
| Observable execution | Stage logs/status, training JSONL, metrics, session summary, hashed exports | Abrupt kernel death can lose buffered/native output; Kaggle output is normally retrieved after saved-run termination. |
| Shared results CSV | `Results/comparison.csv`, rebuilt atomically from verified downloads and operational status | Separate `record_type` values distinguish training, evaluation, failure and status rows; missing measurements remain missing. |
| Shareable checkpoints | `Models/teammate_N_models/<Model>/<run-id>/` | Only completed train/evaluate stages may package. Completion is not convergence or scientific validity. |
| Safe retrieval | SHA/size validation, teammate/run identity, no overwrite of different existing bundle | Kaggle CLI is latest-run scoped; a repeat of the same version cannot be uniquely excluded. |
| Git collaboration | Shared source repository and recorded source revision | **No automatic Git commit or push exists.** Weights/raw results are not intended for Git. |

## Assignments

1. DeepJSCC, ADJSCC, SwinJSCC, ISEC.
2. HJSCC, JCM spatial reconstruction variant, MDJCM, DeepJSCC.
3. SQ-GAN, SharedSemanticCodebook, SwinJSCC, ADJSCC.

The JCM variant is not an unchanged native baseline. SQ-GAN's current path is a source codec, without radio-channel evaluation. No claim that all models have identical tasks or all requested metrics is justified. Channel/bandwidth, digital payload, and source codec metrics must retain their distinct units and scopes.

## Corrections in this audit

- Model export now produces teammate/model/run folders, portable constructor assets/configuration, metrics, provenance and integrity metadata.
- Packaging requires completed training and evaluation stage receipts. Failed/partial checkpoints remain in raw run outputs rather than successful model folders.
- Bundle import checks teammate, model, run path safety and hashes; conflicting existing runs are never silently overwritten.
- A bundle loader reconstructs through the shared adapters. A real CPU DeepJSCC round trip is checked separately from fixture-only packaging tests. Other author-specific loaders are not thereby certified.
- Ordered model selection now controls actual execution, rather than only filtering statically ordered cells.
- The CSV includes training events, summaries and failures, not just successful per-image evaluation.
- Final session export preserves initialization failures and stage summaries even when no model completes.

## Sharing and collaboration

Zip `Models/teammate_1_models`, `Models/teammate_2_models`, or `Models/teammate_3_models` and share via Drive/OneDrive. Each model has a run subfolder to avoid overwriting experiments. Keep the bundle intact. Raw logs and results remain under `Results/team_downloads/<person>/<run-id>/`; the master CSV is `Results/comparison.csv`.

Do not commit checkpoint files, credentials, virtual environments or downloaded datasets. Commit reviewed source/notebook/configuration changes separately from a small, curated metrics update. There is no automatic merge/push from simultaneous teammates. Use separate branches and review changes before merging; do not let launch scripts mutate the shared source checkout.

## Coverage and remaining checks

Inspected active notebook generation, numbered launch scripts, launch/retrieval, runtime/export, dataset identity, training, evaluation, SQ-GAN training, model sharing and CSV collection. Existing regression tests exercise representative paths. Archived EncDec, historical jobs, every external author implementation and every stored checkpoint were not all executed or scientifically revalidated. No new remote GPU run occurred.

Before handoff: publish the reviewed migration, pin notebooks to that commit, and run a short fresh-kernel Kaggle pilot with the required inputs/assets. Validate each model's actual shapes, strict checkpoint reload, two-GPU observations, finite metrics and persisted retrieval before any long training. Successful local tests do not replace this gate.

## Local validation results

- `.venv-team/Scripts/python.exe -m pytest tests/current -q`: **58 passed**, two upstream/profiler warnings; 17.64 seconds.
- Focused workflow/sharing suite: **27 passed**, including model-order dispatch, artifact checks, failed-run rejection, idempotent import and a real CPU DeepJSCC saved-model round trip.
- Python compilation passed for `src`, `Infrastructure`, `scripts` and `tests/current`.
- Model checkpoint paths are Git-ignored; teammate-folder README files remain trackable.
- `Results/comparison.csv` exists with a header and **zero new verified run records**. Historical results were not silently imported as new teammate trials.
- Whole-working-tree `git diff --check` flags a pre-existing trailing blank line in `Docs/KAGGLE_RUNS.md`; unrelated historical changes were preserved.

These are local tests, not twelve completed GPU training runs. No Git commit/push or Kaggle job was performed in this audit.
