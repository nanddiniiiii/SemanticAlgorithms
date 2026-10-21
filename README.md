# SemanticAlgorithms

Research code for reproducing and comparing semantic image communication models.

## Status

This repository is an implementation and evaluation work in progress. The
archive includes historical notebooks, tables, plots, logs, and checkpoints
from earlier work. Their presence does not certify author fidelity, complete
training, or a fair matched comparison. The current master notebook is a
bounded-run launcher, not a completed all-model run. Local CPU regression tests
cover helper code only; they do not establish GPU compatibility, model quality,
or completed training.

For the model-by-model history, prior artifacts, and exact next fixes, start at
[`MODEL_STATUS.md`](MODEL_STATUS.md). It separates the first three-model batch,
later model additions, and audit-only models; “has a run/checkpoint” is not
treated as “verified and comparison-ready.”

Current gates include:

- DeepJSCC uses a pinned third-party implementation; original-author code for
  the 2019 paper has not been verified.
- ADJSCC requires its original TensorFlow Compression native operations. The
  Linux/GPU environment remains unvalidated and the launcher must fail closed.
- NTSCC+ is the available released project. It must not be called NTSCC++;
  NTSCC++ online inference is not included in that release.
- SwinJSCC uses the official base model without the optional SA/RA modules.
- SQ-GAN's original three-stage trainer has not yet been integrated into the
  shared runtime. Label/data helpers and historical checkpoints are not
  evidence of a currently validated trainable pipeline.
- DeepJSCC-Q is pinned to a candidate implementation whose paper/authorship
  provenance still needs confirmation.
- DiT-JSCC and DDM-SSCC are paper/code audits only; they are not trained here.

Historical outputs are preserved and are not silently promoted to results for
the new matched contract. Do not compare or cite a historical file as a valid
model result without checking its provenance, full-image evaluation, actual
channel-use ledger, training coverage, and checkpoint lineage. No new GPU run
or final six-model comparison is claimed by this release.

## Repository layout

- `SharedCode/reproduction/` — pinned-source manifest, common experiment
  contract, runtime adapters, channel/accounting helpers, bounded-run engine,
  and Kaggle notebook delivery utilities.
- `Notebooks/AuthorReproduction/SemanticAlgorithms_Master.ipynb` — one
  parameterized Kaggle notebook for source refresh, preflight, bounded runs,
  evaluation, and artifact handling.
- `Docs/` — source-fidelity notes, runtime limitations, and Kaggle CLI guide.
- `sources.lock.json` — immutable upstream source revisions and file hashes.
- `Models/` and `Results/` — local model notes, adaptations, historical
  checkpoints, and result artifacts. Incomplete `.part` transfers are excluded.

Upstream author code is fetched at its pinned revision. Its license and usage
terms remain those of the upstream project; the repository does not repackage
unlicensed source snapshots. Raw datasets, credentials, machine-local manifests,
virtual environments, caches, and incomplete partial downloads are excluded.
Check the license and provenance before redistributing any model weights.

## Reproduction contract

The intended matched study uses 800 distinct DIV2K training images per complete
epoch, a separate 100-image validation split, and Kodak24 for final evaluation.
Its proposed common protocol, stopping rule, channels, rate accounting, metrics,
and checkpoint requirements are recorded in
[`SharedCode/reproduction/experiment_contract.json`](SharedCode/reproduction/experiment_contract.json).
Those settings are a target contract; they are not evidence that all models
currently implement or pass it.

See [`Docs/MODEL_RUNTIME.md`](Docs/MODEL_RUNTIME.md),
[`Docs/NTSCC_RUNTIME.md`](Docs/NTSCC_RUNTIME.md), and
[`Docs/KAGGLE_CLI_GUIDE.md`](Docs/KAGGLE_CLI_GUIDE.md) for execution gates.

## License

No umbrella license is provided yet. Treat the project's original adaptation
code as all-rights-reserved until a license is added. Each upstream repository
retains its own license; check it before use or redistribution.
