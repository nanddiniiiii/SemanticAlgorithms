# Teammate handoff

Repository: https://github.com/nanddiniiiii/Semantic_Lossless

| Person | Notebook | Script | Assigned order |
|---|---|---|---|
| 1 | Notebooks/1.ipynb | scripts/1.py | DeepJSCC, ADJSCC, SwinJSCC, ISEC |
| 2 | Notebooks/2.ipynb | scripts/2.py | HJSCC, JCM spatial variant, MDJCM, DeepJSCC |
| 3 | Notebooks/3.ipynb | scripts/3.py | SQ-GAN, SharedSemanticCodebook, SwinJSCC, ADJSCC |

These twelve slots contain nine distinct models. They are locally tested runners, not twelve completed GPU experiments. Default: AWGN10. Preserve the frozen common split and documented model adaptations.

## Share with each person

- Repository URL, their person number, this guide and `Docs/RESULT_COLLECTION.md`.
- Person 3 additionally needs the approved `classifier.pt` through a private Kaggle Dataset attachment. This asset is **not in Git**. Do not substitute another classifier. Obtain the exact asset and provenance from the coordinator.
- A shared Drive/OneDrive folder for their zipped `Models/teammate_N_models` after retrieval. Drive upload is manual.

Author initializers are downloaded and hash-checked by the existing asset code. If upstream downloads fail, attach the exact documented author assets and configure overrides; do not silently replace pretrained weights with random initialization.

## Local setup

Use Python 3.11/3.12 where available. Local CUDA is unnecessary for collection.

```powershell
git clone https://github.com/nanddiniiiii/Semantic_Lossless.git
cd Semantic_Lossless
python -m venv .venv-collector
.venv-collector/Scripts/Activate.ps1
python -m pip install -e . kaggle
kaggle auth login
```

Linux/WSL activation is `source .venv-collector/bin/activate`. Authenticate your own Kaggle account. For GitHub result publication, install GitHub CLI and run `gh auth login`, or configure `GH_TOKEN` locally with repository Contents write permission. Without collaborator access/authentication, results remain queued locally. Never put credentials in chat, notebooks or Git.

## Choose one launch route

**Local submission and automatic collection:** replace N with your number:

```powershell
python scripts/N.py --username YOUR_KAGGLE_USERNAME
```

This submits the corresponding stable `semantic-notebook-N` identity, attaches the common Datasets and requests T4 x2. Inspect the printed link and actual accelerator/quota. For additional private assets, add `--asset OWNER/DATASET_SLUG`. Do not also submit a manual notebook copy.

**Manual Kaggle:** import `Notebooks/N.ipynb`, enable Internet and T4 x2, and attach these inputs before running:

- `soumikrakshit/div2k-high-resolution-images`
- `daehoyang/flickr2k`
- `drxinchengzhu/kodak24`
- Person 3: the private approved classifier Dataset.

Run setup and initialization/pilot before long training. Execute cells in order. Four four-hour training budgets mean 16 hours **plus** setup/evaluation; select a smaller assigned subset when session quota is insufficient. A 150-epoch target is not a convergence/time guarantee.

The automatic collector recovers versions that it submitted and recorded. It does not adopt unrelated manual notebooks. For manual runs retrieve published outputs through the Kaggle UI/CLI; do not start the submission script merely to monitor a manual copy.

## Recovery and outputs

```powershell
python scripts/N.py --username YOUR_KAGGLE_USERNAME --resume
python scripts/N.py --username YOUR_KAGGLE_USERNAME --resume --once
python scripts/N.py --publish-only
```

Re-running the normal command recovers unfinished local receipts. `--new-run` explicitly starts another training run. `--no-publish` retains local-only results. Polling defaults to 30 minutes. With saved notebook outputs only, files can be downloaded once Kaggle publishes them, normally at run termination; live `/kaggle/working` is unavailable. Runtime reset may lose unpublished files. Collector recovery is not exact training resume.

- Raw evidence: `Results/team_downloads/`.
- Local ledger: `Results/comparison.csv`.
- Models: `Models/teammate_N_models/<Model>/<run-id>/`.
- Remote results when permitted: `Results/published/<event-id>/` and merged `Results/comparison.csv`.

Zip the entire teammate model folder for Drive/OneDrive. Weights stay outside Git. Preserve errors and partial evidence; report algorithm, failing cell, source revision, traceback and hardware. Do not change input geometry, losses, datasets or model definitions just to bypass failures. A completed run does not establish scientific validity.

## Message to each teammate

> We are using https://github.com/nanddiniiiii/Semantic_Lossless. You are person **N**. Your notebook is `Notebooks/N.ipynb`; your local launcher is `scripts/N.py`. Read `Docs/TEAMMATE_HANDOFF.md` and give your Codex the prompt below. Use your own Kaggle T4 x2 session, the shared frozen datasets and default AWGN10. Do not start duplicate runs or alter algorithms. The script downloads saved outputs and publishes result files when you have GitHub write access; otherwise retain them locally. At completion zip `Models/teammate_N_models` and share it in our Drive/OneDrive folder. Person 3 also needs our exact classifier asset.

## Prompt for their Codex

> I am person **N**. Work in the canonical clone of https://github.com/nanddiniiiii/Semantic_Lossless. Read AGENTS.md, Docs/TEAMMATE_HANDOFF.md, Docs/RESULT_COLLECTION.md, configs/README.md and Datasets/README.md. Inspect my numbered notebook/script and verify its pinned source exists remotely. Set up my local collector environment and my own Kaggle authentication; verify actual T4 x2 availability and exact required input/author/classifier assets. Keep secrets out of chat/logs/Git. Preserve assigned order, frozen splits, default AWGN10, model dimensions and documented adaptations. Run preflight before training. Use the existing stable notebook identity, never duplicate submissions. Help me run scripts/N.py and recover via its receipts when authorized. Preserve failures and report evidence before changing architecture/dependencies/data/metrics. Retrieve verified outputs, keep weights under Models/teammate_N_models, and use the existing result-only GitHub queue when write access is available. Otherwise retain everything locally. Do not commit weights, datasets or credentials, and do not claim untested GPU paths have passed.
