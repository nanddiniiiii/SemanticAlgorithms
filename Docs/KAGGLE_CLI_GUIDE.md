# Kaggle CLI workflow

Use `Notebooks/AuthorReproduction/SemanticAlgorithms_Master.ipynb` for every version of the same private Kaggle notebook. Public code belongs in the reviewed `SemanticAlgorithms` GitHub repository. Checkpoints, attached datasets and execution outputs remain private. Never copy credentials, Kaggle configuration, tokens, account metadata or trained weights into the public repository.

The wrapper uses the Kaggle CLI executable available in the active environment. Run authentication yourself in an interactive PowerShell terminal:

```powershell
.\SharedCode\reproduction\delivery\kaggle.ps1 -Action Auth
```

This invokes `kaggle auth login --force`; complete the browser sign in. Tokens are managed by Kaggle outside the repository. Do not paste them into the notebook or terminal commands.

Create a separate private upload directory containing a copy of the master notebook and `kernel-metadata.json`. Supply your actual owner/slug and private dataset slugs:

```json
{
  "id": "YOUR_OWNER/semantic-algorithms-master",
  "title": "SemanticAlgorithms Master",
  "code_file": "SemanticAlgorithms_Master.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
```

Attach the real DIV2K training800, validation100, original Kodak24 and any prior private checkpoint dataset. Set `PUBLIC_REPOSITORY_URL` to the actual public GitHub `SemanticAlgorithms` URL, and set `MODEL`, `MODE`, dataset roots, `CHUNK_SECONDS` and explicit `RESUME` checkpoint in the parameter cell/environment. Default mode is preflight. Each run clones current main into a fresh timestamp directory and records its SHA. Model upstream revisions must be pinned by the published lock/manifests. Source reset preserves previous artifacts.

```powershell
.\SharedCode\reproduction\delivery\kaggle.ps1 -Action Validate -Folder C:\private\semantic-upload
.\SharedCode\reproduction\delivery\kaggle.ps1 -Action Push -Folder C:\private\semantic-upload -ExpectedUsername YOUR_CONFIRMED_USERNAME
.\SharedCode\reproduction\delivery\kaggle.ps1 -Action Status -Kernel YOUR_OWNER/semantic-algorithms-master
.\SharedCode\reproduction\delivery\kaggle.ps1 -Action Files -Kernel YOUR_OWNER/semantic-algorithms-master
.\SharedCode\reproduction\delivery\kaggle.ps1 -Action Output -Kernel YOUR_OWNER/semantic-algorithms-master -OutputDirectory C:\private\downloads\version-001
```

`Push` selects official accelerator `NvidiaTeslaT4` (Kaggle T4 x2). Runtime verifies two T4 GPUs before training. Validate checks private GPU metadata and notebook presence; it does not certify model readiness. Push starts paid-quota-consuming execution only when you invoke it. No Colab or paid GPU fallback is configured.

`Files` and `Output` follow every next-page token, detect token loops and propagate CLI errors. Use a fresh download directory per version. Record the version from the push response/status in your private run ledger and download only after that version completes; the installed CLI output endpoint addresses the current kernel session, so this wrapper deliberately rejects version-suffixed identifiers rather than pretending to retrieve a historical version. Archive each completed download before pushing the next version.

The notebook writes `artifact_sha256.json` even when runtime fails. `Output` requires exactly one such manifest, verifies every listed downloaded file against the run-produced SHA256, and writes `download_verification.json`. This verifies content integrity against the downloaded run manifest, not publisher authenticity. A missing or mismatching artifact is an error. A preflight pass or incomplete training chunk never counts as a measured result.

Training contract: global batch16, training SNR10, validation early stopping after at least50 and at most150 epochs, patience20 and relative improvement .001. Evaluate original Kodak24 at SNR0/5/10/15/20 with10 independent channel trials and measured CBR/rate curves. SQGAN trains its semantic/image/finetune codec stages once then evaluates both digital channel transports. DiT/DDM remain audit only. The runtime must preserve optimizer, scheduler, RNG and sampler state for bounded chunks; unsupported/missing author implementations stop before training.

Push requires an explicitly confirmed username. It compares that username with both metadata owner and the authenticated Kaggle identity; any mismatch stops submission. Channel is an explicit notebook parameter; run each channel/rate with the same master notebook. Optional consolidation validates full measured grids, contract hash, geometry/rate accounting and receipts, then exports unranked partial evidence. It never promotes receipt flags to a publication ranking.

The notebook provisions a model-specific isolated environment, reads its recorded interpreter from environment.json, and runs data/hardware checks and model runtime in that interpreter. Environment preparation is labelled READY_ENVIRONMENT_UNVALIDATED; the subsequent actual GPU preflight must pass before training. ADJSCC fails closed when the required Linux native TensorFlow Compression bundle is unavailable. Set STAGE explicitly for SQGAN semantic/image/finetune stages; other models use main.
