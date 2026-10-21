# Public repository scope

The public repository is [`nanddiniiiii/SemanticAlgorithms`](https://github.com/nanddiniiiii/SemanticAlgorithms).
The local working folder remains `SemanticLossless`.

## Published scope

The requested public release includes the model-development code, project
notebooks, documentation, historical result tables/plots/logs, and available
checkpoints from the local workspace. The local originals are preserved. LFS
stores checkpoint binaries; incomplete `.part` transfers are omitted.

## Excluded from the public repository

- Kaggle credentials, tokens, private upload metadata, and machine-local
  manifests.
- Raw DIV2K/Kodak datasets, virtual environments, caches, and incomplete
  `.part`/`.partial` transfers.
- Upstream source snapshots without a redistribution license. The source lock
  pins the upstream commits and file hashes; the runtime fetches those sources.
- Private signed download URLs and any file that cannot be sanitized safely.

## Fidelity and result policy

The shared protocol is a proposed matched benchmark, not an identical
reproduction of each paper's original experiment. Historical outputs are
preserved for traceability but are not thereby certified for comparison. Keep
each model's upstream source commit, license, architecture, loss, data
transforms, and adaptation explicit. Label third-party or paper-based
reproductions accurately. Do not present preflight, synthetic CPU checks,
partial epochs, or incomplete metric grids as training results. Final scores
require artifact hashes, checkpoint lineage, channel-use ledgers, and
evaluation receipts.

## Local archive

The local `SemanticLossless` folder may contain research history beyond the
public release. Do not recursively add the whole folder to Git. Stage only the
reviewed public paths and preserve private files locally.
