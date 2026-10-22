# Active project

Use this clone of `nanddiniiiii/Semantic_Lossless` for new project work.
The parent workspace's research documents and the old zip extraction are history,
not alternate implementation roots. Do not delete them without a user request.

Read `Docs/RETAINED_ENCDEC.md` and `Docs/MODEL_SELECTION.md` before changing the
retained codecs. The EncDec import is documented by per-file hashes in
`Docs/encdec_import_manifest.json`. The minimal INFRA package supplies codec
contracts/channels only; do not import the scheduler project to make codecs run.

Keep retained checkpoints immutable. Some local checkpoint paths are NTFS hard
links, so always train into a new algorithm/run directory. Preserve baseline and
modified results separately. Do not call a calibrated cache a trained model.

Pin upstream code and record architecture, checkpoint hash, native/adapted data,
training coverage, channel convention, actual transmitted overhead and runtime.
Synthetic CPU smoke checks are not quality benchmarks or GPU compatibility tests.
Only inference FLOPs are required; unsupported operations remain explicit.

The public repository still contains historical experiments and unhydrated LFS
archive pointers. An archived path or a planned adapter does not establish model
availability. Keep unavailable implementations and weights clearly labeled.
