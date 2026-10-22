# Retained EncDec models in the canonical clone

Date: 21 October 2026. Canonical repository:
https://github.com/nanddiniiiii/Semantic_Lossless.

The working clone is the `Semantic_Lossless` directory under the original
SemanticLossless workspace. The earlier inspection clone named Nandini_Work was
renamed and checked out here. New code and results belong in this clone.

## What was transferred

The import receipt lists 197 files/assets, totaling 928,302,119 logical bytes,
with source-relative paths and SHA-256 identities. The scheduler source HEAD
was `032c23a3dafc7bfeb39a84ba55193b6b6f3a22a0`; individual working-file hashes,
rather than HEAD alone, identify the actual copied code.

- `EncDec/Research/`: DeepJSCC/NTSCC wrappers, Windows entropy-model compatibility,
  metrics and existing codebook research helpers.
- `EncDec/Models/DeepJSCC/` and `EncDec/Models/SwinJSCC/`: codec implementations.
- `EncDec/Codebooks/`: communication VQ/codebook implementations.
- Minimal `INFRA/` contracts/channels/registries required by the retained model
  interfaces. The scheduling runtime was not imported.
- Checkpoint-compatible source snapshots under `Models/DeepJSCC/checkpoint_source`,
  `Models/NTSCC/official` and `Models/SwinJSCC/local_source`.
- DeepJSCC production and CIFAR checkpoints; original NTSCC production weights;
  Swin encoder, decoder, full checkpoint and inference manifest; existing Swin VQ
  codebook; historical ADJSCC PyTorch artifact and calibration metadata.

The two original production weights were copied from the supplied checkpoint
package because the scheduler's configured relative paths did not contain them.
Their hashes match the scheduler's expected identities. Runtime model execution
now uses only the copies/source snapshots inside this clone.

| Retained model | Main asset | Identity and important condition |
|---|---|---|
| DeepJSCC | `Models/trained_checkpoints/DeepJSCC/production/imagenet_10_0.33_200.00_32_19.pth` | SHA-256 `00b33bac630555d9a0d72a76d0fea9103cef9a38df0d5547be53c5320a451d44`; training SNR setting 200 dB; retained legacy channel is not the common calibrated experiment |
| Original NTSCC | `Models/trained_checkpoints/NTSCC/production/ntscc_hyperprior_quality_1_psnr.pth` | SHA-256 `7a8d1990dbd678ea5d3daa02c3aeaaf4459b347a96d6aa81a5c02eae6bfc7323`; training SNR setting 10 dB; not NTSCC+ or ++ |
| Adaptive SwinJSCC | `Models/trained_checkpoints/SwinJSCC/encdec_sa_ra/` | SA+RA, AWGN, rate grid 32/64/96/128/192; source and checkpoint validation use the retained manifest |

The ADJSCC `adjscc_r16.pt` is an older PyTorch artifact. It must not be confused
with the teammate's 150-epoch TensorFlow `.weights.h5` experiments. The new ADJSCC
training request remains a separate run.

Calibration caches and the existing Swin codebook are retained history. The NTSCC
cache filename mentions Kodak; do not use it as fresh training/calibration data
for a held-out Kodak comparison. The scheduler calibration provenance is retained
as `encdec_calibration_provenance.json` without overwriting the repository's older
provenance document.

## Storage and Git LFS

D: ran short of space during copying. Same-volume immutable checkpoints are
hard-linked to avoid duplicate allocation. Deleting the old directory entry
does not delete the data as long as this clone's hard link remains. Overwriting
an existing hard-linked file in place can affect both names, so use new output
paths for all training/checkpoint saves. Supplemental C:-drive production weights
are independent copies in this clone.

Git LFS is configured locally. Its duplicate object cache uses
`C:/Users/satya/AppData/Local/SemanticLossless/git-lfs` to avoid exhausting D:.
This cache location is machine-local Git configuration, not a credential or a
runtime dependency on the zip extraction.

Eleven historical archive LFS pointers remained unhydrated at the inspection
following the import. The three retained core codecs' required inference assets
are hydrated and verified. Do not claim that every historical checkpoint in the
repository has been downloaded. Large future downloads need adequate disk space
or a separately selected result-storage location.

## Verification actually performed

`Results/retained_import/asset_verification.json` records all 197 imported hashes
matching. Each core model also passed one synthetic 128x128 RGB CPU forward:

- DeepJSCC: strict load of 29 keys; finite reconstruction with the original shape.
- NTSCC: strict load of 426 keys after explicitly regenerating resolution-dependent
  masks; finite reconstruction with the original shape.
- SwinJSCC: strict encoder/decoder load plus manifest source/runtime validation;
  finite reconstruction, rate 96, 3,072 complex channel uses for this synthetic input.

These are transfer/runtime checks, not Kodak quality measurements, FLOP results,
training convergence, or T4/GPU readiness. DeepJSCC's channel convention and odd
real-latent packing require an explicit common-transport decision before a fair
channel-use benchmark. NTSCC's recorded main-latent uses exclude a complete
side-information/digital serialization ledger.

The checks used Python 3.12.14, PyTorch 2.10.0 CPU and a repository-local timm 0.4.12
installation. The existing model Python interpreter was reused; no new CUDA or
large framework install was performed. Model source and assets were loaded from
this clone, not imported from the old scheduler source tree.

From the clone root:

```powershell
python scripts/check_retained_models.py --verify-assets
python scripts/check_retained_models.py --model deepjscc
python scripts/check_retained_models.py --model ntscc
python scripts/check_retained_models.py --model swinjscc
```

The forward checks require the recorded Python/PyTorch family, torchvision,
numpy, Pillow and timm. The Swin manifest intentionally rejects incompatible
major/minor runtime versions. The NTSCC Windows compatibility layer supports
forward entropy estimates, not a complete CompressAI entropy-coded bitstream.
