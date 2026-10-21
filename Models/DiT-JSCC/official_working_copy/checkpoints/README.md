# Checkpoints

Model binaries are intentionally not committed to Git. This directory and the
public loading interfaces are reserved for a future checkpoint release. When
weights become available, place them here:

```text
checkpoints/
├── autoencoder_kl.pth
├── dit_jscc_256.pth
└── dit_jscc_512.pth
```

Expected contents:

- `dit_jscc_256.pth`: state dict for the 256 × 256 dual-branch DiT-JSCC network.
- `dit_jscc_512.pth`: state dict for the 512 × 512 dual-branch DiT-JSCC network.
- `autoencoder_kl.pth`: Stable Diffusion KL autoencoder state dict compatible with `inference_core/libs/autoencoder.py`.

The loader accepts a plain state dict or a dictionary containing `state_dict`, `model`, `nnet`, or `ema`. Distributed prefixes such as `module.` are removed automatically.

Validate paths without loading model memory:

```bash
python infer.py --check --resolution 256
python infer.py --check --resolution 512
```

If you distribute weights through a GitHub Release or model hub, keep the filenames above stable so the CLI and web demo work without extra arguments. Model files are covered by the Git LFS patterns in `.gitattributes`, but the default `.gitignore` prevents accidental commits until you intentionally change that policy.
