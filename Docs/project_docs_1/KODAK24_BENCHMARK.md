# Kodak-24 shared benchmark

Use [the benchmark notebook](../Notebooks/Kodak24_All_Model_Benchmark.ipynb) as
the single entry point. It downloads/audits exactly `kodim01.png` through
`kodim24.png`, registers local models, runs every image at native resolution,
shows per-image and aggregate tables, and atomically appends measurements to
`results/kodak24_benchmark.csv`.

## Metric contract

- Input and reconstruction are RGB tensors in `[0, 1]`, with no benchmark-side
  resizing. Kodak images remain at `768x512` or `512x768`.
- Encoder and reconstruction latency are measured separately with device
  synchronization and inference mode. Reconstruction includes a registered
  channel/transport plus decoding. The first image can be warmed up; reported
  values are means over repeated timed passes. Quality metrics are outside the
  timed region, and LPIPS runs on CPU so its network does not pollute GPU peaks.
- Complexity includes model/trainable parameter counts, parameter-plus-buffer
  storage, process RSS, peak CUDA allocation, CPU process time, and optional
  model-supplied FLOPs.
- Quality includes RGB PSNR, RGB SSIM, and AlexNet LPIPS.
- `source_bits = H * W * 3 * 8`. `payload_bits`, BPP, and compression ratio are
  present only for serializable payloads. RAW RGB24 is the `1.0x` reference.
- CBR is **only** `complex channel uses / (3 * H * W)`. JPEG and RAW rows leave
  CBR empty unless a specific digital PHY mapping is added. This avoids calling
  stored bits radio symbols.

## Team workflow

1. Pull/rebase the latest main branch.
2. Use a unique run ID and contributor name in the notebook.
3. Register the model and include a checkpoint/version identifier plus its
   operating point (rate, SNR, quality, etc.).
4. Run the audit and full benchmark. A complete method/operating point must
   produce 24 rows.
5. Review the aggregate table, then run the explicit append cell once.
6. Commit the notebook/model integration and `results/kodak24_benchmark.csv`.
   The downloaded `data/Kodak24/` directory stays local and is ignored by Git.

### Repaired DeepJSCC/NTSCC notebook assets

Set `SEMCOM_EXPERIMENT_ROOT` to the external experiment directory before
opening the universal notebook. When present, the notebook loads the two strict
official checkpoints and adds native DeepJSCC, native NTSCC, DeepJSCC K=256,
and rate-conditioned NTSCC PQ4 K=256. The digital codebook packets use the
team's CRC, rate-1/2 LDPC, QPSK and AWGN path. For the full Kodak benchmark,
the NTSCC quantizer uses only its cached CIFAR-10 training calibration records;
Kodak calibration records from the one-image research notebook are excluded.
Because those CIFAR records contain only rate widths divisible by four, the
wrapper also fits the required exact-width-2 global fallback from adjacent
active values in the same CIFAR latents. No Kodak image is used for fitting.

On Nandini's Windows experiment layout, run:

```powershell
.\scripts\run_repaired_kodak24_notebook.ps1
```

The wrapper supplies the experiment root, contributor, unique run ID, a single
timed reconstruction per image, and the explicit append switch. Teammates can
use the same command with `-ExperimentRoot` and `-Contributor` overrides.

The append function de-duplicates by `(run_id, method, operating_point,
image_id)`, uses an exclusive lock, validates the fixed schema, and replaces the
CSV atomically. It does not solve Git conflicts between branches, so always pull
before measuring and before pushing.
