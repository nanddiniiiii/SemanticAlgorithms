# Three-team experiment configuration

This is the operating guide for `Notebooks/TeamSuite/`. The canonical current
decision record remains `Docs/SEMANTIC_LOSSLESS_EXECUTION_PLAN.md`.

## Current validation boundary

The notebooks are configurable launchers, not twelve GPU-certified experiments.
Do not submit a full training queue until its initialization and real-data GPU
pilot pass. Source download, CPU execution and paper reproduction are distinct.
No full training queue has been launched. Bounded GPU pilot versions are
recorded in `Docs/KAGGLE_RUNS.md` and `Docs/TEAM_SUITE_VALIDATION.md`.

## Common data

Attach the same original HR images to each notebook, then fill `DATASET_ROOTS`:

| Key | Contents | Use |
|---|---|---|
| div2k_train | DIV2K's 800 training HR images | Training |
| flickr2k_train | Flickr2K's 2,650 HR images | Training |
| div2k_val | DIV2K's 100 validation HR images | Development only |
| kodak_test | Kodak's 24 RGB images | Final evaluation only |

Membership freezes before training. The manifest checks exact counts, hashes
file bytes and decoded RGB pixels, records duplicates inside training, and rejects
duplicates involving validation or test. All three
teammates must obtain the same manifest identity. It is not a near-duplicate or
scene-identity detector: human review/perceptual duplicate detection is still
needed for stronger leakage claims. DIV2K membership follows the
[official dataset split](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

The dataset preflight decoded and hashed the files mounted from
`soumikrakshit/div2k-high-resolution-images`, `daehoyang/flickr2k`, and
`drxinchengzhu/kodak24`. Attach these three Kaggle datasets. The verified
manifest identity is `add977eb3277909e5ad1447770abdb368458445ab42b2f48a91c24f4c31f7fa0`,
which the generated notebooks now require. It contains 59 repeated training
entries (3,391 distinct training RGB images); no exact duplicate involved a
protected split. File identity does not establish license or near-duplicate
status. Source membership must remain identifiable; do not substitute a combined
folder or LR/upscaled versions. The setup discovers unambiguous known folder
names; paths can also be supplied explicitly.

Training uses native-size random crops: most RGB models256 square, ISEC128
square, SQ-GAN256x512. Development uses center crops; Kodak is evaluated full
resolution, using explicitly charged tiles for fixed-size codecs. Shared
Codebook's256 tiles and SQ-GAN's256x512 tiles are geometry adapters and must be
reported. Native graph and paired-view tasks do not use this common RGB split.

## Setup and source

Git does not automatically execute setup code when cloned. The first notebook
cell creates per-model virtual environments, inheriting Kaggle's managed
Torch/CUDA stack rather than replacing it. Local setup is:

```powershell
python scripts/bootstrap_team_environment.py --managed-runtime
```

`--managed-runtime` is appropriate when the parent interpreter already has
Torch. Without it, setup creates an independent environment and installs Torch.
Heavy model dependencies are separate and have not all been runtime-validated.

The source must be published before a fresh public Git clone contains these
files. The generated launchers pin a reviewed source commit in `REVISION`;
update that pin only for a reviewed new source revision. No reset or pull
happens while another cell trains. Upstreams are fetched by fixed SHA;
their source receipts and project source-file hashes accompany each run.

## Per-model overrides

Use the notebook's `OVERRIDES` dictionary. Required special entries:

```python
OVERRIDES = {
    "shared_semantic_codebook": {"classifier": "/kaggle/input/your-assets/classifier.pt"},
    "deepjscc_wz": {"paired_data_root": "/kaggle/input/native-kitti", "paired_dataset": "KittiStereo"},
    "gbsed": {
        "python": "/absolute/path/to/validated/native/python",
        "training_extraction_config": "/absolute/path/to/task_oriented_extraction_config.yaml",
        "training_learning_config": "/absolute/path/to/task_oriented_learning_config.yaml",
        "evaluation_extraction_config": "/absolute/path/to/pipeline_extraction.yaml",
        "evaluation_learning_config": "/absolute/path/to/pipeline_learning.yaml",
        "receiver_weights": "/absolute/path/to/neural_rx_ofdm_mimo_cdl_final.h5"
    }
}
```

GBSED needs distinct native training and evaluation YAMLs, with absolute input
paths. Runtime copies redirect graph exports and task checkpoints into the new
run; author sources/checkpoints and original YAMLs are never overwritten.

ADJSCC, fixed-C32 SwinJSCC and native JCM initializer downloads have SHA256
checks. SQ-GAN's author archive has an archive digest; its10 parts must strictly
load. Author Baidu-hosted HJSCC/MDJCM assets are not silently treated as retrieved.
Those factories permit explicitly declared scratch training. ISEC scratch
training executes both JSCC and denoiser stages inside one budget.

JCM is the user-approved spatial reconstruction variant; its architecture and
objective changes are in `jcm_spatial.py`. ADJSCC uses DiffCom's PyTorch port.
DeepJSCC currently uses the declared third-party implementation, including its
known kernel difference; do not call it a paper-exact original-author model.
Shared Codebook remains the released SAC ablation, with frozen classifier
pseudo-label construction; it is not the paper's full proposed-loss method.

NTSCC+ requires NATTEN0.14.6. Its author warns that newer versions produce
incorrect checkpoint results. A failed legacy-kernel installation must block
training, not trigger an automatic substitution. Native HJSCC and Swin have
CUDA-only paths. GBSED needs its independent Torch/TF/Sionna/roadscene2vec stack;
its native source encoder compresses graphs and its learning job trains the
downstream risk classifier. DeepJSCC-WZ uses native Lightning DDP, not legacy
DataParallel; its training regime is explicitly adapted to fixed AWGN10.

SQ-GAN fine-tunes pretrained components in native `all` mode with its
adversarial/perceptual/VQ objectives and a frozen Cityscapes segmenter. Teacher
work is included in forward time. This is source-codec training: no AWGN link
has been implemented for SQ-GAN here. A different end-to-end radio variant needs
its own definition and validation. Its source packet bits are measured with
our existing SQ01 serialization, not an author entropy coder.

## Time and GPU use

The default upper bound is14,400 training seconds per model, checked between
steps; saving and an in-flight step may exceed it. Initialization, downloads,
teacher construction, pilots and evaluation add time. Four full-length models
need16 training hours per teammate before that overhead. Do not assume one
Kaggle session permits this: run a subset or reduce the configured budget after
measuring a pilot and checking actual session/quota limits. Automatic cross-
session resume is not implemented. Checkpoints preserve available optimizer
and RNG state but are explicitly labelled not exact mid-epoch resumable.

DataParallel RGB training requires two actual GPUs and observed encoder work on
both. Batch2 is a conservative starting point, not proof of no OOM. On OOM the
run stops, preserving earlier artifacts. Reduce batch/crop only in a separately
declared run; changing input size is not an automatic recovery action.

## Outputs and retrieval

Each notebook writes a unique `/kaggle/working/team_results/<team>/<session>/`
tree. Each algorithm has source/environment/configuration receipts, stage logs,
pilot artifacts, training logs, checkpoints, metrics, reconstructions and a
SHA256 artifact list. Native-task outputs retain their own schemas.

Save an executed Kaggle notebook version to publish its outputs. `/kaggle/working`
alone does not prove persistence. The CLI cannot guarantee instant cell-level
downloads from a running kernel. The three local scripts poll saved-version
status, retrieve terminal outputs, verify artifacts and merge image metrics
plus inference profiles into `Results/team_downloads/comparison.csv`:

```bash
python scripts/download_you.py --kernel OWNER/SLUG --version VERSION
python scripts/download_teammate_1.py --kernel OWNER/SLUG --version VERSION
python scripts/download_teammate_2.py --kernel OWNER/SLUG --version VERSION
```

Run each through its owner's authenticated Kaggle environment, or pass an
isolated `--config-dir`. Never put credentials in source/notebooks. Retrieval
does not submit or retry jobs. Downloads/results are ignored by Git; teammates
must not push binary results onto the shared source branch. The collector uses
an exclusive bounded lock and only verified exported files; repeated collection
rebuilds an idempotent CSV rather than duplicating rows.

## Metric interpretation

Pixel metrics: MSE, MAE, PSNR, SSIM, MS-SSIM, maximum error and exact pixel/sample
fractions. Perceptual: LPIPS. Semantic proxies: frozen ImageNet ResNet18 feature
cosine and predicted-class agreement, explicitly not true semantic accuracy.
Rate: measured complex symbols/CBR and actual source packet bits where available,
with missing radio/FEC/header overhead disclosed. Compute: parameters,
supported-kernel inference FLOPs, warmed latency and GPU memory. Component timing
is measured only where module hooks actually execute. ISEC inference includes
the required latent-gradient refinement, not only one decoder pass.

The default native rate settings are not matched across all twelve models.
ISEC width16, fixed-C32 Swin and spatial JCM spend different channel resources.
Always compare quality versus measured rate, and label unmatched operating
points; a common dataset and SNR alone do not justify a quality winner. A fair
matched-rate sweep requires each model's supported rate/configuration and its
own training/checkpoint budget. One four-hour checkpoint is a bounded training
result, not a reproduced paper rate–distortion curve.

BER, BLER, physical spectral efficiency, airtime and true task fidelity are not
invented when there is no implemented bitstream, waveform or labelled task.
FID/KID are not presented as dependable24-image distributional results.
`metric_coverage.json` records applicable/missing metrics. Native WZ/GBSED
profiling and normalized result ingestion are still incomplete.
