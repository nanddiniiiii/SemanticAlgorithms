# Person 3 execution contracts

HJSCC, OFDM-guided JSCC, DeepJSCC-MIMO and DeepJSCC-WZ are implemented through the common numbered-notebook stage interface. This is not a claim of four completed paper reproductions or GPU runs.

## Models

- HJSCC: existing pinned author hierarchical-VAE adapter, native AWGN10 and objective. Reconstruction hook is checked against the author's reported PSNR. CUDA is required; existing historical T4 pilot evidence applies only to its tested snapshot. Common data/budget changes are adaptations.
- OFDM: implemented author encoder/decoder/CE/EQ, multipath/pilots/CP, audited modern FFT compatibility and native loss/optimizer/scheduler. CPU optimizer/reload checks pass; actual dual-T4 allocation/performance remains a pilot requirement.
- MIMO: implemented released transformer/SVD/perfect-CSI model; source pinned and explicit device moves. Native32x32 geometry preserved through tiled evaluation. CPU optimizer/reload checks pass. It is not a new resolution-adaptive architecture.
- WZ: the native paired Lightning model/optimizer/loss/transform pipeline is executed by scripts/wz_native.py, isolated from this repository's src namespace. Training uses native DDP strategy with two CUDA devices. Bounded pilot completes two steps, saves a checkpoint and explicitly reloads strict learned weights; native fast_dev_run is avoided because it suppresses checkpoint callbacks. Training saves optimizer/scheduler/progress in the Lightning checkpoint; exact resume is not claimed. Evaluation uses a fixed explicitly configured SNR, not the native multirun preset. Native returned metrics, per-image pixel metrics and model profiling are persisted. Clean right-view side information and CSI cost are explicitly excluded from the reported transmitted-left-latent rate.

## Required WZ inputs

Set OVERRIDES['deepjscc_wz']={'paired_data_root':'/kaggle/input/YOUR_NATIVE_PAIRED_DATA','paired_dataset':'KittiStereo'}. For KITTI the root must contain KITTI_stereo_train.txt, KITTI_stereo_val.txt and KITTI_stereo_test.txt, with alternating relative left/right paths. For Cityscape use paired_dataset='Cityscape' and cityscape_dataset/{leftImg8bit,rightImg8bit}/{train,val,test}/<city>/ native filenames. Setup checks all three split layouts, image file existence and path disjointness. It does not infer missing stereo pairs or make the target image into side information. Native transforms produce128x256 RGB; KITTI crops370x740 before resize. Do not add DIV2K/Kodak as fake paired WZ inputs.

WZ is a separate native paired-data comparison. The common dataset manifest is not represented as its actual training/test data. Results/metrics use the native dataset/task identity. All stages retain logs/status/config/checkpoints; successful bundles include native_recipe.json and last.ckpt. load_bundle reconstructs using the paired native model constructor; inference input is ((left,right),csi), not a single RGB tensor. Trusted self-generated Lightning checkpoints require their native Python dependency environment.

## Validation limits

Local tests cover split validation, command/config wiring, native strict-load and metric persistence paths by inspection. WZ's full Lightning/VGG/CompressAI environment and actual paired GPU dataset were not executed on this CPU environment. Its notebook pilot must pass before training; missing data/dependencies are errors, never substituted architectures. OFDM/MIMO/SSC actual CPU optimizer and checkpoint checks are recorded in NATIVE_SYSTEMS_IMPLEMENTATION.md. Real two-GPU DDP/NCCL remains unverified for the changed code.

Latest local validation:82 tests passed across the existing suite, including real CPU OFDM/MIMO optimizer/reload checks and WZ paired-layout/dispatch tests. The WZ-specific native dependency environment is not present locally, so this is not evidence of its full Lightning execution.

## Validation follow-up — 2026-10-28

Fresh full suite before follow-up fixes: 82 passed. Real CPU author OFDM/MIMO/SSC optimizer and strict checkpoint tests executed. WZ bundle round-trip regression uses a small controlled model and does not establish native architecture execution.

Fixed during validation: WZ tuple/path error in bundle loading; missing Hydra Joblib launcher dependency; invalid logger=null override (now ~logger); rank-zero ownership for recipe writes; missing WZ pilot gate; rejection of zero-update training; native split whitespace mismatch. Exact current worker Hydra overrides were executed with Hydra1.3.2, hydra-colorlog1.2.0 and hydra-joblib-launcher1.2.0: composition passed, selecting DeepJSCCWZBaseline2, N256, bw_factor16.

Readiness limits remain: no current full native WZ Lightning/VGG run, no real paired dataset pilot, no current dual-T4 certification of all Person3 pipelines. Kaggle input metadata audit found no verified complete native three-split stereo attachment. See experiments/person3_wz_input_audit/ and the existing Kaggle artifact map. Person2 v1 was not changed or interrupted.
