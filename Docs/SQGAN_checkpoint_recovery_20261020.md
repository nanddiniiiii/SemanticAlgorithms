# SQ-GAN recovery and continuation, 20 October 2026

The official SQ-GAN source is pinned to `0e5cdd7c0240b89749dad684645c53f783ac24cc`. This experiment uses DIV2K pseudo-segmentation and an analog AWGN/Rayleigh quantizer adapter, and must be reported as an adaptation of the official Cityscapes experiment. The training orchestration changes below do not modify the official model source.

## What actually happened

- Kaggle version 15 (`360104654`) was cancelled after 12 hours, as shown by Kaggle version history. Its saved log reaches epoch index 97 at about 42,877 seconds. The browser log had stopped displaying new records much earlier, so the earlier 23-epoch progress report understated the actual run.
- Kaggle's output API lists four saved checkpoint files, including `last.ckpt`. The checkpoint was present despite the notebook page showing output as 0 B.
- The Kaggle CLI implementation opens its local destination before evaluating `response.content`. A large, incomplete download can therefore show a zero-byte local file. That placeholder is not proof that the remote checkpoint is empty.
- A separate private CPU notebook downloaded the original checkpoint on Kaggle, checked its byte count and SHA256, loaded it using PyTorch, and completed successfully. Its durable output is attached to the training continuation.

## Independently verified checkpoint

Recovery notebook: https://www.kaggle.com/code/hmmkk20/sqgan-checkpoint-recovery-20261020

- Bytes: `1086727541`.
- SHA256: `6f3e02b84b7972501f279ddedb182497d2a8b21cecc03071d6195d003360a22d`.
- Saved epoch index: `96`.
- Global step: `77600`.
- Model state tensors: `562`.
- Optimizer states: `2`; learning-rate schedulers: `2`.
- Both saved learning rates: `6.950352863956827e-06`.
- Lightning's saved epoch counters contain `processed=97`, `completed=96`. These counters are recorded as-is; adding one blindly to checkpoint `epoch` is unsafe because checkpoints taken after `fit()` have different boundary semantics.

## Continuation verified on the actual model

Version 16: https://www.kaggle.com/code/hmmkk20/sqgan-div2k-150ep-20261019/log?scriptVersionId=360296846

The live log confirms:

- Two Tesla T4 GPUs.
- The pinned source integrity check passes.
- The CPU two-optimizer stop/save/resume test passes.
- The author `--resume` path restores all states.
- The runtime verification reports epoch index `97`, step `77600`, both original learning rates, and `resume_verified=true`.
- Training advances through epoch index 97 with finite losses.

This is ordinary Lightning state continuation. No claim of bit-for-bit equivalence to an uninterrupted random-number stream is made.

### Additional export issue found during verification

Lightning 1.9 versions its last-checkpoint filename when restoring into a different log directory that already contains `last.ckpt`; it can write `last-v1.ckpt`. Watching only the literal `last.ckpt` therefore misses the new periodic saves. The exporter now selects the newest `last*.ckpt` for each stage and publishes it under the canonical `last.ckpt` name. An isolated filename-selection test verifies this behavior. This changes artifact handling, not the trained architecture.

Version 16 was deliberately stopped after discovering this exporter issue. Its extra updates are not counted as retained training progress. Version 17 is a short real-model publication check, resuming the independently validated checkpoint and stopping normally at cumulative epoch 98. The continuation watcher uses a full-length source snapshot with the original 150-epoch target for subsequent chunks.

## Session and artifact fixes

1. Keep `--max_epochs 150` on every stage invocation so the original cosine schedule retains its total horizon.
2. Apply one absolute 9.5-hour budget to the entire notebook, including transitions between stages. A runtime callback requests a normal stop at an epoch boundary and leaves time for serialization and output publication.
3. Keep the cloned repository, diagnostic images, and top-k checkpoint duplicates under `/tmp/SQ-GAN`. Export one latest complete Lightning checkpoint plus configs per stage to `sqgan_div2k_channel_results/upstream_logs/`.
4. Mirror complete checkpoint writes atomically. Check file stability before and after copying; serialize concurrent mirror operations with a lock.
5. Record real checkpoint counters, optimizer/scheduler state counts, per-stage status, epoch metrics, and output file sizes. Partial stages are never marked complete.
6. Skip completed stages and resume partial stages from the preceding successful notebook output.
7. Run Kodak metrics and FLOPs profiling only when all six stages meet their 150-epoch targets. The training FLOPs profile applies the author's semantic-parameter freeze in the final stage. FLOPs remain explicitly labelled as profiler-counted operations.
8. Export final model weights as `.pth` files after evaluation. Do not claim comparison readiness before the metrics CSVs exist.

## Automatic continuation

`SharedCode/sqgan/continue_training.py` watches the current Kaggle execution using the existing local Kaggle authentication. After a successful chunk it validates the output manifest and submits a new private continuation notebook with the previous successful output mounted as input. No Kaggle account credential is embedded in a notebook.

The runner's state, logs, immutable source snapshot, and downloaded chunk manifests are under `Results/SQGAN/continuation/`. It stops if a notebook ends unsuccessfully, an unexpected edit changes the version, or the manifest is invalid. It never fabricates epochs or advances a failed stage.

This local watcher requires the computer to stay awake and connected. Kaggle continues the currently submitted GPU job independently. Creating `Results/SQGAN/continuation/STOP` stops the local watcher before it submits another chunk; it does not cancel an existing Kaggle execution.

## Remaining runtime estimate

The request requires three author stages per channel, each for 150 epochs: 900 stage-epochs. The recovered first semantic stage has completed 97 training data passes and resumed at index 97. Its recent epochs took approximately 440 seconds. About 53 epochs remain in that stage, or roughly 6.5 hours at the observed rate.

If the other stages took the same time, approximately 98 GPU execution hours would remain. Their image and joint paths have different costs; allow roughly 100–160 active GPU execution hours for planning, plus evaluation and Kaggle queue/quota delays. This is a conditional estimate, not a completion guarantee.

No Git commit or push was made for these changes.


## Live verification, 20 October 2026, 12:17 IST

The isolated private publication test completed successfully in 704.1 seconds with 1.09 GB published output. Its real epoch audit records completed_epochs=98, global_step=78400 and epoch_seconds=464.0285; both optimizer and scheduler states are retained in the exported 1086702902-byte checkpoint. The raw manifest conservatively reports 97 from a lagging Lightning completed counter; the saved epoch is 98 and the epoch audit confirms 98 full passes. Future source snapshot v2 corrects this reporting counter using processed epochs as well. It does not alter training.

A stalled Kaggle API call in the local monitor was cleared, and bounded request timeouts were added. Kaggle normalized the continuation slug from its title; the monitor now follows the returned URL. Active continuation: https://www.kaggle.com/code/hmmkk20/sqgan-div2k-150ep-continuation-02/log . API status is RUNNING; actual restore and batch logs remain to be confirmed. The current notebook is unchanged; subsequent chunks use source_snapshot_v2.ipynb.

Finite losses establish successful numerical execution, not convergence. The saved audit has a very small training discriminator loss and a much larger validation discriminator loss. Further epoch trends and reconstructions are required before declaring model quality healthy. Author losses and architecture remain unchanged.


## Training continuation confirmed

The Kaggle CLI live stream confirmed author source integrity, restoration at epoch_index=98 and global_step=78400 with resume_verified=true, followed by TRAIN PROGRESS epoch_index=98 batch_index=0 global_step=78402 training_batches=400. Both CUDA ranks registered. This proves actual training in the 99th epoch, rather than only RUNNING metadata. No training modification was needed. Evidence: Results/SQGAN/continuation/continuation_02_training_verified.json.
