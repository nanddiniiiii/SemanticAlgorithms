# SQ-GAN stage coordination and artifact merging

## Current deployment

The legacy Kaggle job `hmmkk20/sqgan-div2k-150ep-continuation-02`, version 1,
continues unchanged. Its running Python process contains a broad sequential loop;
that loop cannot be patched from the local filesystem. Stopping it now would
risk losing updates since its last completed epoch and successful publication.

The old local `continue_training.py` watcher was stopped and its STOP marker is
present. Its pre-handoff state was copied to
`Results/SQGAN/stage_coordination/legacy_watcher_state_before_handoff.json`.
This stops automatic submission of another broad sequential worker, not the
running Kaggle session.

`SharedCode/sqgan/coordinate_stages.py` is running locally. Its authoritative
registry is `Results/SQGAN/stage_coordination/registry.json`. Current phase is
`WAITING_FOR_LEGACY_PUBLICATION`. Independent training workers have NOT been
launched yet. The coordinator takes over only after the legacy job exits normally
and its output manifest and checkpoint inventory are available.

The computer must remain awake and connected for dispatch. Existing submitted
Kaggle sessions keep executing independently. The registry stores exact notebook
references, versions, owners, checkpoint progress and parent artifacts.

## Scheduling rules

- At most two GPU workers, each requesting T4 x2. The speed benchmark consumes a
  slot while active. NTSCC_plus stays stopped.
- An ownership reservation is saved before submission. No second worker may own
  an active stage. An interrupted API response does not authorize a duplicate.
- Partially completed stages are resumed before fresh stages. Completed stages
  are skipped. Every worker has an explicit one-stage plan.
- AWGN semantic and image pretraining are independent; likewise for Rayleigh.
- Joint training for a channel is eligible only when BOTH its semantic and image
  pretraining checkpoints have completed 150 full epochs.
- After all six stages complete, evaluation runs once, or in a separate evaluation
  worker if a training chunk had insufficient remaining time.
- Training remains FP32, DDP, one image per GPU, 256x512 geometry, 800 images per
  epoch, 150 epochs per stage, 10 dB. No benchmark optimization is silently enabled.

The first pair depends on the actual published legacy inventory. For example,
if semantic pretraining has finished and image pretraining is partial, the
coordinator resumes that partial image stage alongside an eligible Rayleigh
pretraining stage. It does not assume where the legacy job will finish.

## Merge rules

`stage_artifacts.py` combines artifact directories, NOT independent models by
averaging weights. It validates the pinned author commit, SNR, channel protocol,
batch size, image geometry, matching pseudo-label metadata, checkpoint paths,
both optimizer states and both schedulers.

The SQ-GAN fixed training contract has 400 distributed batches per full epoch and
two optimizer steps per batch. Full-epoch checkpoints therefore have global steps
divisible by 800. The merge code cross-checks the raw saved epoch against that
count and rejects partial-epoch or out-of-range checkpoints. This also avoids the
known lagging Lightning epoch counter in old marker files.

For each stage, select the greatest verified global step. Equal-progress files
with different SHA256 hashes cause a conflict requiring review; no silent choice
is made. Selected checkpoints are copied byte for byte, with raw optimizer and
scheduler states retained for continuation. Stage outputs are kept separately.

When a pretraining stage is complete, its named encoder/decoder/masker/demasker/
quantizer components are extracted from that validated checkpoint. The final
joint stage loads both sets using the author's component-loading mechanism.
It uses the author's joint-training optimizer setup, rather than trying to average
or combine incompatible pretraining optimizer states.

Worker templates are frozen with a hash. Unexpected template, merge-code or
Kaggle version changes require review. Failed or non-advancing workers are not
silently relabelled complete.

## Executed checks

`Results/SQGAN/stage_coordination/local_contract_checks.json` records passing
synthetic tests for highest-progress selection, checkpoint conflict rejection,
partial-stage priority, exclusion of active ownership, joint-stage prerequisites,
one-stage notebook generation and fixed training settings. These are software
tests, not model training results.

A separate CPU Kaggle run executed the merge on a REAL 1,086,702,902-byte checkpoint:

https://www.kaggle.com/code/hmmkk20/sqgan-stage-merge-validation-20261020

- 98 completed epochs and global step 78,400.
- Two optimizer states and two schedulers.
- Source and merged checkpoint SHA256 match exactly:
  `5f4e5c181f95e12caa994aa5746faea8d589075442748f1089f5d81e141df3eb`.
- Zero training updates were performed by validation.

The 98-epoch artifact is a validation fixture. Production resumes from the NEWEST
successful output of the active legacy job, not from this older fixture.

Real two-worker parallel training and joint-stage loading are still pending the
safe handoff. The coordinator performs additional checkpoint validation in each
worker before training. No Git commit or push was performed.
