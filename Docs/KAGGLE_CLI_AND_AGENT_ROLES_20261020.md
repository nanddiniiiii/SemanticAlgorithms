# CLI setup and implementation ownership

## General-purpose CLI

The existing CLI is reusable across projects:

```powershell
& 'C:\Users\Nandini.P\Tools\kaggle-cli\Scripts\kaggle.exe' auth login --force
```

Complete the browser sign-in for the intended account, then enter the verification
code in that same local terminal. Do not paste credentials or verification codes
into chat, notebooks, public Git, or kernel metadata. Account quota and service
rules remain applicable. No training is submitted until the account is verified.

After authentication, these commands prepare, submit, check and retrieve a kernel:

```powershell
$KaggleCli = 'C:\Users\Nandini.P\Tools\kaggle-cli\Scripts\kaggle.exe'
& $KaggleCli auth login
& $KaggleCli kernels push -p 'ABSOLUTE_REVIEWED_KERNEL_DIRECTORY'
& $KaggleCli kernels status 'ACCOUNT/NOTEBOOK_SLUG'
& $KaggleCli kernels output 'ACCOUNT/NOTEBOOK_SLUG' -p 'ABSOLUTE_RESULTS_DIRECTORY'
```

The uppercase arguments are placeholders, not existing submitted jobs. Download
outputs only after publication; verify epoch logs, model files, hashes and metric
manifests. RUNNING is not evidence that 150 epochs completed. Preserve each run's
version and outputs in a separate directory. The reset cell refreshes source in a
new folder and never deletes checkpoints. Latest refers to the benchmark repo;
upstream author versions are pinned for reproducibility.

## Model provenance

| Model | Primary paper | Author code status |
|---|---|---|
| DeepJSCC | https://arxiv.org/abs/1809.01733 | Original author's baseline release not verified; third-party reproductions are not author code. The author's deepJSCC-feedback repository is a different feedback paper. |
| ADJSCC | https://arxiv.org/abs/2012.00533 | https://github.com/alexxu1988/ADJSCC ; TensorFlow/TFC. |
| SwinJSCC | https://arxiv.org/abs/2308.09361 | https://github.com/semcomm/SwinJSCC ; PyTorch. |
| NTSCC+ | Improved Nonlinear Transform Source-Channel Coding to Catalyze Semantic Communications | https://github.com/wsxtyrdd/NTSCC_plus ; PyTorch 1.9/CUDA11.1, NATTEN 0.14.6 explicitly required by authors. |

NTSCC++ online inference is still listed as unreleased in the author README.
The user approved using the released NTSCC+ and labelling it accurately.

## Sol work allocation

**Sol high:** paper/code equivalence; architecture and loss review; channel/CSI and
rate contract; native dependency isolation; two-GPU numerical equivalence;
NTSCC+ variable-rate gathering and auxiliary optimizer; padding and full-image
accounting; FLOP operator coverage; source approval and training launch gates.
These require design judgment and must not be delegated to low effort unchecked.

**Sol medium (spawned):** independent audit, focused CPU tests, bug investigation,
review of checkpoint/resume, source diffs, and validation of measurement helpers.

**Sol low:** implement reviewed schema/CLI wrappers, path handling, safe artifact
copy/download, table/plot formatting, notebook assembly, and documentation.
It must not choose a substitute model, change architecture/loss/channels, invent
metric values, silently change crop geometry, or certify reproduction fidelity.

No low/high agent has been claimed as spawned. The active audit agent uses
gpt-6-sol at medium effort, as requested. Training completion and equal/better
paper results are measured outcomes, not guaranteed by these roles.

## Important limits

DIV2K training in place of an author's original mixed dataset/ImageNet is a
declared controlled benchmark adaptation. Native channels differ across models.
Preserving author code and forcing identical transport are separate comparison
tracks, not interchangeable claims. No data reduction or epoch relabelling is
allowed to fit a 30-hour quota. Use measured runtime estimates before launch.
