# Kaggle CLI guide for the JSCC training notebook

## Current execution status

Kaggle CLI authentication is verified. A private preflight kernel completed successfully on **2 Tesla T4 GPUs** and confirmed the attached DIV2K and Kodak inputs. A private DeepJSCC run has now been submitted with the author-linked PyTorch source and a 150-epoch AWGN + Rayleigh plan. It is currently running: [Kaggle kernel](https://www.kaggle.com/code/hmmkk20/deepjscc-div2k-kodak-150ep). Its output files must be checked before reporting any training results.

A dependency preflight observed Python 3.13.15 / PyTorch 2.11.0+cu128, with TensorFlow present but `tensorflow_compression`, NATTEN, and CompressAI absent. The official TensorFlow Compression release binaries currently document Python through 3.12; an isolated Python 3.11 runtime is under evaluation for ADJSCC. The NTSCC+ dependency installation preflight is still running.

The notebook `JSCC_DeepJSCC_NTSCCpp_ADJSCC_Kaggle_Training.ipynb` currently implements DeepJSCC training and Kodak evaluation. It does **not yet** implement ADJSCC or NTSCC+ training, does not measure FLOPs yet, and is not a completed three-model comparison. The NTSCC++ online editing implementation is not released in the author-linked NTSCC_plus repository; any benchmark using its released training core must be named NTSCC+, not NTSCC++.

Important scope: the DeepJSCC GitHub repository is an author-linked reproduction (its README notes possible differences from the paper). ADJSCC's official release uses TensorFlow Compression custom layers. The NTSCC_plus repository is not the unreleased NTSCC++ online editing implementation. Keep these identities distinct in code and results.

## Install and authenticate

In PowerShell:

```powershell
py -m pip install --upgrade kaggle
kaggle --version
```

In Kaggle, open **Account → API → Create New Token**. Save `kaggle.json` privately, then configure:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle" | Out-Null
Copy-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
```

Keep this token private and out of notebooks, repositories, and shared outputs. Check that authentication works:

```powershell
kaggle competitions list -s image
kaggle kernels list --mine
```

## Prepare datasets

Attach DIV2K and Kodak as Kaggle notebook inputs using the Kaggle UI, or download datasets you have permission to use with the Kaggle CLI. Dataset access and license terms vary. The notebook has explicit path controls at the top; set them to the mounted Kaggle input paths. It checks image counts and refuses to train on synthetic data or evaluate against training images. Keep Kodak held out until final evaluation.

## Upload and run

Kaggle notebooks run on Kaggle-managed accelerators. Create the notebook in Kaggle, choose **Accelerator: GPU T4 x2**, attach DIV2K and Kodak, enable Internet for the official repository clones, upload/import the notebook, and select **Run All**. Kaggle CLI notebook push requires a local Kaggle notebook directory with `kernel-metadata.json`:

```powershell
kaggle kernels init -p .\kaggle_jscc_notebook
```

Edit the generated metadata to use your `username/kernel-slug`, set `kernel_type` to `notebook`, and select GPU in the Kaggle editor's accelerator settings. Place the `.ipynb` in that directory and push:

```powershell
kaggle kernels push -p .\kaggle_jscc_notebook
kaggle kernels status username/kernel-slug
kaggle kernels output username/kernel-slug -p .\kaggle_jscc_output
```

The Kaggle editor is the reliable place to select two GPUs and add private dataset inputs. CLI upload does not grant dataset access or bypass Kaggle accelerator quotas. Training time and whether 150 epochs per model/channel fit within Kaggle session limits depend on model size, image crops, batch size, and current account quotas. The notebook checkpoints after each model/channel run so it can resume after interruption.

## Metrics and interpretation

- PSNR, SSIM, and multi-scale SSIM (MSSSIM) are computed on Kodak reconstructions.
- FLOPs are measured for a representative forward pass during training (forward-operation count; backward FLOPs are reported as an explicitly labelled estimate only if enabled) and for encoder/channel/decoder inference. FLOPs do not measure wall-clock time.
- CBR and channel uses follow each model's own transmission representation. Continuous analog JSCC symbols have complex channel uses, not a packet-bit count. Report symbol count, complex-use count, and CBR separately. For NTSCC++ any coded side information is counted only if the implementation actually serializes it.
- AWGN and Rayleigh results are distinct. The notebook records SNR, seed, channel, checkpoint, and source revision for each row.

Do not present results as exact paper replication unless the dataset, architecture, crop/resolution, SNR/rate points, training schedule, loss, optimizer, channel model, and code revision match the paper. Do not use Kodak for codebook fitting, validation, or checkpoint selection.
