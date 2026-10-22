# Pretrained download ledger

Updated: 2026-10-22

This ledger records author-linked model artifacts fetched into this checkout. It documents file provenance and integrity only; model loading, inference, and benchmark results belong in the corresponding experiment records.

## Complete files

### JCM

Source: [author repository](https://github.com/SJTU-mxtao/Joint-Coding-Modulation-for-Digital-Semantic-Communication), `checkpoints/`.

| Local path | Native artifact | Bytes | SHA-256 |
|---|---|---:|---|
| `Models/pretrained_downloads/JCM/CIFAR_SNR12_Trans128_16qam.pth.tar` | `checkpoints/16qam/CIFAR_SNR12.000_Trans128_16qam.pth.tar` | 57,601,737 | `b7d3a1df66512421b057eabf4f7faa937a3dc22d6fe83632d3f0bca0f0c3b897` |
| `Models/pretrained_downloads/JCM/CIFAR_analog_SNR12_Trans128_2d.pth.tar` | `checkpoints/pretrained_analog/2d/CIFAR_analog_SNR12.000_Trans128.pth.tar` | 40,819,789 | `3264951d226479bbfc376c0cd9d0ae56847515d53c55705cff120a358c46cad0` |

Both are the author's CIFAR checkpoints for the native 12 dB setting. The first is 16-QAM; the second is the 2D analog initialization. The author names contain `SNR12.000`; this checkout uses shortened local names `SNR12`. No model loading or inference was done by the downloader.

### SQ-GAN

Source: [author repository](https://github.com/frapez1/SQ-GAN), [author-linked Drive archive](https://drive.google.com/file/d/10N0pxmfLbm-D2lYTOX-Y0014YicF2u-q/view), `Final_ckpt_parts.zip`.

The 691,499,362-byte ZIP passed archive listing and extraction. It contains ten component checkpoints (747,607,387 bytes total after extraction):

| Local path under `Models/pretrained_downloads/SQ-GAN/extracted/Final_ckpt_parts/` | Bytes | SHA-256 |
|---|---:|---|
| `decoder.ckpt` | 204,883,058 | `a6af82ae24c772790757f93861aa7ac835a684e8251eaeb597a5ecdee39bfe8f` |
| `decoder_semantic.ckpt` | 169,944,100 | `28e1f6f400c3f09f01c77c11a1190b800e7886323f30b876e47bd8ad642b462e` |
| `demasker.ckpt` | 58,375,175 | `87a83124c20d839f6fef65f724d86dc9445ab638994834d58bbf7dd3a5408b65` |
| `demasker_semantic.ckpt` | 58,376,850 | `d6d6dad1331ff8a832f20ad5e0256d6d341cb82d39fcd77a09c7bca54e6dcf1d` |
| `encoder.ckpt` | 117,332,422 | `6afacd86a809fb8f2e2f898934a4a5e8797ac034a50ad90234d12ef36dba0268` |
| `encoder_semantic.ckpt` | 117,322,708 | `e40534ce8c0335d01e4a201fafc47121dc5102f2a217e525cb9973facd61572f` |
| `masker.ckpt` | 8,581,630 | `da3187ab8165c0151f3e7e1efa08f9342c10ec89cfbd7ab03cad734f822ba678` |
| `masker_semantic.ckpt` | 8,583,135 | `0932245c0a1a3e82fb44d99c87017e5983fddd6e558f8966624be7cebe5fda99` |
| `quantize.ckpt` | 2,104,091 | `10bf41064c07bca2d46a3bdd06799f28e48212813553f51028f5be38847bee95` |
| `quantize_semantic.ckpt` | 2,104,218 | `46c4968df648b214c86ca25e648612c59107ab395c01e8819531db45d2dbc20c` |

Archive SHA-256: `7bc059f3334993f9acd9f215132391c77ffbd1484d4750f93d58dec00152facf`.

The archive is a multi-component checkpoint set; pair it with the pinned upstream SQ-GAN code/configuration. The downloader has not inferred an independent serialization schema from these files.

### ADJSCC checkpoint used by DiffCom

Source: [DiffCom author repository](https://github.com/wsxtyrdd/diffcom), [author-linked Drive folder](https://drive.google.com/drive/folders/1N0EzzxCv1wh6JeFr0g8vkmB0Qj23ozZJ?usp=sharing).

| Local path | Drive artifact | Bytes | SHA-256 |
|---|---|---:|---|
| `Models/pretrained_downloads/ADJSCC/ADJSCC_C=2.pth.tar` | `ADJSCC_C=2.pth.tar` | 63,775,944 | `c29d7610eb7b2e475a5ff029d16f0031fc43ac89f316d461b154cef993753c54` |

The Drive response's content length and filename matched the downloaded artifact. ZIP validation found 89 entries and no CRC errors. This is the PyTorch ADJSCC checkpoint linked from DiffCom; it is distinct from the missing local TensorFlow ADJSCC files noted in earlier audits. No model loading or inference was done by the downloader.

### DiffCom ImageNet diffusion prior

Source: [DiffCom author repository](https://github.com/wsxtyrdd/diffcom), [author-linked OpenAI blob](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt).

| Local path | Artifact | Bytes | SHA-256 |
|---|---|---:|---|
| `Models/pretrained_downloads/DiffCom/256x256_diffusion_uncond.pt` | ImageNet 256×256 unconditional prior | 2,211,383,297 | `a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1` |

The final file size matches the source `Content-Length`. The local MD5 (`fd9dd2335b8736d521de0aed54bd90ca`) also matches the source `Content-MD5`. The source does not provide a SHA-256, so the SHA-256 above is the locally computed integrity identifier. No model loading or inference was done by the downloader.

### MambaJSCC

Source: [author public Drive folder](https://drive.google.com/drive/folders/1nrX6FiULN04PomW2eVaWwNDmM3Hhka0Y?usp=sharing).

One native CLIC2021 AWGN pair is downloaded for 256x256, SNR 10 dB, PSNR loss, `OUTCHANS=16`, `extent=MLP`, `SCANnum=2`, `adaptive=no`, `depth=4`, `embed=[128, 192, 256, 320]`, `nums=[2, 2, 6, 2]`.

| Local path | Native artifact | Bytes | SHA-256 |
|---|---|---:|---|
| `Models/pretrained_downloads/MambaJSCC/encoderOUTCHANS16_extentMLP_lossPSNR_SCANnum2_SNR[10]_adpno_typeawgn_depth4_embed[128, 192, 256, 320]_nums[2, 2, 6, 2]_rsl256_CLIC2021.pt` | encoder (same filename as Drive) | 29,279,400 | `a9f55f6a5f85c9034cc1e7d1bb9974e6fb377b112a69215d9e70c739125918c5` |
| `Models/pretrained_downloads/MambaJSCC/decoderOUTCHANS16_extentMLP_lossPSNR_SCANnum2_SNR[10]_adpno_typeawgn_depth4_embed[128, 192, 256, 320]_nums[2, 2, 6, 2]_rsl256_CLIC2021.pt` | decoder (same filename as Drive) | 29,266,598 | `cf21380246bb1d77b8b62b4dff39640a2c64dc125cd289b8eca961f227cc0316` |

Safe archive inspection (without unpickling) shows these are full serialized module objects referencing `models.network.Mamba_encoder` and `models.network.Mamba_decoder`, not plain state dictionaries. The internal archive directory says `rsl512_epoch0_i99999`, which conflicts with the external filename tag `rsl256`; confirm the intended resolution against the pinned source/config before loading. No model load or inference was performed.

#### CSI-ReST (adaptive) Rayleigh checkpoint pair

At the author's Drive folder there is no AWGN `adpssm` pair. The available CSI-ReST pair selected for preservation is C=32, Rayleigh, 256x256, PSNR loss, and SNR values `[1, 3, 5, 7, 9, 11, 13, 15, 17, 20]` dB. It is a separate channel-specific model from the AWGN `adpno` ablation above; do not report it as AWGN-trained.

| Local path | Native artifact | Bytes | SHA-256 |
|---|---|---:|---|
| `Models/pretrained_downloads/MambaJSCC/encoderOUTCHANS32_extentMLP_lossPSNR_SCANnum2_SNR[1, 3, 5, 7, 9, 11, 13, 15, 17, 20]_adpssm_typerayleigh_depth4_embed[128, 192, 256, 320]_nums[2, 2, 6, 2]_rsl256.pt` | encoder (same filename as Drive) | 29,313,858 | `6282cdea3f2e89813d00c25ccde7babbef5d741e3a2a393dabfdea2cfe7d68b9` |
| `Models/pretrained_downloads/MambaJSCC/decoderOUTCHANS32_extentMLP_lossPSNR_SCANnum2_SNR[1, 3, 5, 7, 9, 11, 13, 15, 17, 20]_adpssm_typerayleigh_depth4_embed[128, 192, 256, 320]_nums[2, 2, 6, 2]_rsl256.pt` | decoder (same filename as Drive) | 29,300,842 | `bb394b67804a9de410e21202687ff7579429f5e798c01b057903c894e14e0423` |

Safe archive metadata confirms these are full serialized `models.network.Mamba_encoder` and `models.network.Mamba_decoder` objects. No unpickling, model loading, or inference was performed.

#### Drive availability audit

The public author Drive folder contains 16 files (eight encoder/decoder pairs). All listed AWGN/CLIC2021 PSNR pairs use `adpno` (no CSI adaptation), at OUTCHANS 16, 32, 48, 80, and 115; there is no listed AWGN `adpssm` CSI-ReST pair at C=16 or any other channel count. The `adpssm` pairs are only C=32 Rayleigh at resolution 256 and SNR values `[1, 3, 5, 7, 9, 11, 13, 15, 17, 20]`, with separate LPIPS, MSSSIM, and PSNR variants. Only the PSNR pair is downloaded.

### DiffJSCC OpenImage CBR 1/384

Source: [Hugging Face model repository](https://huggingface.co/Mingyuyang/DiffJSCC-OpenImage-CBR-1-384), pinned revision `58eb8a69fe83318e0a5aa0fae2b6c2ff2c518853`.

The minimal config and README and full `model.ckpt` checkpoint are downloaded from the pinned revision. File size is 9,859,323,853 bytes. SHA-256 `580382b0fe441c3cc516dcc3f458493e6518667b0913510be02cdc01985a4b78` matches the Hugging Face LFS SHA-256; source Xet hash/ETag is `f29159330bb112151af51b6efdbaed26bf573e9aea2d7042df267e473bae0100`.

| Local path | Bytes | SHA-256 |
|---|---:|---|
| `Models/pretrained_downloads/DiffJSCC/config.yaml` | 2,883 | `67463fdfe457697f699c16529092876cb3bf2cf67237bbce6967b6950b222902` |
| `Models/pretrained_downloads/DiffJSCC/README.md` | 31 | `4bcf87ecfbbb8e07a01b21415a970c8b53a5283bf6872b657040d3f45c9241f7` |
| `Models/pretrained_downloads/DiffJSCC/model.ckpt` | 9,859,323,853 | `580382b0fe441c3cc516dcc3f458493e6518667b0913510be02cdc01985a4b78` |

The config identifies a latent-diffusion/ControlNet reconstruction pipeline and enables SNR conditioning (`use_snr: true`, `min_snr: 0`, `snr_scale: 100`). Hugging Face reports a pickle-scan caution for `pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint`; do not load the checkpoint casually, and only use it with the pinned/trusted upstream code in an isolated environment. No model loading or inference was done by the downloader.

## Pending or not downloaded

| Model/artifact | Author source | Status |
|---|---|---|
| DiffCom ADJSCC C=4 and C=6 | [author-linked Drive folder](https://drive.google.com/drive/folders/1N0EzzxCv1wh6JeFr0g8vkmB0Qj23ozZJ?usp=sharing) | Not requested; C=2 is complete above. |
| MDJCM-A checkpoint | [author README](https://github.com/zhang-guangyi/Semantic-MDJCM#pretrained-models), [Baidu share](https://pan.baidu.com/s/1-xiEIH6eG2xnsOPDvlbo3Q) | Not downloaded. The public Baidu endpoint forcibly closed the unauthenticated probe; no account credential was supplied or attempted. |
| MambaJSCC other variants | [author's Drive folder](https://drive.google.com/drive/folders/1nrX6FiULN04PomW2eVaWwNDmM3Hhka0Y?usp=sharing) | AWGN SNR 10 dB C=16 no-CSI pair and Rayleigh CSI-ReST C=32 PSNR pair are downloaded above. No AWGN CSI-ReST pair is listed in the author folder; other AWGN C values and Rayleigh LPIPS/MSSSIM pairs were not downloaded. |
| DiffJSCC OpenImage CBR 1/384 | [Hugging Face model](https://huggingface.co/Mingyuyang/DiffJSCC-OpenImage-CBR-1-384) | Complete above; exact size and LFS SHA-256 verified. HF pickle scan carries a caution noted above. |

No other files in this ledger should be described as locally downloaded or checkpoint-validated. Hashes were computed from the completed local files using SHA-256. No upload, Git push, training, or FLOPs measurement was performed by the downloader.
