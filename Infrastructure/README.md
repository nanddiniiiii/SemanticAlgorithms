# Radio conditions

`configs/environment.json` defines named profiles. Notebook setup resolves each profile through `src.notebook_support.environment_options` and validates the adapter **before** environment installation/training. Stage-specific `train_snr_db` and `test_snr_db` are authoritative. The legacy `snr_db` override applies to both only when no stage override is supplied.

## Channel contract

Real latent shape is `[B, ...]`, with an even number of real values per image. Each image is independently flattened; its first half is I and second half Q. Noise uses complex Es/N0: `N0 = 10**(-snr_db/10)`, with variance `N0/2` in each real component. Symbols are normalized to unit mean energy over **active** symbols. A rate mask must be binary and select matching I/Q pairs; inactive positions are zeroed again after transport. All-zero masks are rejected.

AWGN: `y=x+n`.
Rayleigh: `y=h*x+n`, `h ~ CN(0,1)`; receiver uses perfect CSI, `x_hat=y/h`.

`coherence="symbol"` means independent fading per complex symbol. `coherence="image"` means one complex coefficient shared by the whole image's transmitted block. Perfect CSI and normalization-scale restoration are idealized; pilots, headers and CSI acquisition cost are not included. Deep fades are not clipped and can greatly amplify equalized noise. Neither channel guarantees error-free delivery.

`restore_scale=True` returns decoder features to their pre-normalization scale; ADJSCC explicitly uses `False`, because its decoder expects normalized features. Computation remains differentiable; PyTorch device follows the input tensor.

## Current adapter coverage

| Model | AWGN | Independent train/test + Rayleigh |
|---|---|---|
| DeepJSCC | Infrastructure | Supported |
| ADJSCC | Infrastructure; normalized decoder input | Supported |
| SwinJSCC fixed C32 | Native by default | Explicit `channel_backend="infrastructure"` adaptation |
| SwinJSCC adaptive SA/RA | Native | Infrastructure replacement rejected pending mask integration |
| ISEC/JCM/MDJCM/shared codebook | Native supported contract | Rejected; do not replace their transmission chain silently |
| HJSCC | Native AWGN10 | Rejected |
| SQ-GAN | Source codec | No radio channel or channel SNR measured |

Default remains AWGN10 for the full roster. To run Rayleigh, select compatible models or override unsupported models back to native AWGN. This is simulated baseband transport, not ns-3/Sionna/OFDM.
