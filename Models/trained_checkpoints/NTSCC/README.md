# NTSCC assets

`production/ntscc_hyperprior_quality_1_psnr.pth` is the 10 dB checkpoint used by the repaired benchmark. It was strictly validated against all 426 expected keys and has SHA-256 `7a8d1990dbd678ea5d3daa02c3aeaaf4459b347a96d6aa81a5c02eae6bfc7323`.

`calibration/` contains the current rate-conditioned codebook calibration tensor cache and its provenance. It is a derived calibration artifact, not a neural-network checkpoint. Current Kodak-24 integration uses only the CIFAR-10 portion of this cache so benchmark images are excluded from quantizer fitting.
