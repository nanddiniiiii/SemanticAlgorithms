# DeepJSCC assets

`production/imagenet_10_0.33_200.00_32_19.pth` is the checkpoint used by the repaired Kodak benchmark. It was strictly validated against all 29 expected keys and has SHA-256 `00b33bac630555d9a0d72a76d0fea9103cef9a38df0d5547be53c5320a451d44`.

`cifar10/` contains one latest snapshot (`epoch_999.pkl`) per distinct training configuration found in the workspace. The redundant `epoch_998.pkl` snapshots were intentionally not copied. `configs/` contains the corresponding distinct configuration records.

The production checkpoint was trained/configured at 200 dB, so its 10 dB benchmark result is valid as a measurement but must not be presented as a matched-SNR reproduction.
