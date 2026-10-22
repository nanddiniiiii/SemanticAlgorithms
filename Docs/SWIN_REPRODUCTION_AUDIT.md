# SwinJSCC retained-port audit

The retained SA+RA manifest itself says "corrected modern PyTorch port; not a
paper-exact replication". It is not an author pretrained checkpoint. Native
author sources are separately pinned at
`semcomm/SwinJSCC@a6d0e6da53548976acbe9317839a077ef31f190f`.

## Measured masked-power mismatch

The native author's `net/network.py` calculates active real-coordinate power
as `sum(feature**2)/mask.sum()`, passes it to the channel, and masks received
coordinates before decoding.

The retained port instead normalizes complex-symbol power over its complete
dense tensor, including inactive zeros, while counting only active symbols.
For C active channels out of 192, this produces an active-symbol SNR increase
of `10*log10(192/C)` dB relative to its nominal setting. At nominal 10 dB:

| Active feature channels | Predicted active SNR | Measured synthetic active SNR |
|---:|---:|---:|
| 32 | 17.7815 dB | 17.7801 dB |
| 64 | 14.7712 dB | 14.7650 dB |
| 96 | 13.0103 dB | 13.0023 dB |
| 128 | 11.7609 dB | 11.7463 dB |
| 192 | 10 dB | 9.9933 dB |

`scripts/check_retained_swin_power.py` reproduces this finding without modifying
retained source or weights. Its receipt is in `Docs/validation/`.

## Treatment

The 24-image Kodak run at nominal 10 dB/rate32 is retained as an audit artifact,
not a valid native 10 dB benchmark. It is excluded from the main experiment root.
Generic evaluation and training of this adapter are disabled until a native
author path is verified. Checkpoint loading and computational profiling remain
useful runtime checks, but cannot establish paper quality or training fidelity.

Changing the channel alone would also change the conditions under which this
port's weights were trained. It must not silently become an author baseline.
The native author checkpoint download and integration are separate work.

## Native checkpoint strict-load check

Both author AWGN HR base checkpoints strictly load against the unmodified pinned
configuration: fixed C32 has 28,162,000 parameters and SA+RA has 33,073,488.
The AST-extracted config avoids executing main.py's CLI and CUDA side effects.
No learned weights or model math were altered. This was CPU loading only;
native CUDA forward, rate checks and Kodak quality are still pending.
Receipt: Docs/validation/swin_native_checkpoint_load.json.
