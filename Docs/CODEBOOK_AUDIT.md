# Current codebook implementation audit

This audit compares `semcom_benchmark/codebooks.py` and its complete call path
with the team repository at commit
`6736d926cfe6de9b5c162b93a8b12758ba026634`. No official or team source file
was modified during this audit.

## Bottom line

The implementation is a real, executable K=256 vector-quantization baseline.
It reuses the team's packet schema, fixed-width index serialization, metadata
validation and CRC exactly, and it sends those real packet bits through the
team's LDPC/QPSK/AWGN path. DeepJSCC is represented naturally as one 19-D
vector per spatial latent location.

The NTSCC path is mechanically valid but is **not structurally NTSCC-aware**.
It clusters one zero-padded 256-D vector per spatial token with a single
ordinary K=256 Euclidean codebook, even though NTSCC assigns different active
channel counts to different tokens. The rate indexes are transmitted and the
official decoder receives a correctly remasked tensor, but rate-conditioned
distance, active-dimension-only training and rate-specific codebooks are not
implemented. This is an important explanation for the measured quality loss;
no redesign is made in this pass.

## Exact call path

1. `BenchmarkSession.fit_codebooks()` obtains frozen official-encoder latents
   from eight CIFAR-10 training images resized to 128x128. DeepJSCC and NTSCC
   use separate calibration lists and separate codebooks.
2. `K256LatentCodebook.fit()` calls `tensor_tokens()` and trains a scikit-learn
   `MiniBatchKMeans` with K=256.
3. `BenchmarkSession.prepare_codebook_packets()` encodes either all 24
   DeepJSCC test-image tile latents or the one full-resolution NTSCC channel
   latent.
4. `K256LatentCodebook.encode()` predicts nearest-centroid indices and calls
   the team's `CodebookPacketHeader` and `encode_codebook_packet_bits()`.
5. The noiseless ablation sends the resulting bits directly to
   `K256LatentCodebook.decode()`.
6. The noisy experiment passes the same bits to `TeamDigitalPHY.transmit()`,
   which uses the team's length header, outer CRC, 5G LDPC, QPSK, AWGN, LLR
   demapping and LDPC decoder.
7. A CRC-valid recovered payload is given to the team's
   `decode_codebook_packet_bits()`. Indices select shared centroids, and the
   wrapper restores tensors before calling the frozen official decoders.

## Team logic reused exactly

`semcom_benchmark/codebooks.py` dynamically imports these functions/classes
from the unmodified team `codebook_codec.py`:

- `CodebookPacketHeader`
- `encode_codebook_packet_bits()`
- `decode_codebook_packet_bits()`

Consequently, the following packet behavior is team code, not a reimplementation:

- K/index-count/payload-bit accounting from the header;
- fixed-width index serialization;
- the 128-bit metadata schema containing a 64-bit codebook-ID fingerprint,
  K, active-channel field, aggregate token count, channel dimension and schema
  version/reserved fields;
- optional rate/mask bits placed after indices;
- inner CRC32 generation and validation;
- packet-length, metadata and CRC rejection during decoding.

The physical call path also reuses the team's `frame_payload_bits()` and
`recover_payload_bits()` (32-bit length plus an additional outer CRC32), its
Sionna 5G LDPC encoder/decoder, QPSK mapping/demapping and AWGN functions.

## What the wrapper adds or changes

The wrapper does **not** instantiate the team's `MiniBatchKMeansVQ` class. It
adapts two BCHW model families to the team's packet codec and adds:

- BCHW/CHW-to-token conversion: `[B,C,H,W] -> [B*H*W,C]`;
- scikit-learn `MiniBatchKMeans(K=256, batch_size=8192, n_init=10,
  max_iter=200, random_state=42)` rather than the team's dependency-free online
  Torch centroid updater and its default random seed;
- concatenated fitting/encoding across multiple latent tensors;
- independent DeepJSCC and NTSCC codebooks;
- codebook utilization, used/dead entries, perplexity and quantization MSE;
- `.npz` center storage and SHA-256 hashing;
- explicit four-bit serialization of every NTSCC rate index;
- reconstruction of BCHW tensors and dispatch to the appropriate frozen
  official decoder;
- shared-codebook storage estimates at 32 bits per center value.

The team's `MiniBatchKMeansVQ` convention uses `[B,tokens,channels]` artifacts
and, when present, treats `rate_mask` as a channel mask vector. Our wrapper uses
the packet codec's variable-length mask field as an opaque container for a
four-bit rate index **per spatial NTSCC token**. The team packet codec supports
that bit field, but this is an adapter-specific interpretation rather than the
team VQ class's original mask convention.

The wrapper sets `index_bits=8` directly because K is locked to 256; this is
equal to the team helper's `ceil(log2(256))` result.

## DeepJSCC vectorization, transmission and reconstruction

- Calibration latent per image: `[19,29,29]`, treated as 841 vectors of
  dimension 19. Eight calibration images provide 6,728 real vectors.
- Test image: 24 independently encoded 128x128 tiles, each providing 841
  vectors, for 20,184 indices total.
- Inner codebook packet: `20,184 x 8` index bits + 128 metadata bits + 32 inner
  CRC bits = **161,632 bits**.
- The shared 256x19 center table is not sent per image; its reported 32-bit
  storage is 155,648 bits.
- Team PHY outer framing and LDPC padding produce 323,584 coded bits and
  **161,792 QPSK complex channel uses**.
- After both CRC checks, indices select 19-D centroids. The wrapper uses its
  retained per-tile shapes to restore 24 `[1,19,29,29]` tensors, calls the
  frozen official DeepJSCC decoder on every tile, stitches them, then crops to
  the original image size.

The aggregate header transmits total token count and channel dimension, but the
24 individual tile shapes/positions in `EncodedCodebookPacket.token_shapes`
are local receiver/template state, not bits in the inner team packet.

## NTSCC vectorization, transmission and reconstruction

- Calibration latent per image: `s_masked` at `[1,256,8,8]`, treated as 64
  vectors of dimension 256. Eight calibration images provide only 512 vectors.
- Test latent: `[1,256,32,48]`, treated as 1,536 vectors of dimension 256.
- `s_masked` already contains zeros outside each token's selected rate. The
  ordinary K-means distance nevertheless spans all 256 coordinates.
- One eight-bit centroid index is transmitted for every spatial token.
- Each of the 1,536 official rate indexes is flattened and transmitted in four
  bits, for 6,144 rate-map bits. The Boolean BCHW mask itself is not sent.
- Inner packet: `1,536 x 8` index bits + 6,144 rate bits + 128 metadata bits +
  32 inner CRC bits = **18,592 bits**.
- The shared 256x256 center table is not sent per image; its reported 32-bit
  storage is 2,097,152 bits.
- Team PHY outer framing and LDPC padding produce 38,912 coded bits and
  **19,456 QPSK complex channel uses**.
- After CRC validation, indices select full 256-D centroids. The wrapper
  decodes the four-bit rate indexes, regenerates the official channel mask
  using `config.multiple_rate`, zeros inactive centroid dimensions, restores
  `[1,256,32,48]`, and calls the frozen official NTSCC feature decoder and
  synthesis transform.

As with DeepJSCC, the team metadata contains aggregate token count and channel
dimension, not explicit spatial H/W. Reconstruction uses the locally retained
`token_shapes` and `_ntscc_target` template. Therefore the bitstream is CRC-
protected and internally consistent, but not a fully standalone image packet
without agreed model/image-shape context.

## Is K=256 structurally appropriate for NTSCC?

**As a generic ablation: yes. As an NTSCC-aware representation: no.** It tests
whether a shared 8-bit nearest-centroid token can replace each channel latent
vector while leaving the official encoder/decoder frozen. It is not aligned
with NTSCC's core variable-rate structure for these reasons:

1. Tokens with different active channel counts share one 256-D Euclidean
   space and one codebook.
2. Inactive zero coordinates affect clustering and distance, while the selected
   centroid is masked only after lookup.
3. A fixed eight-bit index is paid for every token regardless of its selected
   NTSCC rate.
4. The calibration set has 512 vectors for 256 centers—only two vectors per
   center on average—and covers eight images.
5. The shared center table is much larger than the per-image packet and its
   availability is assumed rather than transported.
6. Receiver spatial/tile shape remains out-of-band template state.

The executed measurements support, but do not by themselves prove, this
structural diagnosis: NTSCC codebook quantization MSE was 0.559943, utilization
69.53% (178/256 entries), 78 entries were dead, and perplexity was 83.70.
Noiseless and CRC-valid noisy reconstruction PSNR was 12.6616 dB versus
25.7058 dB mean for native NTSCC. At the evaluated 10 dB, all five codebook PHY
frames decoded exactly, so noisy and noiseless codebook quality matched; the
loss came from quantization rather than observed frame errors.
