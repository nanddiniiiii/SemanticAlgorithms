# Model selection: retain, obtain weights, implement

Date: 21 October 2026. This is an execution priority list, not a universal ranking
of reconstruction quality. Pixel fidelity, semantic fidelity, rate, latency and
deployment cost lead to different choices. Family labels overlap: a codebook
codec can also be digital and generative.

## Retain and benchmark first

1. DeepJSCC: retained CNN reference. Checkpoint transfer and CPU smoke passed;
   calibrate channel convention/packing before matched wireless comparisons.
2. Original NTSCC: retained entropy/rate-adaptive reference. Complete side-cost
   accounting before calling its measured rate a full transmission cost.
3. SwinJSCC: retained adaptive SA+RA model and historical fixed base must remain
   separate variants. No automatic fresh training is required for profiling.

See RETAINED_ENCDEC.md and the per-file import manifest for actual local assets.

## Obtain published weights and integrate inference

| Model | Mechanism | Why use weights first | Source |
|---|---|---|---|
| JCM | VAE/probabilistic finite-constellation transmission | Released CIFAR checkpoints provide a manageable digital baseline | https://github.com/SJTU-mxtao/Joint-Coding-Modulation-for-Digital-Semantic-Communication |
| MDJCM | Multi-order digital joint coding/modulation | Author MDJCM-A checkpoint is linked; downloading from Baidu remains a gate | https://github.com/zhang-guangyi/Semantic-MDJCM |
| MambaJSCC | State-space neural JSCC | Released DIV2K checkpoints avoid costly scratch training; CUDA extension compilation remains a gate | https://github.com/Wireless3C-SJTU/MambaJSCC |
| SQ-GAN | Semantic masking, two VQ streams, GAN reconstruction | Author Cityscapes checkpoint avoids restarting three training stages | https://github.com/frapez1/SQ-GAN |
| DiffCom | Diffusion posterior reconstruction conditioned on received measurements | Published codec and diffusion weights; no diffusion scratch training | https://github.com/wsxtyrdd/diffcom |
| DiffJSCC | Diffusion-aided JSCC reconstruction | Author repository links actual Hugging Face weights including OpenImage CBR 1/384 and 1/96 models | https://github.com/mingyuyng/DiffJSCC |

DiffJSCC is a newly verified executable candidate to prioritize over waiting for
DiT-JSCC weights. The Hugging Face API returned the ungated model repository
`Mingyuyang/DiffJSCC-OpenImage-CBR-1-384` with `model.ckpt`. No DiffJSCC weight
download or inference is claimed yet. Small advertised CBR is not proof of exact
recovery or equivalent semantic fidelity, and native CBR definitions must be checked.

## Implement or train

1. ADJSCC: fresh AWGN experiment requested by the user. Reuse the author attention
   core and keep its environment/data adaptation explicit.
2. Shared Semantic-Aware Codebook: highest-priority new codebook implementation.
   Use native labeled STL-10 first; preserve codebook construction, classifier
   and codeword-conditioned codec. It is not a generic patchwise VQ substitution.
   Source: https://github.com/SJTU-mxtao/semantic-communication-w-codebook.
3. SQ-GAN: implement the complete native inference/training integration, establish
   a baseline, then fine-tune. Keep masks, semantic stream and side information
   in the payload ledger. This is source coding; AWGN requires digital transport.
4. Hybrid semantic-complementary prototype: use an existing frozen codec with
   complementary image projections. Count the projection/basis information.
   Label it our paper-inspired implementation, not verified author code.
   Paper: https://arxiv.org/abs/2507.17196.
5. Digital-base/analog-residual HDA reference: a second, clearly labeled hybrid
   system we can construct from existing components. It is a proposed reference
   design, not an existing implementation and not an HDA-DeepSC reproduction.
6. DeepJSCC-Q: lower-priority conditional training. A public trainer exists, but
   paper-variant/quantizer/loss fidelity and pretrained weights remain unverified.
   Source: https://github.com/r0ymanesco/deepjscc_stream.

An improvement must be measured against the unmodified baseline under the same
data/channel/rate/compute conditions. A hypothesis is not novelty. Prefer one or
two convincing ablations over superficial modifications to every model.

## Family coverage and remaining gaps

| Family | Concrete coverage | Present gap |
|---|---|---|
| Continuous neural JSCC | DeepJSCC, ADJSCC, SwinJSCC; Mamba adds state-space architecture | Only three retained models have fresh local smoke receipts |
| Entropy/rate-adaptive JSCC | Original NTSCC; MDJCM adds digital modulation/rate mechanisms | Complete side-information accounting and matched operating points |
| Finite-constellation neural transmission | JCM, MDJCM; DeepJSCC-Q as conditional reproduction | New checkpoints not yet runtime-validated here |
| Shared/discrete codebooks | Shared Semantic-Aware Codebook, SQ-GAN; retained VQ reference | Native codebook and label pipelines still need implementation |
| Generative reconstruction | SQ-GAN, DiffCom, DiffJSCC | GPU memory/latency and semantic hallucination evaluation |
| Hybrid semantic/detail and digital/analog | Complementary-projection prototype plus proposed digital-base/analog-residual reference | Both are work to implement, not completed public-author reproductions |
| Exact reconstruction controls | PNG and JPEG XL lossless plus an explicitly protected bitstream | Exact source decoding does not guarantee zero channel failure |

Add JPEG/JPEG2000 plus channel coding and a learned image codec plus LDPC/QAM as
separation baselines. CompressAI publishes pretrained Cheng2020 models, so this
control need not consume a scratch-training slot. A learned source codec alone
is not an end-to-end semantic communication algorithm.
Source: https://github.com/InterDigitalInc/CompressAI.

CNN, Transformer/Swin, state-space, GAN, VAE and codebook are architectural axes;
analog/digital/hybrid describes transmission. Do not equate a shared codebook
with a knowledge graph. A graph/task-specific system is a later scope extension,
not a missing dependency of these image-reconstruction baselines.

## Research-only for now

- DiT-JSCC: loading interface exists, but JSCC checkpoints are reserved for a
  future release. https://github.com/semcomm/DiTJSCC/tree/main/checkpoints.
- MOC-RVQ: public repository announces future code and weights.
  https://github.com/Albert2X/moc_rvq.
- Masked VQ-VAE: inspected public demo is classification-oriented and comments
  out the VQ and AWGN calls. Do not count it as a ready reconstruction baseline.
  https://github.com/hqyyqh888/RobustSemanComm.
- HDA-DeepSC: no verified author runnable implementation was identified. Its
  paper can guide learning; the proposed HDA reference above must retain its own name.

Native datasets are allowed; maintain separate benchmark tables. Download larger
new weights on Kaggle first and check local capacity before mirroring results.
Preserve all prior checkpoints and the old extraction until the user
removes it; the retained core models no longer require that extraction at runtime.
