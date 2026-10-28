# Semantic Image Communication: A 50-Paper Literature Review for Near-Lossless, One-Shot, and Meaning-Preserving Transmission

**Prepared:** 21 October 2026  
**Scope:** Papers first published or released from 2019–2026 on wireless image semantic communication, learned image JSCC, discrete/codebook and generative transmission, plus a clearly marked set of lossless-image-coding papers that define the exact-recovery side of the project. (DeepJSCC appeared as a 2018 preprint and a 2019 journal paper.)  
**Research aid:** alphaXiv discovery and paper-PDF queries, primary arXiv/venue papers, and the project’s locally held PDF catalogue. The citations below link to primary paper pages wherever available.

> **How to read “top 50.”** This is a curated, project-relevant set, not an objective citation ranking. Selection favors (1) field-shaping milestones, (2) a distinct technical mechanism worth implementing, (3) useful experimental evidence, (4) direct relevance to a one-burst image system, and (5) coverage of the unresolved semantic-versus-pixel-fidelity tradeoff. The final papers on lossless image coding are adjacent enabling work, not semantic-communication systems. Several 2026 items are recent preprints: their claims are preliminary and should not be treated as independently replicated results.

## Executive orientation

There are two different products hiding inside the phrase “semantic image communication.” A **meaning-preserving** link needs to retain the objects, attributes, relations, and task-relevant scene structure. It may deliberately discard pixels. A **near-lossless or lossless** link needs to retain nearly or exactly every source sample. A generative decoder can make a plausible image from a small clue, but it can also invent a texture, alter a face, omit a small object, or change a pixel. Consequently, high CLIP similarity, a convincing picture, high PSNR, and exact pixel equality are four different claims.

The strongest practical direction suggested by this literature is a **single scheduled burst with an explicit payload budget**, not a single radio symbol. The encoder should send a compact semantic base (objects/regions/latent tokens), a progressively useful detail representation, and—when near-exact recovery is required—a protected residual or correction representation. Everything must fit the one-shot budget, including model/codebook identifiers, masks, entropy and framing information, pilots, channel coding, CRC, padding, and any side information. A one-burst design can still have many OFDM symbols and code blocks; “one shot” means no feedback-triggered retransmission after that scheduled transmission.

This is a research target, not an established result: no reviewed method demonstrates all of (a) arbitrary unseen large images, (b) orders-of-magnitude compression, (c) exact or near-exact source recovery, (d) realistic fading and synchronization, (e) one-shot delivery, and (f) practical phone-class encode/decode cost simultaneously. In particular, 10,000× compression and zero distortion are incompatible for unrestricted images unless strong assumptions or shared side information reduce the set of admissible sources. If a representation carries too few bits to distinguish all possible source images, the decoder cannot know which exact image was sent.

The terms should therefore remain separate in the experiment and in the project claim:

| Term | Precise project meaning | Example success test |
|---|---|---|
| Pure semantic transmission | Preserve only a declared meaning/task, such as object identities, attributes, relations, or a VQA answer; pixel changes are allowed | Task accuracy and object/attribute/relation agreement |
| Perceptual image transmission | Produce an image judged visually similar/plausible | LPIPS/DISTS, human study, and semantic checks |
| Near-lossless image transmission | Reconstruct pixels within a strict bound | `max(abs(x−x̂)) ≤ ε`, plus a stated fraction of exact samples |
| Lossless image transmission | Recover every source pixel/sample exactly when delivery succeeds | Byte/pixel equality, with image-level channel failure reported separately |
| One-shot delivery | Complete all planned symbols within one scheduled resource allocation, with no feedback-triggered retry | One burst/resource ledger and delivery-failure probability |

“Pure semantics” has no unique answer until the receiver’s task is specified. A representation sufficient to answer “is there a car?” may not preserve the number of cars, their colors, a license plate, or the exact face. For an image that people must inspect or archive, semantic success cannot replace pixel-fidelity tests.

## The evaluation contract this project should adopt

Before comparing models, freeze what is counted. For an RGB image of `H×W` pixels, report original file bytes, decoded raw pixel bytes, learned latent values, transmitted bits, complex channel symbols, OFDM resource elements, occupied bandwidth/time, and total scheduled resource cost separately. “CBR” is often `k/(3HW)` when `k` is complex channel symbols and the source has `3HW` scalar color values, but papers sometimes count real dimensions or use other conventions. Recompute the formula from each paper rather than comparing ratios by name alone.

For one-shot radio evaluation, include in the budget: model/version or codebook synchronization assumptions; latent indices; entropy-coded lengths; rate masks and headers; pilot/reference resources; FEC parity; CRC; modulation order; OFDM pilots, guards and cyclic prefix; padding; and any packet segmentation required by the physical layer. A neural encoder emitting a 5,000-symbol vector is not by itself proof that a real 5G/NR or WLAN transmission fits one scheduled allocation.

Use at least four outcome families:

| Goal | Measures that answer it | What they do not prove |
|---|---|---|
| Pixel reconstruction | MSE, PSNR, SSIM/MS-SSIM, maximum absolute error, exact pixel match | Human realism or task correctness |
| Perceptual similarity | LPIPS, DISTS, human pairwise judgments | Exact pixels; perceptual metrics can reward plausible changed details |
| Semantic/task fidelity | object/attribute/relation agreement, segmentation/classification/VQA score, CLIP/DINO similarity with careful protocol | Exact reconstruction; text/image embedding agreement can hide local errors |
| Delivery/resource | pre/post-FEC BER, BLER/PER, CRC pass, image-level exact success, CBR/bits-per-pixel/symbols-per-pixel, latency/FLOPs/energy | A good image score does not mean the frame arrived correctly or met its deadline |

Write the project objective as a constrained rate problem rather than an undefined “compression ratio”:

```text
minimize     total scheduled radio resources B(x)
subject to   P(channel/decoder failure) <= delta
             semantic/task distortion D_sem(x, x_hat) <= tau_sem   (semantic mode)
             pixel distortion D_pix(x, x_hat) <= tau_pix           (near-lossless mode)
             max_pixel_error(x, x_hat) <= epsilon                  (strict near-lossless mode)
```

The semantic-only mode may omit the pixel constraints. The near-lossless mode cannot omit the semantic/task constraint if it must also preserve meaning. Exact pixel recovery sets `epsilon=0` for integer pixel samples and requires bitwise equality of decoded pixels; it still does not make channel delivery failure probability zero. A finite-rate code over a noisy channel needs explicit protection and a declared residual error probability.

Report the **complete one-shot success event** as well as conditional image quality: for example, probability that the receiver obtains a valid complete image by the end of the scheduled burst, and image quality conditioned on CRC success. Show distributions over test images and channel realizations, not only a best example or a single mean. Use the same resolution, SNR, channel, total radio resources, source codec operating point, and receiver knowledge for every baseline.

## Chronological paper-by-paper review

The notes below distinguish claims reported by the paper from the review’s interpretation. “Result” statements are deliberately kept at the level the cited source supports; if an exact numerical table could not be verified from the inspected text, no number is invented. First-publication year is used for chronology; a later journal year is noted when it matters.

### I. Learned analog JSCC and channel/rate adaptation

#### 1. Deep Joint Source-Channel Coding for Wireless Image Transmission (2019; first preprint 2018)
[Bourtsoulatze, Kurka & Gündüz](https://arxiv.org/abs/1809.01733)

**Problem and mechanism.** The landmark image DeepJSCC paper replaces the JPEG/JPEG2000 → channel-code → modulation chain with a jointly trained convolutional encoder, a differentiable noisy-channel layer, and a convolutional decoder. The encoder maps pixel values directly to real/imaginary components of unconstrained complex channel inputs, then normalizes average power. The decoder consumes the noisy I/Q samples and reconstructs the image. The objective is expected pixel distortion over training images and channel draws; AWGN and slow Rayleigh fading are evaluated.

**Evidence and comparison.** The paper tests CIFAR-10 and higher-resolution Kodak images, uses PSNR and SSIM, and compares against JPEG/JPEG2000 with capacity-oriented channel coding. It reports gains in low-SNR/low-bandwidth regimes, graceful quality degradation as SNR worsens, and strong slow-Rayleigh behavior. The key experimental contribution is showing that a learned continuous representation can avoid the digital cliff effect at finite blocklength. **Limit:** the latent is not a fixed constellation and is not a bitstream; it assumes a trained model at both ends and a simplified differentiable channel. MSE training blurs fine texture, and good PSNR is not semantic correctness or exact recovery. This remains the baseline to beat for analog one-shot robustness.

#### 2. DeepJSCC-f: Deep Joint Source-Channel Coding of Images with Feedback (2020)
[Kurka & Gündüz](https://arxiv.org/abs/1911.11174)

**Mechanism.** Extends DeepJSCC with receiver-to-transmitter channel output feedback. Instead of deciding the entire image representation before transmission, the encoder sends a sequence of refinement transmissions conditioned on previously observed feedback. This lets later channel uses focus on residual reconstruction error. The work is useful for understanding the value of feedback and progressive refinement.

**Relevance and limitation.** Reported gains show that feedback can improve reconstruction efficiency, particularly when channel quality is uncertain. But feedback rounds violate a strict no-retransmission contract unless all refinement rounds are pre-scheduled in one burst, in which case the transmitter cannot condition later symbols on actual received samples. The paper is therefore a useful upper-bound/control comparison, not a direct one-shot solution. Its use of a learned analog channel representation also leaves interoperability and exact digital delivery open.

#### 3. Bandwidth-Agile Image Transmission with Deep Joint Source-Channel Coding (2020)
[Kurka & Gündüz](https://arxiv.org/abs/2009.12480)

**Mechanism.** Develops a single DeepJSCC network that can provide useful reconstructions from variable numbers of channel symbols, allowing a transmission to be truncated or allocated a different bandwidth. The learned representation is arranged so that the decoder can exploit prefixes/partial channel output instead of requiring a separately trained network for each fixed rate.

**Why it matters.** Variable length is the conceptual bridge from a fixed CBR experiment to a resource scheduler: the link can choose a prefix appropriate to available resources. **Limit:** the receiver needs to know the transmitted length/rate, and performance depends on how well the learned ordering makes early symbols useful. A variable-rate latent is not automatically entropy-optimal, and truncation still needs a real PHY framing contract. Evaluate it under a fixed total resource budget and count rate signalling.

#### 4. Wireless Image Transmission Using Deep Source Channel Coding With Attention Modules (ADJSCC; 2020 preprint, 2022 journal)
[Xu et al.](https://arxiv.org/abs/2012.00533)

**Mechanism.** Adds SNR-aware channel-wise feature attention to a DeepJSCC encoder/decoder. The receiver/transmitter are given the SNR condition; learned channel-wise scale and bias values alter neural feature maps so the model adapts to channel quality. Here “channel-wise” means feature-map channels inside a CNN, while the conditioning variable is the radio SNR. It is not a learned QAM constellation and not a separate radio channel estimator.

**Evidence and limitation.** The purpose is one model across a range of SNRs, avoiding a bank of models trained for each channel point and reducing mismatch. Reported evaluations compare with prior DeepJSCC and conventional compression-plus-channel-code baselines on image datasets. The method assumes channel state/SNR information is available and does not itself learn a practical modem or solve rate signalling. It is a clean adaptation baseline for any proposed hybrid encoder.

#### 5. DeepJSCC-Q: Channel Input Constrained Deep Joint Source-Channel Coding (2021)
[Tung et al.](https://arxiv.org/abs/2111.13042)

**Mechanism.** Constrains the DeepJSCC output to a finite alphabet, addressing a real deployment gap: ordinary DeepJSCC can emit arbitrary complex amplitudes, while standard radios commonly use finite constellations. The model learns a mapping compatible with a specified channel input alphabet and a decoder trained through the noisy channel.

**Finding and limitation.** The paper reports that the constrained model can approach unconstrained DeepJSCC and outperform BPG+LDPC under matched input constraints while retaining graceful degradation. This is an important bridge toward a digital/standardized PHY. However, finite symbol alphabet does not itself guarantee error-free delivery: the learned index sequence can still be corrupted, and mapping/protection/OFDM overhead must be counted. Include it whenever claiming the design is radio-realizable.

#### 6. OFDM-Guided Deep Joint Source-Channel Coding for Wireless Multipath Fading Channels (2021 preprint; journal 2022)
[arXiv:2109.05194](https://arxiv.org/abs/2109.05194)

**Mechanism.** Moves beyond a memoryless AWGN abstraction by incorporating OFDM and frequency-selective multipath fading into the learned transmission chain. The work is relevant because a single complex symbol per latent coordinate does not capture subcarrier allocation, channel frequency response, pilot estimation, cyclic prefix, or inter-symbol interference.

**Evidence and limitation.** The experiments demonstrate learned image JSCC across multipath channel conditions and compare with a digital baseline. The system-level lesson is that channel modeling changes the representation and robustness requirements. Its simulator assumptions still do not equal over-the-air validation, and radio resource elements/pilots/guard overhead must be included when comparing channel usage. Treat OFDM-Guided DeepJSCC as a stepping stone from an ideal channel layer toward a scheduled radio burst.

#### 7. Nonlinear Transform Source-Channel Coding for Semantic Communications (NTSCC; 2021 preprint, JSAC 2022)
[Dai et al.](https://arxiv.org/abs/2112.10961)

**Mechanism.** NTSCC factors learned transmission into a nonlinear analysis transform, a learned latent entropy model/hyperprior, a rate-aware JSCC stage, and synthesis reconstruction. The transform produces a compact latent; hyperprior/context estimates its distribution; those estimates guide how much channel resource the JSCC stage devotes to features. The method joins learned compression and channel-robust transmission rather than mapping raw pixels directly to symbols.

**Evidence and limitation.** The paper evaluates images at varied sizes, uses perceptual/rate-distortion objectives and compares analog DeepJSCC plus separated JPEG/JPEG2000/BPG-style systems. It reports better rate-distortion performance, especially at larger image dimensions and content-dependent rates. Its central insight for this project is that **rate allocation should follow latent importance/uncertainty**, not merely tensor shape. Entropy is only a proxy for semantic value, and entropy-coded representations may become fragile under bit errors unless protection/framing is explicit.

#### 8. Deep Joint Source-Channel Coding for Wireless Image Transmission with Adaptive Rate Control (2021)
[Yang & Kim](https://arxiv.org/abs/2110.04456)

**Mechanism.** Introduces a policy network that selects groups of latent features based on image content and SNR. A Gumbel-Softmax training mechanism makes discrete rate selection differentiable; always-on features provide a base layer and optional groups refine the image. The paper defines channel uses per pixel and combines adaptive rate choice with SNR-aware modulation inside the learned model.

**Reported result and limitation.** The authors report reduced channel use for easy images or high SNR while preserving quality comparable to a separately rate-optimized model. This is directly aligned with “compress harder when the image/channel permits.” But the policy decision/mask must be recoverable at the decoder; the paper itself notes side-information cost can matter for small images. A one-shot implementation needs a robust length/mask header and a hard cap on total symbols.

#### 9. Predictive and Adaptive Deep Coding for Wireless Image Transmission in Semantic Communication (PADC; 2023)
[Zhang et al.](https://doi.org/10.1109/TWC.2023.3234408)

**Mechanism.** PADC combines a variable-code-length DeepJSCC-V encoder, an oracle network (OraNet) that predicts achievable PSNR from image content/SNR/compression ratio, and a compression-ratio optimizer that selects the smallest rate meeting a target quality constraint. It explicitly makes the rate decision depend on both the instance and the channel, rather than choosing a single fixed CBR for all images.

**Evidence and limitation.** The paper reports PSNR prediction error below 0.5 dB on its evaluated data, and near-minimal channel usage subject to a specified PSNR target; DeepJSCC-V has similar PSNR to ADJSCC. This is a strong rate-control design for a near-lossless target if the quality constraint is replaced with a stricter error bound and validated. Its guarantee is learned/predictive, not a mathematical per-image maximum-error guarantee; unseen content and channel mismatch can violate the target.

#### 10. WITT: A Wireless Image Transmission Transformer for Semantic Communications (2022 preprint; ICASSP 2023)
[Dai et al.](https://arxiv.org/abs/2211.00937)

**Mechanism.** Recasts image DeepJSCC around hierarchical Swin Transformer blocks, using shifted-window attention to model spatial relationships efficiently. A spatial modulation module conditions intermediate features on channel state, allowing one network to handle channel conditions. The model is designed for high-resolution semantic image transmission, where local CNN receptive fields may be limiting.

**Finding and limitation.** The paper reports improved transmission quality across resolutions and channel conditions relative to CNN JSCC baselines. It establishes the Transformer lineage that later SwinJSCC develops. Attention increases model complexity and memory cost; the method remains a learned lossy reconstruction system, and a spatial feature map is not a single fixed-size vector independent of resolution. Count the actual transmitted latent dimensions and the metadata used for adaptation.

#### 11. Improved NTSCC to Catalyze Semantic Communications (NTSCC++; 2023)
[Wang et al.](https://arxiv.org/abs/2303.14637)

**Mechanism.** Adds a contextual entropy model to exploit spatial correlations; a response-network architecture supports multiple bandwidth ratios/channel states using one trained model; and online latent feature editing allows rate allocation aligned with a user/task semantic priority. These additions target coding gain, deployment flexibility, and content-guided transmission.

**Evidence and limitation.** The paper reports better rate-distortion efficiency than VTM + 5G LDPC with lower processing latency in its tested setup, and presents a variable-rate model intended for large XR content. Its key project value is the separation of “how compressible is this feature?” from “how important is this feature to the goal?” The reported coding gain depends on the learned entropy/context assumptions; task-priority editing needs explicit priorities and robust signaling. Do not equate latent entropy with semantic value.

#### 12. SwinJSCC: Taming Swin Transformer for Deep Joint Source-Channel Coding (2023 preprint; journal 2025)
[Yang et al.](https://arxiv.org/abs/2308.09361)

**Mechanism.** Uses a hierarchical Swin Transformer as the JSCC backbone and adds channel modulation and rate modulation. Channel modulation conditions feature embeddings on SNR; rate modulation ranks/chooses feature channels to meet the target rate, and can require a mask/side information. This is neural feature modulation, not radio modulation such as QPSK/QAM. The spatial latent can scale with image resolution; target CBR controls how much of it becomes channel symbols.

**Evidence.** Trained on DIV2K crops and tested on Kodak and CLIC2021, the paper compares PSNR/MS-SSIM across AWGN and fast Rayleigh channels against DeepJSCC and BPG+LDPC. On Kodak/AWGN, SNR 7 dB and CBR 1/6, its plotted comparison gives 36.2 dB PSNR for SwinJSCC-base versus 34.4 dB for the listed 14.59M-parameter CNN DeepJSCC(256); complexity is also reported (203.3 vs 494.0 GFLOPs). Its latency table reports 151 ms encode/13 ms decode for its no-adaptation configuration on the authors’ setup, versus substantially longer BPG+LDPC pipeline time. **Caveat:** these are paper-specific hardware/protocol results; rate-mask overhead is non-negligible at low resolution, and the model is not exact.

#### 13. DeepJSCC-1++: Robust and Bandwidth-Adaptive Wireless Image Transmission (2023)
[Bian, Shao & Gündüz](https://doi.org/10.1109/GLOBECOM54140.2023.10436878)

**Mechanism.** Extends the DeepJSCC line with improved robustness and adaptive bandwidth behavior, aiming to support channel variation without training one independent network per operating point. It belongs in the comparison set because it tests whether architectural changes to the baseline CNN can deliver stronger rate flexibility before resorting to codebooks or generative priors.

**Evaluation interpretation.** The paper’s title and publication record identify the target; exact numerical comparisons should be taken from the full proceedings paper and reimplemented under the project’s own common resource accounting before drawing a leaderboard conclusion. The central limitation shared by this family is that robust approximate reconstruction is not exact correction. Treat it as a strong learned analog benchmark and compare at matched image resolution, SNR, and effective radio symbols.

#### 14. Robust Semantic Communications with Masked VQ-VAE Enabled Codebook (2022 preprint; TWC 2023)
[Hu et al.](https://arxiv.org/abs/2206.04011)

**Mechanism.** Builds a discrete codebook shared between transmitter and receiver using masked VQ-VAE. It treats semantic noise—received representations that are misleading for the downstream task—separately from ordinary pixel error. Adversarial weight perturbation augments training; masking is applied where semantic noise is frequent; a feature-importance module suppresses irrelevant/noise-sensitive features; the transmitter sends indices for task-relevant code entries.

**Finding and limitation.** The paper reports improved downstream-task robustness and reduced transmission overhead. It is valuable for learning shared codebooks, masks, and semantic feature selection. However, the demonstrations are task-dependent and do not establish arbitrary-image pixel reconstruction. Codebook synchronization/versioning and index protection are system dependencies; nearest-codeword lookup cannot recover a feature absent from the codebook or correct an undetected index error by itself.

#### 15. Perceptual Learned Source-Channel Coding for High-Fidelity Image Semantic Transmission (2022)
[IEEE conference record](https://ieeexplore.ieee.org/abstract/document/10001359)

**Mechanism and role.** This work places perceptual quality directly into learned source-channel coding for image transmission. It is important because MSE-only systems optimize average pixel error and can wash out visually important edges/textures; perceptual objectives change what the encoder and decoder preserve under a fixed channel budget.

**How to use its evidence.** Compare its perceptual metric and visual results against pixel-distortion metrics rather than interpreting “high fidelity” as exactness. The project should reproduce the paper’s source/channel conditions before importing its loss design. A perceptual objective may improve visual appearance while increasing pixelwise error or inventing detail, so it should be combined with explicit maximum-error/exactness reporting when near-lossless recovery is a requirement.

#### 16. Deep Learning-Based Image Semantic Coding for Semantic Communications (2021)
[Huang, Tao, Gao & Lu](https://doi.org/10.1109/GLOBECOM46510.2021.9685074)

**Mechanism.** Early image semantic-coding work that frames learned image representation as sending useful visual content through a communication system rather than just compressing a file. It is part of the conceptual bridge between conventional learned compression and task-oriented semantic transmission.

**Review note.** The paper is useful for tracing terminology and the design space, but its “semantic” label should be assessed through its actual loss, representation, and end task. If a system is trained only for pixel reconstruction, it is more precise to call it learned JSCC/image coding than to infer that it has explicit object-level semantics. The original work is a historical comparator; report its exact operating point only after reproducing its experimental section.

#### 17. Deep Learning-Based Adaptive Joint Source-Channel Coding using Hypernetworks (2024)
[Xie et al.](https://arxiv.org/abs/2401.11155)

**Mechanism.** Hyper-AJSCC uses a compact hypernetwork to generate/adapt channel-specific parameters for a DJSCC model. Unlike attention-based modulation that adjusts activations, a hypernetwork changes weights conditionally, while using a memory-efficient parameterization. The paper positions it as a generic add-on to existing DJSCC backbones.

**Finding and limitation.** The authors report improved channel adaptability with fewer extra parameters than attention-based baselines. For the project, this raises an important practical comparison: SNR-conditioning can happen through feature scaling, dynamic masks, or generated weights, and these differ in inference cost and receiver state. Hypernetwork output still assumes the transmitter and receiver use compatible channel context/model versions; it does not address finite-alphabet signaling, exact residuals, or symbol-level reliability.

#### 18. Content-Adaptive Distributed JSCC for Image Transmission with Hyperprior (2025)
[Li et al.](https://doi.org/10.1109/TCCN.2024.3438371)

**Mechanism.** Addresses two correlated image views sent by separate transmitters to one receiver. A multilevel cross-attention fusion module combines information from the views; an entropy-mask module allocates different rates to different image contents rather than giving every view equal bandwidth. Experiments use KITTI and InStereo2K, with PSNR and MS-SSIM.

**Reported result and limitation.** The source reports about a 1 dB PSNR gain over the referenced distributed JSCC baseline from improved cross-view fusion and content-adaptive allocation. This is the most relevant cross-layer/cross-source exchange idea for multi-camera or stereo scenarios. It is not directly a single-camera one-shot compressor: source correlation and synchronized multi-transmitter assumptions are essential. A one-image design can borrow adaptive entropy masks, but cannot claim the multi-view gains.

#### 19. DeepJSCC: One-Model-to-Many (2025)
[IEEE publication record](https://doi.org/10.1109/LWC.2025.3547527)

**Mechanism.** Studies a single learned image-JSCC model intended to operate across several settings rather than maintaining separate model checkpoints for each use case. This is relevant to deployment where memory and model-selection signaling matter as much as raw reconstruction quality.

**Project implication.** A one-model-to-many result should be assessed across dimensions that matter to the target: resolution, SNR, fading, rate, source domain, and receiver capability. Multi-condition training often trades peak performance at one point for robust average behavior. It still requires a clear conditioning interface and does not eliminate the need to send model/codec identifiers or to count resource side information.

#### 20. Space-Time Design for Deep Joint Source-Channel Coding of Images over MIMO Channels (2023)
[IEEE publication record](https://ieeexplore.ieee.org/document/10304536)

**Mechanism.** Moves learned image JSCC into a MIMO setting, using spatial/channel structure to improve robustness and exploit multiple antennas. This matters because radio resources are not just scalar “channel uses”: antenna count, spatial streams, channel matrix, precoding, and receiver channel knowledge affect the actual rate and error probability.

**Use and limitation.** It supplies a physical-layer extension point for a future one-shot burst. MIMO simulations must be compared at equal time-frequency-spatial resources and power, not equal latent length alone. Results from a known channel matrix or ideal synchronization should be separated from channel estimation and RF impairments. It is primarily a PHY architecture paper, not a proof of semantic or lossless reconstruction.

#### 21. Alternate Learning-Based Sparse Semantic Communications for Visual Transmission (2023)
[Tong et al.](https://arxiv.org/abs/2309.16681)

**Mechanism.** Studies sparse semantic representations for visual transmission and an alternate-learning strategy for the communication model. Sparsity aims to avoid spending channel resources on weak or redundant latent activations, while alternating optimization can stabilize coupled representation/transmission objectives.

**Use and limitation.** It is a useful bridge between dense analog latents and explicit codebook/index systems. Reproduction should report actual nonzero/index payload, any mask cost, and how sparsity changes as SNR or image content changes. A sparse feature tensor still needs serialization and a robust indication of active positions; reporting only latent sparsity can understate the transmitted rate.

#### 22. On the Role of ViT and CNN in Semantic Communications: Analysis and Prototype Validation (2023)
[Yoo et al.](https://arxiv.org/abs/2306.02759)

**Mechanism and purpose.** Compares convolutional and Transformer vision backbones for semantic transmission and includes a prototype-oriented validation angle. It is valuable as a guard against assuming that attention always beats CNNs: local inductive bias, global context, token count, hardware, and image resolution all matter.

**Project lesson.** Compare backbones at matched channel budget and measured inference conditions, not simply equal parameter count. A prototype validation should be separated from a fully deployed PHY test unless it includes RF impairments, synchronization, and actual over-the-air decoding. Its architectural comparison helps select what to implement before adding codebooks or a generative decoder.

### II. Transformers, variable-rate coding, and semantic importance

#### 23. Deep Learning-Enabled Semantic Communication Systems with Task-Unaware Transmitter and Dynamic Data (2022)
[Zhang et al.](https://arxiv.org/abs/2205.00271)

**Problem and mechanism.** The transmitter may not know the receiver’s eventual task, and actual images may come from a distribution different from the shared training knowledge. The paper proposes receiver-leading training for semantic coding and a data-adaptation network that maps a new observable domain toward the domain the semantic coder expects. The architecture explicitly addresses task mismatch and data shift rather than assuming a fixed shared semantic dataset.

**Importance and limitation.** Experiments report adaptation to observable datasets while maintaining data-recovery and task-execution performance. This is directly relevant to open-set generalization: codebook/model coverage is a distribution problem. The method does not establish open-world guarantees, nor does task preservation imply exact image recovery. Test held-out domains and measure both task outputs and pixel errors.

#### 24. Large AI Model-Based Semantic Communications for Images (2023; revised 2024)
[Jiang et al.](https://arxiv.org/abs/2307.03492)

**Mechanism.** Uses Segment Anything Model (SAM) to build a semantic knowledge base of image regions, an attention-based semantic integration module to weight segments, and adaptive semantic compression to remove redundancy. It is a knowledge-base/foundation-model route to explicit region-level meaning, contrasting with latent-only JSCC.

**Value and limitation.** The paper reports simulations demonstrating feasibility and lower overhead from semantic feature compression. Foundation models add substantial model size, compute, possible segmentation errors, and receiver synchronization requirements. A segment label or embedding can omit color, fine texture, text, identity, or small details. Count transmitted masks/segment descriptors and knowledge-base version; treat semantic reconstruction as lossy.

#### 25. Semantic Segmentation-Based Semantic Communication System for Image Transmission (2023/2024)
[Wu et al.](https://doi.org/10.1016/j.dcan.2023.02.006)

**Mechanism.** Makes object/region labels explicit in the transmission design, allowing the system to prioritize foreground objects or task-relevant image regions differently from background. This is a useful contrast to latent JSCC, where semantic importance is implicit inside learned features.

**Why it matters.** Explicit masks/classes support interpretable unequal protection and task-specific rates. The tradeoff is segmentation errors, label-map payload, and loss of texture/identity details if a region is represented only by a class label. A robust hybrid should transmit object/region semantics as a base layer and add pixel residuals where exact appearance matters.

#### 26. Demo: Real-Time Semantic Communications with a Vision Transformer (2022)
[arXiv:2205.03886](https://arxiv.org/abs/2205.03886)

**Mechanism.** Explores a Vision Transformer in an image semantic-transmission pipeline with the stated goal of real-time communication. It adds an architectural point between convolutional latent coding and the larger SwinJSCC family: patch tokens can model longer-range interactions, but full self-attention scales quadratically with token count.

**Practicality.** “Real-time” depends on image size, hardware, batch size, and whether channel coding and data movement are included. When reproducing, measure encoder and decoder wall-clock latency separately, report FLOPs/parameters and memory, and keep the same device and precision. The semantic label does not imply that the model sends a symbolic scene graph; inspect the actual latent and training loss.

#### 27. Language-Oriented Semantic Latent Representation for Image Transmission (2024)
[Cicchetti et al.](https://arxiv.org/abs/2405.09976)

**Mechanism.** The system sends both an image-to-text description and a compressed image embedding, then fuses them through latent diffusion at the receiver. The paper identifies why caption-only transmission is too coarse: language captures object meaning but often misses spatial location, color, and texture.

**Reported result and limitation.** It reports sending 2.09% of the original image size and better perceptual similarity over noisy channels than a text-only semantic baseline. This is a clear example of complementary semantic and appearance cues. The figure is a source-size ratio, so it must not be interpreted as a CBR or radio-resource ratio unless headers/channel coding are included. Diffusion can still create incorrect details; exactness is not the objective.

#### 28. Goal-Oriented Semantic Communication for Wireless Visual Question Answering (2024)
[Liu et al.](https://arxiv.org/abs/2411.02452)

**Mechanism.** Rather than reconstructing every pixel, the sender extracts and ranks image regions relevant to a question and transmits a compact representation to maximize answer accuracy. Bounding-box/region selection makes the task-conditioned nature of “meaning” explicit.

**Why it matters.** It gives a strong task-oriented comparator for the semantic-only end of the project. It can save bandwidth if the receiver only needs an answer, but it is unsuitable when the user expects a faithful view of the entire image. Report answer accuracy and rate, plus failure on questions that depend on omitted regions. The same image needs different encodings for classification, VQA, and exact reconstruction.

#### 29. A Survey on Semantic Communication for Vision: Categories, Frameworks, Enabling Techniques, and Applications (2026)
[Cheng et al.](https://arxiv.org/abs/2601.22202)

**Contribution.** Systematizes semantic communication for visual data and organizes approaches by semantic preservation, semantic expansion, and semantic refinement. It reviews encoder/decoder designs, knowledge use, and application classes, giving a current map for checking coverage beyond the DeepJSCC lineage.

**Use and limitation.** A survey helps expose neglected categories and terminology, but it is not evidence that one architecture is better. Its classifications help locate primary papers; this review independently separates pixel, perceptual, task, and delivery objectives so unlike metrics are not combined into a false ranking. Since it appeared in 2026, check revisions and venue metadata before formal citation.

#### 30. Improving Learning-Based Semantic Coding Efficiency via a Shared Semantic-Aware Codebook (2024 preprint; TCOMM 2025)
[Zhang, Tao, Sun & Letaief](https://doi.org/10.1109/TCOMM.2024.3450877)

**Mechanism.** Uses a codebook shared at both endpoints to map learned semantic features to discrete indices, reducing the need to transmit full-precision latent vectors. Codebook-based reconstruction makes the receiver’s lookup deterministic if both parties use the same codebook/version. Reported experiments include AWGN and Rayleigh fading and comparisons at multiple channel rates/SNRs.

**Value and limitation.** This is a direct study for learning codebook construction and index-based transmission. Its chief system cost is that the “saved” index payload excludes the storage/synchronization cost of the codebook unless the receiver already has the exact version. Quantization can erase an unseen feature; an index flip can jump to an unrelated entry. Add entropy/error protection and measure index error to image/task error propagation.

### III. Digital, quantized, and codebook-based image SemCom

#### 31. Codebook-Enabled Generative End-to-End Semantic Communication Powered by Transformer (2024)
[Ye et al.](https://arxiv.org/abs/2402.16868)

**Mechanism.** Jointly builds a semantic codec and codebook, then uses a vector-to-index Transformer to make index transmission robust to noise and drive image generation at the receiver. The work explicitly addresses a weakness of naive codebook transmission: numeric index proximity need not correspond to semantic similarity, so an index error can map to an unrelated code vector.

**Result and limitation.** The paper reports improved visual perception versus JPEG+LDPC and traditional JSCC in its experiments. It is a key “codebook as synchronized shared knowledge” architecture. A generative decoder can restore plausible appearance but is not guaranteed to reconstruct the original pixels; performance depends on shared codebook and generative-model agreement. Include model/codebook versioning and a per-image exactness test in any implementation.

#### 32. MOC-RVQ: Multilevel Codebook-Assisted Digital Generative Semantic Communication (2024)
[Zhou et al.](https://arxiv.org/abs/2401.01272)

**Mechanism.** Two-stage method: first, a multi-head octonary codebook compresses the index range and residual vector quantization (RVQ) provides multilevel refinement; second, a Swin-based noise-reduction block uses the codebooks as a semantic knowledge base to restore corrupted features. This reconciles a large expressive codebook with modulation alphabets that offer only a small number of reliable symbols per channel use.

**Evidence and limitation.** The paper uses high-resolution 2K images and reports performance better than BPG/JPEG while approaching analog JSCC at one-sixth the channel bandwidth ratio, under the authors’ evaluation. The result is compelling for digital one-shot compression. It still relies on a shared learned visual prior, and “restored feature” need not equal original feature. Study index bits per code level, RVQ residual behavior, error protection, and exact resource accounting.

#### 33. Semantic Image Communications Using Masked Vector Quantization (SQ-GAN; 2025)
[SQ-GAN](https://arxiv.org/abs/2502.09520)

**Mechanism.** Combines a GAN-style image generator with masked vector quantization to reduce the transmitted representation while using semantic importance to select/retain tokens. It is part of a newer generation that aims to get more perceptual value per channel symbol by transmitting fewer, more useful latent indices rather than dense continuous features.

**How to interpret it.** Evaluate rate, channel noise robustness, codebook utilization/collapse, semantic consistency, perceptual quality, and runtime separately. A masked VQ-GAN can trade pixel fidelity for plausible detail, so its main claim is not losslessness. Inspect whether masks are transmitted, inferred, or shared, and whether total rate counts all token indices and headers. It is an important implementation candidate for codebook-plus-semantic-priority experiments.

#### 34. Low-Rate Semantic Communication with Codebook-Based Conditional Generative Models (2025)
[arXiv:2504.04977](https://arxiv.org/abs/2504.04977)

**Mechanism.** Extracts salient visual content, quantizes it through a compact codebook, and conditions a generative model at the receiver with the transmitted semantic representation. The design targets very low rate by separating a compact semantic cue from the high-detail image that the receiver synthesizes.

**Reported focus and limitation.** The paper evaluates whether saliency/semantic content survives rate and channel noise, including F-measure-style semantic evidence. This is well aligned with “pure semantic meaning” but moves away from exact source recovery. A caption or saliency map can omit instance-specific geometry, text, color, and small objects. Compare at equal total transmitted bits and score both task-level agreement and exact-pixel mismatch.

#### 35. Knowledge-Base-Based Semantic Image Transmission Using CLIP (2025)
[arXiv:2504.01053](https://arxiv.org/abs/2504.01053)

**Mechanism.** Uses CLIP-derived image semantics and a receiver-side vector database (FAISS) as a knowledge base to retrieve or condition image reconstruction. The sender can transmit compact descriptors rather than the full image, while the receiver exploits shared learned visual knowledge.

**Potential and risk.** This can reduce payload when the receiver’s knowledge base contains close matches, but retrieval can fail on novel people, cultures, rare objects, or details absent from the index. The database becomes part of the communication state: version, privacy, update cost, and retrieval confidence must be handled. Report source-image exactness separately; retrieval similarity is not proof that the same image was recovered.

#### 36. VQ-DeepISC: Vector-Quantized Digital Semantic Communication with Channel-Adaptive Image Transmission (2025)
[arXiv:2508.03740](https://arxiv.org/abs/2508.03740)

**Mechanism.** Uses a Swin Transformer to encode hierarchical features, vector quantization to turn features into indices, and channel adaptation to alter index transmission. It regularizes codeword-use distributions to reduce codebook collapse and uses EMA for stable codebook updates. The work describes QPSK/OFDM aligned with IEEE 802.11a as a digital transmission realization.

**Why it matters.** It joins learned semantic features, discrete tokens, and an explicit standard-like PHY, making it highly relevant to practical one-shot design. Reproduction must check the exact standard configuration and count pilots/FEC/packet headers. A paper-level “superior reconstruction fidelity” result should be re-tested on out-of-training-domain images and measured for index errors, not just output-image metrics.

#### 37. VQ-VAE-Based Digital Semantic Communication with Importance-Aware OFDM Transmission (2025)
[arXiv:2508.08686](https://arxiv.org/abs/2508.08686)

**Mechanism.** A shared VQ-VAE codebook converts semantic features into discrete symbols. A gradient-derived feature-importance score allocates more reliable OFDM positions/protection to important features; the receiver rematches noisy features to codebook entries to correct errors.

**Evidence and limitation.** The paper reports better reconstruction than a DeepSC baseline in low-SNR conditions. This is valuable because it treats codebook matching and OFDM resource placement as one system. Nearest-codeword rematching is a denoising heuristic, not an error-correction guarantee; importance gradients are task/training dependent. The project should compare hard index coding, soft analog symbols, and hybrid protection at fixed total resource elements.

#### 38. Generative AI-Based Vector-Quantized End-to-End Semantic Communication for Wireless Image Transmission (2025)
[IEEE TMLCN paper](https://doi.org/10.1109/TMLCN.2025.3607891)

**Mechanism.** Sends quantized latent indices as a shared learned knowledge base; combines edge/energy-aware spatial attention with a Wasserstein critic and contrastive objectives spanning pixel, latent, perceptual, and task-output levels. It evaluates human- and machine-perception utility and studies robustness to semantic/channel noise.

**Reported claim and caveat.** The authors report performance over BPG+LDPC at low SNR and comparable JSCC quality with lower complexity/latency, including high-resolution images. The report also claims high-resolution transmission without extra receiver error correction; this should be tested carefully under independent channel realizations and a defined failure rate. A GAN critic can improve realism without improving source truth. Keep identity/detail and exact error metrics in the test suite.

#### 39. Adapting Diffusion Language Models for Lossless Pixel-Level Image Transmission (DDM-SSCC; 2026)
[Ren et al.](https://arxiv.org/abs/2606.06273)

**Problem and mechanism.** This is a discrete diffusion model adapted to separate source-channel coding (SSCC) for exact image transmission, rather than perceptual semantic reconstruction. A diffusion language model restores pixel tokens in a scheduled order and synchronizes reverse arithmetic coding under bidirectional attention. Halton-guided order improves spatial coverage; a mask-ratio-aware cosine schedule controls denoising pace; temperature calibration improves probability tables for arithmetic coding. Progressive token restoration is designed to supply context for subsequent prediction.

**Evaluation and relevance.** The paper reports experiments on CIFAR-10, DIV2K-LR-X4, and Kodak over AWGN and Rayleigh fading, and claims better exact-recovery performance than selected lossless and semantic-communication baselines. This is the clearest paper in the 50 for the exactness branch. Examine source rate, FEC/channel configuration, exact success probability, and sequential decoder cost; “lossless” describes the source code when decoded correctly, while noisy delivery still has a failure probability. It does not establish zero-error at finite SNR or ultra-low total radio resource use.

#### 40. Joint Coding-Modulation for Digital Semantic Communications via Variational Autoencoder (2023 preprint; 2024 journal)
[Bo et al.](https://arxiv.org/abs/2310.06690)

**Mechanism.** Learns a mapping from semantic representations to digital modulation/channel symbols jointly or in a variational framework, bridging discrete semantic source representation with a standard finite signal alphabet. This addresses a gap left by unconstrained analog DeepJSCC.

**Review point.** The key implementation question is whether codebook/indexing and modulation are optimized together, or whether a quantizer is added after representation learning. Post-hoc quantization can cause a train/test mismatch. Compare against DeepJSCC-Q at identical constellation, FEC, average power, and resource allocation. Do not infer exact recovery from a digital constellation: a digital stream can still have a nonzero block error probability.

### IV. Generative image transmission and hybrid semantic/detail recovery

#### 41. Diffusion-Aided Joint Source-Channel Coding for High-Realism Wireless Image Transmission (DiffJSCC; 2024)
[Yang et al.](https://arxiv.org/abs/2404.17736)

**Mechanism.** Uses a pretrained Stable Diffusion prior to decode noisy wireless features. A first reconstruction stabilizes feature extraction; spatial and textual conditions plus SNR guide diffusion denoising. This directly addresses the blurry outputs of MSE-trained JSCC at very low bandwidth.

**Reported result.** The paper states realistic Kodak reconstructions at 768×512 using 3,072 channel symbols (under 0.008 symbols/pixel) at 1 dB SNR, and reports gains in perceptual measures and downstream tasks over prior JSCC baselines. This is striking rate efficiency for perceptual reconstruction, not near-lossless coding: much of the apparent detail may come from the generative prior. Check object/geometry consistency, rare content, and decoder time; iterative diffusion may be too slow for strict edge latency.

#### 42. DiffCom: Channel-Received Signal as a Natural Condition for Diffusion Posterior Sampling (2024)
[Wang et al.](https://arxiv.org/abs/2406.07390)

**Mechanism.** Uses an off-the-shelf diffusion prior for posterior sampling conditioned on the actual received channel signal. It also develops blind posterior sampling for unknown channel conditions. The paper explores channel mismatch, unseen fading, PAPR reduction, inter-symbol interference, and pilot-free/multipath cases, which are valuable stress tests beyond AWGN.

**Interpretation and limitation.** DiffCom reframes the decoder as searching for a plausible source image consistent with the observation. It reports gains in perceptual metrics such as LPIPS/DISTS/FID. But multiple images may agree with a small noisy latent, and posterior plausibility does not imply the original image is selected. Measure semantic consistency and exactness separately; diffusion inference can require many denoising steps.

#### 43. Channel Denoising Diffusion Models for Wireless Semantic Communications (CDDM; 2023 preprint, 2024 journal)
[Wu et al.](https://arxiv.org/abs/2309.08895)

**Mechanism.** Applies diffusion denoising to channel-corrupted semantic representations, aiming to improve semantic decoding without a conventional decoder architecture alone. It is a channel-noise-aware generative approach rather than merely using diffusion to create sharper pixels.

**What to inspect.** Separate the learned prior’s gain from the effect of extra compute and training data. Test on unseen channel realizations and content, and report the diffusion schedule, number of denoising steps, and latency. For a one-shot use case, diffusion does not imply multiple radio transmissions—the receiver may iteratively compute after one burst—but its inference delay and energy are still part of the system cost.

#### 44. Semantics-Guided Diffusion for Deep JSCC in Wireless Image Transmission (2025)
[arXiv:2501.01138](https://arxiv.org/abs/2501.01138)

**Mechanism.** Incorporates semantic guidance into diffusion-assisted DeepJSCC so the generative decoder is conditioned by meaning-bearing representations rather than a weak generic latent. The objective is to reduce semantic drift at very low rates while retaining diffusion’s perceptual advantage.

**Why it matters.** It highlights that a high-quality generator may hallucinate the wrong object if the channel embedding lacks a discriminative cue. The paper should be read for semantic-conditioning design and compared with DiT-JSCC’s explicit semantic/detail branches. Its generative output remains approximate. For a near-lossless system, use it only as a predictor; transmit and decode a residual that corrects the prediction.

#### 45. Goal-Oriented Semantic Communication for Wireless Image Transmission via Stable Diffusion (2024/2025)
[Li & Deng](https://arxiv.org/abs/2408.00428)

**Mechanism.** A semantic autoencoder compresses image information; Stable Diffusion denoisers then mitigate channel corruption. SD-GSC uses known instantaneous channel gain; PSD-GSC learns/handles unknown gain with a parallel denoiser. This distinguishes channel denoising from simply generating an image from a caption.

**Reported comparison.** Under known channel, the authors report PSNR improvements of 32% and 21% and FID reductions of 40% and 35% relative to ADJSCC and Latent-Diff DNSC, respectively. Under unknown channel, PSD-GSC reports 8% higher PSNR and 17% lower FID than MMSE-equalizer-enhanced SD-GSC. Percent gains in PSNR should be interpreted cautiously because PSNR is logarithmic; reproduce dB values, not just percent statements. Generative denoising is not exact correction.

#### 46. Hybrid Digital-Analog Semantic Communications (2024)
[Zhou et al.](https://arxiv.org/abs/2405.12580)

**Mechanism.** HDA-DeepSC uses analog-digital allocation and fusion: some semantic features are quantized/entropy-coded and protected with digital channel coding/modulation; the remainder is sent as continuous analog symbols. Diffusion-based signal detection refines the received signal. The goal is digital delivery of crucial features plus analog graceful degradation.

**Evaluation.** It trains on 1,000 DIK2K images and tests on Kodak, comparing analog semantic transmission, DeepJSCC-Q, BPG+adaptive modulation/coding, and denoising baselines over AWGN/Rician channels using PSNR/MS-SSIM. The semantic codec uses six Swin Transformer layers, six heads, width 120. It reports matching BPG+LDPC PSNR with a 33% bandwidth-compression-ratio improvement in the authors’ Kodak/AWGN SNR=10 dB setting. **Limit:** perfect CSI and simulations; digital/analog allocation must be signaled and counted. It is not a lossless residual system.

#### 47. Hybrid Semantic-Complementary Transmission for High-Fidelity Image Reconstruction (2025)
[arXiv:2507.17196](https://arxiv.org/abs/2507.17196)

**Mechanism.** Sends a fixed-load semantic representation (SR) from the learned semantic codec and an adjustable complementary representation (CR) that captures image-specific residual information missed by the semantic model. The receiver combines the actual semantic reconstruction with CR to produce a recomposed image. The authors derive a closed-form expression for reconstruction error and an optimal CR allocation under their model.

**Evidence and relevance.** Simulations report substantially lower MSE than semantic communication without CR across tested channels and neural architectures. This directly supports a semantic-base-plus-detail design: vary CR length to trade resource use against fidelity. It is not automatically lossless; exact recovery depends on whether CR is a complete, correctly decoded residual with sufficient precision. For one-shot use, fix the total SR+CR budget before transmission, protect CR framing, and test max error and exact-pixel success.

#### 48. DiT-JSCC: Rethinking Deep JSCC with Diffusion Transformers and Semantic Representations (2026)
[Tan et al.](https://arxiv.org/abs/2601.03112)

**Mechanism.** Uses a visual-foundation-model semantic branch plus a pixel/detail branch, both feeding a conditional latent diffusion Transformer (DiT). A training-free bandwidth allocation rule based on estimated conditional-generation difficulty gives more rate to hard-to-generate semantic content and divides the remaining budget between semantic and detail signals. This directly addresses semantic drift in generative JSCC.

**Reported setup/results.** ImageNet 256/512 crops, a 100-image validation subset, AWGN/Rayleigh, low SNR, CBR roughly 1/384–1/48; compares VTM/BPG+LDPC, SwinJSCC, DiffJSCC and DiffCom. At 256², AWGN, CBR 1/48, 0 dB, the paper reports LPIPS 0.166, DISTS 0.151, CLIP 0.963, DreamSim 0.097, DINOv2 0.958 and FID 60; at 5 dB FID is 49. These are perceptual/semantic scores, not near-lossless results. The decoder is substantial: 21 DiT blocks, 16 heads, hidden size 1024, Stable Diffusion VAE; latency and edge deployment are key open issues.

#### 49. FlowSem: Flow Matching for Adaptive Wireless Image Transmission (2026)
[arXiv:2608.21651](https://arxiv.org/abs/2608.21651)

**Mechanism and setup.** A two-stage system uses an SNR-adaptive DeepJSCC encoder for channel symbols and a conditional flow-matching generator that produces the final image from Gaussian noise conditioned on the coarse reconstruction and SNR. It is evaluated on Cityscapes over AWGN/Rayleigh with a fixed channel-symbol budget, against rate-matched BPG+LDPC, DeepJSCC, DDPM and DDIM.

**Evidence and caution.** The paper reports competitive pixel fidelity and improved structural/perceptual reconstruction. This is generative refinement under a fixed sent-symbol budget; it does not necessarily reduce transmitted symbols further. As a recent preprint, treat results as preliminary until independent reproduction. A flow model can create plausible detail without recovering the actual source. Evaluate exact pixel differences and task-critical object/geometry.

#### 50. TwinViT-DeepJSCC: Adversarially Robust Semantic Image Communication (2026)
[arXiv:2610.08590](https://arxiv.org/abs/2610.08590)

**Mechanism and setup.** This October 2026 preprint uses two complementary ViT-DeepJSCC branches, sensitivity-aware masking at the transmitter, confidence-aware fusion and blind corruption-severity estimation at the receiver, followed by SNR/severity-conditioned DDIM purification. It evaluates CIFAR-100 under FGSM, 20-step PGD, NES, Carlini–Wagner, random jamming, and channel-aware waveforms over AWGN and block-flat Rayleigh. The branches share a fixed aggregate channel-use budget with single-branch baselines; tested SNRs are −5 to 20 dB.

**Reported result and status.** The authors report maximum PSNR gains of about 9.5 dB under 20-step PGD and 10.8 dB under channel-aware waveform attacks on Rayleigh, plus up to about 38 percentage points Top-1 accuracy over the undefended baseline under PGD. This is a useful adversarial-robustness stress test, but it remains an unreviewed preprint as of this report; the threat model is specialized and CIFAR-100 is low resolution. These results do not establish near-lossless recovery or robustness to ordinary channel failures.

## Exact and near-exact image coding: enabling references (not semantic-communication papers)

The preceding 50 review entries concentrate on semantic image transmission and selected adjacent ideas; these lossless codecs are essential for designing the exact branch and should be included in the experimental baseline suite. They are **not additional items in the top-50 count**.

### A. Lossless/near-lossless coding and residual correction

1. **Practical Full Resolution Learned Lossless Image Compression (L3C, 2018)** — [arXiv:1811.12817](https://arxiv.org/abs/1811.12817). A multiscale learned probability model predicts image values and arithmetic-codes the residuals. It demonstrates that learned autoregressive/context models can approach practical lossless compression. Its sequential dependencies and compute cost matter for large images and latency.
2. **Learning Better Lossless Compression Using Lossy Compression (2020)** — [arXiv:2003.10184](https://arxiv.org/abs/2003.10184). Uses a lossy reconstruction as a side-information predictor and entropy-codes the residual needed to recover the original exactly. This motivates a semantic/pixel predictor plus exact correction stream.
3. **Deep Lossy Plus Residual Coding for Lossless and Near-Lossless Image Compression (2022)** — [arXiv:2209.04847](https://arxiv.org/abs/2209.04847). Couples learned lossy compression with a residual code; the residual is what makes exact recovery possible. It gives a direct engineering template for prediction plus correction, though not a wireless error-protected semantic system.
4. **Fitted Neural Lossless Image Compression (FNLIC, CVPR 2025)** — [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Fitted_Neural_Lossless_Image_Compression_CVPR_2025_paper.pdf). Fits a neural representation to the image for lossless coding. Per-image fitting/parameter transmission and encoder time must be counted; a model fitting the image is not automatically a low-latency wireless codec.
5. **CALLIC: Content-Adaptive Learning for Lossless Image Compression (2025)** — [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/download/32494/34649). Content-adaptive learned lossless coding; relevant as an exact-source baseline and as a reminder that pixel entropy and semantic importance are different.
6. **P2-LLM: Large Language Models for Lossless Image Compression—Next-Pixel Prediction in Language Space is All You Need (2024)** — [arXiv:2411.12448](https://arxiv.org/abs/2411.12448). Applies next-token/pixel prediction ideas to exact image coding. Analyze sequential decode latency, model storage, arithmetic coding, and total bitstream size.
7. **Large Language Model for Lossless Image Compression with Visual Prompts (2025)** — [arXiv:2502.16163](https://arxiv.org/abs/2502.16163). Uses visual prompt/context to assist lossless coding. Prompt/context transfer must be counted or explicitly treated as shared preinstalled state.
8. **LUMI: Tokenizer-Agnostic LLM-Based Lossless Image Compression (2026)** — [arXiv:2607.08221](https://arxiv.org/abs/2607.08221). Recent LLM-based lossless-coding direction; useful for testing whether a learned language prior can code images independent of one image tokenizer. It remains a compression method until a wireless channel and delivery protocol are evaluated.

### B. Classical codec anchors

Use raw RGB as a no-compression reference; JPEG as a widely deployed lossy DCT/quantization/entropy-coding reference; JPEG 2000 as a wavelet-based lossy/lossless-capable reference; BPG/VTM as strong engineered lossy baselines; PNG as an exact lossless format; and JPEG XL in both lossy and lossless modes where available. For wireless comparisons, pair bitstreams with a clearly stated channel code and modulation, and use the same total radio resources. PNG being lossless does not mean small: it preserves exact samples but may retain large files. JPEG XL and learned lossless codecs are file compression systems, not error correction; the radio link still needs reliability.

## What the papers collectively show

### 1. The architecture evolved from direct analog mapping to structured learned latents

DeepJSCC shows the core end-to-end mapping: image → neural features → complex-valued symbols → noisy channel → reconstruction. ADJSCC and related conditioning mechanisms add SNR awareness. Variable-rate models add masks/prefixes or an optimizer. NTSCC introduces transform/entropy modeling to allocate rate more efficiently. WITT/SwinJSCC use hierarchical attention to improve large-image representation. Digital/codebook methods discretize latents to fit finite-alphabet systems. Diffusion/DiT systems use powerful priors to create plausible images at extreme rates.

### 2. A smaller representation is not automatically a better semantic code

Vector quantization replaces a feature vector with a codebook index. For a codebook of `K` entries, a fixed-length index requires at least `ceil(log2 K)` bits per codeword (before FEC/framing); the total payload also depends on the number of spatial positions, residual levels, masks, and entropy coding. Codebook rows are learned prototypes in latent space, not a catalogue of whole training images. Generalization depends on the encoder’s feature coverage and decoder’s prior. RREF or row reduction does not increase what the codebook can represent; it solves linear-equation problems and is not a substitute for learned quantization or error correction.

### 3. Semantic, perceptual, and pixel objectives must be evaluated separately

PSNR rewards pixel-wise closeness; LPIPS/DISTS compare learned visual features; CLIP/DINO can indicate high-level representation similarity; FID compares sets/distributions rather than proving a particular source was recovered; task metrics measure the downstream application. A diffusion decoder can improve LPIPS/FID while changing object details. A codebook may preserve class identity but alter identity, text, or fine geometry. Near-lossless requires explicit max error / exact match.

### 4. One-shot has a radio-system meaning

One-shot should mean one scheduled uplink/downlink allocation, with a fixed complete codeword/burst and no HARQ/ARQ retransmission. It can contain many OFDM symbols, resource blocks, code blocks, and packets/fragments. The system can still use FEC and CRC inside that one burst. If an application allows packet fragmentation within a single allocation, that is still one scheduled transmission; if it requires a later retry, it is not no-retransmission. Report image-level failure probability and resource elements, not only “channel symbols” from a neural simulation.

### 5. Cross-layer exchange appears in several forms

“Cross-layer” refers to several different mechanisms in this literature. PADC passes predicted image quality and SNR into a rate controller; NTSCC++ passes entropy/context and semantic guidance into feature allocation; these couple source representation with channel-resource choice. Content-adaptive distributed JSCC exchanges information across correlated image views using multi-layer cross-attention. CDDM inserts a learned diffusion denoiser after channel equalization. DDM-SSCC couples the source probability model and arithmetic coder, then evaluates the resulting source code over a noisy channel. These are concrete encoder/decoder/source/PHY couplings; they do not all mean that every protocol layer exchanges metadata dynamically.

An adjacent, non-image example is the [Knowledge-Enhanced Semantic Communication Receiver](https://arxiv.org/abs/2302.07727), which uses a receiver-side knowledge graph and Transformer retrieval over factual triples on WebNLG. It is useful for understanding shared knowledge and graph-aided decoding, but its text-task results are not image-transmission evidence.

### 6. The likely architecture for this project is a semantic base plus a correction path

Use two clearly measured operating modes, preferably sharing one learned analysis transform:

```text
                         shared model/codebook/version
                                      │
Image ──► analysis ──► semantic base ─┼─► unequal-protection digital indices ─┐
          │                           │                                      │
          ├──► detail / transform coefficients ─► prioritized refinement ────┤
          │                                                                  ▼
          └──► predicted reconstruction ─► quantized residual ─► protected one-burst PHY
                                                                             │
                         one scheduled resource allocation (no HARQ) ◄──────┘
                                                                             ▼
      PHY decode + CRC/status ─► codebook lookup ─► semantic/detail synthesis
                                                  └─► residual addition / exact reconstruction
```

The **semantic-only mode** sends a small semantic base and measures meaning/task success. It is allowed to change pixels and must be labeled lossy/generative. The **near-lossless mode** adds enough detail/residual symbols to bound the source error. The **lossless mode** sends an entropy-coded exact residual or complete lossless bitstream through a protected channel code; it is only exact if the whole codeword is recovered correctly. A shared dictionary can lower recurring payload only if dictionary/model acquisition cost and update consistency are honestly accounted for.

For high compression, reserve a minimum semantic base and allocate additional symbols by a joint score combining (a) semantic/task importance, (b) predicted distortion reduction, (c) code length/entropy, and (d) error sensitivity. Protect identifiers, lengths, semantic anchors, and residual framing more strongly than refinements. Never let generative synthesis silently substitute invented content when an exactness flag or application needs source truth.

## Recommended experimental path

### Stage 1 — Establish the rate and channel ledger

Build a reproducible image set (Kodak/CLIC for comparability; DIV2K or COCO training data for scale; an out-of-domain holdout such as medical, documents, faces, and locally collected images only with consent). Define RGB bit depth/resolution, preprocessing, image splits, SNR/power normalization, channel (AWGN first; then flat Rayleigh; then multipath OFDM), and one scheduled burst’s total time-frequency resources. Freeze whether `k` counts complex symbols or real dimensions. Log every header/pilot/code/FEC/CRC/guard bit.

### Stage 2 — Implement trustworthy baselines

Compare raw, JPEG/JPEG2000/BPG/VTM/JPEG XL/PNG as relevant, JPEG-family bitstreams + LDPC/Polar (document code, rate, modulation, and block errors), DeepJSCC, ADJSCC, NTSCC, and SwinJSCC. Plot rate-quality curves at fixed SNR and quality-SNR curves at fixed total burst size. Include decode failures, latency, FLOPs, parameter count, peak memory, and model size.

### Stage 3 — Learn one codebook/digital path

Implement VQ-VAE quantization on a learned image latent; log codebook size, vector dimension, spatial code count, index bits, empirical entropy, codebook perplexity/usage, dead codes, rate-distortion, and post-channel index errors. Add QPSK/OFDM + FEC before adopting complex analog symbols. Use masked VQ-VAE or MOC-RVQ as deeper targets. Stress test unseen categories and distribution shifts.

### Stage 4 — Add semantic priority and generative decoding

Begin with an explicit semantic signal (segmentation/object labels/CLIP/DINO features) and a deterministic decoder before adding diffusion/DiT. Compare fixed bandwidth, entropy-only allocation, semantic-importance allocation, and combined allocation. Evaluate whether generated content preserves the same objects, attributes, relations, text, count, identity, and geometry. Record decoder latency/energy and sample-to-sample variability.

### Stage 5 — Add a residual and prove near-lossless behavior

Use the semantic/learned codec as a predictor, then encode a quantized residual or lossless residual stream. Report exact pixel equality, maximum absolute error, residual bitrate, channel-decoding success, and end-to-end image success. On channel failure, return an explicit erasure/invalid result rather than a plausible-but-wrong image. To avoid retransmissions, size the one-shot FEC and burst for a declared failure target and show the resulting overhead.

### Recommended paper-to-implementation order

| Order | Paper(s) | What to implement/learn | Why it comes now |
|---:|---|---|---|
| 1 | DeepJSCC; ADJSCC | Analog latent, power normalization, AWGN/fading, SNR feature conditioning | Establish the simplest end-to-end neural baseline |
| 2 | DeepJSCC-Q; OFDM-Guided DeepJSCC | Finite alphabet and multipath/OFDM effects | Expose the gap between differentiable channel simulations and radio symbols |
| 3 | NTSCC; NTSCC++ | Transform, hyperprior/context, variable-rate and feature-priority allocation | Learn learned compression plus channel adaptation |
| 4 | SwinJSCC; PADC | Hierarchical attention, rate selection, and resource/quality controller | Make resolution and instance complexity part of rate selection |
| 5 | Masked VQ-VAE; shared semantic codebook; MOC-RVQ | Codebook training, index transmission, residual quantization, noise rematching | Build a practical discrete semantic representation |
| 6 | VQ-DeepISC; JCM VAE; importance-aware OFDM | Combine indices with finite constellations and OFDM | Connect semantic codes to a plausible digital PHY |
| 7 | DiffJSCC; DiffCom; DiT-JSCC | Generative reconstruction and semantic consistency controls | Study the extreme-rate perceptual ceiling and hallucination risks |
| 8 | DLPR; RC; FNLIC plus selected exact-image models | Add a residual and measure exact recovery | Separate “looks right” from source-correct near-lossless delivery |

Do not begin with the largest diffusion model. First reproduce the rate accounting and deterministic baselines; otherwise a very strong image prior can make a tiny payload look successful while hiding what the channel did not communicate.

## Research-method and quality-control checklist

For each reproduction, preserve paper version, code commit, dataset version/split, preprocessing, random seeds, hardware/software, training compute, channel generator, SNR definition, model checkpoint, and all post-processing. Separate “paper-reported,” “locally reproduced,” “simulated,” and “proposed” results. Run at least three seeds for stochastic training where feasible and enough independent channel draws to place confidence intervals around image-level success. Keep test images out of training, oracle tuning, threshold selection, and codebook fitting. Use paired image/channel draws for baseline comparisons. Publish the full payload ledger, failure cases, codebook occupancy, latency protocol, and scripts.

## Bottom line for the stated goal

For **pure semantic meaning with very low channel use**, prioritize the codebook/conditional-generation and semantic-base papers (masked VQ-VAE, MOC-RVQ, low-rate conditional generation, DiffJSCC/DiT-JSCC), but accept that reconstructed pixels may differ. For **near-lossless/high-fidelity one-shot transmission**, prioritize NTSCC/NTSCC++ or SwinJSCC as a learned predictor, then add a protected residual or exact digital detail path and evaluate at the complete-burst level. For **lossless**, semantic generation alone cannot supply the missing source bits: use an exact residual/entropy code plus a channel code, with a nonzero probability of burst failure at finite SNR unless the rate and protection are sufficient. The best project contribution is likely to make the rate-quality-semantic-reliability tradeoff explicit in one measurable one-shot contract.

## Bibliography notes and source status

The primary links in each heading are the intended citations. The main baseline mechanisms and DiT-JSCC quantitative table were checked against paper text through alphaXiv PDF queries; recent papers were checked against arXiv abstracts or primary venue records. If a section says its exact numerical setup needs full-text verification, retain that warning rather than importing a number from a secondary summary. Venue, preprint, and first-release years may differ; the chronology uses first appearance where known.

## Appendix — implementation cards for all 50 papers

These cards add the requested fields paper by paper. “Focus” distinguishes approximate pixel reconstruction, perceptual/task semantics, and exact/lossless coding; these objectives are not interchangeable. Each flow is a compact explanatory block diagram reconstructed from the method, not copied from a paper figure. NR means the exact value was not reliably available in the source material reviewed for this survey; it is left unfilled rather than guessed. Results are author-reported under their own settings, not results reproduced in our project.

Implementation effort below is my estimate for one researcher building a faithful prototype, not a paper-reported duration: Medium is roughly 3–7 working days; High is roughly 2–4 weeks; Very high is roughly 4–8+ weeks. Existing public code/checkpoints may shorten this; reproducing from scratch, training large models, or setting up a PHY can take longer. These are planning bands, not guarantees. Paper-specific training/inference time is marked NR unless a number is available. Hardware timings are not portable between setups.

### I. Learned joint source-channel coding and adaptation

#### 1. [Deep Joint Source-Channel Coding for Wireless Image Transmission (2019; preprint 2018)](https://arxiv.org/abs/1809.01733)

- **Focus / motivation:** Approximate pixel reconstruction with graceful degradation, addressing the finite-blocklength “cliff” of separate source and channel coding.
- **Method / flow + block diagram:** Image → CNN analysis encoder → power-normalized complex I/Q symbols → AWGN or slow-Rayleigh channel → CNN synthesis decoder → image.
- **Deep learning / architecture:** Convolutional autoencoder, differentiable stochastic channel, power normalization and MSE training; continuous channel symbols, not a bitstream or fixed QAM alphabet.
- **Data / metrics / result:** CIFAR-10 and Kodak; AWGN and slow Rayleigh; PSNR/SSIM against JPEG/JPEG2000 plus channel coding. Reports better quality at low SNR/bandwidth and graceful degradation; exact results depend on operating point and are given in the paper’s plots.
- **Cost / difficulty:** Medium. Best first reproduction. Channel layer is simple; fair digital baselines take work. Runtime/compute: NR in reviewed evidence.

#### 2. [DeepJSCC-f: Deep Joint Source-Channel Coding of Images with Feedback (2020)](https://arxiv.org/abs/1911.11174)

- **Focus / motivation:** Improve fidelity per channel use through receiver feedback and progressive refinement.
- **Method / flow:** Image → staged encoder transmission → receiver reconstruction/error feedback → next refinement symbols → final reconstruction.
- **Deep learning / architecture:** CNN JSCC encoder/decoder with feedback-conditioned, round-wise refinement.
- **Data / metrics / result:** Noisy image channels; exact dataset and numerical values are NR in reviewed notes. The paper reports gains from feedback; quality must be compared alongside total uses and number of rounds.
- **Cost / difficulty:** High. Round synchronization and feedback simulation add complexity. Actual feedback rounds conflict with a one-shot pre-scheduled burst. Runtime: NR.

#### 3. [Bandwidth-Agile Image Transmission with Deep Joint Source-Channel Coding (2020)](https://arxiv.org/abs/2009.12480)

- **Focus / motivation:** Produce useful reconstructions at different available bandwidths, rather than training only for one fixed rate.
- **Method / flow:** Image → bandwidth-agile CNN encoder → selected prefix/length of channel symbols → noisy channel → length-aware decoder → image.
- **Deep learning / architecture:** CNN JSCC with progressive or variable-length latent organization and multi-rate training.
- **Data / metrics / result:** Image benchmarks over noisy channels; exact dataset and numeric operating points are NR here. Its result is useful reconstruction across selected transmission lengths; rate signaling must be counted.
- **Cost / difficulty:** Medium. Builds on DeepJSCC but needs rate-aware training and truncation tests. Runtime: NR.

#### 4. [Wireless Image Transmission Using Deep Source Channel Coding With Attention Modules (ADJSCC; preprint 2020, journal 2022)](https://arxiv.org/abs/2012.00533)

- **Focus / motivation:** Adapt one image codec over a range of SNRs instead of storing one model per channel point.
- **Method / flow:** Image + SNR → CNN features → SNR-conditioned feature-channel attention/scale-bias → symbols → noisy channel → conditioned decoder → image.
- **Deep learning / architecture:** CNN autoencoder, feature-channel attention and SNR conditioning. “Channel attention” here means feature-map channels, not radio channels.
- **Data / metrics / result:** Image transmission over AWGN/fading; compares with DeepJSCC and separated compression/channel coding. Reports improved adaptation over tested conditions; exact common-point values are NR in these notes.
- **Cost / difficulty:** Medium. Straightforward after DeepJSCC; both ends need consistent SNR conditioning. Runtime: NR.

#### 5. [DeepJSCC-Q: Channel Input Constrained Deep Joint Source-Channel Coding (2021)](https://arxiv.org/abs/2111.13042)

- **Focus / motivation:** Make learned transmission compatible with finite symbol alphabets used in digital radios.
- **Method / flow:** Image → learned encoder → finite-alphabet mapper/quantizer → noisy channel → constrained-symbol decoder → image.
- **Deep learning / architecture:** CNN JSCC autoencoder, quantization/alphabet constraint and end-to-end channel training.
- **Data / metrics / result:** Wireless image benchmarks; exact datasets/points are NR in reviewed notes. The authors report performance close to continuous DeepJSCC and better than BPG+LDPC in selected matched settings, with graceful degradation.
- **Cost / difficulty:** High. Training-time quantization must match inference symbols; a radio implementation also needs mapping, FEC, framing and synchronization. Runtime: NR.

#### 6. [OFDM-Guided Deep Joint Source-Channel Coding for Wireless Multipath Fading Channels (preprint 2021; journal 2022)](https://arxiv.org/abs/2109.05194)

- **Focus / motivation:** Move from ideal memoryless channels toward frequency-selective multipath radio.
- **Method / flow:** Image → JSCC encoder → OFDM subcarrier mapping → multipath channel → receive/equalize → image decoder.
- **Deep learning / architecture:** Neural JSCC coupled to an OFDM-guided channel model; symbol-to-resource-element mapping is central.
- **Data / metrics / result:** Image transmission in multipath fading; exact dataset and headline values are NR in reviewed notes. Demonstrates the channel-model extension; simulation is not over-the-air evidence.
- **Cost / difficulty:** High. Requires PHY assumptions, channel estimation/equalization and overhead accounting. Runtime: NR.

#### 7. [Nonlinear Transform Source-Channel Coding for Semantic Communications (NTSCC; preprint 2021, JSAC 2022)](https://arxiv.org/abs/2112.10961)

- **Focus / motivation:** Combine learned compression and channel-aware transmission so rate follows content statistics and perceptual value.
- **Method / flow:** Image → nonlinear analysis transform → latent plus hyperprior/context entropy model → adaptive JSCC → channel → synthesis transform → image.
- **Deep learning / architecture:** Learned analysis/synthesis transform, entropy/context modeling, hyperprior and rate-distortion optimization.
- **Data / metrics / result:** Multiple-resolution image sources and channel conditions; rate-distortion/perceptual metrics against analog DeepJSCC and separated digital coding. Reports improved rate-distortion behavior across tested sources; no single universal gain.
- **Cost / difficulty:** Very high. Transform, entropy model and channel-aware rate path must be reproduced together. Runtime values need full-table verification.

#### 8. [Deep Joint Source-Channel Coding for Wireless Image Transmission with Adaptive Rate Control (2021)](https://arxiv.org/abs/2110.04456)

- **Focus / motivation:** Use fewer channel symbols for easy images or favorable channel conditions while meeting a quality target.
- **Method / flow:** Image + SNR → CNN features → Gumbel-Softmax rate/group selector → base and refinement symbols → channel → rate-aware decoder → image.
- **Deep learning / architecture:** CNN JSCC, differentiable discrete selection via Gumbel-Softmax and SNR-aware rate control.
- **Data / metrics / result:** Image/channel benchmarks; reports reduced channel use at comparable target quality. Exact dataset and rate savings are NR in reviewed notes. Side-information mask/length contributes to the true payload.
- **Cost / difficulty:** High. Selector, decoder and signaling must agree. Runtime: NR.

#### 9. [Predictive and Adaptive Deep Coding for Wireless Image Transmission in Semantic Communication (PADC; 2023)](https://doi.org/10.1109/TWC.2023.3234408)

- **Focus / motivation:** Select the smallest rate predicted to meet a requested PSNR for each image and SNR.
- **Method / flow:** Image + SNR + candidate CR → variable-length DeepJSCC-V → OraNet quality predictor → CR optimizer → chosen symbols → channel/decoder.
- **Deep learning / architecture:** Variable-length JSCC, learned quality oracle and constrained rate optimization.
- **Data / metrics / result:** Reports PSNR prediction error below 0.5 dB and near-minimum channel use for a requested PSNR in tested settings; exact dataset/rate table is NR in these notes. This is a predicted target, not a max-error guarantee.
- **Cost / difficulty:** High. Needs image/SNR/rate coverage and validation of predictor failures. Runtime: NR.

#### 10. [WITT: A Wireless Image Transmission Transformer for Semantic Communications (preprint 2022; ICASSP 2023)](https://arxiv.org/abs/2211.00937)

- **Focus / motivation:** Use hierarchical Transformer context for high-resolution image transmission beyond CNN receptive-field limits.
- **Method / flow:** Image → patch/features → hierarchical Swin encoder → channel symbols → wireless channel → Swin decoder → image.
- **Deep learning / architecture:** Shifted-window hierarchical vision Transformer with JSCC channel layer.
- **Data / metrics / result:** Image quality under channel conditions against CNN-JSCC and digital codecs; reports improved quality in tested settings. Exact dataset and values are NR in this card.
- **Cost / difficulty:** High. Transformer memory grows with resolution; measure rate and latency at matched conditions. Runtime: NR.

#### 11. [Improved NTSCC to Catalyze Semantic Communications (NTSCC++; 2023)](https://arxiv.org/abs/2303.14637)

- **Focus / motivation:** Improve coding efficiency, flexible rates and semantic priority on top of NTSCC.
- **Method / flow:** Image → analysis transform/context entropy model → multi-rate response network + latent editing → channel → synthesis transform → image.
- **Deep learning / architecture:** Context entropy model, response network, latent feature editing and learned rate control.
- **Data / metrics / result:** High-resolution/XR-oriented rate-distortion evaluation; reports gains over VTM + 5G LDPC and lower processing latency in its setup. Exact split/hardware are NR in these notes.
- **Cost / difficulty:** Very high. Adds context modeling and flexible feature control to NTSCC. Runtime table should be checked in the full paper.

#### 12. [SwinJSCC: Taming Swin Transformer for Deep Joint Source-Channel Coding (preprint 2023; journal 2025)](https://arxiv.org/abs/2308.09361)

- **Focus / motivation:** Improve high-resolution quality and allow one model to adapt to rates and SNR.
- **Method / flow:** Image → patch embedding + hierarchical shifted-window Swin stages → rate modulation selects feature channels; channel modulation conditions on SNR → I/Q symbols → channel → inverse modulation/Swin synthesis → image.
- **Deep learning / architecture:** Swin Transformer, rate-modulation network and SNR-conditioned channel-modulation network.
- **Data / metrics / result:** DIV2K training crops; Kodak/CLIC2021 tests; AWGN and fast Rayleigh; PSNR, MS-SSIM, parameters, GFLOPs and latency. Kodak/AWGN, 7 dB, CBR 1/6: 36.2 dB vs 34.4 dB PSNR for listed DeepJSCC(256); 203.3 vs 494.0 GFLOPs and 33.07M vs 14.59M parameters. One no-adaptation setup reports 151 ms encode and 13 ms decode on the authors’ hardware.
- **Cost / difficulty:** High. Open code helps, but rate masks, image padding, resolution, and timing protocol need close control. Lossy, not exact.

#### 13. [DeepJSCC-1++: Robust and Bandwidth-Adaptive Wireless Image Transmission (2023)](https://doi.org/10.1109/GLOBECOM54140.2023.10436878)

- **Focus / motivation:** Improve robustness and rate flexibility within the DeepJSCC family.
- **Method / flow:** Image → enhanced/adaptive JSCC encoder → rate-selected symbols → channel → robust JSCC decoder → image.
- **Deep learning / architecture:** Learned CNN JSCC with robustness and bandwidth adaptation; full details require proceedings text.
- **Data / metrics / result:** Image/channel experiments; exact datasets and numerical comparisons are NR in reviewed notes. Do not rank numerically until its full result tables are extracted.
- **Cost / difficulty:** High. Need proceedings details, rate schedules and matched baselines. Runtime: NR.

#### 14. [Robust Semantic Communications with Masked VQ-VAE Enabled Codebook (preprint 2022; TWC 2023)](https://arxiv.org/abs/2206.04011)

- **Focus / motivation:** Preserve task-relevant semantics against semantic/channel noise while transmitting compact shared codes.
- **Method / flow:** Image → feature encoder + shared VQ-VAE codebook → semantic mask/importance selection → protected indices → channel → codebook lookup + masked restoration → semantic/task output.
- **Deep learning / architecture:** VQ-VAE, masking, feature importance and adversarial weight perturbation.
- **Data / metrics / result:** Task-oriented image communication; paper reports improved downstream robustness and lower overhead against selected baselines. Exact dataset and numeric gains are NR in these notes; it does not establish arbitrary pixel recovery.
- **Cost / difficulty:** High. Codebook, mask, index protection and task metric are coupled. Runtime: NR.

#### 15. [Perceptual Learned Source-Channel Coding for High-Fidelity Image Semantic Transmission (2022)](https://ieeexplore.ieee.org/abstract/document/10001359)

- **Focus / motivation:** Improve visual quality beyond MSE, which can blur perceptually important edges and texture.
- **Method / flow:** Image → perceptual JSCC encoder → noisy latent → perceptual decoder → reconstruction; feature-space loss guides training.
- **Deep learning / architecture:** Learned source-channel autoencoder and perceptual feature objective; exact layer configuration is NR.
- **Data / metrics / result:** Image transmission; exact datasets and numerical PSNR/LPIPS values are NR in reviewed notes. Perceptual realism does not imply exact source detail.
- **Cost / difficulty:** Medium–High. Need to reproduce perceptual loss and matched channel budget. Runtime: NR.

#### 16. [Deep Learning-Based Image Semantic Coding for Semantic Communications (2021)](https://doi.org/10.1109/GLOBECOM46510.2021.9685074)

- **Focus / motivation:** Bridge learned image compression and communication that prioritizes useful visual content.
- **Method / flow:** Image → learned semantic representation/coder → compact payload → channel → semantic/image decoder → output.
- **Deep learning / architecture:** Deep learned image coding and semantic representation; exact backbone/loss needs proceedings verification.
- **Data / metrics / result:** Image semantic coding benchmark; dataset, metrics and numeric results are NR in available notes. Evaluate the actual objective before assuming “semantic” means explicit object understanding.
- **Cost / difficulty:** Medium–High. Reproduction requires extracting missing protocol details. Runtime: NR.

#### 17. [Deep Learning-Based Adaptive Joint Source-Channel Coding using Hypernetworks (2024)](https://arxiv.org/abs/2401.11155)

- **Focus / motivation:** Adapt JSCC to channels without storing a complete separate network at every condition.
- **Method / flow:** SNR/channel condition → compact hypernetwork → generated/adapted JSCC weights → encoder → channel → matching decoder → image.
- **Deep learning / architecture:** Hypernetwork-generated parameters and conditional DJSCC; unlike attention, it changes weights rather than only activations.
- **Data / metrics / result:** Image-channel adaptation; authors report improved adaptability with fewer added parameters than attention baselines. Exact data and numeric gains are NR here.
- **Cost / difficulty:** High. Dynamic weights and model-state agreement add complexity. Runtime and parameter tables require full-paper check.

#### 18. [Content-Adaptive Distributed JSCC for Image Transmission with Hyperprior (2025)](https://doi.org/10.1109/TCCN.2024.3438371)

- **Focus / motivation:** Exploit correlated images from separate cameras and allocate rate according to content.
- **Method / flow:** View A + view B → distributed encoders/hyperpriors → entropy masks → separate channels → multilevel cross-attention fusion → image reconstruction.
- **Deep learning / architecture:** Distributed JSCC, hyperprior, entropy masks and multilevel cross-view attention.
- **Data / metrics / result:** KITTI and InStereo2K; PSNR/MS-SSIM; reports about 1 dB PSNR gain over its distributed-JSCC baseline.
- **Cost / difficulty:** Very high. Requires synchronized multiview data and distributed-rate accounting; not directly a single-camera codec. Runtime: NR.

#### 19. [DeepJSCC: One-Model-to-Many (2025)](https://doi.org/10.1109/LWC.2025.3547527)

- **Focus / motivation:** Reduce checkpoint/model-selection burden with a shared model that serves multiple operating points.
- **Method / flow:** Image + rate/SNR/resolution controls → shared JSCC encoder → channel → condition-aware shared decoder → image.
- **Deep learning / architecture:** Multi-condition training and conditional DeepJSCC; exact backbone and controls require paper verification.
- **Data / metrics / result:** Multiple settings are evaluated; exact dataset and metric values are NR in this review. Report worst-case as well as average performance.
- **Cost / difficulty:** Medium–High. Model is simple to prototype, but broad condition coverage and generalization tests are required. Runtime: NR.

#### 20. [Space-Time Design for Deep Joint Source-Channel Coding of Images over MIMO Channels (2023)](https://ieeexplore.ieee.org/document/10304536)

- **Focus / motivation:** Exploit multiple antennas and space-time structure for robust image transmission.
- **Method / flow:** Image → JSCC symbols → antenna/space-time mapping → MIMO fading matrix → combining/detection → image decoder.
- **Deep learning / architecture:** Deep JSCC integrated with MIMO design; exact backbone details are NR in the reviewed notes.
- **Data / metrics / result:** MIMO simulations; exact image data, antenna configuration and numerical results are NR here. Compare equal time-frequency-spatial resources and power.
- **Cost / difficulty:** High. Requires channel matrix, CSI assumptions and resource accounting. Runtime: NR.

#### 21. [Alternate Learning-Based Sparse Semantic Communications for Visual Transmission (SparseSBC; 2023)](https://arxiv.org/abs/2309.16681)

- **Focus / motivation:** Send sparse semantic bits while avoiding a differentiable channel, motivated by channel non-differentiability and dense redundant latents.
- **Method / flow:** Image → semantic-basis encoder → sparse binary-quantized bits → channel → alternate/self-critic receiver training → semantic reconstruction/task output.
- **Deep learning / architecture:** Alternate transmitter/receiver learning, self-critic stabilization, sparse basis and binary quantization.
- **Data / metrics / result:** Visual transmission over various channel conditions; reports performance above typical semantic baselines. Exact dataset and numeric results are NR in this appendix. Count active-position/mask signaling too.
- **Cost / difficulty:** High. Alternating training and serialization are custom; sparsity percentage is not itself total bitrate. Runtime: NR.

#### 22. [On the Role of ViT and CNN in Semantic Communications: Analysis and Prototype Validation (2023)](https://arxiv.org/abs/2306.02759)

- **Focus / motivation:** Test whether global attention helps compared with CNN local bias and validate a prototype.
- **Method / flow:** Image → CNN or ViT encoder → communication/channel → matching decoder → reconstructed/task output.
- **Deep learning / architecture:** Comparative CNN and Vision Transformer models; prototype-oriented hardware/software validation.
- **Data / metrics / result:** Image quality and SDR/prototype measures; one reported comparison gives about +0.5 dB PSNR for a ViT variant over CNN variants. Exact data/conditions are NR in these notes.
- **Cost / difficulty:** Medium–High. Offline comparison is manageable; hardware replication depends on the original SDR setup. Runtime: hardware-specific.

#### 23. [Deep Learning-Enabled Semantic Communication Systems with Task-Unaware Transmitter and Dynamic Data (2022)](https://arxiv.org/abs/2205.00271)

- **Focus / motivation:** Preserve utility despite unknown receiver tasks and source-domain shift.
- **Method / flow:** New-domain image → data-adaptation network → semantic encoder → channel → receiver-led semantic decoder → task/output.
- **Deep learning / architecture:** Receiver-leading training, domain adaptation and semantic coding.
- **Data / metrics / result:** Multiple tasks/domains; reports adaptation while maintaining recovery and task execution. Exact dataset names and gains are NR here.
- **Cost / difficulty:** High. Needs task/domain-specific training and held-out-domain validation. Task accuracy is not exact image recovery. Runtime: NR.

#### 24. [Large AI Model-Based Semantic Communications for Images (2023; revised 2024)](https://arxiv.org/abs/2307.03492)

- **Focus / motivation:** Send selected region semantics and use shared foundation-model knowledge to reduce payload.
- **Method / flow:** Image → SAM segmentation/region descriptors → attention integration + adaptive compression → channel → shared knowledge base/decoder → semantic image.
- **Deep learning / architecture:** Segment Anything Model, attention, semantic knowledge base and adaptive feature compression.
- **Data / metrics / result:** Simulation reports feasibility and lower semantic overhead; exact dataset and gains are NR in reviewed notes.
- **Cost / difficulty:** Very high. SAM inference, database agreement, region serialization and model storage are heavy; segmentation can omit detail. Runtime: NR.

#### 25. [Semantic Segmentation-Based Semantic Communication System for Image Transmission (2023/2024)](https://doi.org/10.1016/j.dcan.2023.02.006)

- **Focus / motivation:** Prioritize explicit object/region labels for task-oriented delivery.
- **Method / flow:** Image → segmentation network → class/region map + selected appearance → channel → semantic/appearance decoder → output.
- **Deep learning / architecture:** Semantic segmentation and region-level representation; detailed layers are NR in these notes.
- **Data / metrics / result:** Image transmission; exact dataset and numeric metrics are NR here. Segmentation/task quality must be separated from PSNR and exact pixels.
- **Cost / difficulty:** Medium–High. Existing segmentation modules ease prototyping, but map signaling and error propagation need testing. Runtime: NR.


### II. Task-aware representations, foundation models and rate control

#### 26. [Demo: Real-Time Semantic Communications with a Vision Transformer (2022)](https://arxiv.org/abs/2205.03886)

- **Focus / motivation:** Explore ViT-based semantic transmission with a real-time-oriented prototype.
- **Method / flow:** Image patches → ViT encoder/tokens → communication/channel → ViT decoder → image or semantic output.
- **Deep learning / architecture:** Patch embedding and self-attention; attention cost and token count affect latency.
- **Data / metrics / result:** Prototype image transmission; exact dataset, quality values and hardware timing are NR in reviewed notes. “Real-time” applies to the authors’ platform and measured pipeline.
- **Cost / difficulty:** High for prototype replication, Medium for offline modeling. Measure encoder/decoder/channel time separately; runtime values are hardware-specific.

#### 27. [Language-Oriented Semantic Latent Representation for Image Transmission (2024)](https://arxiv.org/abs/2405.09976)

- **Focus / motivation:** Combine language meaning with visual appearance because captions omit layout, color and texture.
- **Method / flow:** Image → caption/text semantics + compressed visual embedding → transmitted text and latent → receiver fusion + latent diffusion → image.
- **Deep learning / architecture:** Captioning, semantic embeddings and latent diffusion.
- **Data / metrics / result:** Image/noisy-channel tests; reports payload about 2.09% of original image size and improved perceptual similarity over a text-only baseline. Exact corpus/metric table is NR here. Source-size ratio is not radio CBR.
- **Cost / difficulty:** Very high. Captioning plus iterative diffusion costs inference time and can generate details absent from the source. Runtime: NR.

#### 28. [Goal-Oriented Semantic Communication for Wireless Visual Question Answering (2024)](https://arxiv.org/abs/2411.02452)

- **Focus / motivation:** Optimize answer correctness for a question instead of reconstructing every image pixel.
- **Method / flow:** Image + question → detect/rank relevant regions → compact visual tokens → channel → VQA decoder → answer.
- **Deep learning / architecture:** Object/region detection, relevance ranking and task-oriented VQA.
- **Data / metrics / result:** VQA-style image/question data; answer accuracy and communication rate are central. Exact datasets and gains are NR in reviewed notes.
- **Cost / difficulty:** High. Requires synchronized detection/VQA modules and question-conditioned transmission; it is not a general whole-image codec. Runtime: NR.

#### 29. [A Survey on Semantic Communication for Vision: Categories, Frameworks, Enabling Techniques, and Applications (2026)](https://arxiv.org/abs/2601.22202)

- **Focus / motivation:** Organize visual semantic communication research; this is a survey rather than a new codec.
- **Method / flow:** Literature search and screening → taxonomy of semantic preservation, expansion and refinement → synthesis of frameworks/applications.
- **Deep learning / architecture:** No single model; reviews encoders, decoders, knowledge use and communication methods.
- **Data / metrics / result:** No image benchmark or model score. Verify latest version/venue before formal citation.
- **Cost / difficulty:** Not applicable as model implementation. Reproducing the literature-search protocol is outside the model cost estimate.

### III. Codebooks, quantization and generative semantic decoding

#### 30. [Improving Learning-Based Semantic Coding Efficiency via a Shared Semantic-Aware Codebook (preprint 2024; TCOMM 2025)](https://doi.org/10.1109/TCOMM.2024.3450877)

- **Focus / motivation:** Reduce transmission of continuous semantic features by sharing a learned codebook at both endpoints.
- **Method / flow:** Image → semantic feature encoder → nearest shared-codebook vectors → indices/bits → channel → protected indices + same codebook lookup → decoder.
- **Deep learning / architecture:** Feature learning, vector quantization, shared dictionary and semantic-aware index coding.
- **Data / metrics / result:** AWGN/Rayleigh and multiple rates/SNRs; evaluates coding efficiency and reconstruction. Exact dataset and headline values are NR in this card.
- **Cost / difficulty:** High. Requires codebook coverage/collapse monitoring, index protection and codebook-version synchronization. Runtime: NR.

#### 31. [Codebook-Enabled Generative End-to-End Semantic Communication Powered by Transformer (2024)](https://arxiv.org/abs/2402.16868)

- **Focus / motivation:** Make compact codebook indices robust to noise and use shared visual knowledge to synthesize a plausible image.
- **Method / flow:** Image → semantic encoder/VQ codebook → indices → Transformer index-noise handling → channel → shared codebook + generative decoder → image.
- **Deep learning / architecture:** Vector quantization, Transformer and generative decoding.
- **Data / metrics / result:** Image-channel experiments; reports visual-perception gains over JPEG+LDPC and traditional JSCC. Exact dataset/metric values are NR in reviewed notes; plausible appearance is not pixel identity.
- **Cost / difficulty:** High. Needs index-aware training and synchronized codebook/generator. Runtime: NR.

#### 32. [MOC-RVQ: Multilevel Codebook-Assisted Digital Generative Semantic Communication (2024)](https://arxiv.org/abs/2401.01272)

- **Focus / motivation:** Reconcile expressive learned codebooks with small digital alphabets using multilevel residual quantization.
- **Method / flow:** Image → multi-head octonary codebook + RVQ indices → digital channel → Swin noise-reduction block + semantic codebook → generative feature/image restoration.
- **Deep learning / architecture:** Multi-head VQ, residual VQ, Swin Transformer denoising and a semantic knowledge base.
- **Data / metrics / result:** Diverse 2K-resolution test images; compares BPG/JPEG and analog JSCC. Reports better quality than BPG/JPEG and quality comparable to analog JSCC at one-sixth the channel-bandwidth ratio under its protocol. Exact image-source names and metric operating points require the full tables.
- **Cost / difficulty:** Very high. Two-stage training, multilevel indices, digital channel and restoration interact; author code is available. Runtime: NR.

#### 33. [SQ-GAN: Semantic Image Communications Using Masked Vector Quantization (2025)](https://arxiv.org/abs/2502.09520)

- **Focus / motivation:** Spend very low-rate tokens on semantic classes relevant to the downstream task, rather than preserving all pixels uniformly.
- **Method / flow:** Image → segmentation map → VQ-GAN features → semantic-conditioned adaptive mask → indices → decoder/generator → image and semantic map.
- **Deep learning / architecture:** VQ-GAN, vector quantization, segmentation, task-weighted losses, adaptive masking and adversarial discriminator.
- **Data / metrics / result:** Multiple image/task tests; compares JPEG2000, BPG and learned codecs using perceptual quality and segmentation accuracy. Reports superior results at extremely low compression rates; exact metric values and dataset split are NR in these notes.
- **Cost / difficulty:** High. Pretrained segmentation and adversarial/VQ training; count mask signaling and all token bits. Code is reported by the authors. Runtime: NR.

#### 34. [Low-Rate Semantic Communication with Codebook-Based Conditional Generative Models (2025)](https://arxiv.org/abs/2504.04977)

- **Focus / motivation:** Preserve salient meaning at very low rate, allowing the receiver’s generator to synthesize omitted appearance.
- **Method / flow:** Image → saliency/semantic features → compact VQ indices → channel → conditional image generator → reconstruction.
- **Deep learning / architecture:** Semantic feature selection, vector quantization and conditional generation.
- **Data / metrics / result:** The authors describe an ultra-low-bitrate setup with average compression ratio **0.57‰**; they report superior visual quality to traditional JSCC and higher saliency similarity than recent generative semantic baselines. The result is perceptual/task-oriented, not exact pixels. Full metric table and image-split names are not extracted in this review.
- **Cost / difficulty:** High. Needs generator conditioning, fair bitrate accounting and semantic failure tests. Runtime: NR.

#### 35. [Knowledge-Base-Based Semantic Image Transmission Using CLIP (2025)](https://arxiv.org/abs/2504.01053)

- **Focus / motivation:** Replace part of image payload with compact semantics and retrieve shared visual knowledge at the receiver.
- **Method / flow:** Image → 512-D CLIP embedding → FAISS/shared vector database → descriptor or entry selection sent → receiver retrieval/conditioned reconstruction.
- **Deep learning / architecture:** CLIP image embeddings, nearest-neighbor search and shared knowledge base.
- **Data / metrics / result:** CIFAR-100; evaluates semantic/category consistency and retrieval behavior. Exact metric values are NR in this card. Retrieval can fail for novel identities, cultures or details missing from the database.
- **Cost / difficulty:** High. Requires database construction/versioning, privacy controls and open-set evaluation. Runtime depends on database/hardware; paper-specific latency is NR.

#### 36. [VQ-DeepISC: Vector-Quantized Digital Semantic Communication with Channel-Adaptive Image Transmission (2025)](https://arxiv.org/abs/2508.03740)

- **Focus / motivation:** Join learned discrete image tokens with a practical digital PHY and adapt to channel conditions.
- **Method / flow:** Image → Swin feature encoder → EMA-updated VQ codebook → indices → QPSK/OFDM-style channel → index/feature correction → Swin decoder → image.
- **Deep learning / architecture:** Swin Transformer, vector quantization, EMA codebook updates and channel adaptation.
- **Data / metrics / result:** Image transmission with a QPSK/OFDM setup aligned by the paper to IEEE 802.11a. Authors report improved reconstruction fidelity; exact dataset/metric values are NR in reviewed notes. Verify index errors and full radio-resource budget.
- **Cost / difficulty:** Very high. Needs VQ plus OFDM and complete pilot/FEC/header accounting. Runtime: NR.

#### 37. [VQ-VAE-Based Digital Semantic Communication with Importance-Aware OFDM Transmission (2025)](https://arxiv.org/abs/2508.08686)

- **Focus / motivation:** Allocate more reliable OFDM positions to important semantic features.
- **Method / flow:** Image → VQ-VAE/codebook → importance scores + indices → importance-aware OFDM mapping/protection → channel → noisy-feature rematching → decoder.
- **Deep learning / architecture:** VQ-VAE, gradient-derived feature importance, OFDM allocation and nearest-codeword correction.
- **Data / metrics / result:** Low-SNR image transmission; reports better reconstruction than a DeepSC baseline. Dataset, PSNR/SSIM values and total PHY budget are NR here. Nearest-codeword rematching is a heuristic, not an error-correction guarantee.
- **Cost / difficulty:** High. Joint importance/resource mapping and fair total-resource comparison are needed. Runtime: NR.

#### 38. [Generative AI-Based Vector-Quantized End-to-End Semantic Communication for Wireless Image Transmission (2025)](https://doi.org/10.1109/TMLCN.2025.3607891)

- **Focus / motivation:** Combine compact VQ indices, semantic priority and generative decoding for human/machine perception.
- **Method / flow:** Image → VQ encoder + edge/energy-aware attention → protected indices → channel → shared VQ/generative decoder → image/task output.
- **Deep learning / architecture:** Vector quantization, spatial attention, Wasserstein critic and contrastive pixel/latent/perceptual/task objectives.
- **Data / metrics / result:** High-resolution image tests; considers human/machine perception, robustness, complexity and latency. Reports low-SNR gains over BPG+LDPC and comparable JSCC quality at lower complexity/latency; exact values are NR in these notes.
- **Cost / difficulty:** Very high. Multi-objective high-resolution VQ-GAN training is demanding. A critic can improve realism without source truth. Runtime: paper-specific table needed.

### IV. Exact coding, diffusion/flow decoders and hybrid systems

#### 39. [Adapting Diffusion Language Models for Lossless Pixel-Level Image Transmission (DDM-SSCC; 2026)](https://arxiv.org/abs/2606.06273)

- **Focus / motivation:** Exact pixel recovery using separate source-channel coding; addresses the fact that semantic reconstruction can differ from original pixels.
- **Method / flow:** Pixels → discrete diffusion token probabilities → synchronized reverse arithmetic coding → bitstream → channel/FEC → arithmetic decoding + scheduled token restoration → exact pixels when stream decodes correctly.
- **Deep learning / architecture:** Discrete diffusion language model, bidirectional attention, Halton-guided token order, mask-aware cosine schedule, temperature calibration and arithmetic coding.
- **Data / metrics / result:** CIFAR-10, DIV2K-LR-X4 and Kodak over AWGN/Rayleigh. Evaluates exact recovery against selected lossless and semantic baselines; claims better exact-recovery performance. Exact bpp, block-success probabilities and table values are NR in reviewed notes. Lossless coding does not mean zero channel error probability.
- **Cost / difficulty:** Very high. Synchronized entropy coding and diffusion decoding are difficult; validate exact decoder agreement and whole-burst success. Runtime: NR.

#### 40. [Joint Coding-Modulation for Digital Semantic Communications via Variational Autoencoder (preprint 2023; journal 2024)](https://arxiv.org/abs/2310.06690)

- **Focus / motivation:** Avoid mismatch between a learned semantic representation and separately appended digital modulation by learning them jointly.
- **Method / flow:** Image/features → VAE encoder + quantized latent → finite-alphabet symbols → channel → learned demodulation/latent decoder → image/task output.
- **Deep learning / architecture:** Variational autoencoder, quantized latent and joint coding/modulation.
- **Data / metrics / result:** Digital semantic tests; exact image dataset, metrics and gains are NR in reviewed notes. Compare against DeepJSCC-Q at identical alphabet, FEC and resource budget.
- **Cost / difficulty:** High. Discrete/continuous training and PHY integration; a digital constellation still has block errors. Runtime: NR.

#### 41. [Diffusion-Aided Joint Source-Channel Coding for High-Realism Wireless Image Transmission (DiffJSCC; 2024)](https://arxiv.org/abs/2404.17736)

- **Focus / motivation:** Improve visual realism where MSE-trained JSCC blurs images at extreme rates.
- **Method / flow:** Image → low-rate JSCC → noisy symbols → coarse reconstruction/features + spatial/text/SNR conditions → Stable Diffusion → image.
- **Deep learning / architecture:** DeepJSCC, pretrained Stable Diffusion and conditional denoising.
- **Data / metrics / result:** Kodak at 768×512 is highlighted; at 1 dB, reports realistic results from 3,072 symbols (<0.008 symbols/pixel) and perceptual/task gains over JSCC baselines. Generated details may not be source-correct.
- **Cost / difficulty:** Very high. Pretrained model memory, iterative decoding, latency and energy dominate. Exact runtime: NR in these notes.

#### 42. [DiffCom: Channel-Received Signal as a Natural Condition for Diffusion Posterior Sampling (2024)](https://arxiv.org/abs/2406.07390)

- **Focus / motivation:** Treat received wireless signal as an observation that conditions diffusion posterior sampling, including unknown-channel cases.
- **Method / flow:** Image → encoder/channel → received signal → channel-consistency observation + diffusion prior → posterior denoising/sampling → image.
- **Deep learning / architecture:** Pretrained diffusion prior, posterior sampling and blind channel handling.
- **Data / metrics / result:** Tests fading/multipath and issues including PAPR/ISI; reports perceptual gains in LPIPS/DISTS/FID. Exact datasets and values are NR in this card.
- **Cost / difficulty:** Very high. Repeated posterior sampling and channel operators cost compute. A plausible posterior sample need not be the original. Runtime: NR.

#### 43. [Channel Denoising Diffusion Models for Wireless Semantic Communications (CDDM; preprint 2023, journal 2024)](https://arxiv.org/abs/2309.08895)

- **Focus / motivation:** Remove channel corruption from semantic representations using diffusion denoising.
- **Method / flow:** Image → semantic encoder → noisy semantic latent → diffusion channel denoiser → semantic/image decoder.
- **Deep learning / architecture:** Forward/reverse diffusion, channel-aware denoising and semantic decoding.
- **Data / metrics / result:** Wireless semantic communication experiments; exact datasets, channels and numeric results are NR in reviewed notes. Record denoising steps and latency in reproduction.
- **Cost / difficulty:** Very high. Diffusion training/sampling adds compute and channel-mismatch tests. Runtime: NR.

#### 44. [Semantics-Guided Diffusion for Deep JSCC in Wireless Image Transmission (2025)](https://arxiv.org/abs/2501.01138)

- **Focus / motivation:** Reduce semantic drift in diffusion-assisted JSCC by supplying more discriminative semantic conditions.
- **Method / flow:** Image → semantic-aware JSCC encoder → channel → received latent + semantic guidance → diffusion decoder → image.
- **Deep learning / architecture:** Semantic feature conditioning, DeepJSCC and diffusion generation.
- **Data / metrics / result:** Low-rate/channel image transmission; exact datasets, metrics and gains are NR in available notes.
- **Cost / difficulty:** Very high. Requires guided diffusion inference and hallucination tests. Runtime: NR.

#### 45. [Goal-Oriented Semantic Communication for Wireless Image Transmission via Stable Diffusion (2024/2025)](https://arxiv.org/abs/2408.00428)

- **Focus / motivation:** Use Stable Diffusion to decode semantic images and handle known or unknown channel gain.
- **Method / flow:** Image → semantic autoencoder → channel → known-gain SD-GSC denoiser or unknown-gain parallel PSD-GSC denoiser → image.
- **Deep learning / architecture:** Semantic autoencoder, Stable Diffusion prior, channel-conditioned denoising and parallel unknown-CSI path.
- **Data / metrics / result:** PSNR/FID against ADJSCC, Latent-Diff DNSC and MMSE-enhanced SD-GSC. Reports known-channel PSNR improvements of 32% and 21% and FID reductions of 40% and 35% versus ADJSCC and Latent-Diff DNSC; unknown-channel PSD-GSC reports 8% higher PSNR and 17% lower FID than the MMSE-enhanced SD-GSC comparison. Recheck PSNR in dB because percentage changes are hard to interpret. Dataset details: NR here.
- **Cost / difficulty:** Very high. Diffusion, CSI branches, latency and energy. Runtime: NR.

#### 46. [Hybrid Digital-Analog Semantic Communications (HDA-DeepSC; 2024)](https://arxiv.org/abs/2405.12580)

- **Focus / motivation:** Use digital reliability for selected semantic features and analog graceful degradation for the remaining representation.
- **Method / flow:** Image → Swin semantic features → digital quantized base + analog features → channel branches → diffusion signal detection/fusion → image.
- **Deep learning / architecture:** Six Swin layers, six heads, width 120; hybrid analog/digital allocation and diffusion-based detector.
- **Data / metrics / result:** 1,000 DIK2K training images and Kodak test; AWGN/Rician; PSNR/MS-SSIM versus semantic analog, DeepJSCC-Q, BPG+AMC and denoising. At Kodak/AWGN/10 dB, reports matching BPG+LDPC PSNR with a 33% bandwidth-compression-ratio improvement under its definition. Assumes perfect CSI in simulation.
- **Cost / difficulty:** Very high. Two transmission branches and fusion require rate/CSI synchronization. Runtime: NR.

#### 47. [Hybrid Semantic-Complementary Transmission for High-Fidelity Image Reconstruction (2025)](https://arxiv.org/abs/2507.17196)

- **Focus / motivation:** Improve fidelity by transmitting a semantic base plus adjustable complementary information missing from the semantic codec.
- **Method / flow:** Image → semantic encoder → fixed semantic representation (SR) + variable complementary residual (CR) → channel → SR decoder + CR recomposition → image.
- **Deep learning / architecture:** Semantic encoder/decoder, learned residual/complementary coding and analytical rate allocation.
- **Data / metrics / result:** Simulated channels and multiple neural architectures; reports lower MSE than semantic-only transmission and derives an optimal CR allocation under its assumptions. Exact data and MSE gains are NR here. CR gives exactness only if complete and decoded exactly.
- **Cost / difficulty:** High. Residual precision, framing and fixed-burst allocation need integration. Runtime: NR.

#### 48. [DiT-JSCC: Rethinking Deep JSCC with Diffusion Transformers and Semantic Representations (2026)](https://arxiv.org/abs/2601.03112)

- **Focus / motivation:** Preserve semantic consistency and visual quality at extreme low rates, where weak conditioning can make a diffusion decoder drift.
- **Method / flow:** Image → semantic branch + detail branch → bandwidth allocator → channel → received conditions → conditional coarse-to-fine DiT → image.
- **Deep learning / architecture:** Visual foundation model, dual branch, latent diffusion; decoder uses 21 DiT blocks, 16 heads, hidden width 1024 and Stable Diffusion VAE.
- **Data / metrics / result:** ImageNet training; 256/512 crops, 100-image validation subset; AWGN/Rayleigh, CBR about 1/384–1/48. At 256², AWGN, CBR 1/48, 0 dB: LPIPS .166, DISTS .151, CLIP .963, DreamSim .097, DINOv2 .958, FID 60; at 5 dB FID 49. These are perceptual/semantic, not lossless metrics.
- **Cost / difficulty:** Very high. Large model, foundation features and expensive inference. Latency/energy values are NR in these notes.

#### 49. [FlowSem: Flow Matching for Adaptive Wireless Image Transmission (2026)](https://arxiv.org/abs/2608.21651)

- **Focus / motivation:** Improve perceptual reconstruction with conditional flow matching after a channel-adaptive coarse transmission.
- **Method / flow:** Image → SNR-adaptive DeepJSCC → fixed symbol budget/channel → coarse image + SNR → conditional flow-matching decoder from noise → image.
- **Deep learning / architecture:** DeepJSCC encoder and conditional flow-matching image generator.
- **Data / metrics / result:** Cityscapes; AWGN/Rayleigh and fixed symbol budget; compared with rate-matched BPG+LDPC, DeepJSCC, DDPM and DDIM. Reports competitive pixel fidelity and improved structure/perceptual quality; exact numeric metrics are NR here. Generated images can differ from source pixels.
- **Cost / difficulty:** Very high. Measure solver steps, latency and source-consistency errors. Runtime: NR.

#### 50. [TwinViT-DeepJSCC: Adversarially Robust Semantic Image Communication (2026 preprint)](https://arxiv.org/abs/2610.08590)

- **Focus / motivation:** Retain quality under adversarial perturbations and channel-aware attacks; this is robustness work, not a new lossless codec.
- **Method / flow:** Image → two ViT-DeepJSCC branches + sensitivity mask → fixed total channel-use budget → attacked/noisy channel → confidence fusion + severity estimation → conditional DDIM purification → image.
- **Deep learning / architecture:** Vision Transformers, dual branches, sensitivity masking, confidence fusion, blind corruption estimation and DDIM.
- **Data / metrics / result:** CIFAR-100; AWGN/block-Rayleigh, SNR −5 to 20 dB; FGSM, 20-step PGD, NES, Carlini–Wagner, random jamming and channel-aware waveform attacks. Reports up to +9.5 dB PSNR under PGD, +10.8 dB under a Rayleigh waveform attack and up to +38 percentage points Top-1 accuracy over an undefended baseline. Recent preprint; independent replication is needed.
- **Cost / difficulty:** Very high. Multiple adversarial loops and diffusion purification multiply compute. CIFAR results do not prove high-resolution or exact-pixel reliability. Runtime: NR.

### Reproduction record to keep for every paper

Record the paper version/code commit, dataset version/split and preprocessing, random seeds, checkpoint and hardware, channel model and SNR definition, CSI assumptions, and inference precision. Keep separate counts for source bytes, compressed bits, payload bits, FEC, headers, pilots, OFDM resource elements and the full scheduled burst. Report PSNR/SSIM/LPIPS/task score alongside exact-pixel success and failure rate when relevant. Measure encoder, decoder and end-to-end wall-clock time with device and batch size. A paper’s “compression ratio” or “channel bandwidth ratio” cannot be compared with another until the denominator and overhead are normalized.

For this project, the most instructive implementation path is DeepJSCC → DeepJSCC-Q/OFDM → NTSCC → SwinJSCC/PADC → shared VQ and MOC-RVQ → SQ-GAN or DiT-JSCC → DDM-SSCC plus a protected residual. The first stages teach channel-aware representation and rate control; codebooks and generative models show how semantic priors reduce payload; the final exactness branch tests source recovery. A semantic generator by itself cannot guarantee the original pixels when those details were never transmitted.


## Appendix — advantages and limitations by paper

In this section, **structural advantage** means the benefit the mechanism is designed to provide under its assumptions; it is not automatically a theorem or a universal guarantee. **Demonstrated advantage** means the authors report evidence for it on their chosen data, channels and metrics. A demonstrated result is still specific to that test setup. Limitations include both fundamental tradeoffs and practical reproduction/deployment issues.

### I. Learned JSCC and adaptive-rate models

#### 1. [DeepJSCC](https://arxiv.org/abs/1809.01733)
- **Structural advantage:** Jointly optimizing image representation and channel robustness can avoid the sharp digital “cliff” and can allocate continuous signal variation to graceful quality changes.
- **Demonstrated advantage:** Reported PSNR/SSIM gains in low-SNR/low-bandwidth regimes and graceful degradation versus separated JPEG/JPEG2000 systems in tested conditions.
- **Disadvantages / limitations:** Continuous I/Q values are not a conventional interoperable bitstream. Training distribution and channel mismatch can hurt; MSE blurs detail; no pixel-exactness or finite-SNR zero-error guarantee.

#### 2. [DeepJSCC-f](https://arxiv.org/abs/1911.11174)
- **Structural advantage:** Feedback lets later transmissions address residual uncertainty after earlier symbols are observed, a form of adaptive refinement.
- **Demonstrated advantage:** The paper reports quality gains from feedback in its tested settings.
- **Disadvantages / limitations:** Requires a return path, multiple rounds and coordination delay. It conflicts with a strict single preplanned burst because the sender cannot use actual feedback if every symbol is sent at once.

#### 3. [Bandwidth-Agile DeepJSCC](https://arxiv.org/abs/2009.12480)
- **Structural advantage:** A useful prefix can support several resource budgets without a wholly separate codec per rate.
- **Demonstrated advantage:** Authors report usable reconstructions at varied channel-symbol lengths.
- **Disadvantages / limitations:** Prefix ordering may not be rate-distortion optimal for every image; the receiver must know the chosen length. It adapts symbol count but does not automatically minimize entropy or include framing overhead.

#### 4. [ADJSCC](https://arxiv.org/abs/2012.00533)
- **Structural advantage:** SNR-conditioned feature attention can change which learned feature channels are emphasized as the channel worsens or improves.
- **Demonstrated advantage:** The paper reports stronger adaptation across channel conditions than a single unconditioned model.
- **Disadvantages / limitations:** Depends on reliable SNR/CSI input and training coverage. Neural feature-channel attention is not radio channel estimation, modulation, or error correction; unseen conditions can cause mismatch.

#### 5. [DeepJSCC-Q](https://arxiv.org/abs/2111.13042)
- **Structural advantage:** A finite input alphabet closes part of the gap between arbitrary learned I/Q and hardware/protocol symbol constraints.
- **Demonstrated advantage:** Authors report approaching unconstrained DeepJSCC and outperforming BPG+LDPC at selected matched operating points.
- **Disadvantages / limitations:** Alphabet compatibility does not eliminate symbol/index errors. Performance depends on the chosen constellation and training surrogate; full FEC, pilots and packet overhead remain.

#### 6. [OFDM-Guided DeepJSCC](https://arxiv.org/abs/2109.05194)
- **Structural advantage:** Incorporating OFDM and frequency-selective fading can make learned representations account for subcarrier-dependent channel effects.
- **Demonstrated advantage:** Demonstrates image transmission over simulated multipath channels beyond memoryless AWGN.
- **Disadvantages / limitations:** Simulator assumptions about CSI, equalization, synchronization and RF impairments limit deployment claims. Resource elements, pilots, cyclic prefix and guards must be counted.

#### 7. [NTSCC](https://arxiv.org/abs/2112.10961)
- **Structural advantage:** A learned transform and entropy/hyperprior model can exploit source redundancy before channel use; content-aware rate allocation is more flexible than a fixed latent budget.
- **Demonstrated advantage:** Reports improved rate-distortion/perceptual performance over analog DeepJSCC and separated digital coding in evaluated settings and resolutions.
- **Disadvantages / limitations:** An entropy model estimates coding likelihood, not semantic importance. Entropy coding can be fragile to bit errors, and transform + hyperprior + JSCC adds training and implementation complexity.

#### 8. [Adaptive Rate Control DeepJSCC](https://arxiv.org/abs/2110.04456)
- **Structural advantage:** A learned selector can spend symbols conditionally on image difficulty and SNR.
- **Demonstrated advantage:** Paper reports lower channel use at comparable target quality for tested cases.
- **Disadvantages / limitations:** Selector mistakes can violate the quality target; masks, length indicators and side information consume resources. A mean quality target is not a per-image maximum-error guarantee.

#### 9. [PADC](https://doi.org/10.1109/TWC.2023.3234408)
- **Structural advantage:** A rate-quality predictor plus optimizer formalizes “use the smallest rate predicted to meet the requested PSNR.”
- **Demonstrated advantage:** Reports less than 0.5 dB PSNR prediction error and near-minimum rate for a target in its tests.
- **Disadvantages / limitations:** The guarantee is empirical and predictor-based, not a strict bound. Distribution shift or channel mismatch can make the selected rate too small; oracle training data and quality calibration are costly.

#### 10. [WITT](https://arxiv.org/abs/2211.00937)
- **Structural advantage:** Hierarchical shifted windows provide local attention with cross-window information exchange, offering a larger spatial modeling capacity than a basic CNN.
- **Demonstrated advantage:** Authors report improved image transmission quality over CNN JSCC baselines for tested resolutions and channels.
- **Disadvantages / limitations:** Attention has memory/compute costs and performance depends on resolution and rate. Better contextual reconstruction remains lossy; exact values should be compared at matched channel use.

#### 11. [NTSCC++](https://arxiv.org/abs/2303.14637)
- **Structural advantage:** Context entropy coding and latent editing can improve compression while letting a single response network support different rates and semantic priorities.
- **Demonstrated advantage:** Reports better rate-distortion efficiency than VTM + 5G LDPC and lower processing latency in the paper’s evaluated setup.
- **Disadvantages / limitations:** Complex entropy/context machinery and multi-rate training make reproduction harder. “Semantic priority” requires a defined task or priority signal, and paper-specific latency may not transfer to another device.

#### 12. [SwinJSCC](https://arxiv.org/abs/2308.09361)
- **Structural advantage:** Hierarchical Swin features capture spatial context; channel modulation handles SNR and rate modulation controls feature-channel use within one model.
- **Demonstrated advantage:** At Kodak/AWGN, 7 dB and CBR 1/6, reports 36.2 dB versus 34.4 dB PSNR for the listed DeepJSCC baseline, with lower reported GFLOPs but more parameters. Paper also reports its own encoder/decoder timings.
- **Disadvantages / limitations:** Still approximate, not lossless. Rate-mask overhead can matter at low resolution; parameter count, adaptation logic and tensor padding complicate deployment. Reported speed is hardware-specific.

#### 13. [DeepJSCC-1++](https://doi.org/10.1109/GLOBECOM54140.2023.10436878)
- **Structural advantage:** Robustness and bandwidth adaptation aim to reduce the need for separate models at every operating point.
- **Demonstrated advantage:** The publication reports robust/adaptive image transmission; exact numeric comparisons need full-paper table extraction before a quantified claim.
- **Disadvantages / limitations:** Its relative gain cannot be inferred from title or abstract. It still belongs to approximate learned JSCC and needs matched image/rate/channel baselines.

#### 14. [Masked VQ-VAE Semantic Communications](https://arxiv.org/abs/2206.04011)
- **Structural advantage:** Shared discrete codewords can reduce latent precision cost; masking and feature importance can suppress semantically unhelpful or noise-sensitive features.
- **Demonstrated advantage:** Reports better task-level robustness and reduced overhead in its evaluated settings.
- **Disadvantages / limitations:** A finite codebook cannot represent every unseen feature exactly; an index error can select a different vector. Results depend on task, mask recovery and shared codebook synchronization, not arbitrary-pixel fidelity.

#### 15. [Perceptual Learned Source-Channel Coding](https://ieeexplore.ieee.org/abstract/document/10001359)
- **Structural advantage:** A perceptual loss can preserve texture/edges that an MSE objective undervalues.
- **Demonstrated advantage:** The work studies perceptual quality as part of learned source-channel coding; exact metric gains need full table verification before quoting.
- **Disadvantages / limitations:** Perceptual similarity can improve while pixel error worsens. Feature losses can favor plausible texture over source truth, so this objective alone is unsuitable for exact recovery.

#### 16. [Deep Learning-Based Image Semantic Coding](https://doi.org/10.1109/GLOBECOM46510.2021.9685074)
- **Structural advantage:** Learned representations can remove source redundancy and prioritize features useful to the selected receiver objective.
- **Demonstrated advantage:** It provides an early image-semantic-coding formulation; exact numerical gains are not independently extracted in this review.
- **Disadvantages / limitations:** “Semantic” is not itself a measurable guarantee. If loss is pixel reconstruction, the method may be learned image coding without explicit task semantics; details need full-paper inspection.

#### 17. [Hypernetwork Adaptive JSCC](https://arxiv.org/abs/2401.11155)
- **Structural advantage:** A small hypernetwork can generate condition-specific parameters instead of storing a full network for every SNR.
- **Demonstrated advantage:** Reports improved channel adaptability with fewer added parameters than attention-based alternatives in its tests.
- **Disadvantages / limitations:** Generated weights add inference work and can be sensitive to conditioning. Both endpoints must agree on model state and channel context; dynamic weights do not solve quantization or exact delivery.

#### 18. [Content-Adaptive Distributed JSCC with Hyperprior](https://doi.org/10.1109/TCCN.2024.3438371)
- **Structural advantage:** Cross-view attention can exploit correlation between cameras and adapt allocation to the joint content.
- **Demonstrated advantage:** Reports roughly 1 dB PSNR gain over its distributed-JSCC baseline on KITTI/InStereo2K evaluations.
- **Disadvantages / limitations:** Gains depend on correlated, synchronized views and do not directly transfer to a single-camera sender. Multiple transmitters and masks create coordination and rate-accounting overhead.

#### 19. [DeepJSCC: One-Model-to-Many](https://doi.org/10.1109/LWC.2025.3547527)
- **Structural advantage:** One model across multiple conditions can lower model storage and model-selection signaling.
- **Demonstrated advantage:** The paper evaluates a shared multi-condition model; detailed numeric gains are NR in this review.
- **Disadvantages / limitations:** Multi-condition training can trade peak performance for average robustness. “Many” conditions must be specified, and unseen resolution/rate/SNR combinations still need testing.

#### 20. [Space-Time DeepJSCC for MIMO](https://ieeexplore.ieee.org/document/10304536)
- **Structural advantage:** Multiple antennas can add spatial diversity or multiplexing opportunities that a scalar-channel model cannot exploit.
- **Demonstrated advantage:** Provides a learned image-JSCC design for MIMO channel simulations; exact gains are not extracted in this review.
- **Disadvantages / limitations:** Benefits depend on antenna count, channel matrix, CSI and equalized spatial resources. More antennas do not automatically mean lower total airtime, power or error probability.

#### 21. [SparseSBC](https://arxiv.org/abs/2309.16681)
- **Structural advantage:** Sparse binary semantic bases can make payload more compact; alternating learning avoids requiring a differentiable physical channel.
- **Demonstrated advantage:** Authors report effective performance across channel settings and advantages over typical semantic baselines.
- **Disadvantages / limitations:** Sparse activations need indices/masks and serialization; overhead can erase nominal sparsity. Task accuracy may survive while image details are lost; alternate training may be sensitive to convergence.

#### 22. [ViT versus CNN Semantic Communication](https://arxiv.org/abs/2306.02759)
- **Structural advantage:** ViT global interactions can represent distant relationships, while CNNs retain efficient local inductive bias; comparison helps choose the right backbone.
- **Demonstrated advantage:** Reports about +0.5 dB PSNR for a ViT variant over CNN variants in one tested condition and includes prototype validation.
- **Disadvantages / limitations:** A single test does not establish general ViT superiority. Token count and hardware affect latency; prototype validation is not necessarily full over-the-air robustness.

#### 23. [Task-Unaware Transmitter and Dynamic Data](https://arxiv.org/abs/2205.00271)
- **Structural advantage:** Receiver-leading training and data adaptation can bridge task uncertainty and source-domain shifts.
- **Demonstrated advantage:** Reports adaptation while preserving data recovery and task execution on evaluated domains.
- **Disadvantages / limitations:** Adaptation does not guarantee open-world generalization. Distribution shift can exceed what the adapter learned, and downstream task success still does not imply pixel-exact reconstruction.

#### 24. [Large AI Model-Based Semantic Communications for Images](https://arxiv.org/abs/2307.03492)
- **Structural advantage:** Segmentation and shared foundation knowledge can send region-level cues instead of all image detail.
- **Demonstrated advantage:** Simulation results report feasibility and lower semantic overhead.
- **Disadvantages / limitations:** SAM adds substantial compute/model storage; segmentation misses or merges regions; masks, database versions and privacy must be handled. Fine appearance, text and identity may be omitted.

#### 25. [Semantic Segmentation-Based Image Communication](https://doi.org/10.1016/j.dcan.2023.02.006)
- **Structural advantage:** Explicit labels make it possible to prioritize foreground objects or regions and interpret what the encoder treats as important.
- **Demonstrated advantage:** Evaluates segmentation-oriented semantic image transmission; numeric benefit is NR in this review.
- **Disadvantages / limitations:** Segmentation errors propagate, label maps cost bits, and class labels discard color, texture and identity. It fits task-specific delivery better than whole-image faithful reconstruction.

### II. Task-aware, foundation-model and codebook approaches

#### 26. [Real-Time Semantic Communications with a Vision Transformer](https://arxiv.org/abs/2205.03886)
- **Structural advantage:** Patch tokens and global self-attention can model image-wide context in one encoder-decoder chain.
- **Demonstrated advantage:** Provides a real-time-oriented ViT communication prototype, demonstrating integration feasibility on the authors’ platform.
- **Disadvantages / limitations:** Real-time claims depend on device, resolution, precision and timing boundary. Self-attention can be expensive at high token counts; prototype performance does not prove robustness on a deployed radio.

#### 27. [Language-Oriented Semantic Latent Representation](https://arxiv.org/abs/2405.09976)
- **Structural advantage:** Text carries high-level meaning while a visual latent retains cues that a caption omits.
- **Demonstrated advantage:** Reports payload about 2.09% of original image size and improved perceptual similarity over text-only transmission in tested noisy settings.
- **Disadvantages / limitations:** The 2.09% source-size ratio is not necessarily complete on-air payload. Captions and diffusion can omit or hallucinate spatial, identity and fine-detail information.

#### 28. [Goal-Oriented Wireless VQA](https://arxiv.org/abs/2411.02452)
- **Structural advantage:** Question-conditioned region selection can spend bits only on evidence needed for the answer.
- **Demonstrated advantage:** Evaluates answer accuracy under communication constraints; exact gains are NR in this review.
- **Disadvantages / limitations:** A representation useful for one question can be useless for another. Missed regions cause unrecoverable answer errors, and this is not a full-image reconstruction codec.

#### 29. [Survey on Semantic Communication for Vision](https://arxiv.org/abs/2601.22202)
- **Structural advantage:** A taxonomy can clarify objectives, identify missing method families and guide paper selection.
- **Demonstrated advantage:** Organizes the field’s categories/frameworks/applications; there is no model benchmark result.
- **Disadvantages / limitations:** Classification quality depends on literature coverage and terminology. A survey cannot prove which architecture performs best; its newest version/venue should be checked before citation.

#### 30. [Shared Semantic-Aware Codebook](https://doi.org/10.1109/TCOMM.2024.3450877)
- **Structural advantage:** Replacing continuous vectors with codebook indices can reduce payload and gives deterministic lookup when codebooks are synchronized.
- **Demonstrated advantage:** Evaluates shared-codebook transmission across AWGN/Rayleigh and several rate/SNR settings.
- **Disadvantages / limitations:** Quantization incurs distortion, unseen features may fall far from all entries, and one bad index can corrupt many reconstructed pixels. Codebook storage/version sync and FEC are real system costs.

#### 31. [Codebook-Enabled Generative Transformer SemCom](https://arxiv.org/abs/2402.16868)
- **Structural advantage:** A Transformer can model dependencies among indices; a shared generator uses prior knowledge to fill missing visual detail.
- **Demonstrated advantage:** Reports improved visual perception over JPEG+LDPC and traditional JSCC in its experiments.
- **Disadvantages / limitations:** Index recovery can fail and a generator may invent plausible but incorrect content. It optimizes appearance/perception, not exact source identity; model synchronization is required.

#### 32. [MOC-RVQ](https://arxiv.org/abs/2401.01272)
- **Structural advantage:** Octonary multi-head codes and residual VQ create compact, progressively refinable digital tokens; Swin denoising exploits the learned codebook prior.
- **Demonstrated advantage:** Reports better performance than BPG/JPEG and analog-JSCC-comparable quality at one-sixth the CBR in the authors’ high-resolution 2K protocol.
- **Disadvantages / limitations:** Two-stage training, multiple index levels and generative restoration make the system complex. Quantization/restoration can alter details; codebook synchronization and symbol protection are essential.

#### 33. [SQ-GAN](https://arxiv.org/abs/2502.09520)
- **Structural advantage:** Task-conditioned masks let the codec spend VQ tokens on semantically important classes and use a generator for omitted content.
- **Demonstrated advantage:** Reports gains over JPEG2000, BPG and learned compression across perceptual and segmentation metrics at very low rates.
- **Disadvantages / limitations:** This is source coding, not a learned radio channel encoder. Task-specific masks can sacrifice non-target details; GAN realism and segmentation accuracy do not mean pixel fidelity or exactness.

#### 34. [Low-Rate Codebook-Based Conditional Generation](https://arxiv.org/abs/2504.04977)
- **Structural advantage:** A compact condition codebook can exploit recurring object/saliency patterns and steer a receiver-side generator.
- **Demonstrated advantage:** Reports average compression ratio 0.57‰, superior visual quality to traditional JSCC, and higher saliency similarity than compared generative SemCom methods.
- **Disadvantages / limitations:** Ultra-low payload depends heavily on shared priors and task-specific saliency. Missing color/layout/identity details cannot be recovered from an insufficient condition; exact metric table and on-air overhead need scrutiny.

#### 35. [CLIP Knowledge-Base Semantic Image Transmission](https://arxiv.org/abs/2504.01053)
- **Structural advantage:** A compact CLIP feature plus retrieval may avoid sending image pixels when category-level meaning is enough.
- **Demonstrated advantage:** CIFAR-100 experiments use category consistency as semantic success, demonstrating a retrieval-based evaluation alternative to PSNR.
- **Disadvantages / limitations:** Category agreement can be high while images differ completely. Retrieval depends on database coverage, version synchronization and privacy; novel identities or rare details may not exist in the knowledge base.

#### 36. [VQ-DeepISC](https://arxiv.org/abs/2508.03740)
- **Structural advantage:** Discrete VQ tokens bridge learned semantic features to a QPSK/OFDM-style digital system; channel adaptation and balanced codeword use target robustness.
- **Demonstrated advantage:** Reports superior reconstruction fidelity over benchmark methods using a QPSK/OFDM setup aligned with IEEE 802.11a.
- **Disadvantages / limitations:** Codebook collapse, index bit errors and standard/PHY configuration details require validation. A paper-aligned simulation is not proof of standards-compliant or over-the-air operation; exact total overhead must be counted.

#### 37. [Importance-Aware VQ-VAE OFDM SemCom](https://arxiv.org/abs/2508.08686)
- **Structural advantage:** Gradient-based importance can place critical features in more reliable OFDM positions; codebook rematching may correct small feature perturbations.
- **Demonstrated advantage:** Reports better low-SNR reconstruction than its DeepSC baseline.
- **Disadvantages / limitations:** Importance is training/task dependent, and nearest-codeword matching is not a reliability guarantee. Pilot spacing, index mapping, FEC and total resource use can change results materially.

#### 38. [Generative AI-Based VQ End-to-End SemCom](https://doi.org/10.1109/TMLCN.2025.3607891)
- **Structural advantage:** VQ reduces transmitted feature precision; attention and multi-level losses can target both human and machine perception.
- **Demonstrated advantage:** Authors report low-SNR gains over BPG+LDPC and comparable JSCC quality with lower complexity/latency in their high-resolution tests.
- **Disadvantages / limitations:** A Wasserstein critic and contrastive losses may improve realism without source correctness. Multi-objective training, codebook state and high-resolution evaluation are costly; independent channel-failure testing is needed.

### III. Exact coding, generative decoders and hybrid systems

#### 39. [DDM-SSCC](https://arxiv.org/abs/2606.06273)
- **Structural advantage:** Discrete token probabilities plus synchronized arithmetic coding can recover the source exactly when the coded stream is decoded correctly; diffusion restoration order supplies bidirectional context.
- **Demonstrated advantage:** Reports improved exact-recovery performance over selected lossless and semantic baselines on CIFAR-10, DIV2K-LR-X4 and Kodak under AWGN/Rayleigh.
- **Disadvantages / limitations:** Source-lossless does not mean channel-error-free. Arithmetic decoder desynchronization can invalidate an object; exact success depends on channel protection and rate. Denoising schedules and probability-model inference may be computationally expensive.

#### 40. [Joint Coding-Modulation VAE](https://arxiv.org/abs/2310.06690)
- **Structural advantage:** Jointly learning the latent and finite-alphabet mapping can reduce mismatch from post-hoc quantization/modulation.
- **Demonstrated advantage:** Establishes a VAE-based digital SemCom design; exact numeric advantage is NR in this review.
- **Disadvantages / limitations:** Discrete latent/channel training is difficult and can depend on surrogate gradients. Digital modulation still has nonzero symbol/block error; exact recovery needs explicit source/channel codes.

#### 41. [DiffJSCC](https://arxiv.org/abs/2404.17736)
- **Structural advantage:** A pretrained diffusion prior can use a small channel observation as conditioning and restore perceptual texture when a conventional decoder blurs.
- **Demonstrated advantage:** Reports realistic Kodak 768×512 reconstruction with 3,072 symbols (<0.008 symbols/pixel) at 1 dB and perceptual/task gains.
- **Disadvantages / limitations:** The prior supplies details that may not have been transmitted; objects, text, geometry or identity can be hallucinated. Iterative diffusion costs latency/energy and does not establish pixel accuracy.

#### 42. [DiffCom](https://arxiv.org/abs/2406.07390)
- **Structural advantage:** Posterior sampling can combine a learned image prior with the actual noisy received observation, potentially handling unknown channels more flexibly.
- **Demonstrated advantage:** Reports perceptual gains on measures such as LPIPS/DISTS/FID and explores mismatch and multipath stress conditions.
- **Disadvantages / limitations:** Many source images can fit the same low-rate observation, so the posterior may select the wrong one. Sampling is expensive; perceptual quality and source identity must be reported separately.

#### 43. [CDDM](https://arxiv.org/abs/2309.08895)
- **Structural advantage:** Diffusion denoising can model complex channel-noise distributions beyond a simple fixed equalizer.
- **Demonstrated advantage:** Evaluates diffusion-based semantic latent denoising in wireless settings; exact numerical gains are NR in this review.
- **Disadvantages / limitations:** Iterative inference adds compute and delay; performance depends on channel/noise training coverage and schedule. A cleaner latent does not guarantee an exact source.

#### 44. [Semantics-Guided Diffusion for DeepJSCC](https://arxiv.org/abs/2501.01138)
- **Structural advantage:** Explicit semantic conditions can constrain diffusion so it is less likely to replace objects or relations while adding detail.
- **Demonstrated advantage:** The method targets reduced semantic drift at low rate; exact numeric gains are NR in reviewed notes.
- **Disadvantages / limitations:** If the transmitted semantic cue is incomplete or corrupted, the generator can still drift. Conditions add payload/model dependencies; exact reconstruction requires a residual or lossless source stream.

#### 45. [Stable-Diffusion Goal-Oriented Image SemCom](https://arxiv.org/abs/2408.00428)
- **Structural advantage:** Channel-aware or parallel denoising lets a large pretrained prior improve images even when CSI is imperfect.
- **Demonstrated advantage:** Reports PSNR/FID changes versus ADJSCC, Latent-Diff DNSC and an MMSE-enhanced variant, including gains for its unknown-channel method.
- **Disadvantages / limitations:** Percent PSNR claims need dB interpretation; known/unknown CSI conditions are not interchangeable. Diffusion remains approximate and incurs heavy inference cost.

#### 46. [HDA-DeepSC](https://arxiv.org/abs/2405.12580)
- **Structural advantage:** Digital features can provide a more stable base while analog features degrade smoothly, combining complementary strengths.
- **Demonstrated advantage:** At Kodak/AWGN/10 dB, reports matching BPG+LDPC PSNR with a 33% bandwidth-compression-ratio improvement under its definition.
- **Disadvantages / limitations:** Requires allocation/fusion signaling and a clear digital/analog split. Perfect CSI and simulation assumptions limit practical claims; this result is not lossless and needs full overhead accounting.

#### 47. [Hybrid Semantic-Complementary Transmission](https://arxiv.org/abs/2507.17196)
- **Structural advantage:** A semantic base captures shared content while a complementary stream targets image-specific residual information; adjustable CR offers a direct rate-fidelity knob.
- **Demonstrated advantage:** Reports lower MSE than semantic-only transmission across tested channels/architectures and derives an optimal CR allocation under its assumptions.
- **Disadvantages / limitations:** An incomplete or noisy complement does not ensure exactness. Analytical allocation assumptions may not match a real packet/OFDM burst; residual overhead and reliability must be included.

#### 48. [DiT-JSCC](https://arxiv.org/abs/2601.03112)
- **Structural advantage:** Separate semantic and detail branches can provide stronger conditions to a generative DiT; conditional complexity allocation aims to spend rate on hard-to-generate content.
- **Demonstrated advantage:** Reports low-rate perceptual/semantic values, including LPIPS .166, DISTS .151, CLIP .963 and FID 60 at the stated 256²/0 dB/CBR 1/48 point; FID 49 at 5 dB.
- **Disadvantages / limitations:** The validation subset is small (100 images), and diffusion/foundation components are large. Strong CLIP/FID does not certify exact source identity, geometry or text; latency and energy remain material.

#### 49. [FlowSem](https://arxiv.org/abs/2608.21651)
- **Structural advantage:** Conditional flow matching can generate details from a coarse JSCC result while conditioning on SNR, potentially improving structure under severe distortion.
- **Demonstrated advantage:** On Cityscapes, reports competitive pixel fidelity and improved structural/perceptual quality against listed baselines at a fixed symbol budget.
- **Disadvantages / limitations:** It adds receiver computation rather than reducing the transmitted budget; generation can hallucinate. Evidence on one domain does not establish general image robustness or exactness.

#### 50. [TwinViT-DeepJSCC](https://arxiv.org/abs/2610.08590)
- **Structural advantage:** Complementary ViT branches, sensitivity masks and confidence fusion can make a fixed-budget representation more resilient; DDIM purification addresses residual attacks.
- **Demonstrated advantage:** On CIFAR-100, reports up to +9.5 dB PSNR under 20-step PGD, +10.8 dB under a Rayleigh channel-aware waveform attack and up to +38 percentage points Top-1 accuracy over an undefended baseline.
- **Disadvantages / limitations:** Results are from a recent preprint and specialized attacks at low resolution. Attack generation and diffusion purification are compute-heavy; these results do not establish normal-channel lossless or high-resolution reliability.

### Cross-paper interpretation

The main theoretical tradeoff is that a stronger learned prior can reduce the number of transmitted bits or symbols only by asking the receiver to infer more of the image from shared knowledge. This is useful for task or perceptual quality, but makes exact source recovery less certain when the payload omits source-specific detail. Digital codebooks make representations countable and interoperable, but introduce quantization and index-error risks. Analog JSCC degrades smoothly, but does not provide bit-exact delivery. Hybrid systems can combine them, but only a complete, sufficiently precise residual plus reliable decoding can close the gap to lossless reconstruction.

Treat every “better than” statement as conditional on matched resolution, dataset split, channel law, SNR, total radio resources, side information and hardware. PSNR, perceptual metrics, task accuracy and exact-pixel success answer different questions and must not be collapsed into one leaderboard.

