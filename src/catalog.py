"""Capabilities describe implemented code, not intended future availability."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    name: str
    family: str
    dataset: str
    backend: str
    trainable: bool = False
    note: str = ""


MODELS = {
    m.name: m for m in [
        Model("deepjscc", "continuous/CNN", "image_folder", "retained", note="Legacy channel; complex-use mapping not certified"),
        Model("swinjscc", "continuous/transformer", "image_folder", "retained", False, "Retained port audit-only for power mismatch; separate native fixed/adaptive Kodak24 evaluations and T4 profiles completed; adaptive quality anomaly unresolved"),
        Model("ntscc", "entropy-adaptive", "image_folder", "retained", note="Original NTSCC; main-latent rate excludes side information"),
        Model("ntscc_plus", "entropy-adaptive/contextual", "image_folder", "pending", note="Improved NTSCC released+ source acquired; legacy NATTEN compatibility gate; ++ online inference unreleased"),
        Model("ofdm_guided_jscc", "continuous/OFDM", "image_folder", "pending", note="Pinned author source acquired; native multipath/CE/EQ pilot pending"),
        Model("isec", "continuous/iterative-decoding", "image_folder", "pending", note="Pinned author JSCC/denoiser source acquired; native weight loading and inference pending"),
        Model("ssc", "digital/residual-VQ/index-modulation", "image_folder", "pending", note="Pinned author source acquired; no verified checkpoint; native FA-IM resources differ from SISO AWGN"),
        Model("gbsed", "graph/task-oriented", "scene_graph", "pending", note="2026 author source; scene-graph reconstruction and risk task, not RGB reconstruction; native dataset/roadscene2vec/Sionna1.1 gates"),
        Model("hsc_projection", "hybrid/semantic-complementary", "image_folder", "pending", note="Paper-derived projection implementation; no verified author release; RGB/float32 serialization is an explicit extension; native neural-base reproduction pending"),
        Model("jcm", "finite-constellation/VAE", "cifar10", "jcm", True, "Native 32x32; reconstruction plus classification objective"),
        Model("adjscc", "continuous/attention", "image_folder", "adjscc", True, "DiffCom PyTorch architecture; per-image AWGN adaptation; original TF lineage stays separate"),
        Model("shared_semantic_codebook", "shared-codebook", "stl10", "shared", True, "SAC without proposed information-theoretic loss term; lossy reconstruction; construct classifier/codebook first"),
        Model("sqgan", "codebook/generative", "cityscapes", "sqgan", note="Ten-part strict checkpoint inference; native trainer pending"),
        Model("vq_deepsc", "digital/multiscale-codebook/CNN-GAN", "image_folder", "unavailable", note="VQ-DeepSC: 2022 preprint, 2023 journal; already in survey; author code/checkpoints not verified; no local implementation"),
        Model("vq_deepisc", "digital/multiscale-codebook/Swin-OFDM", "image_folder", "unavailable", note="VQ-DeepISC: 2025; already in survey; author code/checkpoints not verified; no local implementation"),
        Model("sc_lmkb", "knowledge-base/LLM-assisted", "cross_modal_retrieval", "unavailable", note="SC-LMKB: arXiv2604.05504, 2026; source/channel knowledge generation and hallucination mitigation; cross-modal retrieval reference, not a pixel-reconstruction baseline; code/weights unverified"),
        Model("mdjcm", "digital/rate-adaptive", "image_folder", "pending", note="Checkpoint and native runtime pilot required"),
        Model("amp_sc", "digital/bitstream-mask-attack", "image_folder", "pending", note="Author RGB256 training code verified; pretrained feature-codec prerequisite unavailable; bit-flip native channel; not integrated or trained"),
        Model("mambajscc", "continuous/state-space", "image_folder", "pending", note="Author AWGN/Rayleigh pairs downloaded; native CUDA extension build failed; inference unverified"),
        Model("diffcom", "generative/diffusion", "image_folder", "diffcom", note="Complete T4 posterior pilot passed; AWGN adapter differences remain explicit; no diffusion training"),
        Model("diffjscc", "generative/diffusion", "image_folder", "pending", note="Author Hugging Face checkpoint; inference integration pending"),
        Model("deepjscc_q", "finite-constellation", "cifar10", "pending", note="Candidate mixes paper variants; quantizer evaluation/regularizer defects verified; no faithful trained model yet"),
        Model("hybrid_semantic_complementary", "hybrid", "image_folder", "hybrid", note="Our frozen ADJSCC plus fixed DCT complement reference; not an author reproduction"),
        Model("dit_jscc", "generative/diffusion-transformer", "image_folder", "unavailable", note="JSCC weights reserved for future release"),
        Model("hda_deepsc", "hybrid", "image_folder", "unavailable", note="Verified author executable release not identified"),
        Model("masked_vqvae", "codebook/task-oriented", "cifar10", "unavailable", note="Released classification demo disables VQ/AWGN in inspected path"),
    ]
}


def get_model(name: str) -> Model:
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(f"Unknown algorithm {name!r}; choose {', '.join(MODELS)}") from None
