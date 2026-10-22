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
        Model("swinjscc", "continuous/transformer", "image_folder", "retained", True, "Retained adaptive SA+RA AWGN; fine-tuning enabled"),
        Model("ntscc", "entropy-adaptive", "image_folder", "retained", note="Original NTSCC; main-latent rate excludes side information"),
        Model("jcm", "finite-constellation/VAE", "cifar10", "jcm", True, "Native 32x32; reconstruction plus classification objective"),
        Model("adjscc", "continuous/attention", "image_folder", "adjscc", True, "DiffCom PyTorch architecture; per-image AWGN adaptation; original TF lineage stays separate"),
        Model("shared_semantic_codebook", "shared-codebook", "stl10", "shared", True, "Native SAC w/o information loss; construct classifier/codebook before codec training"),
        Model("sqgan", "codebook/generative", "cityscapes", "sqgan", note="Ten-part strict checkpoint inference; native trainer pending"),
        Model("mdjcm", "digital/rate-adaptive", "image_folder", "pending", note="Checkpoint and native runtime pilot required"),
        Model("mambajscc", "continuous/state-space", "image_folder", "pending", note="CUDA extensions and native checkpoint required"),
        Model("diffcom", "generative/diffusion", "image_folder", "diffcom", note="Complete T4 posterior pilot passed; AWGN adapter differences remain explicit; no diffusion training"),
        Model("diffjscc", "generative/diffusion", "image_folder", "pending", note="Author Hugging Face checkpoint; inference integration pending"),
        Model("deepjscc_q", "finite-constellation", "cifar10", "pending", note="Public trainer needs paper-variant verification"),
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
