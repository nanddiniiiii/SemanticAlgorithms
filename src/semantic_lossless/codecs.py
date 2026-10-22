"""Implemented tensor interfaces. Heavy frameworks are loaded only on demand."""
import contextlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from .artifacts import sha256
from .catalog import get_model
from .paths import project_root, resolve_path
from .sources import source_path, SOURCES


@contextlib.contextmanager
def upstream_import(directory, prefixes):
    old_path = sys.path[:]
    matches = lambda name: any(name == p or name.startswith(p + ".") for p in prefixes)
    saved = {name: value for name, value in sys.modules.items() if matches(name)}
    for name in saved:
        del sys.modules[name]
    sys.path.insert(0, str(directory))
    try:
        yield
    finally:
        sys.path[:] = old_path
        for name in list(sys.modules):
            if matches(name):
                del sys.modules[name]
        sys.modules.update(saved)


def find_jcm_checkpoint(config):
    if config.checkpoint:
        return resolve_path(config.checkpoint)
    root = project_root() / "Models/pretrained_downloads"
    modulation = config.model_options.get("modulation", "16qam")
    candidates = sorted(root.rglob(f"CIFAR_SNR12*_Trans128_{modulation}.pth.tar")) if root.exists() else []
    if len(candidates) != 1:
        raise FileNotFoundError("Set checkpoint to the downloaded native JCM state dict")
    return candidates[0]


def build_codec(config):
    import torch
    from torch import nn
    import torch.nn.functional as F
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = get_model(config.algorithm)
    if config.algorithm == "diffcom":
        from .diffcom import build_diffcom
        return build_diffcom(config)
    if config.algorithm == "hybrid_semantic_complementary":
        from .hybrid import build_hybrid
        return build_hybrid(config)
    if config.algorithm == "shared_semantic_codebook":
        from .shared_codebook import build_shared
        return build_shared(config)
    if config.algorithm == "adjscc":
        from .adjscc import build_adjscc
        return build_adjscc(config)
    if config.algorithm == "sqgan":
        from .sqgan import build_sqgan
        return build_sqgan(config)
    if spec.backend not in {"retained", "jcm"}:
        raise NotImplementedError(f"{config.algorithm}: {spec.note}. No substitute model will be run.")

    class Codec(nn.Module):
        trainable = False
        rate_complete = False
        rate_scope = "unverified"
        metadata = {}
        def loss(self, output, batch):
            raise NotImplementedError("This retained adapter is inference-only")

    class Retained(Codec):
        def __init__(self):
            super().__init__()
            self.name = config.algorithm
            device = torch.device(config.device)
            assets = root / "Models/trained_checkpoints"
            if self.name == "deepjscc":
                from EncDec.Research.model_wrappers import DeepJSCCAdapter
                checkpoint = assets / "DeepJSCC/production/imagenet_10_0.33_200.00_32_19.pth"
                self.adapter = DeepJSCCAdapter(root / "Models/DeepJSCC/checkpoint_source", checkpoint, device, config.snr_db)
                self.core = self.adapter.model
                self.metadata = dict(self.adapter.validation, training_snr_db=200)
                self.rate_scope = "legacy real latent; complex packing not certified"
            elif self.name == "ntscc":
                from EncDec.Research.model_wrappers import NTSCCWindowsAdapter
                checkpoint = assets / "NTSCC/production/ntscc_hyperprior_quality_1_psnr.pth"
                self.adapter = NTSCCWindowsAdapter(root / "Models/NTSCC/official", checkpoint, device, config.snr_db)
                self.core = self.adapter.model
                self.metadata = dict(self.adapter.validation, training_snr_db=10)
                self.rate_scope = "main latent only; excludes side information"
            else:
                from EncDec.Models.SwinJSCC.model_loader import load_stage1_swinjscc
                directory = assets / "SwinJSCC/encdec_sa_ra"
                self.adapter = load_stage1_swinjscc(directory, device=config.device,
                                                  upstream_root=root / "Models/SwinJSCC/local_source")
                self.core = self.adapter.build_training_module()
                self.metadata = {"manifest": json.loads((directory / "stage1_manifest.json").read_text()), "strict_load": True}
                self.metadata.update(comparison_eligible=False,
                    comparison_limitation='Retained AWGN normalizes over dense inactive zeros; nominal SNR is not active-symbol Es/N0 at masked rates. Native-author comparison pending.')
                self.rate_scope = "active complex latent symbols; assumes receiver knows rate/mask protocol"
            self.trainable = False
            if config.checkpoint:
                if self.name != 'swinjscc':
                    raise ValueError("Custom retained checkpoints are currently supported only for SwinJSCC")
                checkpoint = resolve_path(config.checkpoint)
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                from EncDec.Research.model_wrappers import validate_checkpoint
                # Window masks depend on the previous input resolution, not
                # learned weights. Recreate only these registered buffers and
                # retain strict checking for every learned parameter.
                validation = validate_checkpoint(self, state.get("state_dict", state),
                    regenerated_buffer_predicate=lambda key: key.endswith('attn_mask'))
                self.metadata['adapted_checkpoint_validation'] = validation
                self.metadata.update(adapted_checkpoint=str(checkpoint), adapted_sha256=sha256(checkpoint))
            self.eval()
        def optimizer(self):
            if not self.trainable:
                raise NotImplementedError("Retained training is enabled only for SwinJSCC")
            return torch.optim.Adam(self.parameters(), lr=config.learning_rate)
        def loss(self, output, batch):
            if not self.trainable:
                raise NotImplementedError("No verified retained training adapter")
            return F.mse_loss(output["reconstruction"], batch["image"])
        def forward(self, image):
            if len(image) != 1 and self.name != "swinjscc":
                raise ValueError("Retained codec evaluation is batch_size=1")
            h, w = image.shape[-2:]
            ph, pw = (-h) % 128, (-w) % 128
            padded = F.pad(image, (0, pw, 0, ph), mode="replicate") if ph or pw else image
            if self.name == "swinjscc":
                reconstruction, mask = self.core(padded, config.snr_db, config.rate)
                real_count = mask.reshape(len(image), -1).sum(1) if mask is not None else image.new_full((len(image),),padded.shape[-2]*padded.shape[-1]//256*config.rate)
                uses = (real_count / 2).to(torch.long)
            else:
                result = self.adapter.run_once(padded, seed=torch.initial_seed())
                reconstruction = result.reconstruction
                uses = image.new_tensor([result.complex_channel_uses if self.name == "ntscc" else -1], dtype=torch.long)
            return {"reconstruction": reconstruction[..., :h, :w], "complex_uses": uses,
                    "padded_hw": image.new_tensor([[padded.shape[-2], padded.shape[-1]]], dtype=torch.long)}

    class JCM(Codec):
        trainable = True
        rate_scope = "native channel symbols; codec parameters shared, signaling overhead not modeled"
        def __init__(self):
            super().__init__()
            directory = source_path("jcm")
            if not (directory / "source_receipt.json").is_file():
                raise FileNotFoundError("Run prepare --fetch first to retrieve pinned JCM source")
            source = json.loads((directory / "source_receipt.json").read_text())
            if source["commit"] != SOURCES["jcm"][1]:
                raise ValueError("JCM source commit differs")
            for item in source["files"]:
                if sha256(directory / item["path"]) != item["sha256"]:
                    raise ValueError(f"JCM source was modified: {item['path']}")
            modulation = config.model_options.get("modulation", "16qam")
            if modulation not in {"bpsk", "4qam", "16qam", "64qam"}:
                raise ValueError("Unsupported JCM modulation")
            self.native = SimpleNamespace(channel_use=128, mod_method=modulation, mode="test",
                                          snr_train=config.snr_db, snr_test=config.snr_db)
            with upstream_import(directory, ("network", "modules")):
                model_class = importlib.import_module("network").JCM
                self.core = model_class(self.native, torch.device(config.device))
            self.metadata = {"source_commit": source["commit"], "native_modulation": modulation,
                             "native_channel_uses": 128, "classification_classes": 10, "power_scope": "native batch-wide normalization; per replica under DataParallel",
                             "eval_sampling": "author stochastic Gumbel softmax retained"}
            if config.model_options.get("from_scratch", False):
                if config.checkpoint:
                    raise ValueError("from_scratch cannot be combined with checkpoint")
                self.metadata.update(initialization="random_author_initializer", checkpoint=None)
            else:
                checkpoint = find_jcm_checkpoint(config)
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if isinstance(payload, dict) and "state_dict" in payload:
                    payload = payload["state_dict"]
                if payload and all(key.startswith("core.") for key in payload):
                    payload = {key[5:]: value for key, value in payload.items()}
                self.core.load_state_dict(payload, strict=True)
                self.metadata.update(initialization="pretrained_strict_load", checkpoint=str(checkpoint),
                                     checkpoint_sha256=sha256(checkpoint), strict_load=True,
                                     native_training_snr_db=(12 if checkpoint.name.startswith("CIFAR_SNR12") else None))
        def train(self, mode=True):
            super().train(mode)
            if hasattr(self, "native"):
                self.native.mode = "train" if mode else "test"
            return self
        def forward(self, image):
            if image.shape[-2:] != (32, 32):
                raise ValueError("Native JCM accepts only 32x32 input")
            # Native code stores a device attribute used by its constellation.
            # Set it per replica so CUDA:1 does not allocate on CUDA:0 under DP.
            self.core.device = image.device
            _, _, _, logits, native_reconstruction = self.core(image * 2 - 1)
            return {"reconstruction": (native_reconstruction + 1) / 2,
                    "native_reconstruction": native_reconstruction, "logits": logits,
                    "complex_uses": torch.full((len(image),), 128, device=image.device, dtype=torch.long)}
        def loss(self, output, batch):
            labels = batch["label"]
            if (labels < 0).any():
                raise ValueError("Native JCM joint objective requires CIFAR labels")
            weight = config.model_options.get("tradeoff_lambda", 200.0)
            return F.cross_entropy(output["logits"], labels) + weight * F.mse_loss(output["native_reconstruction"], batch["image"] * 2 - 1)
        def optimizer(self):
            probability = list(self.core.prob_convs.parameters())
            ids = {id(p) for p in probability}
            base = [p for p in self.core.parameters() if id(p) not in ids]
            return torch.optim.Adam([{"params": base}, {"params": probability, "lr": config.learning_rate/2}], lr=config.learning_rate)

    return (JCM() if spec.backend == "jcm" else Retained()).to(config.device)
