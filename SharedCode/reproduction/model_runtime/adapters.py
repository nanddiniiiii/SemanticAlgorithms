"""Author architectures and losses, with explicitly labelled transport adapters.

One model per interpreter: upstream packages use colliding absolute module names.
No model source is rewritten, patched, or replaced with a smaller architecture.
"""
import ast
from datetime import datetime
import importlib
import math
from types import SimpleNamespace
import torch
from torch import nn
from .channels import MatchedChannel


def extract_definition(source, filename, name, scope):
    tree = ast.parse((source / filename).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source/filename), "exec"), scope)
    return scope[name]


class TorchAdapter(nn.Module):
    alignment = 4
    crop = 256
    loss_name = "author MSE"

    def __init__(self, model, source, channel="awgn", snr_db=10, device="cuda",rate_cbr=1/6):
        super().__init__()
        self.name = model
        self.transport = MatchedChannel(channel, snr_db)
        self.snr_db = snr_db
        self.model_args = None
        self.target_cbr=rate_cbr
        self.bottleneck_channels=round(rate_cbr*(48 if model=="DeepJSCC" else 1536))
        denominator=48 if model=="DeepJSCC" else 1536
        if self.bottleneck_channels<=0 or not math.isclose(self.bottleneck_channels/denominator,rate_cbr,rel_tol=1e-8):
            raise ValueError("Requested CBR is not realizable by an integer author bottleneck")
        if model == "DeepJSCC":
            # c=8 gives k/n=1/6: 2c real maps at H/4,W/4.
            self.core = importlib.import_module("model").DeepJSCC(c=self.bottleneck_channels, snr=None)
            self.crop = 128
        elif model == "SwinJSCC":
            self.alignment = 128
            args = SimpleNamespace(trainset="DIV2K", testset="kodak", model_size="base",
                model="SwinJSCC_w/o_SAandRA", C=str(self.bottleneck_channels), multiple_snr=str(snr_db),
                channel_type=channel, distortion_metric="MSE")
            cfg = extract_definition(source, "main.py", "config", {
                "torch": torch, "nn": nn, "datetime": datetime, "args": args})
            cfg.device, cfg.CUDA, cfg.pass_channel = torch.device(device), device.startswith("cuda"), True
            self.core = importlib.import_module("net.network").SwinJSCC(args, cfg)
            # Explicit extension at the channel module boundary. Source is unchanged.
            self.core.channel = self.transport
            self.model_args = args
            self.loss_name = "author Distortion(MSE)"
            if device == "cpu":
                self.core.H = self.core.W = 256  # initialized resolution; updater is CUDA-only upstream
        elif model == "DeepJSCC_Q":
            self.alignment = 16
            from modules.feature_encoder_AF import FeatureEncoderAF, FeatureDecoderAF
            from modules.modem import Modem
            self.encoder = FeatureEncoderAF(3, 256, self.bottleneck_channels, reduced=False)
            self.decoder = FeatureDecoderAF(self.bottleneck_channels, 256, 3, (self.bottleneck_channels, 16, 16), reduced=False)
            self.modem = Modem("lm", mod_order=64, return_likelihoods=False, commitment=0.,
                anneal="linear", sigma_start=5., sigma_max=200., sigma_period=1000, sigma_scale=5.)
            from pytorch_msssim import ssim, ms_ssim
            self.author_loss = extract_definition(source, "utils.py", "calc_loss", {
                "torch": torch, "F": torch.nn.functional, "ssim": ssim, "ms_ssim": ms_ssim})
            self.loss_name = "author calc_loss(ssim), released learned 64-symbol constellation"
        else:
            raise ValueError("Use the required-assets gate for this model")

    def encode(self, x):
        if self.name == "DeepJSCC":
            features = self.core.encoder(x)
            return features.unsqueeze(0) if features.ndim == 3 else features
        if self.name == "SwinJSCC":
            h, w = x.shape[-2:]
            if (self.core.H, self.core.W) != (h, w):
                self.core.encoder.update_resolution(h, w)
                self.core.decoder.update_resolution(h//16, w//16)
                self.core.H, self.core.W = h, w
            return self.core.encoder(x, self.snr_db, self.bottleneck_channels, self.model_args.model)
        snr = x.new_full((1, 1), self.snr_db)
        code = self.encoder(x, snr)
        symbols, _ = self.modem.modulate(code)
        return symbols

    def decode(self, received, original_shape):
        if self.name == "DeepJSCC":
            return self.core.decoder(received)
        if self.name == "SwinJSCC":
            return self.core.decoder(received, self.snr_db, self.model_args.model)
        self.decoder.feat_dims = (self.bottleneck_channels, original_shape[-2]//16, original_shape[-1]//16)
        demodulated = self.modem.demodulate(received)
        return torch.sigmoid(self.decoder(demodulated, received.real.new_full((1, 1), self.snr_db)))

    def forward(self, x):
        symbols = self.encode(x)
        received = self.transport(symbols, self.snr_db)
        rec = self.decode(received, x.shape)
        if rec.shape != x.shape:
            raise ValueError("Author reconstruction geometry differs after explicit padding")
        if self.name == "DeepJSCC":
            # train.py explicitly denormalizes BOTH tensors before author MSE.
            loss = self.core.loss(rec*255., x*255.)
        elif self.name == "SwinJSCC":
            loss = self.core.distortion_loss.forward(x, rec.clamp(0, 1)).mean()
        else:
            loss, _ = self.author_loss(rec, x, "ssim")
        # Gather tensors only. Per-replica scalar means are weighted by batch size
        # after gather; Python floats in author return tuples never reach DP gather.
        n = x.shape[0]
        return {"reconstruction": rec, "loss_sum": (loss*n).reshape(1),
                "batch_count": x.new_tensor([n]),
                "complex_uses": torch.full((n,), self.transport.last_complex_uses, device=x.device, dtype=torch.long)}


def reduce_loss(result):
    return result["loss_sum"].sum() / result["batch_count"].sum()


def required_assets(model, stage, contract, mode="preflight"):
    """Fail closed where a runnable matched path has not been implemented/audited."""
    if model == "NTSCC_plus":
        from .ntscc import required_assets as ntscc_assets
        return ntscc_assets(stage,contract,mode)
    if model == "ADJSCC":
        from .adjscc_tf import dependency_gate
        try:
            dependency_gate()
        except (ImportError,RuntimeError) as exc:
            return ["Legacy isolated ADJSCC runtime required: "+str(exc)]
        return []
    if model == "SQGAN":
        assets = contract.get("model_assets", {}).get("SQGAN", {})
        missing = ["Implement/test original Lightning manual GAN update schedule with every generator/discriminator optimizer for semantic/image/finetune stages",
                   "Implement and validate serialized semantic and image index transport, framing/FEC/modulation/error behavior; source is a compression model without a noisy physical channel",
                   "DIV2K/Kodak lack required semantic labels: provide documented frozen pseudo-labeler adaptation or externally labelled train/validation/evaluation manifests"]
        for key in ("semantic_labels", "perceptual_weights", "semantic_checkpoint", "image_checkpoint"):
            if stage != "semantic" or key not in {"semantic_checkpoint", "image_checkpoint"}:
                if not assets.get(key):
                    missing.append("Required asset model_assets.SQGAN." + key)
        return missing
    return []
