"""Adapt the pinned ISEC author modules to the common image runner.

This is an OpenImages-architecture adapter.  The common DIV2K/Flickr training
split and AWGN10 operating point are experiment adaptations, not paper results.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import torch
from torch import nn

from src.codecs import upstream_import


AUTHOR_COMMIT = "43d1a27beaeb89b7057f02bd2fa2dd05c46093e1"


def _per_image_mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).square().flatten(1).mean(1)


def _refine(
    received: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    denoiser: nn.Module,
    *,
    test_snr_db: float,
    training_snr_db: float,
    iterations: int,
    learning_rate: float,
    alpha: float,
    delta: float,
) -> torch.Tensor:
    """Author eval.py's latent update, including its *positive* denoiser term."""
    if iterations < 0 or learning_rate <= 0:
        raise ValueError("iterations must be nonnegative and refinement learning_rate positive")
    sigma = 10 ** (-test_snr_db / 20)
    base_var = 10 ** (-training_snr_db / 10)
    ratio = sigma**2 / base_var
    effective_delta = delta if ratio > 1 else 1.0
    prior_scale = alpha * max(0.1, ratio**2)
    step_size = learning_rate / max(0.1, ratio**effective_delta)
    zt = received.detach()
    for _ in range(iterations):
        with torch.enable_grad():
            zt = zt.detach().requires_grad_(True)
            # The author's criterion uses reduction='sum', including batch.
            nll = (encoder(decoder(zt)) - received.detach()).square().sum() / (2 * sigma**2)
            nll_grad, = torch.autograd.grad(nll, zt)
        with torch.no_grad():
            prior = denoiser(zt)
            zt = zt + step_size * (-nll_grad + prior_scale * prior)
    return zt.detach()


def build(options: dict, device: str, source_dir: str | Path) -> nn.Module:
    """Build native OpenImages encoder, decoder and BF-CNN from supplied source."""
    directory = Path(source_dir)
    if not (directory / "models" / "autoencoders.py").is_file() or not (
        directory / "models" / "bfcnn.py"
    ).is_file():
        raise FileNotFoundError("ISEC author source modules are missing")
    receipt_path = directory / "source_receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("repository") != "changwoolee/isec-deep-jscc" or receipt.get("commit") != AUTHOR_COMMIT:
            raise ValueError("ISEC source receipt is not the approved author revision")
    width = int(options.get("latent_channels", 6))
    if width <= 0:
        raise ValueError("latent_channels must be positive")
    with upstream_import(directory, ("models",)):
        ae = importlib.import_module("models.autoencoders")
        bf = importlib.import_module("models.bfcnn")
        encoder = ae.Encoder(width, 128, 2, 3)
        decoder = ae.Decoder(width, 128, 2, 3)
        denoiser = bf.BF_CNN(1, 64, 3, 20, width)

    class ISEC(nn.Module):
        trainable = True
        needs_grad_evaluation = True
        stage = "jscc"

        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.denoiser = denoiser
            self.metadata = {
                "source_commit": AUTHOR_COMMIT,
                "architecture": "native OpenImages Encoder/Decoder and BF_CNN",
                "latent_channels": width,
                "input_range": "[0,1] mapped to native [-1,1]",
                "training_stages": ["jscc", "denoiser"],
                "rate_scope": "real latent elements / 2; framing excluded",
                "channel": "native real additive Gaussian noise",
                "paper_exact": False,
                "refinement_iterations": int(options.get("iterations", 50)),
            }

        def forward(self, image: torch.Tensor) -> dict:
            if image.ndim != 4 or image.shape[1] != 3:
                raise ValueError("ISEC expects BCHW RGB images")
            native_image = image * 2 - 1
            train_snr = float(options.get("training_snr_db", options.get("snr_db", 10)))
            test_snr = float(options.get("test_snr_db", options.get("snr_db", 10)))
            snr = train_snr if self.training else test_snr
            sigma = 10 ** (-snr / 20)
            if self.training and self.stage == "denoiser":
                self.encoder.eval()
                self.decoder.eval()
                with torch.no_grad():
                    code = self.encoder(native_image)
                    noise = torch.randn_like(code) * sigma
                predicted_noise = self.denoiser(code + noise)
                native_loss = _per_image_mse(predicted_noise, noise)
                with torch.no_grad():
                    recon = self.decoder(code + noise)
            else:
                code = self.encoder(native_image)
                received = code + torch.randn_like(code) * sigma
                if not self.training and self.stage == "denoiser":
                    received = _refine(
                        received, self.encoder, self.decoder, self.denoiser,
                        test_snr_db=test_snr,
                        training_snr_db=train_snr,
                        iterations=int(options.get("iterations", 50)),
                        learning_rate=float(options.get("refinement_lr", 0.002)),
                        alpha=float(options.get("alpha", 1.0)),
                        delta=float(options.get("delta", 1.0)),
                    )
                recon = self.decoder(received)
                native_loss = _per_image_mse(recon, native_image)
            return {
                "reconstruction": (recon + 1) / 2,
                "native_loss": native_loss,
                "complex_uses": image.new_full((len(image),), code[0].numel() / 2),
            }

        def loss(self, output: dict, batch: dict) -> torch.Tensor:
            return output["native_loss"].mean()

        def optimizer(self) -> torch.optim.Optimizer:
            if self.stage == "jscc":
                parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
                return torch.optim.Adam(
                    parameters,
                    lr=float(options.get("learning_rate", 1e-3)),
                    betas=(0.0, 0.9),
                    weight_decay=float(options.get("weight_decay", 0.0)),
                )
            if self.stage == "denoiser":
                return torch.optim.Adam(
                    self.denoiser.parameters(),
                    lr=float(options.get("denoiser_learning_rate", 2e-4)),
                    betas=(0.9, 0.999),
                    weight_decay=float(options.get("weight_decay", 0.0)),
                )
            raise ValueError("ISEC stage must be jscc or denoiser")

        def scheduler(self, optimizer: torch.optim.Optimizer):
            interval = 30 if self.stage == "jscc" else max(1, int(options.get("epochs", 150)) // 2)
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=interval, gamma=0.1)

    return ISEC().to(device)
