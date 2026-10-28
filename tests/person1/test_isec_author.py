"""Focused CPU checks of the ISEC author equations and adapter contract."""
from pathlib import Path

import pytest
import torch
from torch import nn

from src.person1.isec import _refine, build


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x):
        return self.scale * x


@pytest.mark.parametrize("test_snr", [7, 13])
def test_refinement_matches_author_eval_update_with_snr_mismatch(test_snr):
    encoder, decoder, denoiser = Scale(1.3), Scale(0.8), Scale(0.15)
    received = torch.tensor([[[[0.4, -0.2]]]])
    sigma = 10 ** (-test_snr / 20)
    base_var = 10 ** (-10 / 10)
    ratio = sigma**2 / base_var
    native = received.detach().clone().requires_grad_()
    for _ in range(3):
        obj = (encoder(decoder(native)) - received).square().sum() / (2 * sigma**2)
        obj.backward()
        with torch.no_grad():
            # Direct transcription of pinned author eval.py lines 209-220.
            grad = -native.grad + 1.0 * max(0.1, ratio**2) * denoiser(native)
            delta = 1.5 if ratio > 1 else 1.0
            step = 0.002 / max(0.1, ratio**delta)
            native = (native + step * grad).detach().requires_grad_()
    actual = _refine(received, encoder, decoder, denoiser,
                     test_snr_db=test_snr, training_snr_db=10, iterations=3,
                     learning_rate=0.002, alpha=1.0, delta=1.5)
    torch.testing.assert_close(actual, native.detach(), rtol=1e-6, atol=1e-7)


def _fake_source(root: Path):
    package = root / "models"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "autoencoders.py").write_text(
        "import torch\nfrom torch import nn\n"
        "class Encoder(nn.Module):\n"
        " def __init__(self, width, hidden, conv, residual):\n"
        "  super().__init__(); self.layer=nn.Conv2d(3,width,1)\n"
        " def forward(self,x): return self.layer(x)\n"
        "class Decoder(nn.Module):\n"
        " def __init__(self, width, hidden, conv, residual):\n"
        "  super().__init__(); self.layer=nn.Conv2d(width,3,1)\n"
        " def forward(self,x): return self.layer(x)\n"
    )
    (package / "bfcnn.py").write_text(
        "from torch import nn\n"
        "class BF_CNN(nn.Module):\n"
        " def __init__(self,padding,kernels,kernel_size,layers,width):\n"
        "  super().__init__(); self.layer=nn.Conv2d(width,width,1,bias=False)\n"
        " def forward(self,x): return self.layer(x)\n"
    )


def test_stage_losses_optimizers_and_rate(tmp_path):
    _fake_source(tmp_path)
    options = {"latent_channels": 6, "snr_db": 10, "iterations": 1, "epochs": 150}
    model = build(options, "cpu", tmp_path)
    image = torch.rand(2, 3, 4, 4)
    model.train()
    jscc = model(image)
    assert jscc["native_loss"].shape == (2,)
    assert jscc["reconstruction"].shape == image.shape
    torch.testing.assert_close(jscc["complex_uses"], torch.full((2,), 48.0))
    assert model.optimizer().param_groups[0]["betas"] == (0.0, 0.9)
    assert model.scheduler(model.optimizer()).step_size == 30
    model.stage = "denoiser"
    denoiser = model(image)
    assert denoiser["native_loss"].shape == (2,)
    assert model.loss(denoiser, {"image": image}).ndim == 0
    assert model.optimizer().param_groups[0]["betas"] == (0.9, 0.999)
    assert model.scheduler(model.optimizer()).step_size == 75
    model.eval()
    with torch.no_grad():
        refined = model(image)
    assert refined["reconstruction"].shape == image.shape


def test_missing_or_wrong_source_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        build({}, "cpu", tmp_path)
    _fake_source(tmp_path)
    (tmp_path / "source_receipt.json").write_text(
        '{"repository":"another/repo","commit":"deadbeef"}'
    )
    with pytest.raises(ValueError, match="approved author revision"):
        build({}, "cpu", tmp_path)
