"""Synthetic CPU checks; no author architecture or GPU result is certified."""
import random
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_runtime.channels import MatchedChannel
from model_runtime.data import epoch_order
from model_runtime.state import EarlyStop, assert_epoch_coverage, atomic_save, capture_rng, restore_rng


def test_awgn_complex_noise_calibrates_to_declared_snr_and_counts_each_image():
    torch.manual_seed(17)
    transmitted = torch.ones((4, 100000), dtype=torch.complex64)
    channel = MatchedChannel("awgn", 10)
    received = channel(transmitted)
    noise = received - transmitted
    measured_snr = 10 * torch.log10(transmitted.abs().square().mean() / noise.abs().square().mean())
    assert float(measured_snr) == pytest.approx(10, abs=0.1)
    assert abs(float(noise.real.mean())) < 0.002
    assert abs(float(noise.imag.mean())) < 0.002
    assert channel.last_complex_uses == 100000


def test_channel_restores_per_image_scale_and_backpropagates_finite_gradients():
    torch.manual_seed(23)
    source = torch.ones(2, 2, 20000, requires_grad=True)
    features = source * torch.tensor([1., 100.]).view(2, 1, 1)
    channel = MatchedChannel("awgn", 10)
    received = channel(features)
    assert received.shape == features.shape
    normalized_error = (received - features) / torch.tensor([1., 100.]).view(2, 1, 1)
    # Real coordinates each have original scale^2/SNR variance after restoring
    # the paired-complex power, so normalized error variance is 0.1.
    assert normalized_error.square().mean(dim=(1, 2)).tolist() == pytest.approx([.1, .1], abs=.004)
    received.square().mean().backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()
    assert channel.last_complex_uses == 20000


def test_rayleigh_is_finite_differentiable_and_seed_replayable():
    features = torch.linspace(.2, 1., 128).reshape(2, 64).requires_grad_()
    channel = MatchedChannel("rayleigh", 10)
    torch.manual_seed(19)
    first = channel(features)
    torch.manual_seed(19)
    second = channel(features)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    first.square().mean().backward()
    assert torch.isfinite(features.grad).all()
    with pytest.raises(ValueError):
        channel(torch.ones(2, 3))


def test_epoch_order_resume_is_deterministic_and_rejects_duplicate_coverage():
    order = epoch_order(800, 42, 17)
    assert order == epoch_order(800, 42, 17)
    assert order != epoch_order(800, 42, 18)
    assert_epoch_coverage(order)
    with pytest.raises(ValueError):
        assert_epoch_coverage(order[:-1] + [order[0]])
    with pytest.raises(ValueError):
        assert_epoch_coverage(order[:-1])


def test_early_stop_minimum_patience_cap_and_state_resume():
    policy = EarlyStop(min_epochs=5, max_epochs=10, patience=2, relative_delta=.01)
    assert policy.update(1, 1.) == (True, False)
    assert policy.update(2, .995) == (False, False)
    policy = EarlyStop.from_state(policy.state_dict())
    assert policy.update(3, .995) == (False, False)
    assert policy.update(4, .995) == (False, False)
    assert policy.update(5, .995) == (False, True)
    assert policy.best_epoch == 1
    improving = EarlyStop(min_epochs=5, max_epochs=6)
    for epoch in range(1, 7):
        assert improving.update(epoch, 1 / epoch) == (True, epoch == 6)
    with pytest.raises(ValueError):
        improving.update(7, float("nan"))


def test_rng_restore_reproduces_python_numpy_and_torch_cpu():
    state = capture_rng()
    expected = (random.random(), np.random.random(), torch.rand(4))
    restore_rng(state)
    observed = (random.random(), np.random.random(), torch.rand(4))
    assert expected[:2] == observed[:2]
    assert torch.equal(expected[2], observed[2])


def test_atomic_save_keeps_previous_checkpoint_if_serialization_fails(tmp_path, monkeypatch):
    path = tmp_path / "latest.pt"
    atomic_save({"epoch": 1, "rng": capture_rng()}, path)
    previous = path.read_bytes()
    def fail_after_partial_write(payload, stream):
        stream.write(b"incomplete")
        raise OSError("injected serialization failure")
    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    with pytest.raises(OSError, match="injected"):
        atomic_save({"epoch": 2}, path)
    assert path.read_bytes() == previous
    assert torch.load(path, weights_only=False)["epoch"] == 1
