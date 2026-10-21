"""CPU primitive checks, not validation of the original CUDA architecture."""
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SharedCode" / "reproduction"))
from model_runtime.ntscc import (CheckerboardTransport, FLIFRateIndexCodec,
    NTSCCAdapter, NTSCCSideInformationRequired, released_config, released_loss,
    required_assets)

SOURCE = ROOT / "Models" / "author_releases" / "NTSCC_plus" / "765c6f6c11cf3bbe249a26b624dbaf47a4f0676d"


def test_exact_released_loss_and_config_without_importing_author_main():
    config = released_config(SOURCE, "awgn", 10, "cuda")
    assert config.eta == .2 and config.train_lambda == .72
    assert config.lr == 1e-4 and config.aux_lr == .001 and config.clip_max_norm == 1.
    assert config.multiple_rate[0] == 1 and config.multiple_rate[-1] == 320
    target = torch.zeros(2, 3, 4, 4)
    rec = torch.full_like(target, .1, requires_grad=True)
    output = {"x_hat": rec, "x_hat_ntc": torch.full_like(target, .2),
        "k": torch.tensor(7.), "likelihoods": {"y": torch.full((2, 3, 2, 2), .5), "z": torch.full((2, 2, 1, 1), .25)}}
    objective = released_loss(SOURCE, config.train_lambda)(output, target)
    expected_bpp = (24 + 8) / 32
    assert float(objective["bpp_loss"]) == pytest.approx(expected_bpp)
    assert float(objective["loss"].detach()) == pytest.approx(.72 * 255**2 * (.01 + .04) + expected_bpp)
    objective["loss"].backward()
    assert torch.isfinite(rec.grad).all()


def test_active_checkerboards_form_one_codeword_each_and_count_observed_mask():
    class RecordingChannel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.codewords = []
        def forward(self, values):
            self.codewords.append(values.detach().clone())
            return values
    channel = CheckerboardTransport()
    channel.channel = RecordingChannel()
    features = torch.arange(24.).reshape(4, 2, 1, 3).requires_grad_()
    mask = torch.tensor([[[[1, 0, 1]], [[0, 0, 0]]],
                         [[[1, 1, 0]], [[0, 0, 0]]],
                         [[[0, 1, 0]], [[0, 0, 0]]],
                         [[[0, 1, 1]], [[0, 0, 0]]]])
    restored, total = channel(features, mask)
    # Image 0 joins author rows 0 and 2; image 1 joins rows 1 and 3.
    assert channel.channel.codewords[0].tolist() == [[0., 2., 13., 0.]]
    assert channel.channel.codewords[1].tolist() == [[6., 7., 19., 20.]]
    assert channel.last_uses.tolist() == [2, 2] and int(total) == 4
    assert channel.last_active_real.tolist() == [3, 4]
    assert channel.last_padding_real.tolist() == [1, 0]
    assert torch.equal(restored, features * mask)
    restored.sum().backward()
    assert torch.equal(features.grad, mask)


def test_active_transport_seed_replay_and_rayleigh_one_fade_per_image():
    features = torch.arange(1., 49.).reshape(4, 2, 2, 3).requires_grad_()
    mask = torch.ones_like(features)
    channel = CheckerboardTransport("rayleigh", 10)
    torch.manual_seed(42)
    first, count = channel(features, mask)
    first_gain = channel.channel.last_gain.clone()
    torch.manual_seed(42)
    second, _ = channel(features, mask)
    assert torch.equal(first, second) and torch.equal(first_gain, channel.channel.last_gain)
    assert first_gain.shape == (1, 1)  # last invocation is one joined image
    assert channel.last_uses.tolist() == [12, 12] and int(count) == 24
    first.square().mean().backward()
    assert torch.isfinite(features.grad).all()


@pytest.mark.parametrize("mask", [torch.zeros(2, 2, 1, 2), torch.full((2, 2, 1, 2), .5)])
def test_invalid_or_empty_native_masks_fail_explicitly(mask):
    with pytest.raises(ValueError):
        CheckerboardTransport()(torch.ones_like(mask), mask)


def test_flif_commands_use_unique_new_paths_and_roundtrip_framed_packet(tmp_path, monkeypatch):
    executable = tmp_path / "flif"
    executable.write_bytes(b"test-only-executable-placeholder")
    calls = []
    def fake_flif(arguments, **kwargs):
        source, destination = Path(arguments[-2]), Path(arguments[-1])
        assert not destination.exists()
        assert arguments[1] == "--overwrite" and arguments[2] in {"-e", "-d"}
        assert kwargs["check"] is True and kwargs["timeout"] > 0
        calls.append((source, destination))
        # Subprocess is mocked: this validates invocation/framing, not FLIF.
        shutil.copyfile(source, destination)
    monkeypatch.setattr(subprocess, "run", fake_flif)
    codec = FLIFRateIndexCodec(executable, temp_root=tmp_path)
    indexes = torch.arange(12).reshape(3, 4)
    first = codec.encode(indexes, 3, 4)
    second = codec.encode(indexes, 3, 4)
    assert np.array_equal(codec.decode(first.packet), indexes.numpy())
    assert calls[0][0].parent != calls[2][0].parent
    assert first.packet == second.packet
    assert first.ledger["serialized_bits"] == len(first.packet) * 8
    assert first.ledger["complex_channel_uses"] is None
    assert first.ledger["all_transmitted_streams_counted"] is False
    damaged = first.packet[:-1] + bytes([first.packet[-1] ^ 1])
    with pytest.raises(ValueError, match="CRC32"):
        codec.decode(damaged)
    assert not any(path.name.startswith("ntscc-") for path in tmp_path.iterdir())


def test_flif_failure_has_no_png_or_stale_output_fallback(tmp_path, monkeypatch):
    executable = tmp_path / "flif"
    executable.write_bytes(b"test-only")
    codec = FLIFRateIndexCodec(executable)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="nonempty new output"):
        codec.encode(np.zeros(4, dtype=np.uint8), 2, 2)
    with pytest.raises(ValueError):
        codec.encode(np.array([-1, 1, 2, 3]), 2, 2)
    with pytest.raises(ValueError):
        codec.encode(np.array([0., 1., 2., 3.]), 2, 2)


def test_optimizer_sets_disjoint_auxiliary_gradient_and_complete_state_resume():
    # Deliberately synthetic core; this exercises the real adapter optimizer
    # protocol without substituting anything into an author benchmark run.
    class Entropy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.quantiles = torch.nn.Parameter(torch.tensor([1.]))
        def aux_loss(self):
            return self.quantiles.square().sum()
    adapter = NTSCCAdapter.__new__(NTSCCAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.core = torch.nn.Module()
    adapter.core.ntc = Entropy()
    adapter.core.weight = torch.nn.Parameter(torch.tensor([2.]))
    adapter.config = SimpleNamespace(lr=1e-4, aux_lr=.001)
    optimizers, schedulers = adapter.configure_optimizers()
    main_ids = {id(p) for group in optimizers[0].param_groups for p in group["params"]}
    aux_ids = {id(p) for group in optimizers[1].param_groups for p in group["params"]}
    assert not main_ids & aux_ids and main_ids | aux_ids == {id(p) for p in adapter.core.parameters()}
    before = adapter.core.ntc.quantiles.detach().clone()
    adapter.core.weight.square().sum().backward()
    optimizers[0].step()
    optimizers[1].zero_grad()
    adapter.auxiliary_loss().backward()
    optimizers[1].step()
    schedulers[0].step()
    assert not torch.equal(before, adapter.core.ntc.quantiles)
    restored_optimizers, restored_schedulers = adapter.configure_optimizers()
    for restored, original in zip(restored_optimizers, optimizers):
        restored.load_state_dict(original.state_dict())
        assert len(restored.state) == 1
    restored_schedulers[0].load_state_dict(schedulers[0].state_dict())
    assert restored_schedulers[0].last_epoch == 1


def test_transmit_requires_real_flif_configuration_and_native_cpu_is_blocked():
    adapter = NTSCCAdapter.__new__(NTSCCAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.index_codec = None
    with pytest.raises(NTSCCSideInformationRequired):
        adapter.transmit(torch.ones(1, 3, 256, 256))
    if not torch.cuda.is_available():
        assert any("CUDA" in item for item in required_assets("main", {}, mode="train"))
        with pytest.raises(RuntimeError, match="dependency block"):
            NTSCCAdapter(SOURCE, device="cpu")
    assert any("flif_executable" in item for item in required_assets("main", {}, mode="evaluate"))


@pytest.mark.parametrize("outage", [False, True])
def test_custom_transmit_uses_received_indexes_shared_fade_and_explicit_outage(outage, monkeypatch):
    class RateLayer(torch.nn.Module):
        def forward(self, features):
            return features, torch.ones_like(features)
    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rate_adaption_enc = RateLayer()
            self.matched_transport = CheckerboardTransport("rayleigh", 100)
            self.decoded_indexes = None
        def encode(self, x):
            symbols, _ = self.rate_adaption_enc(torch.ones(2, 2, 1, 2))
            return symbols, torch.tensor([0, 1, 2, 3])
        def decode(self, received, indexes):
            self.decoded_indexes = indexes
            return torch.zeros(1, 3, 32, 32)
    class Codec:
        def encode(self, indexes, rows, columns):
            assert (rows, columns) == (2, 2)
            return SimpleNamespace(packet=b"real synthetic test payload", ledger={"framing_bits": 0})
        def decode(self, packet):
            assert packet == b"real synthetic test payload"
            # Distinct receiver vector verifies no oracle encoder-index reuse.
            return np.array([[3, 2], [1, 0]], dtype=np.uint8)
    adapter = NTSCCAdapter.__new__(NTSCCAdapter)
    torch.nn.Module.__init__(adapter)
    adapter.core, adapter.index_codec, adapter.snr_db = Core(), Codec(), 100
    if outage:
        import digital_transport
        def erase(payload, **kwargs):
            return None, {"complex_channel_uses": 100, "crc_ok": False}
        monkeypatch.setattr(digital_transport, "transmit_bytes", erase)
    reconstruction, uses, timing, ledger = adapter.transmit(torch.zeros(1, 3, 32, 32), seed=71)
    assert ledger["complex_channel_uses"] == uses
    assert uses == sum(row["complex_channel_uses"] for row in ledger["streams"])
    assert ledger["all_transmitted_streams_counted"] is True
    assert ledger["side_information_outage"] == outage
    assert timing["end_to_end_seconds"] == pytest.approx(sum(timing[k] for k in ("encoder_seconds", "channel_seconds", "decoder_seconds")))
    if outage:
        assert adapter.core.decoded_indexes is None and torch.all(reconstruction == .5)
    else:
        assert adapter.core.decoded_indexes.tolist() == [3, 2, 1, 0]
        digital = ledger["streams"][1]
        actual_gain = adapter.transport.channel.last_gain.item()
        assert digital["gain_shared_with_primary"] is True
        assert complex(digital["gain_real"], digital["gain_imag"]) == actual_gain
