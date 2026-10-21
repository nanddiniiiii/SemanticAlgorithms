"""CPU regressions for orchestration and measured evaluation boundaries.

Synthetic fixtures exercise contracts; they are never model quality evidence.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import launch_upstream
from matched_evaluation import evaluate_native_image, validate_comparison
from native_geometry import measured_cbr, pad_image, unpad_image
from profiling import training_flops_receipt


def eligible_manifest(model="synthetic_A"):
    return dict(model=model, upstream_authorship_verified=True,
                source_integrity_verified=True, model_runtime_patches=[],
                training_complete=True, evaluation_complete=True,
                review_receipt_verified=True, images_per_epoch=800,
                epochs_per_required_stage=150, dataset_manifest_sha256="fixture",
                evaluation_protocol="native_full_image_v1", training_geometry=[256, 256],
                training_image_exposures=120000, channel_definition="fixture AWGN",
                receiver_csi="none", snr_grid=[10], trials=10,
                metric_implementation="shared", flops_convention="FMA=2",
                rate_budget="matched measured", comparison_track="matched")


def test_native_comparison_can_pass_without_silent_tiling():
    first = eligible_manifest()
    second = eligible_manifest("synthetic_B")
    assert validate_comparison([first, second])["eligible_models"] == ["synthetic_A", "synthetic_B"]


@pytest.mark.parametrize("field,value", [("evaluation_protocol", "matched_tiled_v1"),
                                         ("channel_definition", "different fading"),
                                         ("trials", 1), ("training_complete", False)])
def test_comparison_rejects_mixed_or_incomplete_evidence(field, value):
    first = eligible_manifest()
    second = copy.deepcopy(first)
    second.update(model="synthetic_B", **{field: value})
    with pytest.raises(ValueError):
        validate_comparison([first, second])


@pytest.mark.parametrize("relative,valid", [("data", True), ("../escaped", False),
                                           ("nested/../../escaped", False)])
def test_launch_preparation_is_repeatable_with_external_dataset_symlink(tmp_path, monkeypatch, relative, valid):
    source = tmp_path / "locked"
    source.mkdir()
    entry = source / "adjscc_imagenet.py"
    entry.write_text("# synthetic source fixture\n", encoding="utf-8")
    manifests = tmp_path / "Models/author_releases/ADJSCC"
    manifests.mkdir(parents=True)
    manifest = dict(source_directory="locked", files_sha256={entry.name: hashlib.sha256(entry.read_bytes()).hexdigest()},
                    repository="fixture", commit="fixture")
    (manifests / "fixture.manifest.json").write_text(json.dumps(manifest))
    dataset = tmp_path / "external_data"
    dataset.mkdir()
    probe = tmp_path / "symlink_capability"
    try:
        probe.symlink_to(dataset, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Host cannot create directory symlinks: {exc}")
    probe.unlink()
    arguments = tmp_path / "arguments.json"
    arguments.write_text("[]")
    links = tmp_path / "links.json"
    links.write_text(json.dumps({relative: str(dataset)}))
    destination = tmp_path / "execution"
    monkeypatch.setattr(launch_upstream, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["launch_upstream", "--model", "ADJSCC", "--entrypoint", entry.name,
                                    "--execution-dir", str(destination), "--arguments-json", str(arguments),
                                    "--dataset-links-json", str(links)])
    if not valid:
        with pytest.raises(ValueError):
            launch_upstream.main()
        assert not (tmp_path / "escaped").exists()
        return
    launch_upstream.main()
    launch_upstream.main()
    assert (destination / "data").resolve() == dataset.resolve()
    assert entry.read_text() == "# synthetic source fixture\n"
    assert json.loads((tmp_path / "execution.launch.json").read_text())["execution_started"] is False


def early_stopped_manifest(model="synthetic_B", epochs=70):
    row = eligible_manifest(model)
    row.update(epochs_per_required_stage=epochs, training_image_exposures=epochs * 800,
               early_stopping_receipt=dict(stopped_on_plateau=True, minimum_epochs=50,
                   maximum_epochs=150, patience=20, relative_min_delta=.001,
                   epochs_completed=epochs))
    return row


def test_plateau_receipt_allows_different_measured_exposures_and_disclosed_geometry():
    row = early_stopped_manifest()
    row["training_geometry"] = [128, 128]
    result = validate_comparison([eligible_manifest(), row])
    assert result["eligible_models"] == ["synthetic_A", "synthetic_B"]


@pytest.mark.parametrize("field,value", [("stopped_on_plateau", False), ("minimum_epochs", 10),
                                         ("patience", 2), ("relative_min_delta", .1),
                                         ("epochs_completed", 69)])
def test_plateau_receipt_rejects_wrong_policy_or_epoch(field, value):
    row = early_stopped_manifest()
    row["early_stopping_receipt"][field] = value
    with pytest.raises(ValueError):
        validate_comparison([eligible_manifest(), row])


@pytest.mark.parametrize("epochs", [49, 151, 150.5])
def test_comparison_rejects_epochs_outside_authorized_policy(epochs):
    with pytest.raises(ValueError):
        validate_comparison([eligible_manifest(), early_stopped_manifest(epochs=epochs)])


def test_empty_unsupported_operator_list_does_not_certify_complete_flops():
    row = training_flops_receipt(forward_flops=10, backward_flops=20, optimizer_flops=3,
        unsupported_ops=[], batch_size=1, precision="fp32", profiler="synthetic fixture",
        actual_optimizer_step=True)
    assert row["coverage"] == "partial_supported_operators"


@pytest.mark.parametrize("shape", [(512, 768), (768, 512), (257, 255)])
def test_padding_preserves_every_original_pixel_and_original_denominator(shape):
    image = np.arange(shape[0] * shape[1] * 3, dtype=np.uint32).reshape(*shape, 3)
    padded, info = pad_image(image, multiple=128)
    assert padded.shape[0] % 128 == padded.shape[1] % 128 == 0
    assert np.array_equal(unpad_image(padded, info), image)
    uses = padded.shape[0] * padded.shape[1]
    assert measured_cbr(uses, info) == uses / (3 * shape[0] * shape[1])
    assert padded[-1, -1].tolist() == image[-1, -1].tolist()


@pytest.mark.parametrize("bad", [True, -1, 1.5, float("nan")])
def test_native_evaluator_rejects_invalid_accounting_before_metrics(bad):
    image = np.zeros((8, 8, 3), dtype=np.float32)
    def transmit(padded, context):
        return dict(reconstruction=padded, complex_channel_uses=bad,
                    all_transmitted_streams_counted=True)
    with pytest.raises(ValueError):
        evaluate_native_image(image, transmit, image_id="fixture", channel="awgn",
                              snr_db=10, trial=0, seed=1)


def test_native_shared_metrics_measure_known_nonzero_rgb_error():
    pytest.importorskip("torch")
    pytest.importorskip("pytorch_msssim")
    image = np.zeros((192, 192, 3), dtype=np.float32)
    def transmit(padded, context):
        return dict(reconstruction=np.full_like(padded, 0.25), complex_channel_uses=100,
                    all_transmitted_streams_counted=True)
    row, restored = evaluate_native_image(image, transmit, image_id="fixture", channel="awgn",
                                          snr_db=10, trial=0, seed=1)
    assert row["psnr_db"] == pytest.approx(-10 * np.log10(0.25 ** 2))
    assert 0 <= row["ssim"] < 1 and 0 <= row["msssim"] < 1
    assert np.all(restored == 0.25)
