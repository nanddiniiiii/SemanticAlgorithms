"""Reject malformed complete-grid fixtures; synthetic receipts are not evidence."""
import hashlib
import itertools
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from delivery.consolidate import consolidate


def measurement_fixture(tmp_path):
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"evaluation": {"snr_db": [0, 5, 10, 15, 20],
                                                   "independent_channel_trials_per_image": 10}}))
    source = tmp_path / "measurement"
    source.mkdir()
    identity = {"contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
                "source_commit": "fixture"}
    receipt = dict(identity=identity, model="synthetic", channel="awgn", status="complete",
                   evaluation_complete=True, complete_training_epochs=150, validation_pass=True,
                   checkpoint_sha256="a" * 64, stream_ledger={"fixture": True},
                   all_transmitted_streams_counted=True)
    (source / "measurement_receipt.json").write_text(json.dumps(receipt))
    rows = []
    for image, snr, trial in itertools.product(range(1, 25), [0, 5, 10, 15, 20], range(10)):
        rows.append(dict(image=f"kodim{image:02d}", snr_db=snr, trial=trial,
                         model="synthetic", channel="awgn", source_commit="fixture", measured=True,
                         SSIM=.5, MS_SSIM=.6, LPIPS=.3, MSE=.25, PSNR=6.020599913279624,
                         complex_channel_uses=100, CBR=100 / (3 * 512 * 768),
                         encoder_seconds=.01, decoder_seconds=.01, channel_seconds=.01,
                         end_to_end_seconds=.03, native_full_image=True,
                         all_transmitted_streams_counted=True, original_height=512, original_width=768))
    return source, contract, rows


def write_rows(source, rows):
    (source / "measured_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_complete_well_formed_synthetic_grid_is_exported_unranked(tmp_path):
    source, contract, rows = measurement_fixture(tmp_path)
    write_rows(source, rows)
    output = tmp_path / "export"
    assert consolidate([source], output, contract) == 1200
    report = json.loads((output / "evidence_report.json").read_text())[0]
    assert report["measured_grid_valid"] is True
    assert report["ranking_eligible"] is False


@pytest.mark.parametrize("case", ["negative_uses", "bool_uses", "negative_time",
                                 "invalid_ssim", "missing_msssim", "missing_lpips",
                                 "inconsistent_psnr"])
def test_complete_grid_with_invalid_measurement_is_rejected(tmp_path, case):
    source, contract, rows = measurement_fixture(tmp_path)
    row = rows[0]
    if case == "negative_uses":
        row.update(complex_channel_uses=-100, CBR=-100 / (3 * 512 * 768))
    elif case == "bool_uses":
        row.update(complex_channel_uses=True, CBR=1 / (3 * 512 * 768))
    elif case == "negative_time":
        row["encoder_seconds"] = -1
    elif case == "invalid_ssim":
        row["SSIM"] = 2
    elif case == "missing_msssim":
        row.pop("MS_SSIM")
    elif case == "missing_lpips":
        row.pop("LPIPS")
    elif case == "inconsistent_psnr":
        row["PSNR"] = 100
    write_rows(source, rows)
    with pytest.raises(ValueError, match="No complete validated measured grids"):
        consolidate([source], tmp_path / "export", contract)
