from pathlib import Path

import pytest

from dit_jscc.bandwidth import (
    MinMaxStats,
    allocate_bandwidth,
    analyze_caption,
    kc_score,
)
from dit_jscc.pipeline import DiTJSCCPipeline, InferenceResult, check_environment


def test_cbr_matches_paper_operating_points():
    assert DiTJSCCPipeline._calculate_cbr(256, 8, 8) == pytest.approx(1 / 96)
    assert DiTJSCCPipeline._calculate_cbr(256, 8, 24) == pytest.approx(1 / 48)
    assert DiTJSCCPipeline._calculate_cbr(512, 32, 32) == pytest.approx(1 / 96)


def test_environment_check_does_not_load_models():
    report = check_environment(256)
    assert report["resolution"] == 256
    assert Path(report["config"]).name == "inference_256.py"
    assert "checkpoint_found" in report
    assert "autoencoder_found" in report


def test_invalid_resolution_is_rejected():
    with pytest.raises(ValueError):
        DiTJSCCPipeline(resolution=128)


def test_metadata_excludes_image():
    result = InferenceResult(
        image=object(),
        resolution=256,
        channel_type="awgn",
        snr_db=0,
        semantic_rate=8,
        detail_rate=8,
        cbr=1 / 96,
        sampler="dpm_solver",
        steps=20,
        guidance_scale=2,
        seed=1000,
        elapsed_seconds=1.23456,
    )
    metadata = result.metadata()
    assert "image" not in metadata
    assert metadata["elapsed_seconds"] == 1.235


def test_kc_allocation_preserves_available_total():
    features = analyze_caption("A black dog runs quickly. Snow covers the quiet road.")
    stats = MinMaxStats(
        word_count=(2, 20),
        lexical_diversity=(0.2, 1.0),
        syntactic_complexity=(2, 12),
    )
    score = kc_score(features, stats)
    allocation = allocate_bandwidth(
        score,
        mean_score=0.5,
        baseline_semantic_rate=8,
        total_rate=16,
        eta=1.0,
        semantic_candidates=tuple(range(4, 29, 2)),
        detail_candidates=tuple(range(0, 25, 2)),
    )
    assert allocation.semantic_rate + allocation.detail_rate == 16
