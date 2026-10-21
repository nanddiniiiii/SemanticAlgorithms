"""KC-inspired bandwidth allocation utilities from the DiT-JSCC paper.

Caption generation is deliberately kept outside this lightweight module.  A
caller may use BLIP-2, a cached caption, or any paper-compatible captioning
pipeline, then apply the exact feature normalization and allocation equation
with statistics measured on its evaluation set.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence


_WORD = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
_SENTENCE = re.compile(r"[^.!?]+")


@dataclass(frozen=True)
class CaptionComplexity:
    word_count: float
    lexical_diversity: float
    syntactic_complexity: float


@dataclass(frozen=True)
class MinMaxStats:
    word_count: tuple[float, float]
    lexical_diversity: tuple[float, float]
    syntactic_complexity: tuple[float, float]


@dataclass(frozen=True)
class AllocationResult:
    semantic_rate: int
    detail_rate: int
    requested_total_rate: int
    achieved_total_rate: int
    target_semantic_rate: int
    kc_score: float


def analyze_caption(caption: str) -> CaptionComplexity:
    """Compute WC, LD, and average tokens per sentence."""

    words = [item.lower() for item in _WORD.findall(caption)]
    sentences = [item for item in _SENTENCE.findall(caption) if _WORD.search(item)]
    word_count = len(words)
    lexical_diversity = len(set(words)) / word_count if word_count else 0.0
    syntactic_complexity = (
        sum(len(_WORD.findall(sentence)) for sentence in sentences) / len(sentences)
        if sentences
        else 0.0
    )
    return CaptionComplexity(
        word_count=float(word_count),
        lexical_diversity=float(lexical_diversity),
        syntactic_complexity=float(syntactic_complexity),
    )


def _minmax(value: float, limits: tuple[float, float]) -> float:
    lower, upper = limits
    if upper <= lower:
        raise ValueError(f"Invalid min-max interval: {limits}")
    return min(1.0, max(0.0, (value - lower) / (upper - lower)))


def kc_score(
    features: CaptionComplexity,
    stats: MinMaxStats,
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> float:
    """Calculate the normalized hybrid textual-complexity score."""

    if any(weight < 0 for weight in weights) or not math.isclose(
        sum(weights), 1.0, rel_tol=1e-7, abs_tol=1e-7
    ):
        raise ValueError("weights must be non-negative and sum to 1")
    normalized = (
        _minmax(features.word_count, stats.word_count),
        _minmax(features.lexical_diversity, stats.lexical_diversity),
        _minmax(features.syntactic_complexity, stats.syntactic_complexity),
    )
    return sum(weight * value for weight, value in zip(weights, normalized))


def allocate_bandwidth(
    score: float,
    *,
    mean_score: float,
    baseline_semantic_rate: int,
    total_rate: int,
    eta: float,
    semantic_candidates: Sequence[int],
    detail_candidates: Sequence[int],
) -> AllocationResult:
    """Apply Eq. (KC-BA) and quantize it to a valid candidate-rate pair.

    Candidate pairs preserving ``total_rate`` are preferred. If no exact pair
    exists, the closest total is selected and the discrepancy is reported via
    ``achieved_total_rate``.
    """

    if eta < 0:
        raise ValueError("eta must be non-negative")
    if total_rate < 0 or baseline_semantic_rate < 0:
        raise ValueError("rates must be non-negative")
    target = baseline_semantic_rate + math.ceil(
        eta * (score - mean_score) * baseline_semantic_rate
    )
    pairs = [
        (int(semantic), int(detail))
        for semantic in semantic_candidates
        for detail in detail_candidates
    ]
    if not pairs:
        raise ValueError("candidate sets must not be empty")
    semantic, detail = min(
        pairs,
        key=lambda pair: (
            abs(sum(pair) - total_rate),
            abs(pair[0] - target),
            -pair[0],
        ),
    )
    return AllocationResult(
        semantic_rate=semantic,
        detail_rate=detail,
        requested_total_rate=int(total_rate),
        achieved_total_rate=semantic + detail,
        target_semantic_rate=int(target),
        kc_score=float(score),
    )
