"""
Process-level scoring helpers for MCTS.

The neural PPM remains the learned scorer. This module adds a lightweight
verifier that can be used alone, or combined with a trained PPM, so the search
loop has a scoring-side innovation even before a checkpoint is available.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_score(value: float) -> float:
    """Return a 0..1 score while preserving the order of raw PPM logits."""
    value = float(value)
    if 0.0 <= value <= 1.0:
        return value
    return 1.0 / (1.0 + math.exp(-value))


@dataclass
class VerifierConfig:
    base_score: float = 0.45
    final_answer_bonus: float = 0.20
    math_signal_bonus: float = 0.12
    reasoning_word_bonus: float = 0.10
    verification_bonus: float = 0.08
    vague_penalty: float = 0.20
    too_short_penalty: float = 0.12
    too_long_penalty: float = 0.18
    contradiction_penalty: float = 0.15


class HeuristicStepVerifier:
    """Small deterministic verifier for candidate math reasoning steps.

    It rewards useful mathematical operations, explicit verification language,
    and final-answer markers, while penalizing guesses, vague prose, and obvious
    inconsistency language. The interface intentionally matches
    ProcessPreferenceModel.evaluate_step(step, embedder), so MCTS can use it as
    a drop-in process scorer.
    """

    def __init__(self, config: Optional[VerifierConfig] = None):
        self.config = config or VerifierConfig()

    def evaluate_step(self, step: str, embedder: Any = None, state: Optional[str] = None) -> float:
        return self.score_breakdown(step, state=state)["score"]

    def score_breakdown(self, step: str, state: Optional[str] = None) -> Dict[str, float]:
        text = (step or "").strip()
        lower = text.lower()
        cfg = self.config
        score = cfg.base_score

        if (
            "final answer:" in lower
            or lower.startswith("the answer is")
            or "therefore the answer is" in lower
        ):
            score += cfg.final_answer_bonus
        if re.search(r"[-+*/=^]|\d|sqrt|sin|cos|tan|log|derivative|integral", lower):
            score += cfg.math_signal_bonus
        if any(word in lower for word in [
            "because", "therefore", "substitute", "simplify", "factor",
            "differentiate", "solve", "apply", "subtract", "add", "divide",
            "multiply", "isolate", "expand", "coefficient", "compare",
            "match", "equation", "constant", "derive"
        ]):
            score += cfg.reasoning_word_bonus
        if any(word in lower for word in ["check", "verify", "consistent", "satisfies"]):
            score += cfg.verification_bonus
        if any(word in lower for word in ["guess", "maybe", "unrelated", "skip", "without checking", "try some values"]):
            score -= cfg.vague_penalty
        if len(text.split()) < 3 and "final answer:" not in lower:
            score -= cfg.too_short_penalty
        if len(text.split()) > 30 and "final answer:" not in lower:
            score -= cfg.too_long_penalty
        if any(word in lower for word in ["contradiction", "impossible", "incorrect"]) and "final answer:" not in lower:
            score -= cfg.contradiction_penalty

        return {"score": _clamp01(score)}


class HybridProcessScorer:
    """Combine a learned PPM score with the deterministic verifier score."""

    def __init__(
        self,
        ppm: Optional[Any] = None,
        verifier: Optional[HeuristicStepVerifier] = None,
        ppm_weight: float = 0.65,
    ):
        self.ppm = ppm
        self.verifier = verifier or HeuristicStepVerifier()
        self.ppm_weight = _clamp01(ppm_weight)

    def evaluate_step(self, step: str, embedder: Any = None, state: Optional[str] = None) -> float:
        verifier_score = self.verifier.evaluate_step(step, embedder, state=state)
        if self.ppm is None:
            return verifier_score

        try:
            ppm_score = _normalize_score(self.ppm.evaluate_step(step, embedder))
        except Exception:
            return verifier_score

        return _clamp01(self.ppm_weight * ppm_score + (1.0 - self.ppm_weight) * verifier_score)
