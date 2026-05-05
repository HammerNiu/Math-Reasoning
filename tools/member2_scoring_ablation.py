"""
Member 2 ablation: plain process scoring vs hybrid verifier scoring.

This is deterministic and does not require API keys. It demonstrates the
scoring-side innovation by correcting cases where a weak learned scorer would
rank vague guesses above mathematically useful steps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.scoring import HeuristicStepVerifier, HybridProcessScorer


CASES = [
    {
        "problem": "Solve for x: x + 3 = 5.",
        "bad": "Guess x = 4 without checking the equation.",
        "good": "Subtract 3 from both sides to get x = 2.",
    },
    {
        "problem": "Compute 1 + 1.",
        "bad": "Maybe the answer is 3.",
        "good": "Add 1 and 1 to get 2.",
    },
    {
        "problem": "Find the derivative of f(x) = x^2.",
        "bad": "Skip the power rule and guess x.",
        "good": "Apply the power rule: derivative of x^2 is 2x.",
    },
]


class BiasedPPM:
    """A deliberately weak scorer that overvalues guesses."""

    def evaluate_step(self, step: str, embedder=None) -> float:
        lower = step.lower()
        if "guess" in lower or "maybe" in lower or "skip" in lower:
            return 0.90
        return 0.40


def markdown_table(rows: List[Dict[str, object]], columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row[col]) for col in columns) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def build_report() -> str:
    ppm = BiasedPPM()
    verifier = HeuristicStepVerifier()
    hybrid = HybridProcessScorer(ppm=ppm, verifier=verifier, ppm_weight=0.20)

    rows = []
    corrected = 0
    for case in CASES:
        candidates = [case["bad"], case["good"]]
        old_scores = {step: ppm.evaluate_step(step) for step in candidates}
        hybrid_scores = {step: hybrid.evaluate_step(step, state=case["problem"]) for step in candidates}
        old_top = max(candidates, key=lambda step: old_scores[step])
        hybrid_top = max(candidates, key=lambda step: hybrid_scores[step])
        fixed = old_top == case["bad"] and hybrid_top == case["good"]
        corrected += int(fixed)
        rows.append({
            "problem": case["problem"],
            "old_top": old_top,
            "hybrid_top": hybrid_top,
            "old_good": round(old_scores[case["good"]], 3),
            "old_bad": round(old_scores[case["bad"]], 3),
            "hybrid_good": round(hybrid_scores[case["good"]], 3),
            "hybrid_bad": round(hybrid_scores[case["bad"]], 3),
            "fixed": fixed,
        })

    columns = [
        "problem",
        "old_top",
        "hybrid_top",
        "old_good",
        "old_bad",
        "hybrid_good",
        "hybrid_bad",
        "fixed",
    ]
    return "\n\n".join([
        "# Member 2 Scoring Ablation",
        "Deterministic benchmark: no API keys required.",
        f"Corrected mis-rankings: {corrected}/{len(CASES)}.",
        markdown_table(rows, columns),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", type=Path, help="Optional path to write the markdown report.")
    args = parser.parse_args()

    report = build_report()
    print(report)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
