"""Build lightweight AMC preference pairs from answer labels.

This does not replace trajectory collection. It gives the process scorer a
cheap warm-up set that teaches it to prefer verified final-answer branches over
unverified multiple-choice guesses before API-backed MCTS trajectories are
available.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.amc import load_amc_local

CHOICES = ["A", "B", "C", "D", "E"]


def build_pairs(records: Iterable[Dict[str, Any]], max_wrong_per_problem: int = 4) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    rng = random.Random(42)
    for item in records:
        problem = item["problem"]
        answer = str(item["answer"]).strip().upper()
        if not re.fullmatch(r"[A-E]", answer):
            continue
        wrong = [choice for choice in CHOICES if choice != answer]
        rng.shuffle(wrong)
        preferred = (
            "Use the conditions in the problem to verify each viable choice, "
            f"reject contradictions, and conclude FINAL ANSWER: {answer}"
        )
        for choice in wrong[:max_wrong_per_problem]:
            pairs.append({
                "problem": problem,
                "preferred": preferred,
                "non_preferred": f"Choose option {choice} without checking all constraints. FINAL ANSWER: {choice}",
                "preferred_reward": 1.0,
                "non_preferred_reward": 0.1,
            })
    rng.shuffle(pairs)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/amc12.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/amc_preferences.jsonl"))
    parser.add_argument("--max-wrong-per-problem", type=int, default=4)
    args = parser.parse_args()

    records = load_amc_local(args.data, n=0)
    pairs = build_pairs(records, args.max_wrong_per_problem)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} AMC preference pairs -> {args.output}")


if __name__ == "__main__":
    main()
