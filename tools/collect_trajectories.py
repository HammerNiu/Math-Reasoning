"""
Collect MCTS reasoning trajectories and export preference pairs for PPM training.

Usage:
    python tools/collect_trajectories.py --model deepseek --simulations 5
    python tools/collect_trajectories.py --problems data/problems.txt --output data/traj.jsonl

Output JSONL format (one record per problem):
    {
        "problem": "...",
        "final_action": "...",
        "final_reward": 0.85,
        "preference_pairs": [
            {"preferred": "<good step>", "non_preferred": "<bad step>"},
            ...
        ]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.mcts import MCTS, MCTSConfig
from src.model.model_interface import ModelFactory

# ---------------------------------------------------------------------------
# Built-in example problems (used when no --problems file is given)
# ---------------------------------------------------------------------------

EXAMPLE_PROBLEMS: List[str] = [
    "Solve for x: 2x + 6 = 14",
    "Find the derivative of f(x) = 3x^2 + 2x + 1",
    "What is the sum of the first 10 natural numbers?",
    "Simplify: (x^2 - 4) / (x - 2)",
    "Solve the quadratic: x^2 - 5x + 6 = 0",
    "Compute the integral of sin(x) from 0 to pi",
    "If f(x) = 2x^3 - 3x + 1, find f'(x)",
    "Solve the system: x + y = 5, x - y = 1",
]

_API_KEY_ENVS: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

# ---------------------------------------------------------------------------
# Preference pair extraction
# ---------------------------------------------------------------------------

def extract_preference_pairs(
    trajectory: List[Dict[str, Any]],
    preferred_threshold: float = 0.6,
    non_preferred_threshold: float = 0.35,
) -> List[Dict[str, str]]:
    """Build (preferred, non_preferred) step pairs from a trajectory.

    A step is "preferred" if its node value is high (the search found it
    promising) and "non_preferred" if its value is low.
    """
    preferred: List[str] = []
    non_preferred: List[str] = []

    for entry in trajectory:
        action = (entry.get("action") or "").strip()
        value = float(entry.get("value") or 0.0)
        if not action:
            continue
        if value >= preferred_threshold:
            preferred.append(action)
        elif value <= non_preferred_threshold:
            non_preferred.append(action)

    pairs: List[Dict[str, str]] = []
    for p in preferred:
        for np in non_preferred:
            if p != np:
                pairs.append({"preferred": p, "non_preferred": np})
    return pairs


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def load_problems(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(p) for p in parsed]
    except json.JSONDecodeError:
        pass
    return [line.strip() for line in text.splitlines() if line.strip()]


def collect(args: argparse.Namespace) -> None:
    # Load problem list
    if args.problems:
        problems = load_problems(args.problems)
        print(f"Loaded {len(problems)} problems from {args.problems}")
    else:
        problems = EXAMPLE_PROBLEMS
        print(f"No --problems file given; using {len(problems)} built-in examples")

    # Create model
    name = args.model.lower()
    api_key = os.getenv(_API_KEY_ENVS[name], "")
    model = ModelFactory.create_model(name, api_key)

    # MCTS config
    config = MCTSConfig(
        max_simulations=args.simulations,
        search_strategy=args.strategy,
        max_depth=5,
        num_actions=3,
    )

    # Output file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_pairs = 0

    with args.output.open("w", encoding="utf-8") as out_f:
        for i, problem in enumerate(problems):
            print(f"[{i + 1}/{len(problems)}] {problem[:70]}...")
            mcts = MCTS(config)
            try:
                action, trajectory = mcts.search(problem, model)
                reward = float(mcts.last_stats.get("best_reward", 0.0))
                pairs = extract_preference_pairs(
                    trajectory,
                    preferred_threshold=args.preferred_threshold,
                    non_preferred_threshold=args.non_preferred_threshold,
                )
                record = {
                    "problem": problem,
                    "final_action": action,
                    "final_reward": reward,
                    "preference_pairs": pairs,
                    "stats": {
                        k: mcts.last_stats[k]
                        for k in ("simulations_run", "nodes_expanded", "model_calls")
                        if k in mcts.last_stats
                    },
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_pairs += len(pairs)
                print(f"  reward={reward:.2f}  pairs={len(pairs)}")
            except Exception as exc:
                print(f"  ERROR: {exc}")

    print(f"\nFinished. {total_pairs} preference pairs written to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problems", type=Path, help="Path to problem file (.txt one-per-line, or JSON array)")
    parser.add_argument("--output", type=Path, default=Path("data/trajectories.jsonl"), help="Output JSONL path")
    parser.add_argument("--model", default="openai", choices=list(_API_KEY_ENVS), help="LLM backend")
    parser.add_argument("--simulations", type=int, default=5, help="MCTS simulations per problem")
    parser.add_argument("--strategy", default="adaptive", choices=["baseline", "adaptive"])
    parser.add_argument("--preferred-threshold", type=float, default=0.6,
                        help="Min node value to label a step as preferred")
    parser.add_argument("--non-preferred-threshold", type=float, default=0.35,
                        help="Max node value to label a step as non-preferred")
    args = parser.parse_args()
    collect(args)


if __name__ == "__main__":
    main()
