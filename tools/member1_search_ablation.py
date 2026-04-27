"""
Member 1 ablation: baseline MCTS vs adaptive MCTS.

This script uses a deterministic scripted model so the search behavior can be
tested without API keys. It isolates the Member 1 change: whether MCTS can
prefer a promising final step over a distracting first candidate.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.mcts import MCTS, MCTSConfig


@dataclass(frozen=True)
class AblationCase:
    problem: str
    answer: str
    distractor: str
    useful_step: str


CASES = [
    AblationCase(
        problem="Compute 1 + 1.",
        answer="2",
        distractor="Guess 3 without checking the operation.",
        useful_step="Add 1 and 1 to get 2."
    ),
    AblationCase(
        problem="Solve for x: x + 3 = 5.",
        answer="x = 2",
        distractor="Guess x = 4 and move on.",
        useful_step="Subtract 3 from both sides to get x = 2."
    ),
    AblationCase(
        problem="Find the derivative of f(x) = x^2.",
        answer="2x",
        distractor="Guess the derivative is x.",
        useful_step="Apply the power rule: derivative of x^2 is 2x."
    ),
]


class ScriptedMathModel:
    def __init__(self, cases: Iterable[AblationCase]):
        self.cases: Dict[str, AblationCase] = {case.problem: case for case in cases}

    def generate_response(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        current_state = self._current_state(prompt)
        problem = current_state.splitlines()[0].strip()
        case = self.cases[problem]

        if current_state.strip() == problem:
            steps = [
                case.distractor,
                f"FINAL ANSWER: {case.answer}",
                case.useful_step,
            ]
        elif case.distractor in current_state:
            steps = [
                "Continue the unchecked guess.",
                "FINAL ANSWER: incorrect",
            ]
        else:
            steps = [
                case.useful_step,
                f"FINAL ANSWER: {case.answer}",
            ]
        return "\n".join(f"STEP: {step}" for step in steps)

    def evaluate_reasoning(self, problem: str, solution_steps: List[str]) -> float:
        expected = self.cases[problem].answer.lower()
        joined = " ".join(solution_steps).lower()
        if f"final answer: {expected}" in joined:
            return 1.0
        if "final answer:" in joined:
            return 0.0
        return 0.35

    def embed_text(self, text: str) -> List[float]:
        return [0.0] * 1536

    def _current_state(self, prompt: str) -> str:
        marker = "Current reasoning so far:"
        if marker not in prompt:
            raise ValueError("Unexpected prompt format")
        after_marker = prompt.split(marker, 1)[1]
        return after_marker.split("Generate ", 1)[0].strip()


def run_strategy(name: str, config: MCTSConfig) -> List[Dict[str, object]]:
    rows = []
    for case in CASES:
        model = ScriptedMathModel(CASES)
        mcts = MCTS(config)
        action, trajectory = mcts.search(case.problem, model)
        correct = f"final answer: {case.answer.lower()}" in action.lower()
        rows.append({
            "strategy": name,
            "problem": case.problem,
            "selected_action": action,
            "correct": correct,
            "trajectory_steps": len(trajectory),
            "nodes_expanded": mcts.last_stats["nodes_expanded"],
            "simulations": mcts.last_stats["simulations_run"],
            "latency_ms": round(mcts.last_stats["latency_seconds"] * 1000, 2),
            "model_calls": mcts.last_stats["model_calls"],
            "estimated_tokens": mcts.last_stats["estimated_tokens"],
            "pruned_actions": mcts.last_stats["pruned_actions"],
            "early_stopped": mcts.last_stats["early_stopped"],
        })
    return rows


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    summaries = []
    strategies = sorted({row["strategy"] for row in rows})
    for strategy in strategies:
        subset = [row for row in rows if row["strategy"] == strategy]
        count = len(subset)
        summaries.append({
            "strategy": strategy,
            "accuracy": f"{sum(bool(row['correct']) for row in subset)}/{count}",
            "avg_nodes": round(sum(int(row["nodes_expanded"]) for row in subset) / count, 2),
            "avg_simulations": round(sum(int(row["simulations"]) for row in subset) / count, 2),
            "avg_latency_ms": round(sum(float(row["latency_ms"]) for row in subset) / count, 2),
            "avg_model_calls": round(sum(int(row["model_calls"]) for row in subset) / count, 2),
            "avg_est_tokens": round(sum(int(row["estimated_tokens"]) for row in subset) / count, 2),
            "avg_pruned": round(sum(int(row["pruned_actions"]) for row in subset) / count, 2),
        })
    return summaries


def markdown_table(rows: List[Dict[str, object]], columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def build_report() -> str:
    baseline_config = MCTSConfig(
        max_simulations=3,
        max_depth=3,
        num_actions=3,
        search_strategy="baseline"
    )
    adaptive_config = MCTSConfig(
        max_simulations=3,
        max_depth=3,
        num_actions=3,
        search_strategy="adaptive",
        max_branching_factor=4,
        min_branching_factor=1,
        diversity_weight=0.25,
        prune_threshold=0.25,
        early_stop_reward=0.95,
        early_stop_min_simulations=1
    )

    rows = run_strategy("baseline", baseline_config)
    rows.extend(run_strategy("adaptive", adaptive_config))
    summary = summarize(rows)

    summary_columns = [
        "strategy",
        "accuracy",
        "avg_nodes",
        "avg_simulations",
        "avg_latency_ms",
        "avg_model_calls",
        "avg_est_tokens",
        "avg_pruned",
    ]
    detail_columns = [
        "strategy",
        "problem",
        "selected_action",
        "correct",
        "nodes_expanded",
        "simulations",
        "latency_ms",
        "estimated_tokens",
        "pruned_actions",
        "early_stopped",
    ]

    return "\n\n".join([
        "# Member 1 Search Ablation",
        "Deterministic scripted benchmark: no API keys required.",
        "## Summary",
        markdown_table(summary, summary_columns),
        "## Per-Problem Results",
        markdown_table(rows, detail_columns),
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
