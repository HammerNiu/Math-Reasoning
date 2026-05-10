"""Scan AMC JSONL files for deterministic exact-solver coverage."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.exact_solvers import solve_amc_exact


def _iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _choice_map(problem: str) -> Dict[str, str]:
    marker = re.compile(r"\\textbf\{\s*\(([A-E])\)\s*\}|\(([A-E])\)")
    matches = list(marker.finditer(problem))
    choices: Dict[str, str] = {}
    for index, match in enumerate(matches):
        letter = next(group for group in match.groups() if group)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(problem)
        choices[letter] = problem[start:end]
    return choices


def _normalize(value: str) -> str:
    value = str(value or "").lower()
    value = value.replace("$", " ").replace("\\", " ")
    value = re.sub(r"(textbf|text|mathrm|qquad|quad|frac|sqrt)", " ", value)
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _is_correct(problem: str, expected: str, answer: str) -> Optional[bool]:
    expected = str(expected or "").strip()
    answer = str(answer or "").strip()
    if not expected or not answer:
        return None
    if _normalize(expected) == _normalize(answer):
        return True
    choices = _choice_map(problem)
    if expected in choices:
        return _normalize(answer) in _normalize(choices[expected]).split()
    return None


def scan(inputs: List[Path]) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for path in inputs:
        total = 0
        hits = 0
        correct = 0
        methods: Dict[str, int] = {}
        rows: List[Dict[str, object]] = []
        for record in _iter_jsonl(path):
            total += 1
            problem = str(record.get("problem", ""))
            solution = solve_amc_exact(problem)
            if solution is None:
                continue
            hits += 1
            method = str(solution.get("method", "unknown"))
            answer = str(solution.get("answer", ""))
            ok = _is_correct(problem, str(record.get("answer", "")), answer)
            correct += int(ok is True)
            methods[method] = methods.get(method, 0) + 1
            rows.append(
                {
                    "problem": problem[:120],
                    "expected": record.get("answer", ""),
                    "answer": answer,
                    "method": method,
                    "correct": bool(ok),
                }
            )
        summaries.append(
            {
                "dataset": str(path),
                "total": total,
                "exact_hits": hits,
                "exact_correct": correct,
                "accuracy_on_hits": correct / hits if hits else None,
                "methods": methods,
                "rows": rows,
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[Path("data/amc_holdout50.jsonl"), Path("data/amc12_250.jsonl")],
    )
    parser.add_argument("--output", type=Path, default=Path("data/exact_solver_coverage.json"))
    args = parser.parse_args()

    result = scan(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    total = sum(item["total"] for item in result)
    hits = sum(item["exact_hits"] for item in result)
    correct = sum(item["exact_correct"] for item in result)
    print(f"Exact solver coverage: {correct}/{hits} correct exact hits across {total} records")
    for item in result:
        print(
            f"  {item['dataset']}: {item['exact_correct']}/{item['exact_hits']} "
            f"hits correct over {item['total']} records"
        )


if __name__ == "__main__":
    main()
