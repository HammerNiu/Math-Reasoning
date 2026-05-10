"""Deterministic regression tests for exact AMC-style solvers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.exact_solvers import solve_amc_exact


CASES: List[Dict[str, str]] = [
    {
        "name": "seating four students",
        "expected": "2",
        "problem": (
            "Four students are seated in a row. They chat with the people sitting next to them, "
            "then rearrange themselves so that they are no longer seated next to any of the same "
            "people. How many rearrangements are possible?"
        ),
    },
    {
        "name": "seating five students",
        "expected": "14",
        "problem": (
            "Five students are seated in a row. They rearrange themselves so that no one is next "
            "to the same person as before. How many rearrangements are possible?"
        ),
    },
    {
        "name": "rod quadrilateral original",
        "expected": "17",
        "problem": (
            "Joy has 30 thin rods, one each of every integer length from 1 cm through 30 cm. "
            "She places the rods with lengths 3 cm, 7 cm, and 15 cm on a table. She then wants "
            "to choose a fourth rod that she can put with these three to form a quadrilateral "
            "with positive area. How many remaining rods can she choose?"
        ),
    },
    {
        "name": "rod quadrilateral variant",
        "expected": "15",
        "problem": (
            "A box has 20 thin rods, one each of every integer length from 1 cm through 20 cm. "
            "A student places the rods with lengths 4 cm, 6 cm, and 9 cm on a table. She then "
            "wants to choose a fourth rod that she can put with these three to form a "
            "quadrilateral with positive area. How many remaining rods can she choose?"
        ),
    },
    {
        "name": "integer triangle side count",
        "expected": "9",
        "problem": (
            "A triangle has two sides of length 5 and 9. If the third side has integer length, "
            "how many possible values can the third side have?"
        ),
    },
    {
        "name": "integer triangle side sum",
        "expected": "49",
        "problem": (
            "A triangle has two sides of length 4 and 7. If the third side has integer length, "
            "what is the sum of all possible third side lengths?"
        ),
    },
    {
        "name": "same-area integer triangle",
        "expected": "24",
        "problem": (
            "Two non-congruent triangles have the same area. Each triangle has sides of length "
            "8 and 9, and the third side of each triangle has integer length. What is the sum "
            "of the lengths of the third sides?"
        ),
    },
    {
        "name": "circle sum top digit",
        "expected": "5",
        "problem": (
            "In the figure below, each circle will be filled with a digit from 1 to 6. Each "
            "digit must appear exactly once. The sum of the digits in neighboring circles is "
            "shown in the box between them. The box sums around the loop are 9, 10, 8, 5, 4, "
            "and 6, starting from the top-left edge and moving clockwise. What digit must be "
            "placed in the top circle?"
        ),
    },
]


def run() -> Dict[str, object]:
    rows = []
    for case in CASES:
        solution = solve_amc_exact(case["problem"])
        answer = str(solution.get("answer", "")) if solution else ""
        ok = answer == case["expected"]
        rows.append({
            "name": case["name"],
            "expected": case["expected"],
            "prediction": answer or "NO_MATCH",
            "method": solution.get("method", "none") if solution else "none",
            "correct": ok,
        })
    return {
        "total": len(rows),
        "correct": sum(int(row["correct"]) for row in rows),
        "accuracy": sum(int(row["correct"]) for row in rows) / len(rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/exact_solver_regression.json"))
    args = parser.parse_args()

    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Exact solver regression: {result['correct']}/{result['total']} ({result['accuracy']:.1%})")
    for row in result["rows"]:
        mark = "OK" if row["correct"] else "NO"
        print(f"  {mark} {row['name']}: expected={row['expected']} got={row['prediction']} method={row['method']}")


if __name__ == "__main__":
    main()
