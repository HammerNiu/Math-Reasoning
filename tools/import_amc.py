"""Import AMC problems into a local JSONL file.

Examples:
    .venv/bin/python tools/import_amc.py --output data/amc12.jsonl --n 200
    .venv/bin/python tools/import_amc.py --input my_amc.csv --output data/amc_local.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.amc import AMC12_HF_DATASET, load_amc, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Optional local .jsonl/.json/.csv AMC file")
    parser.add_argument("--output", type=Path, default=Path("data/amc12.jsonl"))
    parser.add_argument("--dataset-id", default=AMC12_HF_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=0, help="Number of problems to export; 0 means all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-difficulty", type=int)
    parser.add_argument("--max-difficulty", type=int)
    args = parser.parse_args()

    problems = load_amc(
        n=args.n,
        local_path=args.input,
        dataset_id=args.dataset_id,
        split=args.split,
        seed=args.seed,
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
    )
    write_jsonl(problems, args.output)
    print(f"Imported {len(problems)} AMC problems -> {args.output}")


if __name__ == "__main__":
    main()

