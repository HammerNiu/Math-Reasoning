"""Generate AMC process-preference pairs with an OpenAI teacher model.

The script uses the official AMC answer as an anchor and asks the teacher to
produce concise preferred reasoning steps plus plausible wrong process steps.
It is resumable: if the output file already has records, those problem ids are
skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.data.amc import load_amc_local
from src.model.model_interface import ModelConfig, ModelFactory


PROMPT = """\
You are creating process-supervision training data for an AMC math solver.

Problem:
{problem}

Official answer choice: {answer}

Return JSON only with this schema:
{{
  "preferred_steps": [
    "one concise correct reasoning step",
    "another concise correct reasoning step",
    "FINAL ANSWER: {answer}, <matching value or expression>"
  ],
  "non_preferred_steps": [
    "a plausible but wrong or incomplete reasoning step",
    "another plausible wrong shortcut"
  ]
}}

Rules:
- Use the official answer as ground truth.
- Preferred steps must solve or verify the answer, not merely state the answer.
- Non-preferred steps should be realistic AMC mistakes: ignoring order, missing a case,
  using the wrong equation, choosing an option without checking, or mishandling units.
- Keep each step under 35 words.
- Do not include markdown fences or commentary outside JSON.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/amc12_250.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/amc_openai_preferences.jsonl"))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--model", default=os.getenv("OPENAI_TEACHER_MODEL", "gpt-4o"))
    parser.add_argument("--reasoning-effort", default=os.getenv("OPENAI_REASONING_EFFORT", "none"))
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay between API calls")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        sys.exit("OPENAI_API_KEY not set.")

    items = load_amc_local(args.data, n=args.n, seed=42)
    model = ModelFactory.create_model(
        "openai",
        api_key,
        config=ModelConfig(
            model=args.model,
            temperature=0.2,
            max_tokens=900,
            timeout=90.0,
            reasoning_effort=args.reasoning_effort,
        ),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = _loaded_problem_ids(args.output)
    print(f"Loaded {len(items)} AMC problems. Existing teacher records: {len(done)}")

    generated = 0
    with args.output.open("a", encoding="utf-8") as f:
        for index, item in enumerate(items, 1):
            problem_id = _problem_key(item)
            if problem_id in done:
                continue
            prompt = PROMPT.format(problem=item["problem"], answer=item["answer"])
            payload = None
            last_error = ""
            for attempt in range(1, args.max_retries + 1):
                try:
                    response = model.generate_response(prompt, temperature=0.2, max_tokens=900)
                    payload = _parse_teacher_json(response)
                    break
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    time.sleep(min(2 ** attempt, 8))
            if payload is None:
                print(f"[{index}/{len(items)}] ERROR {problem_id}: {last_error}")
                continue

            pairs = _pairs_from_payload(item, payload)
            record = {
                "problem_id": problem_id,
                "problem": item["problem"],
                "answer": item["answer"],
                "source": item.get("source", "amc"),
                "teacher_model": args.model,
                "preference_pairs": pairs,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            generated += 1
            print(f"[{index}/{len(items)}] {problem_id}: {len(pairs)} pairs")
            if args.sleep:
                time.sleep(args.sleep)

    print(f"Generated {generated} new OpenAI teacher records -> {args.output}")


def _loaded_problem_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("problem_id"):
                ids.add(str(record["problem_id"]))
    return ids


def _problem_key(item: Dict[str, Any]) -> str:
    return str(item.get("problem_id") or abs(hash(item["problem"])))


def _parse_teacher_json(response: str) -> Dict[str, List[str]]:
    text = (response or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.IGNORECASE).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    payload = json.loads(text)
    preferred = [str(s).strip() for s in payload.get("preferred_steps", []) if str(s).strip()]
    non_preferred = [str(s).strip() for s in payload.get("non_preferred_steps", []) if str(s).strip()]
    if not preferred or not non_preferred:
        raise ValueError("teacher JSON missing preferred/non_preferred steps")
    return {"preferred_steps": preferred[:5], "non_preferred_steps": non_preferred[:5]}


def _pairs_from_payload(item: Dict[str, Any], payload: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for preferred in payload["preferred_steps"]:
        for non_preferred in payload["non_preferred_steps"]:
            if preferred == non_preferred:
                continue
            pairs.append({
                "problem": item["problem"],
                "preferred": preferred,
                "non_preferred": non_preferred,
                "preferred_reward": 1.0,
                "non_preferred_reward": 0.0,
            })
    return pairs[:20]


if __name__ == "__main__":
    main()

