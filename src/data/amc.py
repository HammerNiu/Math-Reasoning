"""AMC dataset loading and normalization utilities.

The primary supported remote source is the Hugging Face dataset
``edev2000/amc12-full``. Local JSON/JSONL/CSV exports with common field names
are also supported so benchmark runs can be reproduced without network access.
"""
from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

AMC12_HF_DATASET = "edev2000/amc12-full"


def normalize_amc_item(item: Dict[str, Any], source: str = "amc") -> Optional[Dict[str, Any]]:
    problem = (
        item.get("question")
        or item.get("problem")
        or item.get("prompt")
        or item.get("text")
        or ""
    )
    answer = (
        item.get("answer")
        or item.get("final_answer")
        or item.get("label")
        or item.get("target")
        or ""
    )
    problem = str(problem).strip()
    answer = str(answer).strip()
    if not problem or not answer:
        return None

    answer = _normalize_answer(answer)
    difficulty = _parse_difficulty(item)
    problem_id = (
        item.get("problem_id")
        or item.get("id")
        or item.get("name")
        or item.get("exam")
        or ""
    )
    year = item.get("year") or _parse_year(str(problem_id)) or _parse_year(problem)

    return {
        "problem": problem,
        "answer": answer,
        "subject": "amc",
        "source": source,
        "difficulty": difficulty,
        "problem_id": str(problem_id),
        "year": year,
    }


def load_amc_hf(
    n: int,
    dataset_id: str = AMC12_HF_DATASET,
    split: str = "train",
    seed: int = 42,
    min_difficulty: Optional[int] = None,
    max_difficulty: Optional[int] = None,
) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install Hugging Face datasets first: pip install datasets") from exc

    ds = load_dataset(dataset_id, split=split)
    source = dataset_id.split("/")[-1]
    pool = [
        normalized
        for raw in ds
        if (normalized := normalize_amc_item(dict(raw), source=source)) is not None
    ]
    return _sample_and_filter(pool, n, seed, min_difficulty, max_difficulty)


def load_amc_local(
    path: Path,
    n: int,
    seed: int = 42,
    min_difficulty: Optional[int] = None,
    max_difficulty: Optional[int] = None,
) -> List[Dict[str, Any]]:
    records = list(_read_local_records(path))
    pool = [
        normalized
        for raw in records
        if (normalized := normalize_amc_item(raw, source=path.stem)) is not None
    ]
    return _sample_and_filter(pool, n, seed, min_difficulty, max_difficulty)


def load_amc(
    n: int,
    local_path: Optional[Path] = None,
    dataset_id: str = AMC12_HF_DATASET,
    split: str = "train",
    seed: int = 42,
    min_difficulty: Optional[int] = None,
    max_difficulty: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if local_path:
        return load_amc_local(local_path, n, seed, min_difficulty, max_difficulty)
    return load_amc_hf(n, dataset_id, split, seed, min_difficulty, max_difficulty)


def write_jsonl(items: Iterable[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _sample_and_filter(
    pool: List[Dict[str, Any]],
    n: int,
    seed: int,
    min_difficulty: Optional[int],
    max_difficulty: Optional[int],
) -> List[Dict[str, Any]]:
    if min_difficulty is not None:
        pool = [item for item in pool if int(item.get("difficulty", 0)) >= min_difficulty]
    if max_difficulty is not None:
        pool = [item for item in pool if int(item.get("difficulty", 0)) <= max_difficulty]
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n] if n and n > 0 else pool


def _read_local_records(path: Path) -> Iterable[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("data") or data.get("records") or data.get("items") or [data]
        if not isinstance(data, list):
            raise ValueError(f"JSON file must contain a list of records: {path}")
        for item in data:
            if isinstance(item, dict):
                yield item
        return
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f)
        return
    raise ValueError(f"Unsupported AMC file type: {path.suffix}. Use .jsonl, .json, or .csv.")


def _normalize_answer(answer: str) -> str:
    stripped = answer.strip()
    match = re.search(r"\b([A-E])\b", stripped.upper())
    if match and len(stripped) <= 6:
        return match.group(1)
    boxed = re.search(r"\\boxed\{([^{}]+)\}", stripped)
    if boxed:
        return boxed.group(1).strip()
    return stripped


def _parse_difficulty(item: Dict[str, Any]) -> int:
    raw = item.get("difficulty")
    if raw not in (None, ""):
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    problem_id = str(item.get("problem_id") or item.get("id") or "")
    match = re.search(r"(?:P|-)(\d{1,2})\b", problem_id)
    if match:
        index = int(match.group(1))
        if index <= 10:
            return 2
        if index <= 20:
            return 3
        return 4
    return 3


def _parse_year(text: str) -> Optional[int]:
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return int(match.group(0)) if match else None

