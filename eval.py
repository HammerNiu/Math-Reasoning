"""
Simple evaluation script for MATH and OlympiadBench datasets.

Usage:
    python eval.py --dataset math --n 50
    python eval.py --dataset olympiad --n 20
    python eval.py --dataset math --n 100 --subject algebra --level "Level 5"
    python eval.py --dataset all --n 30

Supported models: openai, anthropic, deepseek  (set API keys in .env)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

_here = Path(__file__).resolve().parent
ROOT = _here if (_here / "src").exists() else _here.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.model.model_interface import ModelFactory

# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

MATH_SUBJECTS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]

def _extract_boxed(text: str) -> str:
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return ""
    start = matches[-1].end()
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1].strip()


def load_math(n: int, subjects: Optional[List[str]] = None, level: Optional[str] = None, seed: int = 42) -> List[Dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Run: pip install datasets")

    subjects = subjects or MATH_SUBJECTS
    pool = []
    for subj in subjects:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", subj, split="test")
        except Exception as e:
            print(f"  Warning: cannot load '{subj}': {e}")
            continue
        for item in ds:
            if level and item.get("level") != level:
                continue
            answer = _extract_boxed(item.get("solution", ""))
            if not answer:
                continue
            pool.append({
                "problem": item["problem"].strip(),
                "answer": answer,
                "subject": subj,
                "level": item.get("level", ""),
                "source": "MATH",
            })

    import random
    random.seed(seed)
    random.shuffle(pool)
    selected = pool[:n]
    print(f"  MATH: {len(pool)} problems available, using {len(selected)}")
    return selected


def load_olympiad(n: int, seed: int = 42) -> List[Dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Run: pip install datasets")

    try:
        ds = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
    except Exception as e:
        sys.exit(f"Cannot load OlympiadBench: {e}")

    pool = []
    for item in ds:
        problem = item.get("question") or ""
        answer = item.get("final_answer") or ""
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not problem.strip() or not str(answer).strip():
            continue
        pool.append({
            "problem": problem.strip(),
            "answer": str(answer).strip(),
            "subject": item.get("subfield", "") or item.get("subject", ""),
            "source": "OlympiadBench",
        })

    import random
    random.seed(seed)
    random.shuffle(pool)
    selected = pool[:n]
    print(f"  OlympiadBench: {len(pool)} problems available, using {len(selected)}")
    return selected

# ---------------------------------------------------------------------------
# Answer checking
# ---------------------------------------------------------------------------

def _latex_to_plain(s: str) -> str:
    # 去掉 $...$ 和 $$...$$ 包裹
    s = re.sub(r"\$\$(.+?)\$\$", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"\$(.+?)\$", r"\1", s)
    s = re.sub(r"\\dfrac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt", "sqrt", s)
    s = re.sub(r"\\cdot|\\times", "*", s)
    s = re.sub(r"\\pi", "pi", s)
    s = re.sub(r"\\infty", "inf", s)
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = re.sub(r"[{}\$]", "", s)
    return s.strip().lower().replace(" ", "")


def _to_float(s: str) -> Optional[float]:
    s = re.sub(r"[^0-9.\-/]", "", s)
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            return float(a) / float(b)
        return float(s) if s else None
    except (ValueError, ZeroDivisionError):
        return None


def check_answer(prediction: str, expected: str) -> bool:
    # prefer explicit "FINAL ANSWER: X" marker (last occurrence wins)
    fa_matches = list(re.finditer(r"FINAL ANSWER[:\s]+(.+)", prediction, re.IGNORECASE))
    if fa_matches:
        candidate = fa_matches[-1].group(1).strip()
    else:
        boxed = _extract_boxed(prediction)
        candidate = boxed if boxed else prediction.strip()

    plain_exp = _latex_to_plain(expected)
    plain_cand = _latex_to_plain(candidate)

    if plain_exp == plain_cand:
        return True
    if plain_exp and plain_exp in plain_cand:
        return True

    fexp = _to_float(plain_exp)
    fcand = _to_float(plain_cand)
    if fexp is not None and fcand is not None:
        return abs(fexp - fcand) < 1e-4

    exp_tok = set(re.findall(r"[a-z0-9]+", plain_exp))
    cand_tok = set(re.findall(r"[a-z0-9]+", plain_cand))
    if exp_tok and exp_tok == cand_tok:
        return True

    return False

# ---------------------------------------------------------------------------
# Solve a single problem
# ---------------------------------------------------------------------------

SOLVE_PROMPT = """\
Solve the following math problem step by step. At the end, state your final answer clearly as:
FINAL ANSWER: <answer>

Problem: {problem}"""


def solve(problem: str, model) -> str:
    prompt = SOLVE_PROMPT.format(problem=problem)
    return model.generate_response(prompt, temperature=0.0, max_tokens=2048)

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(problems: List[Dict], model, label: str) -> Dict:
    correct = 0
    errors = 0
    results = []
    t0 = time.perf_counter()

    for i, item in enumerate(problems):
        try:
            prediction = solve(item["problem"], model)
            ok = check_answer(prediction, item["answer"])
        except Exception as e:
            prediction = f"ERROR: {e}"
            ok = False
            errors += 1

        correct += int(ok)
        results.append({
            "problem": item["problem"][:80],
            "expected": item["answer"],
            "prediction": prediction[-300:],
            "correct": ok,
            "subject": item.get("subject", ""),
            "source": item.get("source", ""),
        })

        mark = "OK" if ok else "NO"
        subj = f"[{item.get('subject','')[:10]}]" if item.get("subject") else ""
        print(f"  [{i+1:>3}/{len(problems)}] {mark} {subj:<12} exp={item['answer'][:20]:<22} got={prediction[:40]!r}")

    elapsed = time.perf_counter() - t0
    n = len(problems)
    accuracy = correct / n if n else 0.0

    print(f"\n  {label}: {correct}/{n} correct  ({accuracy:.1%})  {elapsed:.0f}s total\n")
    return {
        "label": label,
        "correct": correct,
        "total": n,
        "accuracy": accuracy,
        "errors": errors,
        "elapsed_s": round(elapsed, 1),
        "results": results,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset",  choices=["math", "olympiad", "all"], default="math")
    parser.add_argument("--n",        type=int, default=50, help="problems per dataset")
    parser.add_argument("--model",    default="openai",
                        choices=["openai", "anthropic", "deepseek", "ollama"])
    parser.add_argument("--subject",  nargs="+", choices=MATH_SUBJECTS,
                        help="MATH subjects to include (default: all)")
    parser.add_argument("--level",    choices=["Level 1","Level 2","Level 3","Level 4","Level 5"],
                        help="filter MATH by difficulty level")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--output",   type=Path, default=Path("data/eval_results.json"))
    args = parser.parse_args()

    # Load model
    key_map = {
        "openai":    "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek":  "DEEPSEEK_API_KEY",
        "ollama":    "",
    }
    api_key = os.getenv(key_map[args.model], "")
    if not api_key and args.model != "ollama":
        sys.exit(f"{key_map[args.model]} not set — check your .env file.")
    model = ModelFactory.create_model(args.model, api_key)
    print(f"\nModel: {args.model}  |  Dataset: {args.dataset}  |  N: {args.n}\n")

    all_results = []

    if args.dataset in ("math", "all"):
        print("Loading MATH benchmark...")
        problems = load_math(args.n, subjects=args.subject, level=args.level, seed=args.seed)
        label = f"MATH" + (f" {args.level}" if args.level else "")
        r = evaluate(problems, model, label)
        all_results.append(r)

    if args.dataset in ("olympiad", "all"):
        print("Loading OlympiadBench...")
        problems = load_olympiad(args.n, seed=args.seed)
        r = evaluate(problems, model, "OlympiadBench")
        all_results.append(r)

    # Summary
    print("=" * 60)
    print(f"  {'Dataset':<25} {'Accuracy':>10} {'Correct':>10} {'Time':>8}")
    print(f"  {'-'*55}")
    for r in all_results:
        print(f"  {r['label']:<25} {r['accuracy']:>10.1%} "
              f"{r['correct']:>4}/{r['total']:<5} {r['elapsed_s']:>6.0f}s")
    print("=" * 60)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {args.output}\n")


if __name__ == "__main__":
    main()
