"""
Evaluation script for MATH, OlympiadBench, and AMC datasets.
Supports three solving strategies: direct, mcts, mcts+ppm.

Usage:
    python eval.py --dataset math --n 50
    python eval.py --dataset math --n 50 --strategy mcts
    python eval.py --dataset math --n 50 --strategy mcts+ppm --ppm-checkpoint checkpoints/ppm.pt
    python eval.py --dataset olympiad --n 50 --strategy mcts
    python eval.py --dataset amc --amc-path data/amc12.jsonl --n 50 --strategy mcts+ppm

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

from src.model.model_interface import ModelConfig, ModelFactory
from src.data.amc import AMC12_HF_DATASET, load_amc
from src.core.diagram import diagram_summary
from src.core.exact_solvers import format_exact_final_answer, solve_geometry_inequality_exact

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


def load_amc_benchmark(
    n: int,
    path: Optional[Path] = None,
    dataset_id: str = AMC12_HF_DATASET,
    split: str = "train",
    seed: int = 42,
) -> List[Dict]:
    selected = load_amc(
        n=n,
        local_path=path,
        dataset_id=dataset_id,
        split=split,
        seed=seed,
    )
    source = str(path) if path else dataset_id
    print(f"  AMC: using {len(selected)} problems from {source}")
    return selected

# ---------------------------------------------------------------------------
# Answer checking
# ---------------------------------------------------------------------------

def _latex_to_plain(s: str) -> str:
    s = re.sub(r"\$\$(.+?)\$\$", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"\$(.+?)\$", r"\1", s)
    s = s.replace(r"\(", "").replace(r"\)", "")
    s = s.replace(r"\[", "").replace(r"\]", "")
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


def check_answer(prediction: str, expected: str, problem: str = "") -> bool:
    fa_matches = list(re.finditer(r"FINAL ANSWER[:\s]+(.+)", prediction, re.IGNORECASE))
    if fa_matches:
        candidate = fa_matches[-1].group(1).strip()
    else:
        boxed = _extract_boxed(prediction)
        candidate = boxed if boxed else prediction.strip()

    plain_exp = _latex_to_plain(expected)
    plain_cand = _latex_to_plain(candidate)

    if re.fullmatch(r"[a-e]", plain_exp):
        letter = _extract_choice_letter(candidate)
        if letter:
            return letter.lower() == plain_exp
        option_text = _extract_choice_text(problem, plain_exp.upper())
        if option_text and _plain_answer_match(candidate, option_text):
            return True

    if _plain_answer_match(candidate, expected):
        return True

    return False


def _plain_answer_match(candidate: str, expected: str) -> bool:
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


def _extract_choice_letter(text: str) -> str:
    boxed = _extract_boxed(text)
    target = boxed or text
    final = re.search(r"final\s+answer[:\s]+(?:\\boxed\{)?\(?([A-E])\)?", target, re.IGNORECASE)
    if final:
        return final.group(1).upper()
    standalone = re.search(r"(?:answer\s+is|choose|option|choice)\s+\(?([A-E])\)?", target, re.IGNORECASE)
    if standalone:
        return standalone.group(1).upper()
    compact = target.strip().upper()
    return compact.strip("()") if re.fullmatch(r"\(?[A-E]\)?", compact) else ""


def _extract_choice_text(problem: str, letter: str) -> str:
    labels = list(re.finditer(r"(?:\\(?:textbf|mathrm|text)\s*\{\s*)?\(([A-E])\)\s*(?:\})?", problem))
    if not labels:
        return ""
    for idx, match in enumerate(labels):
        if match.group(1).upper() != letter.upper():
            continue
        start = match.end()
        end = labels[idx + 1].start() if idx + 1 < len(labels) else len(problem)
        text = problem[start:end]
        text = re.sub(r"\\qquad|\\quad", " ", text)
        text = text.strip(" $,\n\t")
        return text.strip()
    return ""

# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

SOLVE_PROMPT = """\
Solve the following math problem step by step. At the end, state your final answer clearly as:
FINAL ANSWER: <answer>

If the problem is multiple choice, compute the mathematical value first, then
map it to the matching answer choice and write the final answer as the choice
letter followed by the value, e.g. FINAL ANSWER: D, 1/7.

Diagram context:
{diagram_context}

Problem: {problem}"""


def solve_direct(problem: str, model) -> tuple[str, dict]:
    prompt = SOLVE_PROMPT.format(problem=problem, diagram_context=diagram_summary(problem) or "No parsed diagram context.")
    answer = model.generate_response(prompt, temperature=0.0, max_tokens=2048)
    return answer, {}


def solve_mcts(problem: str, model, simulations: int = 3, ppm=None, top_k: int = 2) -> tuple[str, dict]:
    from src.core.mcts import MCTS, MCTSConfig
    cfg = MCTSConfig(
        search_strategy="adaptive",
        max_simulations=simulations,
        max_depth=4,
        num_actions=2,
        eval_cache=True,
        max_state_steps=8,
        max_retries=3,
        retry_delay=2.0,
        top_k_prune=top_k if ppm else 0,
        generation_temperature=0.0,
        seed=20240509,
    )
    mcts = MCTS(cfg)
    if ppm is not None:
        mcts.set_ppm(ppm)
    _, trajectory = mcts.search(problem, model)
    stats = {
        "api_calls": mcts.last_stats.get("model_calls", 0),
        "ppm_pruned": mcts.last_stats.get("ppm_pruned", 0),
        "best_reward": mcts.last_stats.get("best_reward", 0.0),
    }

    # 从 trajectory 中找包含 FINAL ANSWER 的完整推理状态
    # trajectory 按时间顺序记录，取最后一个含 FINAL ANSWER 的 state
    full_solution = ""
    for entry in reversed(trajectory):
        state = entry.get("state", "")
        if "final answer" in state.lower():
            full_solution = state
            break

    # 如果 MCTS 没到终止状态，追加一步让模型直接给出答案
    if not full_solution:
        best_state = trajectory[-1].get("state", problem) if trajectory else problem
        followup = (
            f"{best_state}\n\n"
            f"Diagram context:\n{diagram_summary(problem) or 'No parsed diagram context.'}\n\n"
            f"Complete the solution above and state the answer on the last line exactly as:\n"
            f"FINAL ANSWER: <your answer here>\n"
            f"For multiple-choice problems, include the option letter and the matching value.\n"
            f"Do not add any text after FINAL ANSWER."
        )
        full_solution = model.generate_response(followup, temperature=0.0, max_tokens=512)
        stats["api_calls"] += 1

    return full_solution, stats


def load_ppm(checkpoint: str) -> object:
    from src.core.ppm import load_ppm_checkpoint
    ppm = load_ppm_checkpoint(checkpoint)
    print(f"  PPM loaded from {checkpoint}")
    return ppm

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(problems: List[Dict], model, label: str, strategy: str,
             simulations: int = 3, ppm=None, top_k: int = 2) -> Dict:
    correct = 0
    errors = 0
    total_api_calls = 0
    results = []
    t0 = time.perf_counter()

    for i, item in enumerate(problems):
        try:
            exact_solution = solve_geometry_inequality_exact(item["problem"])
            if exact_solution is not None:
                prediction = format_exact_final_answer(exact_solution)
                stats = {
                    "api_calls": 0,
                    "exact_solver": exact_solution.get("method", "geometry inequality exact solver"),
                }
            elif strategy == "direct":
                prediction, stats = solve_direct(item["problem"], model)
            else:
                prediction, stats = solve_mcts(item["problem"], model,
                                               simulations=simulations, ppm=ppm, top_k=top_k)
            ok = check_answer(prediction, item["answer"], problem=item["problem"])
            total_api_calls += stats.get("api_calls", 1)
        except Exception as e:
            prediction = f"ERROR: {e}"
            stats = {}
            ok = False
            errors += 1
            total_api_calls += 1

        correct += int(ok)
        results.append({
            "problem": item["problem"][:80],
            "expected": item["answer"],
            "prediction": prediction[-300:],
            "correct": ok,
            "subject": item.get("subject", ""),
            "source": item.get("source", ""),
            "stats": stats,
        })

        mark = "OK" if ok else "NO"
        subj = f"[{item.get('subject','')[:10]}]" if item.get("subject") else ""
        extra = f"  calls={stats.get('api_calls','')}" if strategy != "direct" else ""
        # 显示提取出的最终答案，而非完整推理过程的开头
        fa = list(re.finditer(r"FINAL ANSWER[:\s]+(.+)", prediction, re.IGNORECASE))
        boxed = _extract_boxed(prediction)
        extracted = fa[-1].group(1).strip()[:30] if fa else (boxed[:30] if boxed else prediction[:30])
        print(f"  [{i+1:>3}/{len(problems)}] {mark} {subj:<12} exp={item['answer'][:20]:<22} got={extracted!r}{extra}")

    elapsed = time.perf_counter() - t0
    n = len(problems)
    accuracy = correct / n if n else 0.0
    avg_calls = total_api_calls / n if n else 0.0

    print(f"\n  {label}: {correct}/{n} correct  ({accuracy:.1%})  "
          f"avg_calls={avg_calls:.1f}  {elapsed:.0f}s total\n")
    return {
        "label": label,
        "strategy": strategy,
        "correct": correct,
        "total": n,
        "accuracy": accuracy,
        "avg_api_calls": round(avg_calls, 1),
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
    parser.add_argument("--dataset",    choices=["math", "olympiad", "amc", "all"], default="math")
    parser.add_argument("--n",          type=int, default=20, help="题目数量")
    parser.add_argument("--model",      default="openai",
                        choices=["openai", "anthropic", "deepseek", "ollama"])
    parser.add_argument("--openai-model", default=os.getenv("OPENAI_MODEL", "gpt-5.2"))
    parser.add_argument("--reasoning-effort", default=os.getenv("OPENAI_REASONING_EFFORT", "high"))
    parser.add_argument("--strategy",   default="direct",
                        choices=["direct", "mcts", "mcts+ppm"],
                        help="direct=直接回答  mcts=树搜索  mcts+ppm=树搜索+PPM剪枝")
    parser.add_argument("--simulations",type=int, default=3, help="MCTS模拟次数（默认3）")
    parser.add_argument("--top-k",      type=int, default=2, help="PPM保留的top-k步（默认2）")
    parser.add_argument("--ppm-checkpoint", type=str, default="checkpoints/ppm_math_level5.pt",
                        help="PPM模型路径（mcts+ppm时使用）")
    parser.add_argument("--subject",    nargs="+", choices=MATH_SUBJECTS)
    parser.add_argument("--level",      choices=["Level 1","Level 2","Level 3","Level 4","Level 5"])
    parser.add_argument("--amc-path",   type=Path, help="Local AMC .jsonl/.json/.csv file")
    parser.add_argument("--amc-dataset-id", default=AMC12_HF_DATASET)
    parser.add_argument("--amc-split",  default="train")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--output",     type=Path, default=Path("data/eval_results.json"))
    args = parser.parse_args()

    # 加载模型
    key_map = {
        "openai":    "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek":  "DEEPSEEK_API_KEY",
        "ollama":    "",
    }
    api_key = os.getenv(key_map[args.model], "")
    if not api_key and args.model != "ollama":
        sys.exit(f"{key_map[args.model]} not set — check your .env file.")
    model_kwargs = {}
    if args.model == "openai":
        model_kwargs["config"] = ModelConfig(
            model=args.openai_model,
            temperature=0.0,
            max_tokens=1800,
            timeout=90.0,
            reasoning_effort=args.reasoning_effort,
        )
    model = ModelFactory.create_model(args.model, api_key, **model_kwargs)

    # 加载 PPM（如需要）
    ppm = None
    if args.strategy == "mcts+ppm":
        ckpt = ROOT / args.ppm_checkpoint
        if not ckpt.exists():
            sys.exit(f"PPM checkpoint not found: {ckpt}\n"
                     f"请先运行 run_experiment.py 训练 PPM，或改用 --strategy mcts")
        ppm = load_ppm(str(ckpt))

    print(f"\nModel: {args.model}  |  Strategy: {args.strategy}  |  "
          f"Dataset: {args.dataset}  |  N: {args.n}")
    if args.strategy != "direct":
        print(f"MCTS simulations: {args.simulations}" +
              (f"  |  PPM top-k: {args.top_k}" if ppm else ""))
    print()

    all_results = []

    if args.dataset in ("math", "all"):
        print("Loading MATH benchmark...")
        problems = load_math(args.n, subjects=args.subject, level=args.level, seed=args.seed)
        label = f"MATH [{args.strategy}]" + (f" {args.level}" if args.level else "")
        r = evaluate(problems, model, label, args.strategy,
                     simulations=args.simulations, ppm=ppm, top_k=args.top_k)
        all_results.append(r)

    if args.dataset in ("olympiad", "all"):
        print("Loading OlympiadBench...")
        problems = load_olympiad(args.n, seed=args.seed)
        label = f"OlympiadBench [{args.strategy}]"
        r = evaluate(problems, model, label, args.strategy,
                     simulations=args.simulations, ppm=ppm, top_k=args.top_k)
        all_results.append(r)

    if args.dataset in ("amc", "all"):
        print("Loading AMC benchmark...")
        problems = load_amc_benchmark(
            args.n,
            path=args.amc_path,
            dataset_id=args.amc_dataset_id,
            split=args.amc_split,
            seed=args.seed,
        )
        label = f"AMC [{args.strategy}]"
        r = evaluate(problems, model, label, args.strategy,
                     simulations=args.simulations, ppm=ppm, top_k=args.top_k)
        all_results.append(r)

    # 汇总
    print("=" * 65)
    print(f"  {'Dataset':<30} {'Accuracy':>10} {'Correct':>10} {'Avg Calls':>10} {'Time':>6}")
    print(f"  {'-'*60}")
    for r in all_results:
        print(f"  {r['label']:<30} {r['accuracy']:>10.1%} "
              f"{r['correct']:>4}/{r['total']:<5} "
              f"{r['avg_api_calls']:>10.1f} "
              f"{r['elapsed_s']:>5.0f}s")
    print("=" * 65)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {args.output}\n")


if __name__ == "__main__":
    main()
