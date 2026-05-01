"""
End-to-end comparison experiment: Baseline vs Adaptive vs PPM-Pruned MCTS.
Uses MATH Level 5 competition problems (hendrycks/competition_math via HuggingFace).
Estimated runtime: ~12-15 minutes with DeepSeek API.

Pipeline:
  Phase 1 — Collect trajectories (20 problems, adaptive MCTS, no PPM)
  Phase 2 — Train PPM on collected preference pairs  (~1 min, no API calls)
  Phase 3 — Evaluate 3 configs on 10 held-out problems:
               A. Baseline MCTS    (simple MCTS, LLM scoring)
               B. Adaptive MCTS    (keyword heuristics, no PPM)
               C. Adaptive + PPM   (Method 1: PPM pre-scoring, top_k_prune=2)

Usage:
    python experiments/run_experiment.py
    python experiments/run_experiment.py --train 15 --test 8
    python experiments/run_experiment.py --subjects algebra number_theory
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.core.mcts import MCTS, MCTSConfig
from src.core.ppm import PPMConfig, PPMTrainer, ProcessPreferenceModel
from src.model.model_interface import LocalEmbedder, ModelFactory

# ---------------------------------------------------------------------------
# MATH dataset loader
# ---------------------------------------------------------------------------

# Subjects with mostly numeric/expression answers — best for answer checking
DEFAULT_SUBJECTS = ["algebra", "number_theory", "counting_and_probability"]
ALL_SUBJECTS     = ["algebra", "counting_and_probability", "geometry",
                    "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]


def _extract_boxed(solution: str) -> str:
    """Extract the content of the last \\boxed{...} in a LaTeX solution string.
    Handles nested braces correctly.
    """
    matches = list(re.finditer(r"\\boxed\{", solution))
    if not matches:
        return ""
    start = matches[-1].end()
    depth, i = 1, start
    while i < len(solution) and depth > 0:
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
        i += 1
    return solution[start : i - 1].strip()


def _latex_to_plain(s: str) -> str:
    """Best-effort conversion of LaTeX math to a plain comparable string."""
    s = re.sub(r"\\dfrac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt", "sqrt", s)
    s = re.sub(r"\\cdot", "*", s)
    s = re.sub(r"\\times", "*", s)
    s = re.sub(r"\\pi", "pi", s)
    s = re.sub(r"\\infty", "inf", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)   # strip remaining commands
    s = re.sub(r"[{}\$]", "", s)        # strip braces and dollar signs
    return s.strip().lower().replace(" ", "")


def load_math_level5(
    subjects: List[str] = DEFAULT_SUBJECTS,
    n_total: int = 40,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Load Level 5 problems from the MATH benchmark.

    Returns a list of dicts with keys: problem, answer, subject.
    Filters to problems whose boxed answer can be extracted and is non-empty.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Install HuggingFace datasets:  pip install datasets")

    pool: List[Dict[str, str]] = []
    print(f"  Loading MATH Level 5 from subjects: {subjects}")
    for subj in subjects:
        try:
            ds_train = load_dataset("EleutherAI/hendrycks_math", subj, split="train")
            ds_test  = load_dataset("EleutherAI/hendrycks_math", subj, split="test")
            combined = list(ds_train) + list(ds_test)
        except Exception as exc:
            print(f"  Warning: could not load subject '{subj}': {exc}")
            continue
        for item in combined:
            if item.get("level") != "Level 5":
                continue
            answer = _extract_boxed(item.get("solution", ""))
            if not answer:
                continue
            pool.append({
                "problem": item["problem"].strip(),
                "answer":  answer,
                "subject": subj,
            })

    if not pool:
        sys.exit("No Level 5 problems found. Check dataset availability.")

    random.seed(seed)
    random.shuffle(pool)
    selected = pool[:n_total]
    print(f"  Loaded {len(pool)} Level 5 problems total; using {len(selected)}")
    return selected


# ---------------------------------------------------------------------------
# Answer checking
# ---------------------------------------------------------------------------

def _extract_candidate(mcts_output: str) -> str:
    """Pull the answer from FINAL ANSWER: ... or fall back to last line."""
    fa = re.search(r"final answer[:\s]+(.+)", mcts_output, re.IGNORECASE)
    if fa:
        return fa.group(1).strip()
    lines = [l.strip() for l in mcts_output.splitlines() if l.strip()]
    return lines[-1] if lines else mcts_output.strip()


def check_answer(mcts_output: str, expected: str) -> bool:
    """Multi-strategy answer comparison for LaTeX competition math."""
    candidate = _extract_candidate(mcts_output)

    plain_exp = _latex_to_plain(expected)
    plain_cand = _latex_to_plain(candidate)

    # 1. Plain-text exact match
    if plain_exp == plain_cand:
        return True

    # 2. Substring match (candidate contains expected)
    if plain_exp and plain_exp in plain_cand:
        return True

    # 3. Numeric comparison (works for integer / decimal answers)
    def _to_float(s: str) -> Optional[float]:
        s = re.sub(r"[^0-9.\-/]", "", s)
        try:
            if "/" in s:
                a, b = s.split("/", 1)
                return float(a) / float(b)
            return float(s)
        except (ValueError, ZeroDivisionError):
            return None

    fexp = _to_float(plain_exp)
    fcand = _to_float(plain_cand)
    if fexp is not None and fcand is not None:
        return abs(fexp - fcand) < 1e-4

    # 4. LaTeX-normalized match (strip all formatting, compare tokens)
    def _tokens(s: str) -> set:
        return set(re.findall(r"[a-z0-9]+", s.lower()))

    exp_tok = _tokens(plain_exp)
    cand_tok = _tokens(plain_cand)
    if exp_tok and exp_tok == cand_tok:
        return True

    return False


# ---------------------------------------------------------------------------
# Phase 1 — Trajectory collection
# ---------------------------------------------------------------------------

def collect_trajectories(
    problems: List[Dict[str, str]],
    model,
    simulations: int = 3,
) -> List[Dict[str, Any]]:
    config = MCTSConfig(
        search_strategy="adaptive",
        max_simulations=simulations,
        max_depth=4,
        num_actions=3,
        eval_cache=True,
        max_state_steps=8,
        max_retries=3,
        retry_delay=2.0,
    )
    records = []
    for i, item in enumerate(problems):
        subj = item.get("subject", "")
        print(f"  [{i+1}/{len(problems)}] [{subj}] {item['problem'][:65]}...")
        mcts = MCTS(config)
        try:
            action, trajectory = mcts.search(item["problem"], model)
            reward = mcts.last_stats.get("best_reward", 0.0)
            records.append({
                "problem": item["problem"],
                "final_action": action,
                "final_reward": reward,
                "trajectory": trajectory,
            })
            print(f"         reward={reward:.2f}  "
                  f"api_calls={mcts.last_stats.get('model_calls',0)}  "
                  f"cache_hits={mcts.last_stats.get('cache_hits',0)}")
        except Exception as exc:
            print(f"         ERROR: {exc}")
    return records


def build_preference_pairs(records: List[Dict]) -> List[Dict[str, str]]:
    """Cross-problem preference pairs.

    Steps from trajectories where the model succeeded (reward >= 0.7) are
    'preferred'; steps from trajectories where the model failed (reward <= 0.4)
    are 'non_preferred'.  Pairs every preferred step with every non-preferred
    step across different problems, capped at 2000 to keep training fast.
    """
    preferred_steps: List[str] = []
    non_preferred_steps: List[str] = []

    for rec in records:
        reward = float(rec.get("final_reward", 0.0))
        steps = [
            (entry.get("action") or "").strip()
            for entry in rec.get("trajectory", [])
            if (entry.get("action") or "").strip()
        ]
        if reward >= 0.8:
            preferred_steps.extend(steps)
        elif reward <= 0.6:
            non_preferred_steps.extend(steps)

    pairs: List[Dict[str, str]] = [
        {"preferred": p, "non_preferred": n}
        for p in preferred_steps
        for n in non_preferred_steps
        if p != n
    ]
    random.shuffle(pairs)
    return pairs[:2000]


# ---------------------------------------------------------------------------
# Phase 2 — PPM training
# ---------------------------------------------------------------------------

def train_ppm(pairs: List[Dict[str, str]], epochs: int = 80) -> ProcessPreferenceModel:
    embedder   = LocalEmbedder.get()
    ppm_config = PPMConfig(
        input_dim=embedder.dim,
        hidden_dim=256,
        learning_rate=5e-4,
        batch_size=max(4, min(32, len(pairs) // 4)),
    )
    ppm     = ProcessPreferenceModel(ppm_config)
    trainer = PPMTrainer(ppm)

    random.shuffle(pairs)
    n_val      = max(1, int(len(pairs) * 0.1)) if len(pairs) > 8 else 0
    val_data   = pairs[:n_val]
    train_data = pairs[n_val:]

    print(f"  Training on {len(train_data)} pairs, validating on {len(val_data)}")
    history = trainer.train(
        train_data, embedder,
        num_epochs=epochs,
        validation_data=val_data if val_data else None,
    )
    print(f"  Final train loss: {history['train_losses'][-1]:.4f}", end="")
    if history.get("val_losses"):
        print(f"  |  val loss: {history['val_losses'][-1]:.4f}", end="")
    print()
    return ppm


# ---------------------------------------------------------------------------
# Phase 3 — Evaluation
# ---------------------------------------------------------------------------

def run_config(
    label: str,
    config: MCTSConfig,
    problems: List[Dict[str, str]],
    model,
    ppm: Optional[ProcessPreferenceModel] = None,
) -> Dict[str, Any]:
    correct, total_calls, total_latency, total_ppm_pruned = 0, 0, 0.0, 0
    details: List[Dict] = []

    for item in problems:
        mcts = MCTS(config)
        if ppm is not None:
            mcts.set_ppm(ppm)
        t0 = time.perf_counter()
        try:
            action, _ = mcts.search(item["problem"], model)
            latency   = time.perf_counter() - t0
            ok        = check_answer(action, item["answer"])
            correct        += int(ok)
            total_calls    += mcts.last_stats.get("model_calls", 0)
            total_latency  += latency
            total_ppm_pruned += mcts.last_stats.get("ppm_pruned", 0)
            cand = _extract_candidate(action)
            details.append({"problem": item["problem"][:60], "correct": ok,
                            "got": cand[:40], "expected": item["answer"][:30],
                            "subject": item.get("subject", "")})
        except Exception as exc:
            details.append({"problem": item["problem"][:60], "correct": False,
                            "got": f"ERROR: {exc}"[:40],
                            "expected": item["answer"][:30],
                            "subject": item.get("subject", "")})
            total_latency += time.perf_counter() - t0

    n = max(1, len(problems))
    return {
        "label":       label,
        "correct":     correct,
        "total":       len(problems),
        "accuracy":    correct / n,
        "avg_calls":   total_calls / n,
        "avg_latency": total_latency / n,
        "ppm_pruned":  total_ppm_pruned,
        "details":     details,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_W = 74

def print_results(results: List[Dict[str, Any]]) -> None:
    print(f"\n{'='*_W}")
    print("  COMPARISON RESULTS — MATH Level 5")
    print(f"{'='*_W}")
    print(f"  {'Config':<30} {'Accuracy':>10} {'Avg Calls':>11} {'Avg Time':>10} {'PPM Pruned':>11}")
    print(f"  {'-'*70}")
    for r in results:
        acc = f"{r['correct']}/{r['total']} ({r['accuracy']:.0%})"
        print(f"  {r['label']:<30} {acc:>10} {r['avg_calls']:>11.1f}"
              f" {r['avg_latency']:>9.1f}s {r['ppm_pruned']:>11}")

    print(f"\n  {'-'*70}")
    print("  PER-PROBLEM BREAKDOWN")
    for r in results:
        print(f"\n  +-- {r['label']}")
        for d in r["details"]:
            mark = "OK" if d["correct"] else "NO"
            subj = f"[{d['subject'][:12]}]" if d.get("subject") else ""
            print(f"  |  {mark} {subj:<14} {d['problem']:<42}"
                  f"  exp={d['expected']:<14}  got={d['got']}")

    print(f"\n{'='*_W}")
    best = max(results, key=lambda x: (x["accuracy"], -x["avg_calls"]))
    print(f"  Best:  {best['label']}  —  {best['correct']}/{best['total']} correct")
    if len(results) >= 3:
        delta_acc   = results[2]["accuracy"] - results[0]["accuracy"]
        delta_calls = results[0]["avg_calls"] - results[2]["avg_calls"]
        print(f"  Accuracy change  (Baseline → PPM): {delta_acc:+.0%}")
        print(f"  API call saving  (Baseline → PPM): {delta_calls:+.1f} calls/problem")
    print(f"{'='*_W}\n")


def save_results(results: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Full results saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train",       type=int,   default=20,
                        help="# problems for PPM training (default 20)")
    parser.add_argument("--test",        type=int,   default=10,
                        help="# problems for evaluation (default 10)")
    parser.add_argument("--simulations", type=int,   default=3,
                        help="MCTS simulations per problem (default 3)")
    parser.add_argument("--ppm-epochs",  type=int,   default=80,
                        help="PPM training epochs (default 80)")
    parser.add_argument("--top-k",       type=int,   default=2,
                        help="top_k_prune for Config C (default 2)")
    parser.add_argument("--subjects",    nargs="+",  default=DEFAULT_SUBJECTS,
                        choices=ALL_SUBJECTS, metavar="SUBJECT",
                        help=f"MATH subjects to sample from (default: {DEFAULT_SUBJECTS})")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--output",      type=Path,
                        default=Path("data/experiment_results.json"))
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        sys.exit("DEEPSEEK_API_KEY not set — check your .env file.")
    model = ModelFactory.create_model("deepseek", api_key)

    # ── Load dataset ─────────────────────────────────────────────────────
    print(f"\n{'='*_W}")
    print("  MCTS + PPM Experiment — MATH Level 5")
    print(f"  Train: {args.train}  |  Test: {args.test}  "
          f"|  Simulations: {args.simulations}  |  PPM epochs: {args.ppm_epochs}")
    print(f"{'='*_W}\n")

    print("Loading MATH Level 5 dataset...")
    all_problems = load_math_level5(
        subjects=args.subjects,
        n_total=args.train + args.test,
        seed=args.seed,
    )
    train_problems = all_problems[:args.train]
    test_problems  = all_problems[args.train : args.train + args.test]
    print(f"  Train pool: {len(train_problems)}  |  Test pool: {len(test_problems)}\n")

    # ── Phase 1: collect trajectories ────────────────────────────────────
    t0 = time.perf_counter()
    print("Phase 1/3: Collecting trajectories (adaptive MCTS, no PPM)...")
    records = collect_trajectories(train_problems, model, simulations=args.simulations)
    pairs   = build_preference_pairs(records)
    print(f"  → {len(pairs)} preference pairs in {time.perf_counter()-t0:.0f}s\n")

    # ── Phase 2: train PPM ────────────────────────────────────────────────
    ppm: Optional[ProcessPreferenceModel] = None
    if len(pairs) >= 2:
        t1 = time.perf_counter()
        print("Phase 2/3: Training PPM (local, no API)...")
        ppm = train_ppm(pairs, epochs=args.ppm_epochs)
        ckpt = Path("checkpoints/ppm_math_level5.pt")
        ckpt.parent.mkdir(exist_ok=True)
        ppm.save_model(str(ckpt))
        print(f"  Checkpoint → {ckpt}  ({time.perf_counter()-t1:.0f}s)\n")
    else:
        print(f"Phase 2/3: Only {len(pairs)} pairs — skipping PPM training.\n"
              "  Config C will use adaptive heuristics instead of PPM.\n")

    # ── Phase 3: evaluate ────────────────────────────────────────────────
    print("Phase 3/3: Evaluating on test problems...\n")

    shared = dict(max_depth=4, num_actions=3, eval_cache=True,
                  max_state_steps=8, max_retries=3, retry_delay=2.0,
                  max_simulations=args.simulations)

    configs = [
        ("A. Baseline MCTS",           MCTSConfig(search_strategy="baseline", **shared), None),
        ("B. Adaptive MCTS",           MCTSConfig(search_strategy="adaptive", **shared), None),
        ("C. Adaptive + PPM (Meth.1)", MCTSConfig(search_strategy="adaptive",
                                                   top_k_prune=args.top_k, **shared),   ppm),
    ]

    all_results = []
    for label, cfg, use_ppm in configs:
        tag = " [PPM pre-scoring]" if use_ppm else ""
        print(f"  Running {label}{tag}...")
        r = run_config(label, cfg, test_problems, model, ppm=use_ppm)
        all_results.append(r)
        print(f"  → {r['correct']}/{r['total']} correct  "
              f"{r['avg_calls']:.1f} calls/problem  {r['avg_latency']:.1f}s/problem\n")

    print_results(all_results)
    save_results(all_results, args.output)


if __name__ == "__main__":
    main()
