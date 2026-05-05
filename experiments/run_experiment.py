"""
End-to-end comparison experiment: Baseline vs Adaptive vs PPM-Pruned MCTS.

Supports multiple math training datasets with optional curriculum ordering.

Pipeline:
  Phase 1 — Collect trajectories (adaptive MCTS, no PPM)
  Phase 2 — Train ContextAwarePPM on collected preference pairs (~1 min, no API)
  Phase 3 — Evaluate 3 configs on held-out problems:
               A. Baseline MCTS    (simple MCTS, LLM scoring)
               B. Adaptive MCTS    (keyword heuristics, no PPM)
               C. Adaptive + PPM   (PPM pre-scoring + context-aware evaluation)

Usage:
    python experiments/run_experiment.py
    python experiments/run_experiment.py --train 40 --test 15 --dataset math_all gsm8k
    python experiments/run_experiment.py --dataset math_all olympiad aime --curriculum
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
from src.core.ppm import PPMConfig, PPMTrainer, ProcessPreferenceModel, ContextAwarePPM
from src.model.model_interface import LocalEmbedder, ModelFactory

# ── Dataset registry ──────────────────────────────────────────────────────────

DEFAULT_SUBJECTS = ["algebra", "number_theory", "counting_and_probability"]
ALL_SUBJECTS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]

DATASET_REGISTRY: Dict[str, str] = {
    "math_l5":  "MATH Level 5 only (original setting)",
    "math_all": "All MATH levels 1-5, all 7 subjects",
    "gsm8k":    "GSM8K grade-school math (easy warm-up for curriculum)",
    "numina":   "NuminaMath-CoT (large-scale, randomly sampled subset)",
    "olympiad": "OlympiadBench competition problems",
    "aime":     "AIME 1983-2024 historical problems",
}


# ── Answer extraction helpers ─────────────────────────────────────────────────

def _extract_boxed(solution: str) -> str:
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
    s = re.sub(r"\\dfrac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt", "sqrt", s)
    s = re.sub(r"\\cdot", "*", s)
    s = re.sub(r"\\times", "*", s)
    s = re.sub(r"\\pi", "pi", s)
    s = re.sub(r"\\infty", "inf", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = re.sub(r"[{}\$]", "", s)
    return s.strip().lower().replace(" ", "")


# ── Per-dataset loaders ───────────────────────────────────────────────────────

def _load_math_all(subjects: List[str], seed: int) -> List[Dict[str, Any]]:
    """All MATH levels 1-5 across specified subjects."""
    from datasets import load_dataset
    pool: List[Dict[str, Any]] = []
    for subj in subjects:
        for split in ("train", "test"):
            try:
                ds = load_dataset("EleutherAI/hendrycks_math", subj, split=split)
            except Exception as exc:
                print(f"  Warning: {subj}/{split}: {exc}")
                continue
            for item in ds:
                answer = _extract_boxed(item.get("solution", ""))
                if not answer:
                    continue
                try:
                    level = int(item.get("level", "Level 3").split()[-1])
                except ValueError:
                    level = 3
                pool.append({
                    "problem": item["problem"].strip(),
                    "answer": answer,
                    "subject": subj,
                    "source": "math_all",
                    "difficulty": level,
                })
    return pool


def load_math_level5(
    subjects: List[str] = DEFAULT_SUBJECTS,
    n_total: int = 40,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Original loader — MATH Level 5 only (kept for backward compatibility)."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Install HuggingFace datasets:  pip install datasets")

    pool: List[Dict[str, Any]] = []
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
                "source":  "math_l5",
                "difficulty": 5,
            })

    if not pool:
        sys.exit("No Level 5 problems found.")

    random.seed(seed)
    random.shuffle(pool)
    selected = pool[:n_total]
    print(f"  Loaded {len(pool)} Level 5 problems total; using {len(selected)}")
    return selected


def _load_gsm8k(seed: int) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    try:
        from datasets import load_dataset
        for split in ("train", "test"):
            ds = load_dataset("openai/gsm8k", "main", split=split)
            for item in ds:
                m = re.search(r"####\s*(.+)", item.get("answer", ""))
                answer = m.group(1).strip() if m else ""
                if answer:
                    pool.append({
                        "problem": item["question"].strip(),
                        "answer": answer,
                        "subject": "gsm8k",
                        "source": "gsm8k",
                        "difficulty": 1,
                    })
    except Exception as exc:
        print(f"  Warning: GSM8K load failed: {exc}")
    return pool


def _load_numina(seed: int, max_n: int = 5000) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    try:
        from datasets import load_dataset
        ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
        rng = random.Random(seed)
        sampled: List[Any] = []
        for item in ds:
            sampled.append(item)
            if len(sampled) >= max_n * 5:
                break
        rng.shuffle(sampled)
        for item in sampled[:max_n]:
            answer = _extract_boxed(item.get("solution", ""))
            if not answer:
                continue
            pool.append({
                "problem": item["problem"].strip(),
                "answer": answer,
                "subject": item.get("source", "numina"),
                "source": "numina",
                "difficulty": 3,
            })
    except Exception as exc:
        print(f"  Warning: NuminaMath-CoT load failed: {exc}")
    return pool


def _load_olympiad(seed: int) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    try:
        from datasets import load_dataset
        ds = load_dataset("math-ai/OlympiadBench", split="train")
        for item in ds:
            problem = item.get("problem", item.get("question", "")).strip()
            answer = str(item.get("answer", item.get("final_answer", ""))).strip()
            if problem and answer:
                pool.append({
                    "problem": problem,
                    "answer": answer,
                    "subject": "olympiad",
                    "source": "olympiad",
                    "difficulty": 5,
                })
    except Exception as exc:
        print(f"  Warning: OlympiadBench load failed: {exc}")
    return pool


def _load_aime(seed: int) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    for ds_name in ("Maxwell-Jia/AIME_1983_2024", "di-zhang-fdu/AIME_1983_2024"):
        try:
            from datasets import load_dataset
            ds = load_dataset(ds_name, split="train")
            for item in ds:
                problem = item.get("Problem", item.get("problem", "")).strip()
                answer = str(item.get("Answer", item.get("answer", ""))).strip()
                if problem and answer:
                    pool.append({
                        "problem": problem,
                        "answer": answer,
                        "subject": "aime",
                        "source": "aime",
                        "difficulty": 6,
                    })
            break
        except Exception as exc:
            print(f"  Warning: {ds_name} load failed: {exc}")
    return pool


def load_multi_dataset(
    sources: List[str] = ("math_l5",),
    subjects: List[str] = DEFAULT_SUBJECTS,
    n_total: int = 200,
    seed: int = 42,
    curriculum: bool = False,
) -> List[Dict[str, Any]]:
    """Load and merge problems from multiple math datasets.

    sources    : list drawn from DATASET_REGISTRY keys
    n_total    : total problems to return (sampled uniformly across sources)
    curriculum : if True, sort ascending by difficulty before returning
                 (easy → hard; enables curriculum learning during trajectory collection)
    """
    pool: List[Dict[str, Any]] = []
    print(f"  Datasets requested: {sources}")

    _loaders = {
        "math_all": lambda: _load_math_all(ALL_SUBJECTS, seed),
        "math_l5":  lambda: load_math_level5(subjects, n_total=99999, seed=seed),
        "gsm8k":    lambda: _load_gsm8k(seed),
        "numina":   lambda: _load_numina(seed),
        "olympiad": lambda: _load_olympiad(seed),
        "aime":     lambda: _load_aime(seed),
    }

    for src in sources:
        key = src.lower()
        if key not in _loaders:
            print(f"  Warning: unknown source '{src}' — choose from {list(DATASET_REGISTRY)}")
            continue
        sub = _loaders[key]()
        print(f"  {key}: {len(sub)} problems loaded")
        pool.extend(sub)

    if not pool:
        print("  Warning: all loaders failed — falling back to MATH Level 5 defaults")
        return load_math_level5(subjects, n_total=n_total, seed=seed)

    if curriculum:
        pool.sort(key=lambda x: x.get("difficulty", 3))
        print(f"  Curriculum mode: difficulty {pool[0].get('difficulty')} → {pool[-1].get('difficulty')}")
    else:
        random.seed(seed)
        random.shuffle(pool)

    selected = pool[:n_total]
    src_counts = {}
    for p in selected:
        src_counts[p.get("source", "?")] = src_counts.get(p.get("source", "?"), 0) + 1
    print(f"  Total pool: {len(pool)} | Using: {len(selected)} — {src_counts}")
    return selected


# ── Answer checking ───────────────────────────────────────────────────────────

def _extract_candidate(mcts_output: str) -> str:
    fa = re.search(r"final answer[:\s]+(.+)", mcts_output, re.IGNORECASE)
    if fa:
        return fa.group(1).strip()
    lines = [l.strip() for l in mcts_output.splitlines() if l.strip()]
    return lines[-1] if lines else mcts_output.strip()


def check_answer(mcts_output: str, expected: str) -> bool:
    candidate = _extract_candidate(mcts_output)
    plain_exp = _latex_to_plain(expected)
    plain_cand = _latex_to_plain(candidate)

    if plain_exp == plain_cand:
        return True
    if plain_exp and plain_exp in plain_cand:
        return True

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

    def _tokens(s: str) -> set:
        return set(re.findall(r"[a-z0-9]+", s.lower()))

    if (exp_tok := _tokens(plain_exp)) and exp_tok == _tokens(plain_cand):
        return True
    return False


# ── Phase 1: trajectory collection ───────────────────────────────────────────

def collect_trajectories(
    problems: List[Dict[str, Any]],
    model: Any,
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
    records: List[Dict[str, Any]] = []
    for i, item in enumerate(problems):
        subj = item.get("subject", "")
        src  = item.get("source", "")
        print(f"  [{i+1}/{len(problems)}] [{src}/{subj}] {item['problem'][:60]}...")
        mcts = MCTS(config)
        try:
            action, trajectory = mcts.search(item["problem"], model)
            reward = mcts.last_stats.get("best_reward", 0.0)
            records.append({
                "problem": item["problem"],
                "final_action": action,
                "final_reward": reward,
                "trajectory": trajectory,
                "source": src,
            })
            print(f"         reward={reward:.2f}  "
                  f"api_calls={mcts.last_stats.get('model_calls',0)}  "
                  f"cache_hits={mcts.last_stats.get('cache_hits',0)}")
        except Exception as exc:
            print(f"         ERROR: {exc}")
    return records


def build_preference_pairs(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Cross-problem preference pairs with reward metadata for soft margin loss.

    Each pair now carries:
      preferred_reward / non_preferred_reward  — used by Innovation 2
      problem                                  — used by ContextAwarePPM (Innovation 1)
    """
    preferred_pool:     List[Dict[str, Any]] = []
    non_preferred_pool: List[Dict[str, Any]] = []

    for rec in records:
        reward = float(rec.get("final_reward", 0.0))
        problem = rec.get("problem", "")
        steps = [
            (entry.get("action") or "").strip()
            for entry in rec.get("trajectory", [])
            if (entry.get("action") or "").strip()
        ]
        if reward >= 0.8:
            for s in steps:
                preferred_pool.append({"step": s, "reward": reward, "problem": problem})
        elif reward <= 0.6:
            for s in steps:
                non_preferred_pool.append({"step": s, "reward": reward, "problem": problem})

    pairs: List[Dict[str, Any]] = [
        {
            "preferred":             p["step"],
            "non_preferred":         n["step"],
            "preferred_reward":      p["reward"],
            "non_preferred_reward":  n["reward"],
            "problem":               p["problem"],
        }
        for p in preferred_pool
        for n in non_preferred_pool
        if p["step"] != n["step"]
    ]
    random.shuffle(pairs)
    return pairs[:2000]


# ── Phase 2: PPM training ─────────────────────────────────────────────────────

def train_ppm(pairs: List[Dict[str, Any]], epochs: int = 80) -> ProcessPreferenceModel:
    """Train a ContextAwarePPM with all innovations enabled."""
    embedder = LocalEmbedder.get()
    ppm_config = PPMConfig(
        hidden_dim=256,
        learning_rate=5e-4,
        batch_size=max(4, min(32, len(pairs) // 4)),
        dropout_rate=0.1,
        use_scheduler=True,
        scheduler_T_max=epochs,
        use_hard_negatives=len(pairs) >= 32,
        hard_negative_ratio=0.3,
    )
    ppm = ContextAwarePPM(embedding_dim=embedder.dim, config=ppm_config)
    trainer = PPMTrainer(ppm)

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * 0.1)) if len(pairs) > 8 else 0
    val_data   = pairs[:n_val]
    train_data = pairs[n_val:]

    print(f"  ContextAwarePPM | train={len(train_data)} val={n_val} | "
          f"hard_negatives={ppm_config.use_hard_negatives} scheduler=cosine")
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


# ── Phase 3: evaluation ───────────────────────────────────────────────────────

def run_config(
    label: str,
    config: MCTSConfig,
    problems: List[Dict[str, Any]],
    model: Any,
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
            latency = time.perf_counter() - t0
            ok = check_answer(action, item["answer"])
            correct            += int(ok)
            total_calls        += mcts.last_stats.get("model_calls", 0)
            total_latency      += latency
            total_ppm_pruned   += mcts.last_stats.get("ppm_pruned", 0)
            details.append({
                "problem":  item["problem"][:60],
                "correct":  ok,
                "got":      _extract_candidate(action)[:40],
                "expected": item["answer"][:30],
                "subject":  item.get("subject", ""),
                "source":   item.get("source", ""),
            })
        except Exception as exc:
            details.append({
                "problem":  item["problem"][:60],
                "correct":  False,
                "got":      f"ERROR: {exc}"[:40],
                "expected": item["answer"][:30],
                "subject":  item.get("subject", ""),
                "source":   item.get("source", ""),
            })
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


# ── Reporting ─────────────────────────────────────────────────────────────────

_W = 76

def print_results(results: List[Dict[str, Any]]) -> None:
    print(f"\n{'='*_W}")
    print("  COMPARISON RESULTS — MATH Reasoning")
    print(f"{'='*_W}")
    print(f"  {'Config':<32} {'Accuracy':>10} {'Avg Calls':>11} {'Avg Time':>10} {'PPM Pruned':>11}")
    print(f"  {'-'*72}")
    for r in results:
        acc = f"{r['correct']}/{r['total']} ({r['accuracy']:.0%})"
        print(f"  {r['label']:<32} {acc:>10} {r['avg_calls']:>11.1f}"
              f" {r['avg_latency']:>9.1f}s {r['ppm_pruned']:>11}")

    print(f"\n  {'-'*72}")
    print("  PER-PROBLEM BREAKDOWN")
    for r in results:
        print(f"\n  +-- {r['label']}")
        for d in r["details"]:
            mark = "OK" if d["correct"] else "NO"
            tag = f"[{d.get('source','')}/{d['subject'][:8]}]"
            print(f"  |  {mark} {tag:<18} {d['problem']:<40}"
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train", type=int, default=20,
                        help="# problems for PPM training (default 20)")
    parser.add_argument("--test", type=int, default=10,
                        help="# problems for evaluation (default 10)")
    parser.add_argument("--simulations", type=int, default=3,
                        help="MCTS simulations per problem (default 3)")
    parser.add_argument("--ppm-epochs", type=int, default=80,
                        help="PPM training epochs (default 80)")
    parser.add_argument("--top-k", type=int, default=2,
                        help="top_k_prune for Config C (default 2)")
    parser.add_argument(
        "--dataset", nargs="+", default=["math_l5"],
        choices=list(DATASET_REGISTRY),
        metavar="DATASET",
        help=(
            f"Training datasets to merge (default: math_l5). "
            f"Choices: {list(DATASET_REGISTRY)}"
        ),
    )
    parser.add_argument("--curriculum", action="store_true",
                        help="Sort training problems by difficulty (easy → hard)")
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS,
                        choices=ALL_SUBJECTS, metavar="SUBJECT",
                        help=f"MATH subjects when using math_l5/math_all (default: {DEFAULT_SUBJECTS})")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=Path("data/experiment_results.json"))
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        sys.exit("DEEPSEEK_API_KEY not set — check your .env file.")
    model = ModelFactory.create_model("deepseek", api_key)

    print(f"\n{'='*_W}")
    print("  MCTS + ContextAwarePPM Experiment — Math Reasoning")
    print(f"  Train: {args.train}  |  Test: {args.test}  "
          f"|  Simulations: {args.simulations}  |  PPM epochs: {args.ppm_epochs}")
    print(f"  Datasets: {args.dataset}  |  Curriculum: {args.curriculum}")
    print(f"{'='*_W}\n")

    print("Loading training dataset...")
    all_problems = load_multi_dataset(
        sources=args.dataset,
        subjects=args.subjects,
        n_total=args.train + args.test,
        seed=args.seed,
        curriculum=args.curriculum,
    )
    train_problems = all_problems[:args.train]
    test_problems  = all_problems[args.train : args.train + args.test]
    print(f"  Train pool: {len(train_problems)}  |  Test pool: {len(test_problems)}\n")

    # Phase 1
    t0 = time.perf_counter()
    print("Phase 1/3: Collecting trajectories (adaptive MCTS, no PPM)...")
    records = collect_trajectories(train_problems, model, simulations=args.simulations)
    pairs   = build_preference_pairs(records)
    print(f"  → {len(pairs)} preference pairs in {time.perf_counter()-t0:.0f}s\n")

    # Phase 2
    ppm: Optional[ProcessPreferenceModel] = None
    if len(pairs) >= 2:
        t1 = time.perf_counter()
        print("Phase 2/3: Training ContextAwarePPM (local, no API)...")
        ppm = train_ppm(pairs, epochs=args.ppm_epochs)
        ckpt = Path("checkpoints/ppm_math.pt")
        ckpt.parent.mkdir(exist_ok=True)
        ppm.save_model(str(ckpt))
        print(f"  Checkpoint → {ckpt}  ({time.perf_counter()-t1:.0f}s)\n")
    else:
        print(f"Phase 2/3: Only {len(pairs)} pairs — skipping PPM training.\n"
              "  Config C will use adaptive heuristics instead of PPM.\n")

    # Phase 3
    print("Phase 3/3: Evaluating on test problems...\n")
    shared = dict(max_depth=4, num_actions=3, eval_cache=True,
                  max_state_steps=8, max_retries=3, retry_delay=2.0,
                  max_simulations=args.simulations)

    configs = [
        ("A. Baseline MCTS",           MCTSConfig(search_strategy="baseline", **shared), None),
        ("B. Adaptive MCTS",           MCTSConfig(search_strategy="adaptive", **shared), None),
        ("C. Adaptive + ContextPPM",   MCTSConfig(search_strategy="adaptive",
                                                   top_k_prune=args.top_k, **shared), ppm),
    ]

    all_results: List[Dict[str, Any]] = []
    for label, cfg, use_ppm in configs:
        tag = " [ContextAwarePPM]" if use_ppm else ""
        print(f"  Running {label}{tag}...")
        r = run_config(label, cfg, test_problems, model, ppm=use_ppm)
        all_results.append(r)
        print(f"  → {r['correct']}/{r['total']} correct  "
              f"{r['avg_calls']:.1f} calls/problem  {r['avg_latency']:.1f}s/problem\n")

    print_results(all_results)
    save_results(all_results, args.output)


if __name__ == "__main__":
    main()
