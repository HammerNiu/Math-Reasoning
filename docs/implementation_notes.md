# Math Reasoning with MCTS + PPM — Implementation Notes

> Project: rStar-Math inspired math reasoning system
> Course: COMS 6895 Advanced Big Data and AI

---

## 1. System Architecture

This project adapts the core ideas of rStar-Math into a framework that can call commercial LLM APIs.

```
Problem Input
   │
   ▼
┌─────────────────────────────────────────────────┐
│  MCTS Search  (src/core/mcts.py)                │
│                                                 │
│  get_possible_actions()                         │
│    └─ Generator model produces candidate steps  │◄── set_model()   large model
│                                                 │
│  _prepare_actions()                             │
│    └─ PPM pre-scores → keep top-k only          │◄── set_ppm()     trained PPM  [Method 1]
│       (falls back to keyword heuristics)        │
│                                                 │
│  evaluate_state()                               │
│    └─ PPM > Critic model > Generator model      │◄── set_critic()  cheap model  [Method 2]
└─────────────────────────────────────────────────┘
   │
   ▼
Best reasoning trajectory → preference pairs
   │
   ▼
PPM training  (tools/train_ppm.py)
   │
   ▼
Improved search quality next round (self-evolution)
```

---

## 2. Implemented Optimisation Methods

### Method 1: Pruned MCTS with PPM Pre-scoring

**Idea**: Before expanding a tree node, score every candidate step with the trained PPM and expand only the top-k highest-scoring branches instead of all candidates.

**Location**: `src/core/mcts.py` — `_prepare_actions()` → `_ppm_prune()`

**Flow**:
```
Generate N candidate steps (LLM)
         │
         ▼  De-duplicate (Jaccard similarity)
         │
    PPM attached & top_k_prune > 0?
         │
  ───────┼────────────────────────────
  Yes    │                          No
         ▼                           ▼
  ppm.evaluate_step() per step    Keyword heuristic scoring (fallback)
         │
         ▼
  Sort → keep top-k only
  Pruned branches → never expanded → no API cost
```

**Configuration**:
```python
config = MCTSConfig(
    search_strategy="adaptive",
    top_k_prune=2,          # expand only the 2 highest-PPM-scored steps
    parallel_actions=True,  # score steps concurrently to reduce latency
)
mcts = MCTS(config)
mcts.set_model(generator_model)
mcts.set_ppm(trained_ppm)   # PPM must be trained first
```

**Effect analysis**:

| Scenario | No pruning (baseline) | PPM pruning (top_k=2) |
|---|---|---|
| Branches expanded per node | max_branching_factor (4) | fixed 2 |
| Downstream LLM calls | 4× | 2× |
| Branch selection quality | keyword heuristics | learned signal |
| With untrained PPM | usable | equivalent to random pruning — disable |

**Key constraint**: An untrained PPM outputs near-random scores. Keep `top_k_prune=0` until after running `collect_trajectories.py` + `train_ppm.py` for at least one round.

**Stats**: `mcts.last_stats["ppm_pruned"]` records branches removed by PPM each search.

---

### Method 2: Asymmetric Policy–Reward Design

**Idea**: Use different models for generation (policy) and scoring (reward). A large model generates diverse, high-quality candidate steps; a small model or PPM evaluates step quality cheaply.

**Location**: `src/core/mcts.py` — `set_critic()` + `evaluate_state()`

**Scoring priority** (highest to lowest):
```
1. PPM          — fastest; most accurate once trained
2. Critic model — lightweight LLM registered via set_critic()
3. Main model   — generator doubles as evaluator; most expensive
4. Heuristic    — fallback for non-terminal nodes
```

**Configuration**:
```python
from src.model.model_interface import ModelFactory

generator = ModelFactory.create_model("deepseek", deepseek_key)  # DeepSeek-V3, generates steps
critic    = ModelFactory.create_model("openai",   openai_key)    # GPT-4o-mini, scores steps

mcts = MCTS(MCTSConfig(search_strategy="adaptive"))
mcts.set_model(generator)
mcts.set_critic(critic)
# If PPM is also set, it takes priority over the critic
```

**Effect analysis**:

| Scoring method | Cost per call | Accuracy | Best used when |
|---|---|---|---|
| Generator self-eval | High (large model price) | Medium (self-serving bias) | No PPM/critic available |
| Critic (small LLM) | Medium | Medium-high | Before PPM is trained |
| PPM | Very low (embedding + small net) | High (after training) | PPM sufficiently trained |

**Note**: A weak LLM critic introduces more scoring noise than the generator self-evaluating. Prefer going directly to the PPM route; use the critic only as a stop-gap before PPM training.

---

### Supporting optimisations (used alongside both Methods)

#### State-value caching (`eval_cache=True`, on by default)
MCTS revisits the same nodes during backpropagation. The cache avoids duplicate API calls.
- Stat: `last_stats["cache_hits"]`

#### Token budget (`token_budget=N`)
Hard-stops the search once estimated token usage reaches the budget.
- Recommended: DeepSeek ≈ `4000` (~$0.001/problem), GPT-4o ≈ `2000` (~$0.05/problem)

#### Context truncation (`max_state_steps=N`)
Keeps only the problem statement + last N reasoning steps, preventing unbounded token growth as search depth increases.
- Recommended: `6`

#### Parallel action generation (`parallel_actions=True`)
Generates candidate steps concurrently; wall-clock time drops from O(k) to O(1).

#### Retry with exponential backoff (`max_retries=3, retry_delay=2.0`)
All LLM calls automatically retry on rate-limit / network errors with 2 s → 4 s → 8 s backoff.

---

## 3. Full Workflow

### Stage 1: Cold start (PPM not yet trained)

```bash
# Set environment variables (see Section 5)

# Collect initial trajectories on ~20–200 problems
python tools/collect_trajectories.py \
    --model deepseek \
    --simulations 5 \
    --output data/trajectories_round1.jsonl

# Train first PPM version
python tools/train_ppm.py \
    --data data/trajectories_round1.jsonl \
    --epochs 50 \
    --output checkpoints/ppm_round1.pt
```

### Stage 2: PPM-guided search (Method 1 active)

```python
from src.core.mcts import MCTS, MCTSConfig
from src.core.ppm import ProcessPreferenceModel, PPMConfig
from src.model.model_interface import ModelFactory

ppm = ProcessPreferenceModel(PPMConfig(input_dim=384))
ppm.load_model("checkpoints/ppm_round1.pt")

config = MCTSConfig(
    search_strategy="adaptive",
    top_k_prune=2,       # Method 1
    eval_cache=True,
    token_budget=4000,
    max_state_steps=6,
)
mcts = MCTS(config)
mcts.set_model(ModelFactory.create_model("deepseek", DEEPSEEK_KEY))
mcts.set_ppm(ppm)        # Method 1 + 2 (PPM handles both pruning and evaluation)

action, trajectory = mcts.search("Solve for x: 3x - 7 = 14")
```

### Stage 3: Iterative improvement

```bash
# Collect higher-quality trajectories with PPM-guided search
python tools/collect_trajectories.py \
    --model deepseek \
    --simulations 8 \
    --output data/trajectories_round2.jsonl

# Retrain PPM on combined data
python tools/train_ppm.py \
    --data data/trajectories_round2.jsonl \
    --epochs 80 \
    --output checkpoints/ppm_round2.pt
```

---

## 4. Supported Models

| Name | Type | model_type | Env var | Best used as |
|---|---|---|---|---|
| DeepSeek-V3 | Cloud API | `"deepseek"` | `DEEPSEEK_API_KEY` | Generator |
| DeepSeek-R1 | Cloud API | `"deepseek"` + `model="deepseek-reasoner"` | `DEEPSEEK_API_KEY` | Generator (with chain-of-thought) |
| GPT-4o-mini | Cloud API | `"openai"` | `OPENAI_API_KEY` | Critic / Embedding |
| Claude Haiku | Cloud API | `"anthropic"` | `ANTHROPIC_API_KEY` | Critic |
| Qwen2.5-Math:7b | Local (Ollama) | `"ollama"` | none | Generator + Critic (free) |

**Embeddings**: DeepSeek, Anthropic, and Ollama all use **local sentence-transformers** (`all-MiniLM-L6-v2`, dim=384) — no OpenAI key needed. PPM `input_dim` is set to 384 accordingly.

---

## 5. Quick-Start Experiment

Run the full pipeline (collect → train → compare) in ~10 minutes:

```bash
# .env must contain: DEEPSEEK_API_KEY=sk-...
python experiments/run_experiment.py

# Faster (fewer problems):
python experiments/run_experiment.py --train 10 --test 5
```

Compares three configs on the same test problems:
- **A. Baseline MCTS** — simple search, LLM scoring
- **B. Adaptive MCTS** — keyword heuristic pruning
- **C. Adaptive + PPM** — Method 1 PPM pre-scoring (`top_k_prune=2`)

---

## 6. Key File Index

```
src/core/mcts.py                  MCTS: search, PPM pruning, asymmetric scoring
src/core/ppm.py                   PPM network architecture and trainer
src/model/model_interface.py      LLM interfaces: OpenAI / Anthropic / DeepSeek / Ollama
                                  + LocalEmbedder (sentence-transformers, free)
backend/main.py                   FastAPI: /solve, /compare-models
tools/collect_trajectories.py     Data collection: MCTS trajectories → preference pairs JSONL
tools/train_ppm.py                PPM training: preference pairs → model weights
tools/member1_search_ablation.py  Ablation study: baseline vs adaptive (no API key needed)
experiments/run_experiment.py     End-to-end comparison experiment
config/default.json               Default config for all models and MCTS parameters
```
