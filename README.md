# Math Reasoning with MCTS + PPM

## Overview

The system trains on the full **AIME dataset** and uses the learned PPM to steer MCTS search toward higher-quality reasoning paths at inference time. No model fine-tuning is required — the PPM runs as a lightweight scoring layer on top of any capable LLM.

```
AIME Problems
     │
     ▼
MCTS Trajectory Collection
     │  (DeepSeek-V3 generates candidate reasoning steps)
     ▼
Preference Pair Construction
     │  (high-reward trajectories vs. low-reward trajectories)
     ▼
PPM Training  (local, free — sentence-transformers + small MLP)
     │
     ▼
PPM-Guided MCTS Search
     │  (PPM pre-scores and prunes weak branches before expansion)
     ▼
Best Reasoning Trajectory + Final Answer
```
## Setup

```bash
git clone <repo>
cd Math-Reasoning
pip install -r requirements.txt
```

Create a `.env` file with your DeepSeek API key:

```
DEEPSEEK_API_KEY=sk-...
```


## Running the Experiment

```bash
# Default: 20 train / 10 test, MATH Level 5 only
python experiments/run_experiment.py

# Custom size
python experiments/run_experiment.py --train 40 --test 15

# Full multi-dataset training with curriculum learning
python experiments/run_experiment.py \
    --dataset math_all gsm8k numina olympiad aime \
    --curriculum --train 200 --test 30
```

This runs the full pipeline:
1. Collects MCTS trajectories on training problems (from selected datasets)
2. Builds preference pairs with reward metadata and trains a **ContextAwarePPM** locally
3. Compares Baseline vs Adaptive vs PPM-guided MCTS on held-out problems

### Supported Training Datasets

| Flag | Dataset | Scale | Difficulty |
|---|---|---|---|
| `math_l5` | MATH Level 5 (default) | ~2K | ⭐⭐⭐⭐⭐ |
| `math_all` | All MATH levels 1–5 | ~12K | ⭐–⭐⭐⭐⭐⭐ |
| `gsm8k` | GSM8K grade-school | ~8.5K | ⭐ |
| `numina` | NuminaMath-CoT (sampled) | 860K → 5K | ⭐⭐⭐ |
| `olympiad` | OlympiadBench | ~8K | ⭐⭐⭐⭐⭐ |
| `aime` | AIME 1983–2024 | ~900 | ⭐⭐⭐⭐⭐⭐ |


### PPM Innovations

| Innovation | Where | Effect |
|---|---|---|
| Context-Aware Dual Encoder | `ContextAwarePPM` | Joint (problem, step) encoding for context-sensitive scoring |
| Reward-Proportional Soft Margin | `train_step()` | Loss margin scales with actual reward gap; high-confidence pairs push harder |
| Hard Negative Mining | `_apply_hard_negatives()` | 30% of batch replaced with cosine-similar but lower-quality steps |
| Multi-Dataset Curriculum | `run_experiment.py --curriculum` | Easy→Hard ordering across 6 math datasets |
| Dropout + Cosine Annealing | `PPMConfig`, `PPMTrainer` | Regularization for small-dataset settings |
| PPM-Guided Rollout | `_select_rollout_action()` | Softmax sampling over PPM scores restores Monte Carlo property |

See [docs/ppm_innovations.md](docs/ppm_innovations.md) for the full bilingual writeup.

### Supporting Optimizations

| Optimization | Config Key | Effect |
|---|---|---|
| Evaluation caching | `eval_cache=True` | Avoids duplicate LLM calls for revisited states |
| Token budget | `token_budget=4000` | Hard-stops search before runaway token usage |
| Context truncation | `max_state_steps=6` | Keeps only last N steps in prompt |
| Parallel generation | `parallel_actions=True` | Concurrent step generation, O(1) latency |
| Retry with backoff | `max_retries=3` | Handles rate limits and transient API errors |

## FastAPI Backend

```bash
uvicorn backend.main:app --reload
```

Endpoints:
- `POST /solve` — solve a single problem
- `POST /compare-models` — run multiple configs on the same problem
- `GET /health` — service health check

## Project Structure

```
src/core/mcts.py                  MCTS: search, PPM pruning, asymmetric scoring
src/core/ppm.py                   PPM network architecture and trainer
src/model/model_interface.py      LLM interfaces: DeepSeek / Anthropic / Ollama
                                  + LocalEmbedder (sentence-transformers, free)
backend/main.py                   FastAPI: /solve, /compare-models
tools/collect_trajectories.py     Trajectory collection → preference pairs JSONL
tools/train_ppm.py                PPM training: preference pairs → model weights
experiments/run_experiment.py     End-to-end comparison experiment
config/default.json               Default MCTS and model configuration
docs/implementation_notes.md      Method details and workflow
docs/experiment_analysis.md       Full experiment results and analysis
```

## Supported Models

| Model | Type | Best Used As |
|---|---|---|
| DeepSeek-V3 | Cloud API (`"deepseek"`) | Generator |
| DeepSeek-R1 | Cloud API (`"deepseek"`) | Generator (chain-of-thought) |
| Claude Haiku | Cloud API (`"anthropic"`) | Critic |
| Qwen2.5-Math:7b | Local via Ollama (`"ollama"`) | Generator + Critic (free) |

## Citation

```bibtex
@article{rstar2024,
  title={rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking},
  author={Li, Xinyu and others},
  year={2024}
}
```
