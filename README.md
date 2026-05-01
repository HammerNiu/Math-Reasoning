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
# Full pipeline: collect → train PPM → evaluate 3 configs
python experiments/run_experiment.py

# Or
python experiments/run_experiment.py --train num --test num
```

This runs the full pipeline:
1. Collects MCTS trajectories on training problems
2. Builds preference pairs and trains the PPM locally
3. Compares Baseline vs Adaptive vs PPM-guided MCTS on held-out problems


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
