# Math Reasoning with MCTS + Process Scoring

## Overview

This project builds a math-reasoning system that compares direct one-path reasoning with a search-based pipeline. The improved system uses Monte Carlo Tree Search (MCTS) to explore multiple candidate reasoning paths and a process-level scorer to prune weak branches before they consume more model calls.

The repository supports three progressively stronger modes:

```text
Problem
  -> Baseline MCTS
      simple expansion and LLM evaluation
  -> Adaptive MCTS
      adaptive branching, duplicate pruning, diversity scoring, early stopping
  -> Adaptive MCTS + Process Scorer
      PPM-compatible scorer, verifier fallback, top-k pruning
  -> Selected reasoning path and final answer
```

The implementation is designed for the EECS E6895 final project rubric: it includes a reproducible codebase, report source, presentation/demo materials, deterministic ablations, and scripts for full MATH/OlympiadBench evaluation when API keys are available.

## Main Contributions

| Requirement from task split | Implementation |
|---|---|
| Baseline reproduction | `MCTSConfig(search_strategy="baseline")` and command-line/demo paths |
| Member 1 search innovation | Adaptive MCTS in `src/core/mcts.py`: adaptive branching, pruning, diversity scoring, early stopping, caching, token budget |
| Member 2 scoring innovation | `src/core/scoring.py`: heuristic verifier plus hybrid PPM+verifier scorer |
| Member 3 system/demo integration | Streamlit side-by-side comparison UI and FastAPI endpoints |
| Quantitative evidence | Deterministic ablations in `tools/member1_search_ablation.py` and `tools/member2_scoring_ablation.py`; full benchmark scripts in `experiments/` and `eval.py` |
| Final report and rubric mapping | `report/final_report.tex`, `docs/experiment_analysis.md`, `docs/rubric_checklist.md` |

## Setup

```bash
git clone https://github.com/HammerNiu/Math-Reasoning.git
cd Math-Reasoning
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Create `.env` for cloud models:

```bash
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
```

For local inference, install Ollama and pull a math model:

```bash
ollama pull qwen2-math:7b
```

## Quick Verification

These checks do not require API keys:

```bash
.venv/bin/python -m compileall src backend experiments tools eval.py example.py app.py streamlit.py
.venv/bin/python tools/member1_search_ablation.py
.venv/bin/python tools/member2_scoring_ablation.py
.venv/bin/python -c "from backend.main import app; print(app.title)"
```

The deterministic ablations can also write report-ready Markdown:

```bash
.venv/bin/python tools/member1_search_ablation.py --write docs/member1_search_ablation.md
.venv/bin/python tools/member2_scoring_ablation.py --write docs/member2_scoring_ablation.md
```

## Run the Demo

Streamlit comparison demo:

```bash
.venv/bin/streamlit run app.py
```

The UI defaults to `demo`, a deterministic no-key model that finishes quickly
and shows the baseline/improved comparison without spending API credits. Switch
the model dropdown to `openai`, `deepseek`, `anthropic`, or local
`qwen2-math:7b` via Ollama only when you want a real model run; for cloud
models, start with `Simulations = 1` because each search simulation can make
several model calls.

The first screen lets you run the same problem through:

- Baseline MCTS
- Improved MCTS with verifier-only scoring
- Improved MCTS with hybrid PPM+verifier scoring, if a checkpoint is supplied

FastAPI backend:

```bash
.venv/bin/uvicorn backend.main:app --reload
```

Endpoints:

- `POST /solve` solves one problem with MCTS and optional process scoring.
- `POST /compare-models` compares available model backends on the same problem.
- `GET /health` checks service status.

## Train and Evaluate PPM

Collect trajectories:

```bash
.venv/bin/python tools/collect_trajectories.py \
  --model deepseek \
  --simulations 5 \
  --output data/trajectories.jsonl
```

Train PPM:

```bash
.venv/bin/python tools/train_ppm.py \
  --data data/trajectories.jsonl \
  --epochs 50 \
  --output checkpoints/ppm.pt
```

Run full comparison on MATH Level 5:

```bash
.venv/bin/python experiments/run_experiment.py \
  --train 20 \
  --test 10 \
  --simulations 3 \
  --output data/experiment_results.json
```

Run direct/MCTS/MCTS+PPM evaluation on MATH or OlympiadBench:

```bash
.venv/bin/python eval.py --dataset math --n 20 --strategy direct --model openai
.venv/bin/python eval.py --dataset math --n 20 --strategy mcts --model openai
.venv/bin/python eval.py --dataset math --n 20 --strategy mcts+ppm --model openai --ppm-checkpoint checkpoints/ppm.pt
```

## Project Structure

```text
src/core/mcts.py                  MCTS, adaptive search, PPM/verifier pruning
src/core/ppm.py                   PPM architecture and trainer
src/core/scoring.py               Verifier and hybrid process scorer
src/model/model_interface.py      OpenAI, Anthropic, DeepSeek, Ollama, local embedder
backend/main.py                   FastAPI demo service
app.py                            Side-by-side comparison demo
streamlit.py                      Convenience launcher for app.py
tools/collect_trajectories.py     Trajectory collection for preference data
tools/train_ppm.py                PPM training
tools/member1_search_ablation.py  Search-side ablation, no API key required
tools/member2_scoring_ablation.py Scoring-side ablation, no API key required
experiments/run_experiment.py     End-to-end MATH Level 5 experiment
eval.py                           MATH/OlympiadBench evaluation script
docs/implementation_notes.md      System design and method notes
docs/experiment_analysis.md       Results and analysis
docs/rubric_checklist.md          Rubric compliance checklist
report/final_report.tex           ACM-style final report source
```

## Submission Materials

- Final report source/PDF: `report/final_report.tex`, `report/final_report.pdf`
- GitHub repository link: https://github.com/HammerNiu/Math-Reasoning
- Demo artifact link, once pushed: https://github.com/HammerNiu/Math-Reasoning/blob/main/docs/demo_video.gif
- Presentation deck in this workspace: `math_reasoning_visual_polish.pptx`

Before CourseWorks submission, verify that both GitHub and demo links are publicly accessible.
