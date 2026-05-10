cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400   1) # Math Reasoning with MCTS + Process Scoring
a9d51488 (Hammer       2026-04-21 01:29:20 -0400   2) 
b9cb1e12 (cinnnamooon  2026-05-01 17:10:03 -0400   3) ## Overview
a9d51488 (Hammer       2026-04-21 01:29:20 -0400   4) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400   5) This project builds a math-reasoning system that compares direct one-path reasoning with a search-based pipeline. The improved system uses Monte Carlo Tree Search (MCTS) to explore multiple candidate reasoning paths and a process-level scorer to prune weak branches before they consume more model calls.
a9d51488 (Hammer       2026-04-21 01:29:20 -0400   6) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400   7) The repository supports three progressively stronger modes:
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400   8) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400   9) ```text
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  10) Problem
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  11)   -> Baseline MCTS
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  12)       simple expansion and LLM evaluation
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  13)   -> Adaptive MCTS
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  14)       adaptive branching, duplicate pruning, diversity scoring, early stopping
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  15)   -> Adaptive MCTS + Process Scorer
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  16)       PPM-compatible scorer, verifier fallback, top-k pruning
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  17)   -> Selected reasoning path and final answer
b9cb1e12 (cinnnamooon  2026-05-01 17:10:03 -0400  18) ```
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  19) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  20) The implementation is designed for the EECS E6895 final project rubric: it includes a reproducible codebase, report source, presentation/demo materials, deterministic ablations, and scripts for full MATH/OlympiadBench evaluation when API keys are available.
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  21) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  22) ## Main Contributions
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  23) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  24) | Requirement from task split | Implementation |
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  25) |---|---|
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  26) | Baseline reproduction | `MCTSConfig(search_strategy="baseline")` and command-line/demo paths |
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  27) | Member 1 search innovation | Adaptive MCTS in `src/core/mcts.py`: adaptive branching, pruning, diversity scoring, early stopping, caching, token budget |
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  28) | Member 2 scoring innovation | `src/core/scoring.py`: heuristic verifier plus hybrid PPM+verifier scorer |
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  29) | Member 3 system/demo integration | Streamlit side-by-side comparison UI and FastAPI endpoints |
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  30) | Quantitative evidence | Deterministic ablations in `tools/member1_search_ablation.py` and `tools/member2_scoring_ablation.py`; full benchmark scripts in `experiments/` and `eval.py` |
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  31) | Final report and rubric mapping | `report/final_report.tex`, `docs/experiment_analysis.md`, `docs/rubric_checklist.md` |
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  32) 
a9d51488 (Hammer       2026-04-21 01:29:20 -0400  33) ## Setup
a9d51488 (Hammer       2026-04-21 01:29:20 -0400  34) 
a9d51488 (Hammer       2026-04-21 01:29:20 -0400  35) ```bash
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  36) git clone https://github.com/HammerNiu/Math-Reasoning.git
a9d51488 (Hammer       2026-04-21 01:29:20 -0400  37) cd Math-Reasoning
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  38) python -m venv .venv
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  39) .venv/bin/pip install -r requirements.txt
a9d51488 (Hammer       2026-04-21 01:29:20 -0400  40) ```
a9d51488 (Hammer       2026-04-21 01:29:20 -0400  41) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  42) Create `.env` for cloud models:
b9cb1e12 (cinnnamooon  2026-05-01 17:10:03 -0400  43) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  44) ```bash
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  45) OPENAI_API_KEY=sk-...
b9cb1e12 (cinnnamooon  2026-05-01 17:10:03 -0400  46) DEEPSEEK_API_KEY=sk-...
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  47) ANTHROPIC_API_KEY=sk-...
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  48) ```
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  49) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  50) For local inference, install Ollama and pull a math model:
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  51) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  52) ```bash
6884bedc (superwayne66 2026-05-05 15:24:40 -0400  53) ollama pull qwen2-math:7b
a9d51488 (Hammer       2026-04-21 01:29:20 -0400  54) ```
a9d51488 (Hammer       2026-04-21 01:29:20 -0400  55) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  56) ## Quick Verification
a9d51488 (Hammer       2026-04-21 01:29:20 -0400  57) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  58) These checks do not require API keys:
a9d51488 (Hammer       2026-04-21 01:29:20 -0400  59) 
a9d51488 (Hammer       2026-04-21 01:29:20 -0400  60) ```bash
55990bed (cinnnamooon  2026-05-05 13:44:00 -0400  61) # Default: 20 train / 10 test, MATH Level 5 only
b9cb1e12 (cinnnamooon  2026-05-01 17:10:03 -0400  62) python experiments/run_experiment.py
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  63) 
55990bed (cinnnamooon  2026-05-05 13:44:00 -0400  64) # Custom size
55990bed (cinnnamooon  2026-05-05 13:44:00 -0400  65) python experiments/run_experiment.py --train 40 --test 15
b9cb1e12 (cinnnamooon  2026-05-01 17:10:03 -0400  66) 
55990bed (cinnnamooon  2026-05-05 13:44:00 -0400  67) # Full multi-dataset training with curriculum learning
55990bed (cinnnamooon  2026-05-05 13:44:00 -0400  68) python experiments/run_experiment.py \
55990bed (cinnnamooon  2026-05-05 13:44:00 -0400  69)     --dataset math_all gsm8k numina olympiad aime \
55990bed (cinnnamooon  2026-05-05 13:44:00 -0400  70)     --curriculum --train 200 --test 30
a9d51488 (Hammer       2026-04-21 01:29:20 -0400  71) ```
a9d51488 (Hammer       2026-04-21 01:29:20 -0400  72) 
b9cb1e12 (cinnnamooon  2026-05-01 17:10:03 -0400  73) This runs the full pipeline:
55990bed (cinnnamooon  2026-05-05 13:44:00 -0400  74) 1. Collects MCTS trajectories on training problems (from selected datasets)
55990bed (cinnnamooon  2026-05-05 13:44:00 -0400  75) 2. Builds preference pairs with reward metadata and trains a **ContextAwarePPM** locally
b9cb1e12 (cinnnamooon  2026-05-01 17:10:03 -0400  76) 3. Compares Baseline vs Adaptive vs PPM-guided MCTS on held-out problems
b9cb1e12 (cinnnamooon  2026-05-01 17:10:03 -0400  77) 
bc8c386b (cinnnamooon  2026-05-05 13:58:25 -0400  78) ### Training Datasets
b9cb1e12 (cinnnamooon  2026-05-01 17:10:03 -0400  79) 
bc8c386b (cinnnamooon  2026-05-05 13:58:25 -0400  80) | Flag | Dataset
bc8c386b (cinnnamooon  2026-05-05 13:58:25 -0400  81) | `math_l5` | MATH Level 5 (default) 
bc8c386b (cinnnamooon  2026-05-05 13:58:25 -0400  82) | `math_all` | All MATH levels 1–5  
bc8c386b (cinnnamooon  2026-05-05 13:58:25 -0400  83) | `gsm8k` | GSM8K grade-school 
bc8c386b (cinnnamooon  2026-05-05 13:58:25 -0400  84) | `numina` | NuminaMath-CoT (sampled) 
bc8c386b (cinnnamooon  2026-05-05 13:58:25 -0400  85) | `olympiad` | OlympiadBench 
bc8c386b (cinnnamooon  2026-05-05 13:58:25 -0400  86) | `aime` | AIME 1983–2024 
a9d51488 (Hammer       2026-04-21 01:29:20 -0400  87) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  88) ## Run the Demo
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  89) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  90) Streamlit comparison demo:
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  91) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  92) ```bash
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  93) .venv/bin/streamlit run app.py
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  94) ```
b9cb1e12 (cinnnamooon  2026-05-01 17:10:03 -0400  95) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  96) The UI defaults to `demo`, a deterministic no-key model that finishes quickly
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400  97) and shows the baseline/improved comparison without spending API credits. Switch
6884bedc (superwayne66 2026-05-05 15:24:40 -0400  98) the model dropdown to `openai`, `deepseek`, `anthropic`, or local
6884bedc (superwayne66 2026-05-05 15:24:40 -0400  99) `qwen2-math:7b` via Ollama only when you want a real model run; for cloud
6884bedc (superwayne66 2026-05-05 15:24:40 -0400 100) models, start with `Simulations = 1` because each search simulation can make
6884bedc (superwayne66 2026-05-05 15:24:40 -0400 101) several model calls.
b9cb1e12 (cinnnamooon  2026-05-01 17:10:03 -0400 102) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 103) The first screen lets you run the same problem through:
b9cb1e12 (cinnnamooon  2026-05-01 17:10:03 -0400 104) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 105) - Baseline MCTS
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 106) - Improved MCTS with verifier-only scoring
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 107) - Improved MCTS with hybrid PPM+verifier scoring, if a checkpoint is supplied
b9cb1e12 (cinnnamooon  2026-05-01 17:10:03 -0400 108) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 109) FastAPI backend:
b9cb1e12 (cinnnamooon  2026-05-01 17:10:03 -0400 110) 
c0b89c29 (superwayne66 2026-04-27 06:05:29 -0400 111) ```bash
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 112) .venv/bin/uvicorn backend.main:app --reload
c0b89c29 (superwayne66 2026-04-27 06:05:29 -0400 113) ```
b9cb1e12 (cinnnamooon  2026-05-01 17:10:03 -0400 114) 
b9cb1e12 (cinnnamooon  2026-05-01 17:10:03 -0400 115) Endpoints:
c0b89c29 (superwayne66 2026-04-27 06:05:29 -0400 116) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 117) - `POST /solve` solves one problem with MCTS and optional process scoring.
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 118) - `POST /compare-models` compares available model backends on the same problem.
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 119) - `GET /health` checks service status.
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 120) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 121) ## Train and Evaluate PPM
a9d51488 (Hammer       2026-04-21 01:29:20 -0400 122) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 123) Collect trajectories:
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 124) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 125) ```bash
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 126) .venv/bin/python tools/collect_trajectories.py \
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 127)   --model deepseek \
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 128)   --simulations 5 \
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 129)   --output data/trajectories.jsonl
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 130) ```
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 131) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 132) Train PPM:
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 133) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 134) ```bash
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 135) .venv/bin/python tools/train_ppm.py \
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 136)   --data data/trajectories.jsonl \
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 137)   --epochs 50 \
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 138)   --output checkpoints/ppm.pt
a9d51488 (Hammer       2026-04-21 01:29:20 -0400 139) ```
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 140) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 141) ## Project Structure
a9d51488 (Hammer       2026-04-21 01:29:20 -0400 142) 
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 143) ```text
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 144) src/core/mcts.py                  MCTS, adaptive search, PPM/verifier pruning
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 145) src/core/ppm.py                   PPM architecture and trainer
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 146) src/core/scoring.py               Verifier and hybrid process scorer
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 147) src/model/model_interface.py      OpenAI, Anthropic, DeepSeek, Ollama, local embedder
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 148) backend/main.py                   FastAPI demo service
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 149) app.py                            Side-by-side comparison demo
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 150) streamlit.py                      Convenience launcher for app.py
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 151) tools/collect_trajectories.py     Trajectory collection for preference data
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 152) tools/train_ppm.py                PPM training
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 153) tools/member1_search_ablation.py  Search-side ablation, no API key required
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 154) tools/member2_scoring_ablation.py Scoring-side ablation, no API key required
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 155) experiments/run_experiment.py     End-to-end MATH Level 5 experiment
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 156) eval.py                           MATH/OlympiadBench evaluation script
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 157) docs/implementation_notes.md      System design and method notes
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 158) docs/experiment_analysis.md       Results and analysis
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 159) docs/rubric_checklist.md          Rubric compliance checklist
cda0f1e2 (superwayne66 2026-05-05 13:16:04 -0400 160) report/final_report.tex           ACM-style final report source
a9d51488 (Hammer       2026-04-21 01:29:20 -0400 161) ```
