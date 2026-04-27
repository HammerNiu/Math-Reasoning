# Math Reasoning

A math reasoning system inspired by the [rStar-Math](https://arxiv.org/pdf/2501.04519) framework — using MCTS + Process Preference Model to guide LLMs through step-by-step math reasoning.

## How It Works

1. **MCTS** explores multiple reasoning paths for a given problem
2. **PPM** scores each step's confidence using text embeddings
3. The best trajectory is returned as the solution

## Setup

```bash
git clone <repo>
cd Math-Reasoning
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

```

Create a `.env` file:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
```

## Running

**Command-line demo:**
```bash
python example.py
```

**Streamlit UI:**
```bash
streamlit run streamlit.py
```
Opens at `http://localhost:8501`

**Member 1 MCTS ablation:**
```bash
python3 tools/member1_search_ablation.py
```
See `docs/member1_mcts.md` for the modified MCTS method, experiment table,
slide notes, and demo example.

## Project Structure

```
├── src/
│   ├── core/          # MCTS, PPM
│   └── models/        # LLM interfaces
├── backend/           # FastAPI (WIP)
├── config/            # default.json
├── example.py         # CLI demo
└── streamlit.py       # Web UI
```

## Citation

```bibtex
@article{rstar2024,
  title={rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking},
  year={2024}
}
```
