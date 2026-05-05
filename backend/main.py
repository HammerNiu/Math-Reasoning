import os
import time
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.core.scoring import HeuristicStepVerifier, HybridProcessScorer
from src.core.mcts import MCTS, MCTSConfig
from src.core.ppm import ProcessPreferenceModel, PPMConfig
from src.models.model_interface import LocalEmbedder, ModelFactory

app = FastAPI(title="Math Reasoning Demonstrator API")

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class MathProblem(BaseModel):
    problem_text: str
    model_name: str = "openai"          # "openai" | "anthropic" | "deepseek"
    mcts_simulations: int = 3
    search_strategy: str = "adaptive"   # "baseline" | "adaptive"
    temperature: float = 0.7
    ppm_checkpoint: Optional[str] = None  # path to a trained PPM .pt file
    use_verifier: bool = True
    top_k_prune: int = 2


class SolutionResponse(BaseModel):
    solution_steps: List[str]
    confidence_score: float
    reasoning_path: List[Dict[str, Any]]
    execution_time: float
    model_name: str
    stats: Dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_API_KEY_ENVS: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

def _build_model(model_name: str, ollama_model: str = "qwen2.5-math:7b"):
    name = model_name.lower()
    if name == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ModelFactory.create_model("ollama", ollama_model=ollama_model, base_url=base_url)
    env = _API_KEY_ENVS.get(name)
    if env is None:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(_API_KEY_ENVS)} or 'ollama'")
    api_key = os.getenv(env, "")
    return ModelFactory.create_model(name, api_key)


def _load_ppm(checkpoint_path: Optional[str]) -> Optional[ProcessPreferenceModel]:
    if not checkpoint_path:
        return None
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"PPM checkpoint not found: {checkpoint_path}")

    config = PPMConfig(input_dim=384)
    try:
        import torch
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        saved_config = checkpoint.get("config")
        if isinstance(saved_config, PPMConfig):
            config = saved_config
        elif isinstance(saved_config, dict):
            config = PPMConfig(**saved_config)
    except Exception:
        try:
            config = PPMConfig(input_dim=LocalEmbedder.get().dim)
        except Exception:
            pass

    ppm = ProcessPreferenceModel(config)
    ppm.load_model(checkpoint_path)
    ppm.eval()
    return ppm


def _build_process_scorer(ppm: Optional[ProcessPreferenceModel], use_verifier: bool):
    if ppm is not None and use_verifier:
        return HybridProcessScorer(ppm=ppm, verifier=HeuristicStepVerifier())
    if ppm is not None:
        return ppm
    if use_verifier:
        return HeuristicStepVerifier()
    return None


def _run_mcts(problem: MathProblem):
    model = _build_model(problem.model_name)
    ppm = _load_ppm(problem.ppm_checkpoint)
    process_scorer = _build_process_scorer(ppm, problem.use_verifier)

    config = MCTSConfig(
        max_simulations=max(1, min(problem.mcts_simulations, 20)),
        search_strategy=problem.search_strategy,
        max_depth=5,
        eval_cache=True,
        max_state_steps=8,
        top_k_prune=max(0, problem.top_k_prune) if process_scorer is not None else 0,
    )
    mcts = MCTS(config)
    if process_scorer is not None:
        mcts.set_ppm(process_scorer)

    t0 = time.perf_counter()
    action, trajectory = mcts.search(problem.problem_text, model)
    elapsed = time.perf_counter() - t0

    steps = [line.strip() for line in action.split('\n') if line.strip()]
    confidence = max((float(t.get('value', 0.0)) for t in trajectory), default=0.5)
    confidence = max(0.0, min(1.0, confidence))

    return SolutionResponse(
        solution_steps=steps,
        confidence_score=confidence,
        reasoning_path=trajectory,
        execution_time=round(elapsed, 3),
        model_name=problem.model_name,
        stats=mcts.last_stats,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/solve", response_model=SolutionResponse)
async def solve_problem(problem: MathProblem) -> SolutionResponse:
    """Solve a math problem using MCTS + the chosen LLM."""
    try:
        return _run_mcts(problem)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare-models")
async def compare_models(problem: MathProblem) -> Dict[str, Any]:
    """Run the same problem through all available models and compare results."""
    results: Dict[str, Any] = {}
    all_models = list(_API_KEY_ENVS) + ["ollama"]
    for name in all_models:
        if name != "ollama":
            api_key = os.getenv(_API_KEY_ENVS[name], "")
            if not api_key:
                results[name] = {"skipped": "API key not set"}
                continue
        try:
            p = problem.model_copy(update={"model_name": name})
            r = _run_mcts(p)
            results[name] = {
                "solution_steps": r.solution_steps,
                "confidence_score": r.confidence_score,
                "execution_time": r.execution_time,
                "model_calls": r.stats.get("model_calls", 0),
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    return {"problem": problem.problem_text, "results": results}


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
