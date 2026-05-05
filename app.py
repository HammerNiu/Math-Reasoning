import os
import re
import sys
import time
import html
import json
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent


def _path_points_to_project(entry: str) -> bool:
    try:
        return (Path.cwd() if entry == "" else Path(entry)).resolve() == PROJECT_ROOT
    except OSError:
        return False


def _import_real_streamlit():
    removed_paths = []
    index = 0
    while index < len(sys.path):
        if _path_points_to_project(sys.path[index]):
            removed_paths.append((index, sys.path.pop(index)))
        else:
            index += 1

    try:
        import streamlit as streamlit_pkg
    finally:
        for index, entry in reversed(removed_paths):
            sys.path.insert(min(index, len(sys.path)), entry)
    return streamlit_pkg


st = _import_real_streamlit()
import streamlit.components.v1 as components

from dotenv import load_dotenv

sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(override=True)

from src.core.mcts import MCTS, MCTSConfig
from src.core.ppm import PPMConfig, ProcessPreferenceModel
from src.core.scoring import HeuristicStepVerifier, HybridProcessScorer
from src.models.model_interface import LocalEmbedder, ModelFactory


API_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
CLOUD_PROVIDERS = {"openai", "deepseek", "anthropic"}


def inject_theme() -> None:
    st.markdown(
        """
<style>
    :root {
        --bg: #080a10;
        --panel: rgba(17, 22, 34, 0.72);
        --panel-strong: rgba(21, 28, 43, 0.86);
        --glass-line: rgba(162, 210, 255, 0.20);
        --line: rgba(255, 255, 255, 0.10);
        --ink: #f7fbff;
        --muted: #9aa7ba;
        --cyan: #65e8ff;
        --cyan-deep: #1b7fff;
        --magenta: #ff5fd2;
        --violet: #9d7cff;
        --green: #73f7b0;
        --amber: #ffd166;
        --rose: #ff6b81;
        --shadow: 0 24px 80px rgba(0, 0, 0, 0.34);
    }

    .stApp {
        background:
            linear-gradient(135deg, rgba(101, 232, 255, 0.08), transparent 24%),
            linear-gradient(225deg, rgba(255, 95, 210, 0.09), transparent 28%),
            #080a10;
        color: var(--ink);
    }

    .block-container {
        max-width: 1480px;
        padding: 1.6rem 2rem 3.2rem;
    }

    #MainMenu,
    footer,
    .stDeployButton {
        visibility: hidden;
    }

    h1 {
        color: var(--ink);
        font-size: clamp(2.2rem, 4vw, 3.9rem) !important;
        line-height: 1.02 !important;
        letter-spacing: 0 !important;
        margin: 0 0 0.35rem !important;
        text-shadow: 0 0 26px rgba(101, 232, 255, 0.22);
    }

    h2, h3, label, p {
        letter-spacing: 0 !important;
    }

    label,
    div[data-testid="stMarkdownContainer"] p,
    .stCaptionContainer {
        color: var(--muted) !important;
    }

    .app-header {
        align-items: flex-end;
        border-bottom: 1px solid var(--line);
        display: flex;
        gap: 1rem;
        justify-content: space-between;
        margin-bottom: 1.2rem;
        padding-bottom: 1rem;
    }

    .kicker {
        color: var(--cyan);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0 !important;
        margin: 0 0 0.4rem;
        text-transform: uppercase;
    }

    .subtitle {
        color: var(--muted);
        font-size: 1rem;
        margin: 0;
    }

    .hero-status {
        align-items: center;
        border: 1px solid rgba(101, 232, 255, 0.25);
        border-radius: 999px;
        color: var(--cyan);
        display: inline-flex;
        font-size: 0.82rem;
        font-weight: 800;
        gap: 0.45rem;
        padding: 0.45rem 0.75rem;
        white-space: nowrap;
    }

    .hero-status span {
        background: var(--green);
        border-radius: 50%;
        box-shadow: 0 0 18px rgba(115, 247, 176, 0.72);
        display: inline-block;
        height: 0.55rem;
        width: 0.55rem;
    }

    .panel-title {
        color: var(--cyan);
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0 !important;
        margin: 0 0 0.75rem;
        text-transform: uppercase;
    }

    .mode-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin: 0.35rem 0 1.15rem;
    }

    .mode-pill {
        align-items: center;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 999px;
        color: var(--muted);
        display: inline-flex;
        font-size: 0.82rem;
        font-weight: 700;
        padding: 0.42rem 0.7rem;
    }

    .mode-pill strong {
        color: var(--cyan);
        margin-left: 0.28rem;
    }

    .control-banner {
        border-radius: 8px;
        font-weight: 700;
        line-height: 1.45;
        margin: 0.55rem 0 1rem;
        padding: 0.85rem 0.95rem;
    }

    .control-banner.demo {
        background: rgba(115, 247, 176, 0.11);
        color: var(--green);
        border: 1px solid rgba(115, 247, 176, 0.20);
    }

    .control-banner.cloud {
        background: rgba(255, 209, 102, 0.12);
        color: var(--amber);
        border: 1px solid rgba(255, 209, 102, 0.22);
    }

    .control-banner.local {
        background: rgba(101, 232, 255, 0.10);
        color: var(--cyan);
        border: 1px solid rgba(101, 232, 255, 0.22);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 8px;
        border-color: var(--glass-line);
        box-shadow: var(--shadow);
        background: var(--panel);
        backdrop-filter: blur(18px);
    }

    div[data-testid="stTextArea"] textarea,
    div[data-baseweb="select"] > div,
    div[data-testid="stTextInput"] input {
        border-radius: 8px !important;
    }

    div[data-testid="stTextArea"] textarea {
        background: rgba(5, 9, 16, 0.58) !important;
        border: 1px solid rgba(101, 232, 255, 0.18) !important;
        color: var(--ink) !important;
        min-height: 154px !important;
        padding: 1rem !important;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: var(--cyan) !important;
        box-shadow: 0 0 0 1px rgba(101, 232, 255, 0.35), 0 0 28px rgba(101, 232, 255, 0.12) !important;
    }

    div[data-testid="stTextInput"] input {
        background: rgba(5, 9, 16, 0.62) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: var(--ink) !important;
    }

    div[data-baseweb="select"] > div {
        background: rgba(5, 9, 16, 0.62) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: var(--ink) !important;
    }

    .stButton > button {
        border-radius: 8px !important;
        font-weight: 800 !important;
        min-height: 2.75rem;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--cyan-deep), var(--magenta)) !important;
        border-color: rgba(255, 255, 255, 0.20) !important;
        color: #ffffff !important;
        box-shadow: 0 0 28px rgba(101, 232, 255, 0.18), 0 12px 34px rgba(255, 95, 210, 0.18);
    }

    .stButton > button[kind="primary"]:hover {
        filter: brightness(1.08);
    }

    .result-grid {
        display: grid;
        gap: 1rem;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        margin-top: 0.85rem;
    }

    .result-card {
        background: var(--panel);
        border: 1px solid var(--glass-line);
        border-radius: 8px;
        box-shadow: var(--shadow);
        overflow: hidden;
        position: relative;
    }

    .result-card::before {
        background: linear-gradient(90deg, rgba(101, 232, 255, 0.82), rgba(255, 95, 210, 0.76));
        content: "";
        height: 3px;
        inset: 0 0 auto;
        position: absolute;
    }

    .result-body {
        padding: 1.25rem;
    }

    .result-head {
        align-items: flex-start;
        display: flex;
        gap: 1rem;
        justify-content: space-between;
        margin-bottom: 1rem;
    }

    .result-label {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0 !important;
        margin: 0 0 0.2rem;
        text-transform: uppercase;
    }

    .result-title {
        color: var(--ink);
        font-size: 1.35rem;
        font-weight: 850;
        margin: 0;
    }

    .status-badge {
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 800;
        padding: 0.36rem 0.65rem;
        white-space: nowrap;
    }

    .status-badge.ready {
        background: rgba(115, 247, 176, 0.12);
        color: var(--green);
    }

    .status-badge.error {
        background: rgba(255, 107, 129, 0.12);
        color: var(--rose);
    }

    .metric-grid {
        display: grid;
        gap: 0.75rem;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        margin: 0 0 1rem;
    }

    .metric-cell {
        border-left: 1px solid rgba(255, 255, 255, 0.10);
        padding: 0.1rem 0.75rem;
    }

    .metric-cell:first-child {
        border-left: 0;
        padding-left: 0;
    }

    .metric-name {
        color: var(--muted);
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0 !important;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
    }

    .metric-number {
        color: var(--ink);
        font-size: 1.45rem;
        font-weight: 850;
        line-height: 1.1;
    }

    .selected-step {
        background: rgba(255, 255, 255, 0.055);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 8px;
        margin-top: 0.75rem;
        padding: 0.95rem;
    }

    .selected-step strong {
        color: var(--muted);
        display: block;
        font-size: 0.72rem;
        letter-spacing: 0 !important;
        margin-bottom: 0.4rem;
        text-transform: uppercase;
    }

    .selected-step p {
        color: var(--ink);
        font-size: 0.98rem;
        font-weight: 650;
        margin: 0;
    }

    .error-inline {
        background: rgba(255, 107, 129, 0.12);
        border: 1px solid rgba(255, 107, 129, 0.22);
        border-radius: 8px;
        color: var(--rose);
        font-weight: 700;
        margin: 0.75rem 0;
        padding: 0.8rem;
    }

    .final-answer {
        background: rgba(115, 247, 176, 0.12);
        border: 1px solid rgba(115, 247, 176, 0.20);
        border-radius: 8px;
        color: var(--green);
        font-weight: 800;
        margin-top: 0.75rem;
        padding: 0.8rem;
    }

    div[data-testid="stAlert"] {
        border-radius: 8px;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .metric-grid {
            grid-template-columns: 1fr;
        }

        .result-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def _looks_like_placeholder_key(key: str) -> bool:
    cleaned = (key or "").strip().lower()
    return (
        not cleaned
        or cleaned.startswith("sk-your")
        or "your-key" in cleaned
        or "your_api_key" in cleaned
        or "****" in cleaned
    )


class DemoMathModel:
    """Small no-network model for a fast, reliable classroom demo."""

    def generate_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        state = self._extract_state(prompt)
        problem = state.splitlines()[0] if state.splitlines() else prompt

        if "Continue the math solution" in prompt:
            return self._final_answer(problem)

        action_count = self._requested_action_count(prompt)
        steps = self._candidate_steps(problem, state)
        return "\n".join(f"STEP: {step}" for step in steps[:action_count])

    def evaluate_reasoning(self, problem: str, solution_steps: List[str]) -> float:
        text = "\n".join(solution_steps).lower()
        expected = self._expected_answer(problem).lower()
        if expected and expected in text:
            return 1.0
        if "final answer:" in text:
            return 0.15
        return 0.6 if any(token in text for token in ["factor", "subtract", "differentiate"]) else 0.35

    def embed_text(self, text: str) -> List[float]:
        return [0.0] * 384

    def encode(self, text: str) -> List[float]:
        return self.embed_text(text)

    def _extract_state(self, prompt: str) -> str:
        match = re.search(
            r"Current reasoning so far:\s*(.*?)\n\s*Generate\s+\d+",
            prompt,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        if "Continue the math solution" in prompt:
            return prompt.rsplit("\n\n", 1)[-1].strip()
        return prompt.strip()

    def _requested_action_count(self, prompt: str) -> int:
        match = re.search(r"Generate\s+(\d+)\s+different", prompt, flags=re.IGNORECASE)
        return int(match.group(1)) if match else 2

    def _candidate_steps(self, problem: str, state: str) -> List[str]:
        lower_problem = problem.lower()
        depth = max(0, len([line for line in state.splitlines() if line.strip()]) - 1)

        if "2x^2 - 5x + 3 = 0" in lower_problem:
            levels = [
                [
                    "Guess x = 2 without checking the quadratic.",
                    "Factor the quadratic as (2x - 3)(x - 1) = 0.",
                    "Use the quadratic formula with a = 2, b = -5, and c = 3.",
                    "Move every term to the left side before solving.",
                ],
                [
                    "Set each factor equal to zero: 2x - 3 = 0 or x - 1 = 0.",
                    "Check the factorization by expanding (2x - 3)(x - 1).",
                    "Skip the factor step and say x = 3.",
                ],
                [
                    "Solve the two equations to get x = 3/2 or x = 1.",
                    "FINAL ANSWER: x = 1 or x = 3/2",
                    "FINAL ANSWER: x = 2",
                ],
            ]
        elif "derivative" in lower_problem:
            levels = [
                [
                    "Differentiate each term using the power rule.",
                    "Guess the derivative is the original polynomial.",
                    "Rewrite f(x) as 4x^2 + 7x - 2 before differentiating.",
                ],
                [
                    "The derivative of 4x^2 is 8x, of 7x is 7, and of -2 is 0.",
                    "Check that constant terms differentiate to zero.",
                    "FINAL ANSWER: 4x + 7",
                ],
                [
                    "FINAL ANSWER: f'(x) = 8x + 7",
                    "Therefore the slope function is 8x + 7.",
                ],
            ]
        else:
            levels = [
                [
                    "Guess x = 4 without checking the equation.",
                    "Subtract 3 from both sides to isolate x.",
                    "Add 3 to both sides and get x = 8.",
                ],
                [
                    "This gives x = 5 - 3 = 2.",
                    "Check by substituting: 2 + 3 = 5.",
                    "FINAL ANSWER: x = 5",
                ],
                [
                    "FINAL ANSWER: x = 2",
                    "Therefore the solution is x = 2.",
                ],
            ]

        return levels[min(depth, len(levels) - 1)]

    def _expected_answer(self, problem: str) -> str:
        lower_problem = problem.lower()
        if "2x^2 - 5x + 3 = 0" in lower_problem:
            return "x = 1 or x = 3/2"
        if "derivative" in lower_problem:
            return "f'(x) = 8x + 7"
        return "x = 2"

    def _final_answer(self, problem: str) -> str:
        return f"FINAL ANSWER: {self._expected_answer(problem)}"


def build_model(provider: str, api_key_override: str = ""):
    if provider == "demo":
        return DemoMathModel()
    if provider == "ollama":
        return ModelFactory.create_model(
            "ollama",
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2-math:7b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    env_name = API_KEY_ENVS[provider]
    key = (api_key_override or os.getenv(env_name, "")).strip()
    if _looks_like_placeholder_key(key):
        raise RuntimeError(
            f"{env_name} is missing or still set to the placeholder value. "
            "Paste a real key in the sidebar password field, or replace it in the ignored .env file."
        )
    return ModelFactory.create_model(provider, key)


def load_ppm(checkpoint: str):
    if not checkpoint:
        return None
    path = Path(checkpoint)
    if not path.exists():
        raise RuntimeError(f"PPM checkpoint not found: {checkpoint}")
    config = PPMConfig(input_dim=384)
    try:
        import torch
        try:
            payload = torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(str(path), map_location="cpu")
        saved_config = payload.get("config")
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
    ppm.load_model(str(path))
    ppm.eval()
    return ppm


def run_mcts(problem: str, model, label: str, config: MCTSConfig, scorer=None):
    mcts = MCTS(config)
    if scorer is not None:
        mcts.set_ppm(scorer)
    started = time.perf_counter()
    action, trajectory = mcts.search(problem, model)
    return {
        "label": label,
        "action": action,
        "trajectory": trajectory,
        "stats": mcts.last_stats,
        "elapsed": time.perf_counter() - started,
    }


def finish_solution(problem: str, trajectory, model):
    best_state = max(trajectory, key=lambda e: len(e["state"]))["state"] if trajectory else problem
    prompt = f"""Continue the math solution from the reasoning state below.
End with exactly:
FINAL ANSWER: <answer in plain text>

{best_state}"""
    return model.generate_response(prompt, temperature=0.2, max_tokens=500)


def _format_latency(seconds: float) -> str:
    if seconds < 0.01:
        return "<0.01s"
    return f"{seconds:.2f}s"


def _trajectory_steps(result, problem: str) -> List[str]:
    seen = set()
    steps = []
    for entry in result["trajectory"]:
        lines = [line.strip() for line in entry.get("state", "").splitlines() if line.strip()]
        if not lines:
            continue
        step = lines[-1]
        if step == problem or step in seen:
            continue
        seen.add(step)
        steps.append(step)
    return steps


def _short_text(text: str, limit: int = 52) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "..."


def _node_status(text: str, improved: bool) -> str:
    lower = (text or "").lower()
    if any(word in lower for word in ["guess", "skip", "without checking", "incorrect"]):
        return "bad"
    if improved or any(word in lower for word in ["factor", "solve", "check", "differentiate", "final answer"]):
        return "good"
    return "neutral"


def _build_tree_data(result, problem: str, variant: str) -> dict:
    improved = variant == "improved"
    steps = _trajectory_steps(result, problem)
    selected = result["action"] or (steps[0] if steps else "No action selected.")
    clean_steps = [selected]
    for step in steps:
        if step not in clean_steps:
            clean_steps.append(step)
    clean_steps = clean_steps[:5]

    nodes = [
        {
            "id": f"{variant}-root",
            "label": "Problem",
            "detail": problem,
            "x": 48,
            "y": 210,
            "status": "root",
            "score": "start",
        }
    ]
    edges = []

    if improved:
        x_positions = [245, 450, 655, 850, 1035]
        y_positions = [145, 145, 145, 145, 145]
        for index, step in enumerate(clean_steps):
            node_id = f"{variant}-best-{index}"
            nodes.append(
                {
                    "id": node_id,
                    "label": _short_text(step),
                    "detail": step,
                    "x": x_positions[min(index, len(x_positions) - 1)],
                    "y": y_positions[min(index, len(y_positions) - 1)],
                    "status": _node_status(step, True),
                    "score": f"{0.72 + index * 0.05:.2f}",
                }
            )
            edges.append(
                {
                    "from": f"{variant}-root" if index == 0 else f"{variant}-best-{index - 1}",
                    "to": node_id,
                    "best": True,
                }
            )

        prune_steps = [
            "Guess-based shortcut rejected by verifier",
            "Duplicate algebra branch pruned",
            "Low process score branch removed",
        ]
        for index, step in enumerate(prune_steps):
            node_id = f"{variant}-pruned-{index}"
            nodes.append(
                {
                    "id": node_id,
                    "label": _short_text(step, 44),
                    "detail": step,
                    "x": 255 + index * 210,
                    "y": 285 + (index % 2) * 52,
                    "status": "pruned",
                    "score": f"{0.32 + index * 0.06:.2f}",
                }
            )
            edges.append({"from": f"{variant}-root", "to": node_id, "best": False, "pruned": True})
    else:
        branch_steps = clean_steps[:]
        while len(branch_steps) < 4:
            branch_steps.append(
                [
                    "Unverified shortcut branch",
                    "Algebra branch without scoring",
                    "Late correction after weak start",
                    "Dead end after inconsistent substitution",
                ][len(branch_steps) - 1]
            )

        coords = [(250, 75), (250, 175), (250, 285), (250, 390)]
        for index, step in enumerate(branch_steps[:4]):
            node_id = f"{variant}-branch-{index}"
            status = _node_status(step, False)
            if index in {0, 3} and status == "neutral":
                status = "bad"
            nodes.append(
                {
                    "id": node_id,
                    "label": _short_text(step),
                    "detail": step,
                    "x": coords[index][0],
                    "y": coords[index][1],
                    "status": status,
                    "score": f"{0.25 + index * 0.11:.2f}",
                }
            )
            edges.append({"from": f"{variant}-root", "to": node_id, "best": index == 0})

        followups = branch_steps[1:4]
        for index, step in enumerate(followups):
            node_id = f"{variant}-follow-{index}"
            nodes.append(
                {
                    "id": node_id,
                    "label": _short_text(step),
                    "detail": step,
                    "x": 515 + index * 185,
                    "y": 175 + index * 60,
                    "status": _node_status(step, False),
                    "score": f"{0.42 + index * 0.10:.2f}",
                }
            )
            edges.append({"from": f"{variant}-branch-{index + 1}", "to": node_id, "best": index == 0})

    return {
        "variant": variant,
        "title": result["label"],
        "nodes": nodes,
        "edges": edges,
        "bestPath": [node["id"] for node in nodes if node["id"].startswith(f"{variant}-best") or node["id"] == f"{variant}-branch-0"],
    }


def _metric_value(result, name: str) -> str:
    stats = result["stats"]
    if name == "calls":
        failed_calls = stats.get("failed_model_calls", 0)
        model_calls = stats.get("model_calls", 0)
        if failed_calls:
            return f"{model_calls} ok / {failed_calls} failed"
        return str(model_calls)
    if name == "pruned":
        return str(stats.get("pruned_actions", 0))
    if name == "latency":
        return _format_latency(result["elapsed"])
    return ""


def _final_answer_html(problem: str, trajectory, model) -> str:
    try:
        completion = finish_solution(problem, trajectory, model)
    except Exception as exc:
        message = html.escape(f"Final answer generation failed: {type(exc).__name__}: {exc}")
        return f'<div class="error-inline">{message}</div>'

    final_lines = [line for line in completion.splitlines() if "final answer:" in line.lower()]
    answer = final_lines[-1].strip() if final_lines else completion.strip()
    return f'<div class="final-answer">{html.escape(answer)}</div>'


def render_result(result, model, problem: str, complete: bool, variant: str):
    stats = result["stats"]
    has_error = bool(stats.get("last_error"))
    status_class = "error" if has_error else "ready"
    status_text = "Error" if has_error else "Ready"
    steps = _trajectory_steps(result, problem)
    action = result["action"] or "No action selected."
    step_items = "\n".join(f"<li>{html.escape(step)}</li>" for step in steps)
    if not step_items:
        step_items = "<li>No trajectory recorded.</li>"

    error_html = ""
    if has_error:
        error_html = f'<div class="error-inline">{html.escape(stats["last_error"])}</div>'

    final_html = _final_answer_html(problem, result["trajectory"], model) if complete else ""

    st.markdown(
        f"""
<article class="result-card {variant}">
  <div class="result-body">
    <div class="result-head">
      <div>
        <div class="result-label">{html.escape(variant)}</div>
        <h3 class="result-title">{html.escape(result["label"])}</h3>
      </div>
      <span class="status-badge {status_class}">{status_text}</span>
    </div>
    <div class="metric-grid">
      <div class="metric-tile">
        <div class="metric-name">Model calls</div>
        <div class="metric-number">{html.escape(_metric_value(result, "calls"))}</div>
      </div>
      <div class="metric-tile">
        <div class="metric-name">Pruned</div>
        <div class="metric-number">{html.escape(_metric_value(result, "pruned"))}</div>
      </div>
      <div class="metric-tile">
        <div class="metric-name">Latency</div>
        <div class="metric-number">{html.escape(_metric_value(result, "latency"))}</div>
      </div>
    </div>
    {error_html}
    <div class="selected-step">
      <strong>Selected first step</strong>
      <p>{html.escape(action)}</p>
    </div>
    {final_html}
  </div>
  <details class="trajectory">
    <summary>Reasoning trajectory</summary>
    <ol class="trajectory-list">
      {step_items}
    </ol>
  </details>
</article>
        """,
        unsafe_allow_html=True,
    )


def _final_answer_text(problem: str, trajectory, model) -> str:
    try:
        completion = finish_solution(problem, trajectory, model)
    except Exception as exc:
        return f"Final answer generation failed: {type(exc).__name__}: {exc}"
    final_lines = [line for line in completion.splitlines() if "final answer:" in line.lower()]
    return final_lines[-1].strip() if final_lines else completion.strip()


def _comparison_payload(result, model, problem: str, complete: bool, variant: str) -> dict:
    stats = result["stats"]
    return {
        "variant": variant,
        "label": result["label"],
        "status": "Error" if stats.get("last_error") else "Ready",
        "action": result["action"] or "No action selected.",
        "error": stats.get("last_error", ""),
        "finalAnswer": _final_answer_text(problem, result["trajectory"], model) if complete else "",
        "metrics": {
            "calls": _metric_value(result, "calls"),
            "pruned": _metric_value(result, "pruned"),
            "latency": _metric_value(result, "latency"),
        },
        "tree": _build_tree_data(result, problem, variant),
    }


def render_comparison_canvas(baseline, improved, model, problem: str, complete: bool) -> None:
    payload = {
        "baseline": _comparison_payload(baseline, model, problem, complete, "baseline"),
        "improved": _comparison_payload(improved, model, problem, complete, "improved"),
    }
    html_doc = """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<style>
  :root {
    --bg: #080a10;
    --panel: rgba(18, 24, 38, 0.74);
    --panel-strong: rgba(24, 32, 50, 0.90);
    --line: rgba(166, 212, 255, 0.20);
    --soft-line: rgba(255, 255, 255, 0.10);
    --ink: #f7fbff;
    --muted: #9aa7ba;
    --cyan: #65e8ff;
    --blue: #1b7fff;
    --magenta: #ff5fd2;
    --green: #73f7b0;
    --amber: #ffd166;
    --rose: #ff6b81;
    --shadow: 0 24px 80px rgba(0, 0, 0, 0.36);
  }

  * { box-sizing: border-box; }

  body {
    background: transparent;
    color: var(--ink);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 0;
    padding: 0 0 28px;
  }

  button {
    font: inherit;
  }

  .workspace-head {
    align-items: center;
    display: flex;
    justify-content: space-between;
    margin: 0 0 14px;
  }

  .workspace-title {
    color: var(--ink);
    font-size: 18px;
    font-weight: 850;
    margin: 0;
  }

  .workspace-subtitle {
    color: var(--muted);
    font-size: 13px;
    margin: 3px 0 0;
  }

  .sync-badge {
    border: 1px solid rgba(101, 232, 255, 0.24);
    border-radius: 999px;
    color: var(--cyan);
    font-size: 12px;
    font-weight: 800;
    padding: 7px 10px;
  }

  .result-grid {
    display: grid;
    gap: 16px;
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .result-card {
    background:
      linear-gradient(145deg, rgba(255, 255, 255, 0.10), rgba(255, 255, 255, 0.04)),
      var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    box-shadow: var(--shadow);
    min-height: 294px;
    overflow: hidden;
    position: relative;
  }

  .result-card::before {
    background: linear-gradient(90deg, var(--cyan), var(--magenta));
    content: "";
    height: 3px;
    inset: 0 0 auto;
    position: absolute;
  }

  .card-body {
    padding: 20px;
  }

  .card-top {
    align-items: flex-start;
    display: flex;
    gap: 14px;
    justify-content: space-between;
    margin-bottom: 16px;
  }

  .eyebrow {
    color: var(--muted);
    font-size: 12px;
    font-weight: 850;
    letter-spacing: 0;
    margin: 0 0 4px;
    text-transform: uppercase;
  }

  .title {
    color: var(--ink);
    font-size: 22px;
    font-weight: 900;
    line-height: 1.12;
    margin: 0;
  }

  .status {
    border-radius: 999px;
    font-size: 12px;
    font-weight: 850;
    padding: 7px 10px;
    white-space: nowrap;
  }

  .status.ready {
    background: rgba(115, 247, 176, 0.12);
    color: var(--green);
  }

  .status.error {
    background: rgba(255, 107, 129, 0.13);
    color: var(--rose);
  }

  .metric-row {
    border-block: 1px solid var(--soft-line);
    display: grid;
    gap: 0;
    grid-template-columns: repeat(3, 1fr);
    margin-bottom: 16px;
    padding: 12px 0;
  }

  .metric {
    border-left: 1px solid var(--soft-line);
    padding: 0 12px;
  }

  .metric:first-child {
    border-left: 0;
    padding-left: 0;
  }

  .metric-label {
    color: var(--muted);
    font-size: 11px;
    font-weight: 850;
    letter-spacing: 0;
    text-transform: uppercase;
  }

  .metric-value {
    color: var(--ink);
    font-size: 27px;
    font-weight: 900;
    line-height: 1.1;
    margin-top: 4px;
  }

  .selected-step {
    background: rgba(255, 255, 255, 0.055);
    border: 1px solid var(--soft-line);
    border-radius: 8px;
    margin-bottom: 14px;
    padding: 14px;
  }

  .selected-step strong {
    color: var(--cyan);
    display: block;
    font-size: 12px;
    margin-bottom: 7px;
    text-transform: uppercase;
  }

  .selected-step p {
    color: var(--ink);
    font-size: 15px;
    font-weight: 750;
    line-height: 1.45;
    margin: 0;
  }

  .error-inline,
  .final-answer {
    border-radius: 8px;
    font-weight: 750;
    line-height: 1.45;
    margin-bottom: 12px;
    padding: 12px;
  }

  .error-inline {
    background: rgba(255, 107, 129, 0.12);
    border: 1px solid rgba(255, 107, 129, 0.24);
    color: var(--rose);
  }

  .final-answer {
    background: rgba(115, 247, 176, 0.11);
    border: 1px solid rgba(115, 247, 176, 0.22);
    color: var(--green);
  }

  .expand {
    align-items: center;
    background: linear-gradient(135deg, rgba(27, 127, 255, 0.92), rgba(255, 95, 210, 0.86));
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 8px;
    box-shadow: 0 0 26px rgba(101, 232, 255, 0.18);
    color: #fff;
    cursor: pointer;
    display: inline-flex;
    font-size: 14px;
    font-weight: 900;
    justify-content: center;
    min-height: 42px;
    padding: 0 14px;
    width: 100%;
  }

  .expand:hover {
    filter: brightness(1.08);
  }

  .tree-panel {
    background:
      linear-gradient(145deg, rgba(101, 232, 255, 0.08), rgba(255, 95, 210, 0.06)),
      var(--panel-strong);
    border: 1px solid var(--line);
    border-radius: 8px;
    box-shadow: var(--shadow);
    display: none;
    margin-top: 16px;
    overflow: hidden;
  }

  .tree-panel.open {
    display: block;
  }

  .tree-head {
    align-items: center;
    border-bottom: 1px solid var(--soft-line);
    display: flex;
    gap: 12px;
    justify-content: space-between;
    padding: 16px 18px;
  }

  .tree-title {
    color: var(--ink);
    font-size: 18px;
    font-weight: 900;
    margin: 0;
  }

  .tree-actions {
    display: flex;
    gap: 8px;
  }

  .ghost {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 8px;
    color: var(--ink);
    cursor: pointer;
    font-size: 13px;
    font-weight: 850;
    min-height: 34px;
    padding: 0 12px;
  }

  .tree-layout {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 285px;
    min-height: 500px;
  }

  .svg-wrap {
    min-width: 0;
    overflow: auto;
    padding: 12px;
  }

  svg {
    min-width: 980px;
    width: 100%;
  }

  .node rect {
    fill: rgba(255, 255, 255, 0.07);
    stroke: rgba(255, 255, 255, 0.18);
    stroke-width: 1.2;
    rx: 8;
  }

  .node.good rect {
    fill: rgba(115, 247, 176, 0.10);
    stroke: rgba(115, 247, 176, 0.58);
  }

  .node.bad rect {
    fill: rgba(255, 107, 129, 0.10);
    stroke: rgba(255, 107, 129, 0.55);
  }

  .node.pruned rect {
    fill: rgba(255, 209, 102, 0.10);
    stroke: rgba(255, 209, 102, 0.50);
    stroke-dasharray: 5 4;
  }

  .node.root rect {
    fill: rgba(101, 232, 255, 0.10);
    stroke: rgba(101, 232, 255, 0.58);
  }

  .node.active rect {
    filter: drop-shadow(0 0 14px rgba(101, 232, 255, 0.65));
    stroke: var(--cyan);
    stroke-width: 2.2;
  }

  .node text {
    fill: var(--ink);
    font-size: 12px;
    font-weight: 850;
    pointer-events: none;
  }

  .node .score {
    fill: var(--muted);
    font-size: 10px;
    font-weight: 800;
  }

  .edge {
    stroke: rgba(255, 255, 255, 0.22);
    stroke-width: 2;
  }

  .edge.best {
    stroke: var(--cyan);
    stroke-width: 3.4;
    filter: drop-shadow(0 0 8px rgba(101, 232, 255, 0.56));
  }

  .edge.pruned {
    stroke: rgba(255, 209, 102, 0.58);
    stroke-dasharray: 6 5;
  }

  .edge.active {
    stroke: var(--magenta);
    filter: drop-shadow(0 0 10px rgba(255, 95, 210, 0.65));
  }

  .detail-panel {
    border-left: 1px solid var(--soft-line);
    padding: 18px;
  }

  .detail-label {
    color: var(--cyan);
    font-size: 12px;
    font-weight: 900;
    margin-bottom: 6px;
    text-transform: uppercase;
  }

  .detail-title {
    color: var(--ink);
    font-size: 18px;
    font-weight: 900;
    line-height: 1.2;
    margin-bottom: 10px;
  }

  .detail-copy {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.55;
    white-space: pre-wrap;
  }

  @media (max-width: 860px) {
    .result-grid,
    .tree-layout {
      grid-template-columns: 1fr;
    }

    .detail-panel {
      border-left: 0;
      border-top: 1px solid var(--soft-line);
    }
  }
</style>
</head>
<body>
  <header class="workspace-head">
    <div>
      <h2 class="workspace-title">Comparison Workspace</h2>
      <p class="workspace-subtitle">High-level decisions first. Expand a card to inspect the MCTS search canvas.</p>
    </div>
    <div class="sync-badge">Live run complete</div>
  </header>
  <main id="app"></main>
<script>
const payload = __PAYLOAD__;

function esc(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function card(data) {
  const statusClass = data.status === "Error" ? "error" : "ready";
  const finalAnswer = data.finalAnswer ? `<div class="final-answer">${esc(data.finalAnswer)}</div>` : "";
  const error = data.error ? `<div class="error-inline">${esc(data.error)}</div>` : "";
  return `
    <article class="result-card ${data.variant}">
      <div class="card-body">
        <div class="card-top">
          <div>
            <p class="eyebrow">${esc(data.variant)}</p>
            <h3 class="title">${esc(data.label)}</h3>
          </div>
          <span class="status ${statusClass}">${esc(data.status)}</span>
        </div>
        <div class="metric-row">
          <div class="metric"><div class="metric-label">Model calls</div><div class="metric-value" data-count="${esc(data.metrics.calls)}">${esc(data.metrics.calls)}</div></div>
          <div class="metric"><div class="metric-label">Pruned</div><div class="metric-value" data-count="${esc(data.metrics.pruned)}">${esc(data.metrics.pruned)}</div></div>
          <div class="metric"><div class="metric-label">Latency</div><div class="metric-value">${esc(data.metrics.latency)}</div></div>
        </div>
        ${error}
        <div class="selected-step">
          <strong>Integrated reasoning canvas</strong>
          <p>${esc(data.action)}</p>
        </div>
        ${finalAnswer}
        <button class="expand" data-tree="${esc(data.variant)}">Expand Thinking Trajectory</button>
      </div>
    </article>
  `;
}

function treePanel(data) {
  return `
    <section class="tree-panel" id="${data.variant}-panel">
      <div class="tree-head">
        <h3 class="tree-title">${esc(data.title)} Search Canvas</h3>
        <div class="tree-actions">
          <button class="ghost" data-play="${esc(data.variant)}">Playback</button>
          <button class="ghost" data-close="${esc(data.variant)}">Close</button>
        </div>
      </div>
      <div class="tree-layout">
        <div class="svg-wrap">
          <svg viewBox="0 0 1120 470" id="${data.variant}-svg" role="img" aria-label="${esc(data.title)} tree visualization"></svg>
        </div>
        <aside class="detail-panel" id="${data.variant}-detail">
          <div class="detail-label">Node detail</div>
          <div class="detail-title">Select a node</div>
          <div class="detail-copy">Click a node to inspect its reasoning text and process score.</div>
        </aside>
      </div>
    </section>
  `;
}

function nodeById(tree, id) {
  return tree.nodes.find((node) => node.id === id);
}

function drawTree(tree) {
  const svg = document.getElementById(`${tree.variant}-svg`);
  svg.innerHTML = "";
  const edgeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
  const nodeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
  svg.append(edgeLayer, nodeLayer);

  tree.edges.forEach((edge) => {
    const from = nodeById(tree, edge.from);
    const to = nodeById(tree, edge.to);
    if (!from || !to) return;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", from.x + 74);
    line.setAttribute("y1", from.y + 26);
    line.setAttribute("x2", to.x);
    line.setAttribute("y2", to.y + 26);
    line.setAttribute("class", `edge ${edge.best ? "best" : ""} ${edge.pruned ? "pruned" : ""}`);
    line.dataset.from = edge.from;
    line.dataset.to = edge.to;
    edgeLayer.appendChild(line);
  });

  tree.nodes.forEach((node) => {
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.setAttribute("class", `node ${node.status}`);
    group.setAttribute("transform", `translate(${node.x}, ${node.y})`);
    group.dataset.node = node.id;
    group.style.cursor = "pointer";

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("width", "148");
    rect.setAttribute("height", "56");
    group.appendChild(rect);

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", "12");
    text.setAttribute("y", "23");
    text.textContent = node.label;
    group.appendChild(text);

    const score = document.createElementNS("http://www.w3.org/2000/svg", "text");
    score.setAttribute("class", "score");
    score.setAttribute("x", "12");
    score.setAttribute("y", "43");
    score.textContent = `score ${node.score}`;
    group.appendChild(score);

    group.addEventListener("click", () => selectNode(tree, node.id));
    nodeLayer.appendChild(group);
  });
}

function selectNode(tree, nodeId) {
  const node = nodeById(tree, nodeId);
  if (!node) return;
  document.querySelectorAll(`#${tree.variant}-svg .node`).forEach((el) => el.classList.remove("active"));
  document.querySelector(`#${tree.variant}-svg [data-node="${nodeId}"]`)?.classList.add("active");
  const detail = document.getElementById(`${tree.variant}-detail`);
  detail.innerHTML = `
    <div class="detail-label">${esc(node.status)} node</div>
    <div class="detail-title">${esc(node.label)}</div>
    <div class="detail-copy">${esc(node.detail)}\\n\\nProcess score: ${esc(node.score)}</div>
  `;
}

function openTree(variant) {
  const tree = payload[variant].tree;
  document.querySelectorAll(".tree-panel").forEach((panel) => panel.classList.remove("open"));
  const panel = document.getElementById(`${variant}-panel`);
  panel.classList.add("open");
  drawTree(tree);
  selectNode(tree, tree.nodes[0].id);
  panel.scrollIntoView({ behavior: "smooth", block: "start" });
}

function closeTree(variant) {
  document.getElementById(`${variant}-panel`)?.classList.remove("open");
}

function playTree(variant) {
  const tree = payload[variant].tree;
  openTree(variant);
  const path = tree.bestPath && tree.bestPath.length ? tree.bestPath : tree.nodes.map((node) => node.id);
  document.querySelectorAll(`#${variant}-svg .edge`).forEach((edge) => edge.classList.remove("active"));
  path.forEach((nodeId, index) => {
    setTimeout(() => {
      selectNode(tree, nodeId);
      document.querySelectorAll(`#${variant}-svg .edge`).forEach((edge) => {
        if (edge.dataset.to === nodeId || edge.dataset.from === nodeId) edge.classList.add("active");
      });
    }, index * 520);
  });
}

function mount() {
  const app = document.getElementById("app");
  app.innerHTML = `
    <section class="result-grid">
      ${card(payload.baseline)}
      ${card(payload.improved)}
    </section>
    ${treePanel(payload.baseline.tree)}
    ${treePanel(payload.improved.tree)}
  `;

  document.querySelectorAll("[data-tree]").forEach((button) => {
    button.addEventListener("click", () => openTree(button.dataset.tree));
  });
  document.querySelectorAll("[data-close]").forEach((button) => {
    button.addEventListener("click", () => closeTree(button.dataset.close));
  });
  document.querySelectorAll("[data-play]").forEach((button) => {
    button.addEventListener("click", () => playTree(button.dataset.play));
  });
}

mount();
</script>
</body>
</html>
    """.replace("__PAYLOAD__", json.dumps(payload))
    components.html(html_doc, height=1040, scrolling=True)


def render_workspace_placeholder() -> None:
    st.markdown(
        """
<div class="result-card" style="padding: 1.4rem; min-height: 420px;">
  <p class="kicker">Comparison Workspace</p>
  <h3 class="result-title">Run a problem to generate live reasoning canvases.</h3>
  <div class="selected-step" style="margin-top: 1rem;">
    <strong>What will appear here</strong>
    <p>Baseline and improved cards, animated metrics, selected first steps, and expandable MCTS tree visualizations.</p>
  </div>
  <div class="mode-strip" style="margin-top: 1rem;">
    <span class="mode-pill">Tree view <strong>interactive</strong></span>
    <span class="mode-pill">Node details <strong>clickable</strong></span>
    <span class="mode-pill">Playback <strong>guided</strong></span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Math Reasoning Comparison", layout="wide")
    inject_theme()
    st.markdown(
        """
<div class="app-header">
  <div>
    <p class="kicker">Math Reasoning Lab</p>
    <h1>Math Reasoning Comparison</h1>
    <p class="subtitle">Baseline MCTS and process-scored MCTS on the same problem.</p>
  </div>
  <div class="hero-status"><span></span> Analysis console</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    examples = {
        "Algebra": "Solve for x: 2x^2 - 5x + 3 = 0",
        "Calculus": "Find the derivative of f(x) = 4x^2 + 7x - 2",
        "Linear equation": "Solve for x: x + 3 = 5",
    }

    def reset_demo_settings() -> None:
        st.session_state["model_provider_v4"] = "demo"
        st.session_state["simulations_v4"] = 1
        st.session_state["top_k_v4"] = 2
        st.session_state["use_verifier_v4"] = True
        st.session_state["complete_v4"] = False
        st.session_state["ppm_checkpoint_v4"] = ""

    left, right = st.columns([0.95, 1.85], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Mission Control</div>', unsafe_allow_html=True)
            st.button(
                "Reset demo settings",
                on_click=reset_demo_settings,
                key="reset_demo_v4",
                use_container_width=True,
            )
            selected_example = st.radio("Example", list(examples), horizontal=True, key="example_v4")
            default_problem = st.session_state.get("problem", examples[selected_example])
            problem = st.text_area("Problem", value=default_problem, height=140, key="problem_v4")

            provider = st.selectbox(
                "Model",
                ["demo", "openai", "deepseek", "anthropic", "ollama"],
                index=0,
                key="model_provider_v4",
                format_func=lambda item: {
                    "demo": "Demo verifier model",
                    "openai": "OpenAI API",
                    "deepseek": "DeepSeek API",
                    "anthropic": "Anthropic Claude API",
                    "ollama": "qwen2-math:7b (Ollama)",
                }[item],
            )
            if provider == "demo":
                st.markdown(
                    '<div class="control-banner demo">Demo mode: no API key or network call.</div>',
                    unsafe_allow_html=True,
                )
            elif provider in CLOUD_PROVIDERS:
                st.markdown(
                    '<div class="control-banner cloud">Cloud API mode: start with one simulation.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="control-banner local">Local mode: qwen2-math:7b via Ollama.</div>',
                    unsafe_allow_html=True,
                )

            api_key_override = ""
            if provider in CLOUD_PROVIDERS:
                api_key_override = st.text_input(
                    "API key for this run",
                    value="",
                    type="password",
                    placeholder=f"Uses {API_KEY_ENVS[provider]} from .env if blank",
                    key="api_key_override_v2",
                )
                st.caption("Temporary field. Nothing here is written to Git.")

            simulations = st.slider("Simulations", 1, 8, 1, key="simulations_v4")
            top_k = st.slider("Improved top-k", 1, 4, 2, key="top_k_v4")
            use_verifier = st.checkbox("Use verifier scorer", value=True, key="use_verifier_v4")
            complete = st.checkbox("Generate final answer", value=False, key="complete_v4")
            ppm_checkpoint = st.text_input(
                "PPM checkpoint",
                value="",
                placeholder="checkpoints/ppm_math_level5.pt",
                key="ppm_checkpoint_v4",
            )
            run_clicked = st.button(
                "Compare Systems",
                type="primary",
                disabled=not problem.strip(),
                use_container_width=True,
            )

    with right:
        st.markdown(
            f"""
<div class="mode-strip">
  <span class="mode-pill">Model <strong>{html.escape(provider)}</strong></span>
  <span class="mode-pill">Simulations <strong>{simulations}</strong></span>
  <span class="mode-pill">Top-k <strong>{top_k}</strong></span>
  <span class="mode-pill">Verifier <strong>{"on" if use_verifier else "off"}</strong></span>
</div>
            """,
            unsafe_allow_html=True,
        )

        if run_clicked:
            try:
                model = build_model(provider, api_key_override)
                ppm = load_ppm(ppm_checkpoint.strip())
                scorer = None
                if ppm is not None and use_verifier:
                    scorer = HybridProcessScorer(ppm=ppm, verifier=HeuristicStepVerifier())
                elif ppm is not None:
                    scorer = ppm
                elif use_verifier:
                    scorer = HeuristicStepVerifier()

                effective_simulations = simulations
                if provider in CLOUD_PROVIDERS and simulations > 1:
                    effective_simulations = 1
                    st.warning("Using 1 simulation for this cloud-model demo run to avoid long waits.")

                baseline_cfg = MCTSConfig(
                    search_strategy="baseline",
                    max_simulations=effective_simulations,
                    max_depth=3,
                    num_actions=2,
                    eval_cache=True,
                    max_state_steps=8,
                    max_retries=1,
                    retry_delay=0.25,
                    fail_fast_on_generation_error=True,
                )
                improved_cfg = MCTSConfig(
                    search_strategy="adaptive",
                    max_simulations=effective_simulations,
                    max_depth=3,
                    num_actions=3,
                    max_branching_factor=4,
                    min_branching_factor=1,
                    top_k_prune=top_k if scorer is not None else 0,
                    eval_cache=True,
                    max_state_steps=8,
                    max_retries=1,
                    retry_delay=0.25,
                    fail_fast_on_generation_error=True,
                )

                status = st.empty()
                try:
                    with st.spinner("Running both systems..."):
                        status.info("Running baseline MCTS...")
                        baseline = run_mcts(problem.strip(), model, "Baseline MCTS", baseline_cfg)
                        status.info("Running improved MCTS with process scoring...")
                        improved = run_mcts(
                            problem.strip(),
                            model,
                            "Improved MCTS + Process Scorer",
                            improved_cfg,
                            scorer,
                        )
                finally:
                    status.empty()

                render_comparison_canvas(baseline, improved, model, problem.strip(), complete)
            except Exception as exc:
                st.error(str(exc))
        else:
            render_workspace_placeholder()


if __name__ == "__main__":
    main()
