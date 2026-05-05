import os
import re
import sys
import time
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

from dotenv import load_dotenv

sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

from src.core.mcts import MCTS, MCTSConfig
from src.core.ppm import PPMConfig, ProcessPreferenceModel
from src.core.scoring import HeuristicStepVerifier, HybridProcessScorer
from src.models.model_interface import LocalEmbedder, ModelFactory


API_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}
CLOUD_PROVIDERS = {"openai", "anthropic", "deepseek"}


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


def build_model(provider: str):
    if provider == "demo":
        return DemoMathModel()
    if provider == "ollama":
        return ModelFactory.create_model(
            "ollama",
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5-math:7b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    key = os.getenv(API_KEY_ENVS[provider], "")
    if not key:
        raise RuntimeError(f"{API_KEY_ENVS[provider]} is not set.")
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


def render_result(result, model, problem: str, complete: bool):
    st.subheader(result["label"])
    stats = result["stats"]
    c1, c2, c3 = st.columns(3)
    failed_calls = stats.get("failed_model_calls", 0)
    call_text = str(stats.get("model_calls", 0))
    if failed_calls:
        call_text = f"{call_text} ok / {failed_calls} failed"
    c1.metric("Model calls", call_text)
    c2.metric("Pruned", stats.get("pruned_actions", 0))
    c3.metric("Latency", f"{result['elapsed']:.2f}s")

    if stats.get("last_error"):
        st.error(stats["last_error"])

    st.markdown("**Selected first step**")
    st.write(result["action"] or "No action selected.")

    if complete:
        completion = finish_solution(problem, result["trajectory"], model)
        final_lines = [line for line in completion.splitlines() if "final answer:" in line.lower()]
        st.markdown("**Final answer**")
        st.success(final_lines[-1].strip() if final_lines else completion.strip())

    with st.expander("Reasoning trajectory"):
        seen = set()
        for entry in result["trajectory"]:
            lines = [line.strip() for line in entry.get("state", "").splitlines() if line.strip()]
            if not lines:
                continue
            step = lines[-1]
            if step == problem or step in seen:
                continue
            seen.add(step)
            st.write(step)


def main() -> None:
    st.set_page_config(page_title="Math Reasoning Comparison", layout="wide")
    st.title("Math Reasoning Comparison")

    examples = {
        "Algebra": "Solve for x: 2x^2 - 5x + 3 = 0",
        "Calculus": "Find the derivative of f(x) = 4x^2 + 7x - 2",
        "Linear equation": "Solve for x: x + 3 = 5",
    }

    def reset_demo_settings() -> None:
        st.session_state["model_provider_v3"] = "demo"
        st.session_state["simulations_v3"] = 1
        st.session_state["top_k_v3"] = 2
        st.session_state["use_verifier_v3"] = True
        st.session_state["complete_v3"] = False
        st.session_state["ppm_checkpoint_v3"] = ""

    left, right = st.columns([2, 1])
    with right:
        st.button("Reset demo settings", on_click=reset_demo_settings, key="reset_demo_v3")
        provider = st.selectbox(
            "Model",
            ["demo", "openai", "deepseek", "anthropic", "ollama"],
            index=0,
            key="model_provider_v3",
        )
        if provider == "demo":
            st.success("Demo model selected: no API key, no network calls.")
        elif provider in CLOUD_PROVIDERS:
            st.warning("Cloud model selected: this will call a real API. Start with one simulation.")
        else:
            st.info("Ollama selected: make sure the local Ollama server is running.")
        simulations = st.slider("Simulations", 1, 8, 1, key="simulations_v3")
        top_k = st.slider("Improved top-k", 1, 4, 2, key="top_k_v3")
        use_verifier = st.checkbox("Use verifier scorer", value=True, key="use_verifier_v3")
        complete = st.checkbox("Generate final answer", value=False, key="complete_v3")
        ppm_checkpoint = st.text_input(
            "PPM checkpoint",
            value="",
            placeholder="checkpoints/ppm_math_level5.pt",
            key="ppm_checkpoint_v3",
        )

    with left:
        selected_example = st.radio("Example", list(examples), horizontal=True, key="example_v3")
        default_problem = st.session_state.get("problem", examples[selected_example])
        problem = st.text_area("Problem", value=default_problem, height=120, key="problem_v3")

    if st.button("Compare Baseline And Improved", type="primary", disabled=not problem.strip()):
        try:
            model = build_model(provider)
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
                status.empty()

            col_a, col_b = st.columns(2)
            with col_a:
                render_result(baseline, model, problem.strip(), complete)
            with col_b:
                render_result(improved, model, problem.strip(), complete)
        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
