import os
import sys
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
load_dotenv()

from src.models.model_interface import OpenAIModel
from src.core.mcts import MCTS
from src.core.ppm import ProcessPreferenceModel, PPMConfig

st.title("Math Reasoning")

col1, col2 = st.columns(2)
if col1.button("Algebra example"):
    st.session_state.problem = "Solve for x: 2x² - 5x + 3 = 0"
if col2.button("Calculus example"):
    st.session_state.problem = "Find the derivative of f(x) = 4x² + 7x - 2"

problem = st.text_area("Problem", value=st.session_state.get("problem", ""),
                        placeholder="e.g. Find the derivative of f(x) = x² + 3x")

if st.button("Solve", type="primary", disabled=not problem.strip()):
    model = OpenAIModel(os.getenv("OPENAI_API_KEY"))
    mcts = MCTS.from_config_file("config/default.json")
    ppm = ProcessPreferenceModel(PPMConfig(input_dim=1536, hidden_dim=256))

    with st.spinner("Thinking…"):
        action, trajectory = mcts.search(problem.strip(), model)

        # Pick the deepest state MCTS found, then ask the model to finish
        if trajectory:
            best_state = max(trajectory, key=lambda e: len(e["state"]))["state"]
        else:
            best_state = problem.strip()

        completion_prompt = f"""You are solving a math problem. Continue from where the solution left off and write the remaining steps to reach the final answer.

Problem and steps so far:
{best_state}

Continue step by step. End with:
FINAL ANSWER: <answer in plain text, no LaTeX>"""
        completion = model.generate_response(completion_prompt, temperature=0.3, max_tokens=500)

    def render(text):
        return text.replace(r"\(", "$").replace(r"\)", "$").replace(r"\[", "$$").replace(r"\]", "$$")

    seen = set()
    for step in trajectory:
        lines = [l.strip() for l in step["state"].split("\n") if l.strip()]
        new_step = lines[-1]
        if new_step in seen or new_step == problem.strip():
            continue
        seen.add(new_step)
        confidence = float(1 / (1 + abs(ppm.evaluate_step(new_step, model))))
        st.markdown(render(new_step))
        st.progress(confidence, text=f"Confidence: {confidence:.0%}")

    for line in completion.strip().splitlines():
        if "final answer:" in line.lower():
            st.success(render(line.strip()))
            break
