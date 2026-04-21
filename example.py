import os
from dotenv import load_dotenv
from src.models.model_interface import OpenAIModel
from src.core.mcts import MCTS
from src.core.ppm import ProcessPreferenceModel, PPMConfig

load_dotenv()

model = OpenAIModel(os.getenv("OPENAI_API_KEY"))
mcts = MCTS.from_config_file("config/default.json")
ppm = ProcessPreferenceModel(PPMConfig(input_dim=1536, hidden_dim=256))

problem = "Solve for x: 2x² - 5x + 3 = 0"
print(f"Problem: {problem}\n")

action, trajectory = mcts.search(problem, model)

seen = set()
for step in trajectory:
    lines = [l.strip() for l in step["state"].split("\n") if l.strip()]
    new_step = lines[-1]
    if new_step in seen or new_step == problem.strip():
        continue
    seen.add(new_step)
    confidence = float(1 / (1 + abs(ppm.evaluate_step(new_step, model))))
    print(f"Step: {new_step}")
    print(f"Confidence: {confidence:.2f}\n")

# Complete the solution from the deepest MCTS state
best_state = max(trajectory, key=lambda e: len(e["state"]))["state"] if trajectory else problem
completion = model.generate_response(
    f"Continue solving this math problem step by step from where it left off. End with: FINAL ANSWER: <answer in plain text, no LaTeX>\n\n{best_state}",
    temperature=0.1, max_tokens=500
)
for line in completion.strip().splitlines():
    if "final answer:" in line.lower():
        print(line.strip())
        break
