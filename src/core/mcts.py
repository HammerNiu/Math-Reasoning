"""
Monte Carlo Tree Search (MCTS) implementation for mathematical reasoning
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import json
import re

@dataclass
class MCTSConfig:
    exploration_weight: float = 1.0
    max_simulations: int = 3
    max_depth: int = 5
    num_actions: int = 2  # candidate next steps per node

class MCTSNode:
    def __init__(self, state: str, parent: Optional['MCTSNode'] = None, action: Optional[str] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions: List[str] = []

    def add_child(self, action: str, state: str) -> 'MCTSNode':
        child = MCTSNode(state=state, parent=self, action=action)
        self.children.append(child)
        if action in self.untried_actions:
            self.untried_actions.remove(action)
        return child

    def update(self, reward: float) -> None:
        self.visits += 1
        self.value += (reward - self.value) / self.visits

    def get_ucb_score(self, exploration_weight: float) -> float:
        if self.visits == 0:
            return float('inf')
        exploitation = self.value
        exploration = exploration_weight * np.sqrt(2 * np.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def is_terminal(self) -> bool:
        return _is_terminal(self.state)

    def get_possible_actions(self) -> List[str]:
        return []

class MCTS:
    def __init__(self, config: Optional[MCTSConfig] = None):
        self.config = config or MCTSConfig()
        self._model = None

    def set_model(self, model: Any) -> None:
        self._model = model

    @classmethod
    def from_config_file(cls, config_path: str) -> 'MCTS':
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        config = MCTSConfig(**config_data['mcts'])
        return cls(config)

    def select_action(self, node: MCTSNode) -> Tuple[MCTSNode, str]:
        if not node.children:
            return node, ""
        ucb_scores = [
            child.get_ucb_score(self.config.exploration_weight)
            for child in node.children
        ]
        selected_child = node.children[np.argmax(ucb_scores)]
        return selected_child, selected_child.action

    def expand(self, node: MCTSNode) -> Tuple[MCTSNode, str]:
        if not node.untried_actions:
            node.untried_actions = self.get_possible_actions(node.state)
        if not node.untried_actions:
            return node, ""
        action = node.untried_actions[0]
        new_state = self.apply_action(node.state, action)
        child = node.add_child(action, new_state)
        return child, action

    def simulate(self, state: str, depth: int = 0) -> float:
        if depth >= self.config.max_depth or self.is_terminal_state(state):
            return self.evaluate_state(state)
        actions = self.get_possible_actions(state)
        if not actions:
            return self.evaluate_state(state)
        action = actions[0]
        new_state = self.apply_action(state, action)
        return self.simulate(new_state, depth + 1)

    def backpropagate(self, node: MCTSNode, reward: float) -> None:
        while node is not None:
            node.update(reward)
            node = node.parent

    def search(self, root_state: str, model: Any = None) -> Tuple[str, List[Dict[str, Any]]]:
        if model is not None:
            self._model = model

        root = MCTSNode(state=root_state)
        trajectory = []

        for _ in range(self.config.max_simulations):
            node = root

            # Selection
            while node.children and not node.untried_actions:
                node, action = self.select_action(node)
                trajectory.append({
                    "state": node.state,
                    "action": action,
                    "value": node.value,
                    "visits": node.visits
                })

            # Expansion
            if not node.is_terminal():
                node, action = self.expand(node)
                if action:
                    trajectory.append({
                        "state": node.state,
                        "action": action,
                        "value": node.value,
                        "visits": node.visits
                    })

            # Simulation
            reward = self.simulate(node.state)

            # Backpropagation
            self.backpropagate(node, reward)

        if not root.children:
            # Fallback: generate one direct solution
            actions = self.get_possible_actions(root_state)
            if actions:
                final_state = self.apply_action(root_state, actions[0])
                trajectory.append({"state": final_state, "action": actions[0], "value": 0.0, "visits": 1})
                return actions[0], trajectory
            return "", trajectory

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action, trajectory

    # ── Domain-specific implementations for math reasoning ──────────────────

    def get_possible_actions(self, state: str) -> List[str]:
        """Ask the LLM for candidate next reasoning steps."""
        if self._model is None or self.is_terminal_state(state):
            return []

        prompt = f"""You are solving a math problem step by step.

Current reasoning so far:
{state}

Generate {self.config.num_actions} different candidate next reasoning steps.
Each step should advance the solution. If the problem is solved, write the final step as:
FINAL ANSWER: <answer>

Use plain text only, no LaTeX. Format: output each step on its own line, prefixed with "STEP:" like:
STEP: <step text>
STEP: <step text>"""

        try:
            response = self._model.generate_response(prompt, temperature=0.8, max_tokens=300)
            steps = re.findall(r'STEP:\s*(.+)', response)
            return [s.strip() for s in steps if s.strip()] or [response.strip()]
        except Exception:
            return []

    def apply_action(self, state: str, action: str) -> str:
        """Append a reasoning step to the current state."""
        return f"{state}\n{action}"

    def is_terminal_state(self, state: str) -> bool:
        return _is_terminal(state)

    def evaluate_state(self, state: str) -> float:
        """Score the current reasoning state using the LLM."""
        if self._model is None:
            return 0.5
        # Quick heuristic: terminal states with an answer score higher
        if self.is_terminal_state(state):
            lines = [l for l in state.split('\n') if l.strip()]
            problem = lines[0] if lines else ""
            steps = lines[1:] if len(lines) > 1 else [state]
            try:
                return self._model.evaluate_reasoning(problem, steps)
            except Exception:
                return 0.7
        return 0.3


def _is_terminal(state: str) -> bool:
    # Only check the last non-empty line to avoid matching the problem statement
    lines = [l.strip() for l in state.split('\n') if l.strip()]
    if len(lines) <= 1:
        return False
    last = lines[-1].lower()
    markers = ["final answer:", "therefore the answer is", "the answer is"]
    return any(m in last for m in markers)
