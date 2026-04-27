import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


"""
Monte Carlo Tree Search (MCTS) implementation for mathematical reasoning.

The default "baseline" strategy preserves the simple repo behavior. The
"adaptive" strategy is Member 1's search-side innovation: it asks for more
candidate steps near the root, orders them by terminal promise and diversity,
prunes weak near-duplicates, and stops early when a high-confidence answer is
found.
"""

@dataclass
class MCTSConfig:
    exploration_weight: float = 1.0
    max_simulations: int = 3
    max_depth: int = 5
    num_actions: int = 2  # candidate next steps per node
    search_strategy: str = "baseline"  # "baseline" or "adaptive"
    max_branching_factor: int = 4
    min_branching_factor: int = 1
    diversity_weight: float = 0.25
    prune_threshold: float = 0.25
    early_stop_reward: float = 0.95
    early_stop_min_simulations: int = 1
    seed: Optional[int] = None

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
        parent_visits = max(1, self.parent.visits if self.parent else 1)
        exploration = exploration_weight * np.sqrt(2 * np.log(parent_visits) / self.visits)
        return exploitation + exploration

    def is_terminal(self) -> bool:
        return _is_terminal(self.state)

    def get_possible_actions(self) -> List[str]:
        return []

    def depth(self) -> int:
        depth = 0
        node = self.parent
        while node is not None:
            depth += 1
            node = node.parent
        return depth


class MCTS:
    def __init__(self, config: Optional[MCTSConfig] = None):
        self.config = config or MCTSConfig()
        self._model = None
        self._rng = np.random.default_rng(self.config.seed)
        self.last_stats: Dict[str, Any] = {}

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
            actions = self.get_possible_actions(node.state)
            node.untried_actions = self._prepare_actions(actions, node.state, node)
        if not node.untried_actions:
            return node, ""
        action = node.untried_actions[0]
        new_state = self.apply_action(node.state, action)
        child = node.add_child(action, new_state)
        self._stats_increment("nodes_expanded")
        if child.is_terminal():
            self._stats_increment("terminal_nodes")
        return child, action

    def simulate(self, state: str, depth: int = 0) -> float:
        self.last_stats["max_depth_reached"] = max(
            self.last_stats.get("max_depth_reached", 0),
            depth
        )
        if depth >= self.config.max_depth or self.is_terminal_state(state):
            return self.evaluate_state(state)
        actions = self.get_possible_actions(state)
        actions = self._prepare_actions(actions, state, None)
        if not actions:
            return self.evaluate_state(state)
        action = self._select_rollout_action(actions, state)
        new_state = self.apply_action(state, action)
        return self.simulate(new_state, depth + 1)

    def backpropagate(self, node: MCTSNode, reward: float) -> None:
        while node is not None:
            node.update(reward)
            node = node.parent

    def search(self, root_state: str, model: Any = None) -> Tuple[str, List[Dict[str, Any]]]:
        if model is not None:
            self._model = model

        started_at = time.perf_counter()
        self._init_stats(root_state)
        root = MCTSNode(state=root_state)
        trajectory = []

        for simulation_index in range(self.config.max_simulations):
            self._stats_increment("simulations_run")
            node = root

            # Selection
            while node.children and not node.untried_actions:
                node, action = self.select_action(node)
                trajectory.append(self._trajectory_entry(
                    node=node,
                    action=action,
                    phase="selection",
                    simulation=simulation_index
                ))
                if node.is_terminal():
                    break

            # Expansion
            if not node.is_terminal():
                node, action = self.expand(node)
                if action:
                    trajectory.append(self._trajectory_entry(
                        node=node,
                        action=action,
                        phase="expansion",
                        simulation=simulation_index
                    ))

            # Simulation
            reward = self.simulate(node.state, node.depth())
            self.last_stats["best_reward"] = max(
                self.last_stats.get("best_reward", 0.0),
                reward
            )

            # Backpropagation
            self.backpropagate(node, reward)

            if self._should_early_stop(reward, simulation_index + 1):
                self.last_stats["early_stopped"] = True
                break

        self.last_stats["latency_seconds"] = time.perf_counter() - started_at

        if not root.children:
            # Fallback: generate one direct solution
            actions = self.get_possible_actions(root_state)
            if actions:
                final_state = self.apply_action(root_state, actions[0])
                trajectory.append({"state": final_state, "action": actions[0], "value": 0.0, "visits": 1})
                return actions[0], trajectory
            return "", trajectory

        best_child = self._best_child(root)
        return best_child.action, trajectory

    # Domain-specific implementations for math reasoning

    def get_possible_actions(self, state: str) -> List[str]:
        """Ask the LLM for candidate next reasoning steps."""
        if self._model is None or self.is_terminal_state(state):
            return []

        action_count = self._candidate_count_for_state(state)
        prompt = f"""You are solving a math problem step by step.

Current reasoning so far:
{state}

Generate {action_count} different candidate next reasoning steps.
Each step should advance the solution. If the problem is solved, write the final step as:
FINAL ANSWER: <answer>

Use plain text only, no LaTeX. Format: output each step on its own line, prefixed with "STEP:" like:
STEP: <step text>
STEP: <step text>"""

        try:
            response = self._model.generate_response(prompt, temperature=0.8, max_tokens=300)
            self._record_model_generation(prompt, response)
            steps = re.findall(r'^\s*(?:[-*]\s*)?STEP:\s*(.+)$', response, flags=re.IGNORECASE | re.MULTILINE)
            if not steps:
                steps = [line.strip() for line in response.splitlines() if line.strip()]
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
                self._record_model_evaluation(problem, steps)
                return self._model.evaluate_reasoning(problem, steps)
            except Exception:
                return 0.7
        if self._uses_adaptive_search():
            return self._estimate_state_progress(state)
        return 0.3

    # Member 1 adaptive search helpers

    def _uses_adaptive_search(self) -> bool:
        return self.config.search_strategy.lower() in {"adaptive", "improved", "member1"}

    def _candidate_count_for_state(self, state: str) -> int:
        if not self._uses_adaptive_search():
            return self.config.num_actions
        depth = self._state_depth(state)
        if depth <= 1:
            return max(self.config.num_actions, self.config.max_branching_factor)
        return max(self.config.min_branching_factor, self.config.num_actions)

    def _prepare_actions(
        self,
        actions: Sequence[str],
        state: str,
        node: Optional[MCTSNode]
    ) -> List[str]:
        if not actions:
            return []

        if not self._uses_adaptive_search():
            return [action for action in actions if action.strip()]

        self._stats_increment("generated_actions", len(actions))
        unique_actions = self._dedupe_actions(actions)
        if len(unique_actions) < len(actions):
            self._stats_increment("pruned_actions", len(actions) - len(unique_actions))

        scored_actions = [
            (self._score_action(action, state, node), action)
            for action in unique_actions
        ]
        scored_actions.sort(key=lambda item: item[0], reverse=True)

        if not scored_actions:
            return []

        best_score = scored_actions[0][0]
        min_keep = max(1, self.config.min_branching_factor)
        kept = [
            action for score, action in scored_actions
            if score >= best_score - self.config.prune_threshold
        ]
        if len(kept) < min_keep:
            kept = [action for _, action in scored_actions[:min_keep]]

        max_keep = max(min_keep, self.config.max_branching_factor)
        kept = kept[:max_keep]
        self._stats_increment("pruned_actions", len(scored_actions) - len(kept))
        return kept

    def _dedupe_actions(self, actions: Sequence[str]) -> List[str]:
        seen = set()
        unique = []
        for action in actions:
            cleaned = action.strip()
            if not cleaned:
                continue
            key = self._normalize_text(cleaned)
            if key in seen:
                continue
            seen.add(key)
            unique.append(cleaned)
        return unique

    def _score_action(
        self,
        action: str,
        state: str,
        node: Optional[MCTSNode]
    ) -> float:
        value_score = self._estimate_action_value(action, state)
        diversity_score = self._estimate_action_diversity(action, state, node)
        diversity_weight = min(max(self.config.diversity_weight, 0.0), 1.0)
        return (1.0 - diversity_weight) * value_score + diversity_weight * diversity_score

    def _estimate_action_value(self, action: str, state: str) -> float:
        lower = action.lower()
        score = 0.35

        if _is_terminal(self.apply_action(state, action)):
            score += 0.45
        if re.search(r'[-+*/=^]|\d', action):
            score += 0.10
        if any(word in lower for word in [
            "because", "therefore", "substitute", "simplify", "factor",
            "differentiate", "solve", "check", "verify"
        ]):
            score += 0.08
        if len(action.split()) < 3:
            score -= 0.12
        if any(word in lower for word in ["guess", "maybe", "unrelated", "skip"]):
            score -= 0.18

        return max(0.0, min(1.0, score))

    def _estimate_action_diversity(
        self,
        action: str,
        state: str,
        node: Optional[MCTSNode]
    ) -> float:
        references = [line for line in state.splitlines() if line.strip()]
        if node is not None:
            references.extend(child.action for child in node.children if child.action)
        if not references:
            return 1.0
        max_similarity = max(self._jaccard_similarity(action, ref) for ref in references)
        return 1.0 - max_similarity

    def _select_rollout_action(self, actions: Sequence[str], state: str) -> str:
        return actions[0]

    def _best_child(self, root: MCTSNode) -> MCTSNode:
        if not self._uses_adaptive_search():
            return max(root.children, key=lambda child: child.visits)
        return max(root.children, key=lambda child: (child.value, child.visits))

    def _should_early_stop(self, reward: float, simulations_run: int) -> bool:
        if not self._uses_adaptive_search():
            return False
        if simulations_run < self.config.early_stop_min_simulations:
            return False
        return reward >= self.config.early_stop_reward

    def _estimate_state_progress(self, state: str) -> float:
        depth = self._state_depth(state)
        has_math = bool(re.search(r'[-+*/=^]|\d', state))
        progress = 0.25 + min(depth, self.config.max_depth) * 0.05
        if has_math:
            progress += 0.05
        return max(0.0, min(0.65, progress))

    def _state_depth(self, state: str) -> int:
        return max(0, len([line for line in state.splitlines() if line.strip()]) - 1)

    def _trajectory_entry(
        self,
        node: MCTSNode,
        action: str,
        phase: str,
        simulation: int
    ) -> Dict[str, Any]:
        return {
            "state": node.state,
            "action": action,
            "value": node.value,
            "visits": node.visits,
            "phase": phase,
            "depth": node.depth(),
            "simulation": simulation
        }

    def _init_stats(self, root_state: str) -> None:
        self.last_stats = {
            "strategy": self.config.search_strategy,
            "root_problem": root_state,
            "simulations_run": 0,
            "nodes_expanded": 0,
            "terminal_nodes": 0,
            "generated_actions": 0,
            "pruned_actions": 0,
            "model_calls": 0,
            "estimated_tokens": 0,
            "best_reward": 0.0,
            "max_depth_reached": 0,
            "early_stopped": False,
            "latency_seconds": 0.0
        }

    def _stats_increment(self, key: str, amount: int = 1) -> None:
        if not self.last_stats:
            return
        self.last_stats[key] = self.last_stats.get(key, 0) + amount

    def _record_model_generation(self, prompt: str, response: str) -> None:
        self._stats_increment("model_calls")
        self._stats_increment("estimated_tokens", self._estimate_tokens(prompt) + self._estimate_tokens(response))
        self._record_provider_token_usage()

    def _record_model_evaluation(self, problem: str, steps: Sequence[str]) -> None:
        self._stats_increment("model_calls")
        text = problem + "\n" + "\n".join(steps)
        self._stats_increment("estimated_tokens", self._estimate_tokens(text))
        self._record_provider_token_usage()

    def _record_provider_token_usage(self) -> None:
        if self._model is None:
            return
        usage = getattr(self._model, "last_usage", None) or getattr(self._model, "last_token_usage", None)
        if isinstance(usage, dict):
            total = usage.get("total_tokens") or usage.get("total")
            if isinstance(total, int):
                self.last_stats["provider_tokens"] = self.last_stats.get("provider_tokens", 0) + total

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _normalize_text(self, text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

    def _jaccard_similarity(self, left: str, right: str) -> float:
        left_tokens = set(re.findall(r"[a-z0-9]+", left.lower()))
        right_tokens = set(re.findall(r"[a-z0-9]+", right.lower()))
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _is_terminal(state: str) -> bool:
    # Only check the last non-empty line to avoid matching the problem statement
    lines = [l.strip() for l in state.split('\n') if l.strip()]
    if len(lines) <= 1:
        return False
    last = lines[-1].lower()
    markers = ["final answer:", "therefore the answer is", "the answer is"]
    return any(m in last for m in markers)
