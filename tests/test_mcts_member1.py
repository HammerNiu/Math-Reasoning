import unittest

from src.core.mcts import MCTS, MCTSConfig


class ScriptedModel:
    def generate_response(self, prompt, temperature=0.7, max_tokens=1000):
        if "Guess 3 without checking" in prompt:
            return "STEP: FINAL ANSWER: incorrect"
        return "\n".join([
            "STEP: Guess 3 without checking the operation.",
            "STEP: FINAL ANSWER: 2",
            "STEP: Add the two numbers carefully.",
        ])

    def evaluate_reasoning(self, problem, solution_steps):
        joined = " ".join(solution_steps).lower()
        if "final answer: 2" in joined:
            return 1.0
        if "final answer:" in joined:
            return 0.0
        return 0.35

    def embed_text(self, text):
        return [0.0] * 1536


class MemberOneMCTSTest(unittest.TestCase):
    def test_baseline_preserves_first_candidate_behavior(self):
        mcts = MCTS(MCTSConfig(
            max_simulations=2,
            max_depth=2,
            num_actions=3,
            search_strategy="baseline"
        ))

        action, _ = mcts.search("Compute 1 + 1.", ScriptedModel())

        self.assertEqual(action, "Guess 3 without checking the operation.")
        self.assertFalse(mcts.last_stats["early_stopped"])

    def test_adaptive_search_prefers_promising_terminal_step(self):
        mcts = MCTS(MCTSConfig(
            max_simulations=2,
            max_depth=2,
            num_actions=3,
            search_strategy="adaptive",
            max_branching_factor=4,
            prune_threshold=0.25,
            early_stop_reward=0.95
        ))

        action, trajectory = mcts.search("Compute 1 + 1.", ScriptedModel())

        self.assertEqual(action, "FINAL ANSWER: 2")
        self.assertTrue(mcts.last_stats["early_stopped"])
        self.assertGreaterEqual(mcts.last_stats["pruned_actions"], 1)
        self.assertEqual(trajectory[0]["phase"], "expansion")


if __name__ == "__main__":
    unittest.main()
