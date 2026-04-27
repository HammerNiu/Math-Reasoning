# Member 1: Search / MCTS Lead

## Goal

Own the search-side innovation for the Math Reasoning project. The local repo is a simplified demonstrator inspired by `ai-in-pm/rStar-Math`: MCTS generates reasoning trajectories, PPM/verifier-style scoring rates them, and the demo should compare baseline vs improved behavior.

## Baseline MCTS Flow

1. Selection: choose child nodes with UCB.
2. Expansion: ask the model for candidate next reasoning steps.
3. Simulation: roll out one candidate path until terminal state or max depth.
4. Backpropagation: update visit counts and average rewards.

The original local baseline always expands and simulates the first candidate returned by the model. That is easy to reproduce, but fragile: if the first candidate is a plausible-looking distraction, the search wastes simulations and can return the wrong first action.

## Innovation: Adaptive Diversity-Aware MCTS

Implemented in `src/core/mcts.py` behind `MCTSConfig.search_strategy = "adaptive"`.

What changed:

- Adaptive branching: ask for more candidates near the root with `max_branching_factor`.
- Diversity-aware ranking: deduplicate candidates and prefer steps that add new information instead of repeating existing reasoning.
- Promise heuristic: prioritize terminal final-answer steps, mathematically grounded steps, and verification/simplification language.
- Lightweight pruning: drop weak near-duplicates using `prune_threshold`.
- Early stopping: stop search when a terminal state receives a high reward.
- Metrics: every search exposes `mcts.last_stats` with simulations, nodes expanded, terminal nodes, latency, model calls, estimated tokens, pruned actions, and early-stop status.

The baseline path remains available with `MCTSConfig(search_strategy="baseline")`.

## Ablation

Run:

```bash
python3 tools/member1_search_ablation.py
```

This deterministic benchmark uses a scripted model so it does not need API keys. It isolates search behavior by returning one distracting first candidate and one correct final-answer candidate for each problem.

| strategy | accuracy | avg_nodes | avg_simulations | avg_latency_ms | avg_model_calls | avg_est_tokens | avg_pruned |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive | 3/3 | 1.0 | 1.0 | <1 | 2.0 | 134.33 | 2.0 |
| baseline | 0/3 | 3.0 | 3.0 | <1 | 6.0 | 647.67 | 0.0 |

Interpretation: on this controlled ablation, adaptive MCTS selects the correct terminal step earlier, expands fewer nodes, and uses fewer model calls/tokens. A real-LLM experiment can reuse the same metrics by swapping `ScriptedMathModel` with `OpenAIModel` or another model interface.

## Short Demo Example

Problem:

```text
Solve for x: x + 3 = 5.
```

Scripted candidates:

```text
STEP: Guess x = 4 and move on.
STEP: FINAL ANSWER: x = 2
STEP: Subtract 3 from both sides to get x = 2.
```

Baseline result:

```text
Guess x = 4 and move on.
```

Adaptive result:

```text
FINAL ANSWER: x = 2
```

## Slide 4 Draft: Modified MCTS

Title: Adaptive Diversity-Aware MCTS

Claim: The baseline search follows the first model suggestion too strongly. The modified search ranks and prunes candidate reasoning steps before expansion, which reduces wasted rollout paths and can stop as soon as a high-confidence final answer is found.

Visual structure:

- Left: baseline MCTS tree, first branch selected even when wrong.
- Right: adaptive MCTS tree, duplicate/weak branches pruned, final-answer branch selected.
- Bottom metrics: accuracy, expanded nodes, latency, estimated token use.

Speaker note: This is a search-side improvement, not a new scorer. It is designed to integrate cleanly with Member 2's improved PPM/verifier by replacing the lightweight heuristic score with verifier scores later.
