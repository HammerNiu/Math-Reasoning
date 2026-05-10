# Member 1 Search Ablation

Deterministic scripted benchmark: no API keys required.

## Summary

| strategy | accuracy | avg_nodes | avg_simulations | avg_latency_ms | avg_model_calls | avg_est_tokens | avg_pruned |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive | 3/3 | 1.0 | 1.0 | 0.06 | 2.0 | 908.33 | 2.0 |
| baseline | 0/3 | 3.0 | 3.0 | 0.13 | 6.0 | 4516.33 | 0.0 |

## Per-Problem Results

| strategy | problem | selected_action | correct | nodes_expanded | simulations | latency_ms | estimated_tokens | pruned_actions | early_stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Compute 1 + 1. | Guess 3 without checking the operation. | False | 3 | 3 | 0.25 | 4474 | 0 | False |
| baseline | Solve for x: x + 3 = 5. | Guess x = 4 and move on. | False | 3 | 3 | 0.07 | 4519 | 0 | False |
| baseline | Find the derivative of f(x) = x^2. | Guess the derivative is x. | False | 3 | 3 | 0.06 | 4556 | 0 | False |
| adaptive | Compute 1 + 1. | FINAL ANSWER: 2 | True | 1 | 1 | 0.1 | 899 | 2 | True |
| adaptive | Solve for x: x + 3 = 5. | FINAL ANSWER: x = 2 | True | 1 | 1 | 0.04 | 908 | 2 | True |
| adaptive | Find the derivative of f(x) = x^2. | FINAL ANSWER: 2x | True | 1 | 1 | 0.04 | 918 | 2 | True |
