# Member 1 Search Ablation

Deterministic scripted benchmark: no API keys required.

## Summary

| strategy | accuracy | avg_nodes | avg_simulations | avg_latency_ms | avg_model_calls | avg_est_tokens | avg_pruned |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive | 3/3 | 1.0 | 1.0 | 0.06 | 2.0 | 134.33 | 2.0 |
| baseline | 0/3 | 3.0 | 3.0 | 0.1 | 6.0 | 647.67 | 0.0 |

## Per-Problem Results

| strategy | problem | selected_action | correct | nodes_expanded | simulations | latency_ms | estimated_tokens | pruned_actions | early_stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Compute 1 + 1. | Guess 3 without checking the operation. | False | 3 | 3 | 0.2 | 616 | 0 | False |
| baseline | Solve for x: x + 3 = 5. | Guess x = 4 and move on. | False | 3 | 3 | 0.05 | 652 | 0 | False |
| baseline | Find the derivative of f(x) = x^2. | Guess the derivative is x. | False | 3 | 3 | 0.04 | 675 | 0 | False |
| adaptive | Compute 1 + 1. | FINAL ANSWER: 2 | True | 1 | 1 | 0.1 | 127 | 2 | True |
| adaptive | Solve for x: x + 3 = 5. | FINAL ANSWER: x = 2 | True | 1 | 1 | 0.04 | 135 | 2 | True |
| adaptive | Find the derivative of f(x) = x^2. | FINAL ANSWER: 2x | True | 1 | 1 | 0.04 | 141 | 2 | True |
