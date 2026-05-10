# Member 2 Scoring Ablation

Deterministic benchmark: no API keys required.

Corrected mis-rankings: 3/3.

| problem | old_top | hybrid_top | old_good | old_bad | hybrid_good | hybrid_bad | fixed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Solve for x: x + 3 = 5. | Guess x = 4 without checking the equation. | Subtract 3 from both sides to get x = 2. | 0.4 | 0.9 | 0.616 | 0.476 | True |
| Compute 1 + 1. | Maybe the answer is 3. | Add 1 and 1 to get 2. | 0.4 | 0.9 | 0.616 | 0.476 | True |
| Find the derivative of f(x) = x^2. | Skip the power rule and guess x. | Apply the power rule: derivative of x^2 is 2x. | 0.4 | 0.9 | 0.616 | 0.38 | True |
