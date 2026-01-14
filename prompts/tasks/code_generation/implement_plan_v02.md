# TASK: Implement Reward Function Refinements

You must rewrite the `calculate_reward` function to implement the Researcher's plan.

## Researcher's Plan
{plan}

## Current Implementation
Below is the *current* failing code. **DO NOT output this code.** You must change it.

```python
{current_code}
```
## YOUR INSTRUCTIONS
1. **Normalization**: Ensure the final return value is roughly between -1.0 and +1.0.

2. **No Cliffs**: Avoid step functions (e.g., if y < 0.5: reward += 10). Use continuous gradients.

3. **Output**: Return ONLY the updated function.