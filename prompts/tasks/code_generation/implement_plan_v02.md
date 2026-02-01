# TASK: Implement Reward Function Refinements

You must rewrite the `calculate_reward` function to implement the Researcher's plan.

## The Context
You are writing the reward function for the **Training Phase**. 
- You are defining the **Shaped Reward** that the agent will optimize.
- The environment wrapper handles the sparse goal reward (+100/-100). You must simply add shaping signals.

## Researcher's Plan
{plan}

## Current Implementation
Below is the *current* failing code. **DO NOT output this code.** You must change it.

```python
{current_code}
```
## YOUR INSTRUCTIONS
1. **Guided Implementation**: Follow the Researcher's instructions precisely. Do not apply modifications unless explicitly requested in the Plan.

2. **Reward Signal**: Prefer continuous gradients, but discrete bonuses for landing are permitted.

3. **Output**: Return ONLY the updated function.
