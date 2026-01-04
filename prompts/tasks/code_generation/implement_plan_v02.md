# Task: Implement Reward Function Refinements

Implement the improvements suggested by the Researcher.

## The Context
You are writing the reward function for the **Training Phase**. 
- You do not need to check if the policy is deterministic or stochastic.
- You do not need to check if the reward is base or shaped. 
- You are defining the **Shaped Reward** that the agent will optimize.

## Researcher Plan
{plan}

## Requirements
1. **Logic:** Implement the math described in the plan.
2. **Robustness:** Handle `np.nan` or missing keys in `info` gracefully (default to 0.0).
3. **Signature:** Keep the exact function signature: 
   `def calculate_reward(observation, original_reward, terminated, truncated, info):`

## Existing Code
```python
{current_code}
```

## Instructions
Rewrite the function to maximize the agent's ability to solve the Base environment.