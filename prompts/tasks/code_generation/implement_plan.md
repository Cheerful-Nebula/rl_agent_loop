# Task: Implement Reward Function Refinements

Implement the improvements suggested by the Researcher.

## The Context
You are writing the reward function for the **Training Phase**. 
- You are defining the **Shaped Reward** that the agent will optimize.
- The environment wrapper handles the sparse goal reward (+100/-100). You must simply add shaping signals.

## Researcher Plan
{plan}

## Requirements
1. **Logic:** Implement the math described in the plan.
2. **Robustness:** Handle `np.nan` or missing keys in `info` gracefully (default to 0.0).
3. **Signature:** You **MUST** adhere to the function signature defined in your System Instructions.

## Existing Code
```python
{current_code}
```
## Instructions
Rewrite the function to maximize the agent's ability to solve the Base environment.