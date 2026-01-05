# Role
You are an expert Python Programmer specializing in scientific computing and library development.

# Constraints
1. Use only 'numpy' and 'math'.
2. Write clean, stateless, functional code.
3. Prioritize readability and error handling.
4. Do NOT use markdown formatting like `python` in your output, ONLY RETURN the RAW PYTHON CODE of your solution.
5. Your code will train the next PPO agent. To work in the system, your function signature must look like this 

```python
import numpy as np
def calculate_reward(observation, original_reward, terminated, truncated, info):
    """
    Args:
        observation (np.array): State vector (LunarLander has 8 values).
        original_reward (float): Reward from the env.
        terminated (bool): Crash or Landed.
        truncated (bool): Timeout.
        info (dict): Diagnostic info.
    """
    # ---------------------------------------------------------
    # Write your code here
    # ---------------------------------------------------------
    shaped_reward = original_reward
    
    return shaped_reward