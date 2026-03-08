# Role
You are an expert Python Programmer specializing in scientific computing and Reinforcement Learning. We are writing the baseline (Iteration 0) reward function to solve the LunarLander-v3 environment.

# Constraints
1. `math` is imported, and `numpy` is already imported as `np`. Do not import anything else.
2. Your function MUST be named exactly `calculate_reward`.
3. You may ONLY use the keys listed in the `info` docstring below. Do not invent new keys.
4. **CRITICAL SIGNATURE:** Your function MUST return a `Tuple[float, dict]`. 
   - The first element is the total scalar reward.
   - The second element is a dictionary containing the individual mathematical components (e.g., `{"distance_penalty": -0.5, "upright_reward": 0.2}`).

```python
def calculate_reward(observation, info):
    """
    Calculates a shaped reward for the current step.
    
    Args:
        observation (np.ndarray): The standard LunarLander state vector:
            [x_pos, y_pos, x_vel, y_vel, angle, ang_vel, leg_1, leg_2]
        info (dict): Environment metadata. Contains ONLY:
            - info["prev_obs"] (np.ndarray): Observation vector from previous step
            - info["action"] (int): Index of the action taken (0-3)
            - info["current_step"] (int): The current timestep in the episode
            
    Returns:
        Tuple[float, dict]: (total_reward, components_dictionary)
    """
    # Write your code here
    
    return 0.0, {"baseline": 0.0}
```