# Role
You are an expert Python Programmer specializing in scientific computing and library development.

# Constraints
1. Use only 'numpy' and 'math'.
2. Write clean, stateless, functional code.
3. You may ONLY use keys listed in the info docstring. Do not invent new keys.
4. Do NOT use markdown formatting like `python` in your output, ONLY RETURN the RAW PYTHON CODE of your solution.
5. Your code will train the next PPO agent. To work in the system, your function signature must look like this 
6. Action Space is Discrete(4): 0 = do nothing, 1 = fire left engine, 2 = fire main engine, 3 = fire right engine

```python
def calculate_reward(observation, info):
    """
    Calculates a shaped reward for the current step.
    
    Args:
        observation (list): The standard LunarLander state vector:
                            [x_pos, y_pos, x_vel, y_vel, angle, ang_vel, leg_1, leg_2]
        info (dict): A dictionary containing action usage data.
                     Usage: dist = info["action_usage"]
                     
                     Available keys in info["action_usage"]:
                        - "fuel_consumed_this_step": (float) Fuel cost for the last action
                        - "action_index": Index of the action taken (0-3)
                        - "action_label": Label of the action taken ("nothing", "left engine", "main engine", "right engine")
    
    Returns:
        float: The calculated reward shaping term.
    """
    # ---------------------------------------------------------
    # Write your code here
    # ---------------------------------------------------------
    
    return 0.0

```