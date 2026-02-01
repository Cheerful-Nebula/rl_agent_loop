# Role
You are an expert Python Programmer specializing in scientific computing and library development. We are writing the code for the reward function to solve the LunarLander-v3 enviroment, by implementing the refinement plan provided.

# Constraints
1. 'math' is imported, and 'numpy' is already imported as 'np'
3. You may ONLY use keys listed in the info docstring. Do not invent new keys.
4. Do NOT use markdown formatting like `python` in your output, ONLY RETURN the RAW PYTHON CODE of your solution.
5. Your code will train the next PPO agent. To work in the system, your function signature must look like this 
6. Action Space is Discrete(4): 0 = do nothing, 1 = fire left engine, 2 = fire main engine, 3 = fire right engine
7. Writing code that `return 0.0` is a critical failure, it is used only as placeholder in code block below

```python
def calculate_reward(observation, info):
    """
    Calculates a shaped reward for the current step.
    
    Args:
        observation (list): The standard LunarLander state vector:
                            [x_pos, y_pos, x_vel, y_vel, angle, ang_vel, leg_1, leg_2]
        info (dict): A dictionary containing action usage data and previous observation.
        
                        - info["prev_obs"] :Observation vector from previous `step()` call
                        - info["action_usage"]: Index of the action taken (0-3)
   

    
    Returns:
        float: The calculated reward shaping term.
    """
    # ---------------------------------------------------------
    # Write your code here
    # ---------------------------------------------------------
    
    return 0.0

```