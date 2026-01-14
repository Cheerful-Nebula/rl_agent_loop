# Role
You are an expert Python Programmer specializing in scientific computing and library development.

# Constraints
1. Use only 'numpy' and 'math'.
2. Write clean, stateless, functional code.
3. Prioritize readability and error handling.
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
        info (dict): A dictionary containing derived physics data.
                     Usage: dist = info["raw_physics"]["distance_from_origin"]
                     
                     Available keys in info["raw_physics"]:
                     - "distance_from_origin": (float) Distance from (0,0)
                     - "angular_velocity": (float) Absolute angular velocity
                     - "linear_velocity_mag": (float) Magnitude of linear velocity
                     - "fuel_consumed_this_step": (float) Fuel cost for the last action
    
    Returns:
        shpaed_reward (float): The calculated reward shaping term.
        components (dict): A dictionary with individual reward components for debugging.
            i.e. Component Dictionary (This is what the Researcher sees)
            # TIP: Prefix keys with 'rew_' to avoid collisions in the info dict
            components = {
                "rew_pos": dist_penalty,
                "rew_vel": vel_penalty,
                "rew_fuel": fuel_cost
            }
    """
    # ---------------------------------------------------------
    # Write your code here
    # ---------------------------------------------------------
    
    return 0.0, {}

```