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