# ==========================================
# The logic of the agent's environment/ The Sensor Layer
# ==========================================


import gymnasium as gym
import importlib
import reward_shaping # This assumes reward_shaping.py is in the root

class DynamicRewardWrapper(gym.Wrapper):
    """
    Injects the LLM-generated reward function into the environment.
    """
    def __init__(self, env):
        super().__init__(env)
        # Reload ensures we get the LATEST version of the code without restarting python
        importlib.reload(reward_shaping)

    @staticmethod
    def is_successful_landing(obs):
        """Standard definition of a safe landing"""
        x, y, vx, vy, angle, angular_vel, leg1, leg2 = obs
        on_pad = abs(x) <= 0.2 
        upright = abs(angle) < 0.1 
        stable = abs(vx) < 0.1 and abs(vy) < 0.1 
        feet_down = (leg1 > 0.5) and (leg2 > 0.5) 
        return on_pad and upright and stable and feet_down
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 1. Calculate Concrete Success
        if terminated:
            success_event = self.is_successful_landing(obs)
            info["is_success"] = success_event
            
        # 2. Inject LLM Reward Logic
        try:
            new_reward = reward_shaping.calculate_reward(
                obs, reward, terminated, truncated, info
            )
        except Exception as e:
            # Fallback if LLM code crashes
            # print(f"Error in reward shaping: {e}") 
            new_reward = reward
            
        return obs, new_reward, terminated, truncated, info