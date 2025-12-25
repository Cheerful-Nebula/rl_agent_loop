# src/wrappers.py
import gymnasium as gym
from src import utils  # <--- Needed for loading the module inside the child process

class DynamicRewardWrapper(gym.Wrapper):
    """
    Injects the LLM-generated reward function into the environment.
    """
    def __init__(self, env, reward_code_path=None):
        super().__init__(env)
        self.reward_module = None
        
        # Load the module INSIDE the process that actually runs the environment
        if reward_code_path:
            try:
                # We use the helper from utils to load from the string path
                self.reward_module = utils.load_dynamic_module("current_reward", reward_code_path)
            except Exception as e:
                print(f"⚠️ Wrapper failed to load reward module from {reward_code_path}: {e}")

    def step(self, action):
        # 1. Execute the Action (ONCE)
        obs, reward, terminated, truncated, info = self.env.step(action)
           
        # 2. Inject LLM Reward Logic (If module is loaded)
        if self.reward_module:
            try:
                # We pass the obs/info resulting from the step above
                new_reward = self.reward_module.calculate_reward(obs, action, info)
                return obs, new_reward, terminated, truncated, info

            except Exception as e:
                # If the user's code crashes, fail silently and use default reward
                # (You can enable print here for debugging if needed)
                pass
            
        return obs, reward, terminated, truncated, info