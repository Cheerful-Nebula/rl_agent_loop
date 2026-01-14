# src/wrappers.py
import gymnasium as gym
from src import utils  

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
        
        # Creating the sparse reward problem, removing all rewards except if lander crashed or landed
        reward = reward if (terminated and abs(reward) >= 100) else 0.0 

        lander = self.env.unwrapped.lander
        
        # Inject raw data for the LLM to analyze later
        info["raw_physics"] = {
            "distance_from_origin": float(lander.position.length),
            "angular_velocity": float(lander.angularVelocity),
            "linear_velocity_mag": float(lander.linearVelocity.length),
            "fuel_consumed_this_step": 0.3 if action == 2 else (0.03 if action in [1, 3] else 0)
        }

        # 2. Inject LLM Reward Logic (If module is loaded)
        if self.reward_module:
            try:
                # We pass the obs/info resulting from the step above
                new_reward = self.reward_module.calculate_reward(obs, info)
                new_reward = reward + new_reward
                return obs, new_reward, terminated, truncated, info

            except Exception as e:
                print("*"*20)
                print(f"DynamicRewardWrapper failed:\n {e}")
                print("*"*20)
                pass
            
        return obs, reward, terminated, truncated, info