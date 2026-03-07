import gymnasium as gym
import numpy as np
import sys
from src import utils  

class DynamicRewardWrapper(gym.Wrapper):
    """
    Injects the LLM-generated reward function and manages dual-telemetry logging
    (step-wise for evaluation, episodic for training).
    """
    def __init__(self, env : gym.Env, reward_code_path: str|None = None):
        super().__init__(env)
        self.reward_module = None
        self.last_obs = None
        self.ep_components = {} # Stores the cumulative episode totals
        
        if reward_code_path:
            try:
                self.reward_module = utils.load_dynamic_module("current_reward", reward_code_path)
            except Exception as e:
                print(f"⚠️ Wrapper failed to load reward module: {e}")
                sys.exit()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = np.array(obs)
        self.ep_components = {} # CRITICAL: Reset accumulators for the new episode
        self.current_step = 0
        return obs, info

    def step(self, action):
        prev_obs_safe = self.last_obs.copy() if self.last_obs is not None else np.zeros(8)
        norm_obs, _, terminated, truncated, info = self.env.step(action)
        self.last_obs = np.array(norm_obs)
        
        # Inject Neccessary Information into info Dictionary
        self.current_step += 1
        info['current_step'] = self.current_step
        info["action"] = int(action)
        info["prev_obs"] = prev_obs_safe 
        
        # 1. Execute LLM's Reward Function
        final_reward = 0.0
        step_components = {}
        
        if self.reward_module:
            try:
                shaping_reward, llm_components = self.reward_module.calculate_reward(norm_obs, info)
                final_reward += shaping_reward
                
                # Merge the LLM's components with our base reward component
                for k, v in llm_components.items():
                    step_components[k] = float(v)
                    
            except Exception as e:
                print(f"DynamicRewardWrapper failed:\n {e}")
                sys.exit()

        # 2. Expose Step-Wise Components (For Evaluation CSV!)
        info['step_reward_components'] = step_components

        # 3. Accumulate Episodic Totals
        for key, val in step_components.items():
            self.ep_components[key] = self.ep_components.get(key, 0.0) + val

        # 4. Expose Cumulative Totals (For Training CSV!)
        if terminated or truncated:
            info['terminal_observation'] = norm_obs
            info['reward_components'] = self.ep_components.copy() 
            
        return norm_obs, final_reward, terminated, truncated, info