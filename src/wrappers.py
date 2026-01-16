# src/wrappers.py
import gymnasium as gym
import numpy as np
import sys
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
                sys.exit()

    def build_physical_state(self):
        """
        Extracts raw physics data from Box2D.
        Returns a dictionary of raw non-normalized floats
        Inject returned dict into `info` dict from `step()` method for LLM reasoning
        """
        env = self.env.unwrapped
        lander = env.lander
        pos = lander.position           # Box2D world units
        vel = lander.linearVelocity     # world units / second

        # We return a DICT for the LLM (easier to query than an array index)
        return {
            "x_pos": float(pos.x),                             # world x  
            "y_pos": float(pos.y),                             # world y
            "x_vel": float(vel.x),                             # vx (world units/s)
            "y_vel": float(vel.y),                             # vy (world units/s)
            "angle": float(lander.angle),                      # radians
            "angle_vel": float(lander.angularVelocity),        # rad/s
            "leg_left": float(env.legs[0].ground_contact),     # 0.0 or 1.0
            "leg_right": float(env.legs[1].ground_contact),    # 0.0 or 1.0
            "distance_from_origin": float(lander.position.length),
            "linear_velocity_mag": float(lander.linearVelocity.length)
        }

    def step(self, action):
        # 1. Execute Action
        # We capture the 'norm_obs' (Standard Scaled) for the PPO Agent
        norm_obs, original_reward, terminated, truncated, info = self.env.step(action)
        
        # 2. Create the "Clean Slate" Sparse Reward
        # We strip the environment's default shaping, leaving only the sparse signal
        # This forces the LLM to do the heavy lifting.
        base_reward = original_reward if (terminated and abs(original_reward) >= 100) else 0.0 

        # 3. Inject Raw Data into Info
        # This gives the LLM the "Truth" without confusing the Agent
        info["action_usage"]={"fuel_consumed_this_step":0.3 if action == 2 else (0.03 if action in [1, 3] else 0),
                              "action_index":int(action),
                              "action_label":"nothing" if int(action) == 0 \
                                else "left_engine" if int(action) == 1\
                                else "main_engine" if int(action) == 2\
                                else "right_engine"}

        # 4. Execute LLM Reward Logic
        final_reward = base_reward
        if self.reward_module:
            try:
                shaping_reward = self.reward_module.calculate_reward(norm_obs, info)
                final_reward += shaping_reward
            except Exception as e:
                # Fail gracefully so training doesn't crash
                print(f"DynamicRewardWrapper failed:\n {e}")
                sys.exit()
                pass
            
        # Return NORMALIZED obs to PPO, but use the CUSTOM reward
        return norm_obs, final_reward, terminated, truncated, info