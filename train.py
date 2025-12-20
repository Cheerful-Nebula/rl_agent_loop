import warnings

# Filter out the specific pkg_resources deprecation warning
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import gymnasium as gym
import json
import importlib
import numpy as np
import os
import shutil
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure
from datetime import datetime
from gymnasium.wrappers import RecordVideo  

# Import our custom modules
from config import Config
import reward_shaping
from position_tracking import PositionTracker
import utils


# ---------------------------------------------------------
# 1. The Dynamic Wrapper (The Bridge)
# ---------------------------------------------------------
class DynamicRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        importlib.reload(reward_shaping)


    @staticmethod
    def is_successful_landing(obs):
        """
        Determines if the lander has landed safely based on the observation.
        LunarLander-v2 Obs: [x, y, vx, vy, angle, angular_vel, leg1_contact, leg2_contact]
        """
        x, y, vx, vy, angle, angular_vel, leg1, leg2 = obs

        on_pad = abs(x) <= 0.2 # 1. Is it on the landing pad? (Pad is at x=0, roughly width +/- 0.2)
        upright = abs(angle) < 0.1  # 2. Is it upright? (Angle roughly 0) ~5.7 degrees tolerance
        stable = abs(vx) < 0.1 and abs(vy) < 0.1 # 3. Is it stable? (Velocity is low)
        feet_down = (leg1 > 0.5) and (leg2 > 0.5) # 4. Are both feet touching the ground?

        return on_pad and upright and stable and feet_down
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Calculate concrete success independent of the point score
        success_event = False
        if terminated:
            success_event = self.is_successful_landing(obs)
            # Optional: Overwrite standard success metric in info
            info["is_success"] = success_event
        try:
            new_reward = reward_shaping.calculate_reward(
                obs, reward, terminated, truncated, info
            )
        except Exception as e:
            print(f"Error in reward shaping: {e}") 
            new_reward = reward
        return obs, new_reward, terminated, truncated, info

# ---------------------------------------------------------
# 2. Evaluation & Metrics Generation
# ---------------------------------------------------------
def evaluate_agent(model, num_episodes=10):
    """
    Runs the agent and collects detailed stats.
    Records a video of the FIRST episode of this evaluation batch.
    """
    # READ FROM STATE
    iteration = Config.get_iteration()
    video_folder = f"logs/videos/iter_{iteration:03d}"
    
    # 1. Create Eval Env with Video Recording
    eval_env = gym.make(Config.ENV_ID, render_mode="rgb_array")
    
    # We must wrap the eval env so it calculates 'is_success' exactly like training
    eval_env = DynamicRewardWrapper(eval_env)

    # Trigger: Record only episode 0 (the first one)
    eval_env = RecordVideo(
        eval_env, 
        video_folder, 
        episode_trigger=lambda x: x == 0,
        disable_logger=True
    )
    
    # 2. Setup Tracker
    # Indices: 0= X-Pos, 3=Y-Vel, 4=Angle
    target_indices = [0, 3, 4] 
    tracker = PositionTracker(target_indices, iteration, Config.ALGORITHM, Config.ENV_ID)
    tracker.start_rollout()

    total_rewards = []
    crashes = 0
    landings = 0
    concrete_landings = 0

    # Action-Use tracking
    action_counts = {0:0, 1:0, 2:0, 3:0} 
    total_steps = 0

    print(f"\n🎥 Recording video to: {video_folder}")
    print(f"Running evaluation ({num_episodes} episodes)...")
    
    for i in range(num_episodes):
        tracker.start_episode()
        # set seed so each iteration of agents face same terrain during evaluation
        eval_seed = 42 + i
        obs, _ = eval_env.reset(seed=eval_seed)  
        tracker.track_episode(obs)
        
        done = False
        ep_reward = 0
        
        while not done:
            action, _ = model.predict(obs)

            # Track Action
            # (action is a numpy array of size 1, so we assume discrete scalar)
            act_scalar = int(action)
            action_counts[act_scalar] = action_counts.get(act_scalar, 0) + 1
            total_steps += 1

            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            tracker.track_episode(obs)
            ep_reward += reward
            
            if terminated or truncated:
                # 1. Legacy Check (Box2D Reward Threshold)
                if reward <= -100: crashes += 1
                elif reward >= 100: landings += 1
                
                # 2. Concrete Check (Your new logic)
                # We use .get() because truncated episodes might not have the key
                if info.get('is_success', False):
                    concrete_landings += 1
                    
                done = True
                
        tracker.end_episode(terminated)
        total_rewards.append(ep_reward)
    
    tracker.end_rollout(model.num_timesteps)
    eval_env.close() # Important: Finalizes the video file
    
    # Extract rich stats from tracker
    avg_x_pos = tracker.rollout_pts_agg[0]['avg'] # Center is 0.0
    avg_y_vel = tracker.rollout_pts_agg[3]['avg']
    avg_angle = tracker.rollout_pts_agg[4]['avg']
    raw_stats = tracker.get_episode_metrics()
    std_y_vel = raw_stats[3]['std_dev'] # Index 3 is Y-Velocity
    std_x_vel = raw_stats[0]['std_dev'] # Index 0 is X-Velocity

    # Calculate Engine Usage Ratios
    # 0: Do Nothing, 1: Left Eng, 2: Main Eng, 3: Right Eng
    if total_steps > 0:
        main_engine_usage = action_counts[2] / total_steps
        side_engine_usage = (action_counts[1] + action_counts[3]) / total_steps
    else:
        main_engine_usage = 0.0
        side_engine_usage = 0.0

    return {
        "mean_reward": float(np.mean(total_rewards)),
        "success_rate": landings / num_episodes,          # The "Game Score" success
        "concrete_success_rate": concrete_landings / num_episodes, # The "Physics" success
        "crash_rate": crashes / num_episodes,
        "diagnostics": {
            "avg_x_position": float(avg_x_pos),
            "avg_descent_velocity": float(avg_y_vel),
            "avg_tilt_angle": float(avg_angle),
            "main_engine_usage": float(main_engine_usage),
            "side_engine_usage": float(side_engine_usage),
            "vertical_stability_index": float(std_y_vel), # Lower is better (more consistent)
            "horizontal_stability_index": float(std_x_vel)  # Lower is better (more consistent)
        }
    }

# ---------------------------------------------------------
# 3. Main Training Loop
# ---------------------------------------------------------
def run_training_cycle():
    # Setup
    env = make_vec_env(
            Config.ENV_ID,
            n_envs=Config.N_ENVS,
            vec_env_cls=SubprocVecEnv,
            wrapper_class=DynamicRewardWrapper # SB3 will instantiate this wrapper inside each separate process.
        )
    
    iteration = Config.get_iteration()
    # Create a StableBaselines3 logger
    sb3_dir = os.path.join(Config.SB3_DIR,f"iter_{iteration:03d}")
    # Configure the logger to output to terminal (stdout) AND json
    new_logger = configure(sb3_dir, ["stdout", "json"])                     

    # 2. MATH: Scaling for the 3080
    # Calculations:
    # Buffer Size = 256 steps * 32 envs = 8,192 transitions per update.
    # Batch Size  = 2048. 
    # This means the 3080 will process the buffer in 4 massive chunks (8192 / 2048 = 4).
    # Tune PPO for the Crowd
    model = PPO(
        "MlpPolicy",
        env,
        device=Config.get_device(),          # The 3080 does the thinking
        n_steps=256,                         # Keep this short to update frequently
        batch_size=2048,                     # Larger batch for the 3080
        n_epochs=10,                         # More epochs because we have a larger, diverse batch
        gamma=0.999,                         # Long horizon for landing
        gae_lambda=0.98,
        ent_coef=0.01,                       # Encourage exploration
        verbose=0
    )
    
    # Attach the logger to the model
    model.set_logger(new_logger)
    
    print(f"Training on {Config.N_ENVS} envs with Total Buffer: {256*Config.N_ENVS}")
    print(f"Starting training for {Config.TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=Config.TOTAL_TIMESTEPS)
    
    # Save Model
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
    model.save(Config.MODEL_SAVE_PATH)
    
    # Evaluate (includes Video Recording now)
    stats = evaluate_agent(model)
    
    # Create JSON Payload
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "total_timesteps": Config.TOTAL_TIMESTEPS,
            "algorithm": Config.ALGORITHM
        },
        "performance": stats
    }
 
    # Save "Hot" File (The Contract)
    with open(Config.METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)

    # Archive metrics with Iteration ID 
    utils.save_metrics(Config.get_iteration()) 
 
    
    print(f"\nTRAINING COMPLETE.")
    print(f"Mean Reward: {stats['mean_reward']:.2f}")
    print(f"Crash Rate: {stats['crash_rate']:.2f}")
    print(f"Main Engine Usage: {stats['diagnostics']['main_engine_usage']:.2f}")

if __name__ == "__main__":
    run_training_cycle()