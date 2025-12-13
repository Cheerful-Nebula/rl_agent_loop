import gymnasium as gym
import json
import importlib
import numpy as np
import os
import shutil
import glob
from stable_baselines3 import PPO
from datetime import datetime
from gymnasium.wrappers import RecordVideo  

# Import our custom modules
from config import Config
import reward_shaping
from position_tracking import PositionTracker


# ---------------------------------------------------------
# 1. The Dynamic Wrapper (The Bridge)
# ---------------------------------------------------------
class DynamicRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        importlib.reload(reward_shaping)
        print(f"Loaded Reward Module: {reward_shaping.__file__}")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            new_reward = reward_shaping.calculate_reward(
                obs, reward, terminated, truncated, info
            )
        except Exception as e:
            # print(f"Error in reward shaping: {e}") 
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
    # render_mode="rgb_array" is REQUIRED for video recording
    eval_env = gym.make(Config.ENV_ID, render_mode="rgb_array")
    
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

    # NEW: Action tracking
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

            obs, reward, terminated, truncated, _ = eval_env.step(action)
            
            tracker.track_episode(obs)
            ep_reward += reward
            
            if terminated or truncated:
                if reward <= -100: crashes += 1
                elif reward >= 100: landings += 1
                done = True
                
        tracker.end_episode(terminated)
        total_rewards.append(ep_reward)
    
    tracker.end_rollout(model.num_timesteps)
    eval_env.close() # Important: Finalizes the video file
    
    # Extract rich stats from tracker
    avg_x_pos = tracker.rollout_pts_agg[0]['avg'] # Center is 0.0
    avg_y_vel = tracker.rollout_pts_agg[3]['avg']
    avg_angle = tracker.rollout_pts_agg[4]['avg']

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
        "success_rate": landings / num_episodes,
        "crash_rate": crashes / num_episodes,
        "diagnostics": {
            "avg_x_position": float(avg_x_pos),
            "avg_descent_velocity": float(avg_y_vel),
            "avg_tilt_angle": float(avg_angle),
            "main_engine_usage": float(main_engine_usage),
            "side_engine_usage": float(side_engine_usage)
        }
    }

# ---------------------------------------------------------
# 3. Main Training Loop
# ---------------------------------------------------------
def run_training_cycle():
    # Setup
    env = gym.make(Config.ENV_ID)
    env = DynamicRewardWrapper(env)
    
    # Train
    model = PPO("MlpPolicy", env, verbose=1) #, device=Config.get_device())
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

    # Archive with Iteration ID (Consistency)
    iteration = Config.get_iteration() # Get the number we used for everything else 
    archive_dir = "logs/metrics_history"
    os.makedirs(archive_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = os.path.join(archive_dir, f"metrics_gen{iteration:03d}_{timestamp_str}.json")
    shutil.copy(Config.METRICS_FILE, archive_path)
    
    print(f"\nTRAINING COMPLETE.")
    print(f"Metrics saved to: {Config.METRICS_FILE}")
    print(f"History archived to: {archive_path}")
    print(f"Mean Reward: {stats['mean_reward']:.2f}")
    print(f"Crash Rate: {stats['crash_rate']:.2f}")
    print(f"Main Engine Usage: {stats['diagnostics']['main_engine_usage']:.2f}")

if __name__ == "__main__":
    run_training_cycle()