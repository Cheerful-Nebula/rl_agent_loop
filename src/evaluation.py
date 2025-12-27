# ==========================================
# Evaluation Module for Agentic RL / The Exam Layer
# ==========================================
import os 
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo

# -- custom imports --
from src.workspace_manager import ExperimentWorkspace
from src.position_tracking import PositionTracker
from src.config import Config

def evaluate_agent(model, run_id, num_episodes=10):
    """
    Runs the agent on the STANDARD environment to check true performance.
    Records a video of the FIRST episode.
    """
    # Define video path using the run_id passed from train.py
    # Note: run_id here is expected to be "Iter_005" or similar
    # We reconstruct the full path assuming standard workspace structure
    # But since we are inside python, we can just use the path passed by train.py if we wanted.
    # For now, we rely on the relative path which works if run from root.
    ws = ExperimentWorkspace()
    video_folder = ws.get_path("videos", run_id, "video")
    
    # 1. Create Eval Env (STANDARD - No Reward Shaping)
    eval_env = gym.make(Config.ENV_ID, render_mode="rgb_array")
    
    # Trigger recording only for episode 0
    eval_env = RecordVideo(
        eval_env, 
        video_folder, 
        episode_trigger=lambda x: x == 0,
        disable_logger=True
    )
    
    # 2. Setup Tracker
    target_indices = [0, 3, 4] # X-Pos, Y-Vel, Angle
    tracker = PositionTracker(target_indices, 0, Config.ALGORITHM, Config.ENV_ID)
    tracker.start_rollout()

    total_rewards = []
    crashes = 0
    reward_successes = 0
    position_landings = 0
    action_counts = {0:0, 1:0, 2:0, 3:0} 
    total_steps = 0

    print(f"🎥 Recording evaluation video to: {video_folder}")
    
    for i in range(num_episodes):
        tracker.start_episode()
        obs, _ = eval_env.reset(seed=42+i)
        tracker.track_episode(obs)
        
        done = False
        ep_reward = 0
        
        while not done:
            action, _ = model.predict(obs)
            
            # Metrics Logic
            act_scalar = int(action)
            action_counts[act_scalar] = action_counts.get(act_scalar, 0) + 1
            total_steps += 1

            obs, reward, terminated, truncated, info = eval_env.step(action)
            tracker.track_episode(obs)
            ep_reward += reward
            
            if terminated or truncated:
                if reward <= -100: crashes += 1
                elif reward >= 100: reward_successes += 1
                # Ported from ComprehensiveEvalCallback
                term_obs = info.get("terminal_observation")
                # Fallback: Use current obs if terminal_observation is missing
                if term_obs is None: 
                    term_obs = obs
                
                if term_obs is not None:
                    # Logic: Center < 0.2, Upright < 0.1 rad, Both Legs Touching
                    is_centered = abs(term_obs[0]) < 0.2
                    is_upright = abs(term_obs[4]) < 0.1
                    legs_down = term_obs[6] > 0.5 and term_obs[7] > 0.5
                    
                    if is_centered and is_upright and legs_down:
                        position_landings += 1
                done = True
                
        tracker.end_episode(terminated)
        total_rewards.append(ep_reward)
    
    tracker.end_rollout()
    eval_env.close() 
    
    # 3. Calculate Final Stats
    if total_steps > 0:
        main_use = action_counts[2] / total_steps
        side_use = (action_counts[1] + action_counts[3]) / total_steps
    else:
        main_use, side_use = 0, 0

    raw_stats = tracker.get_episode_metrics()
    
    return {
        "mean_reward": float(np.mean(total_rewards)),
        "reward_success_rate": reward_successes / num_episodes,
        "position_success_rate": position_landings / num_episodes,
        "crash_rate": crashes / num_episodes,
        "diagnostics": {
            "avg_x_position": float(tracker.rollout_pts_agg[0]['avg']),
            "avg_descent_velocity": float(tracker.rollout_pts_agg[3]['avg']),
            "avg_tilt_angle": float(tracker.rollout_pts_agg[4]['avg']),
            "main_engine_usage": float(main_use),
            "side_engine_usage": float(side_use),
            "vertical_stability_index": float(raw_stats[3]['std_dev']),
            "horizontal_stability_index": float(raw_stats[0]['std_dev'])
        }
    }