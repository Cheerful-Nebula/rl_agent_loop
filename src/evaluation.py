# =========================================================
# FILE: src/evaluation.py
# =========================================================

import os 
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from src.workspace_manager import ExperimentWorkspace
from src.position_tracking import PositionTracker
from src.config import Config
from src import utils
from src.workspace_manager import ExperimentWorkspace

def summarize_eval(iteration: int,stats: dict, tracker : PositionTracker) -> dict:
    final_stats = {"Iteration":iteration,
        "deterministic_flag": stats["deterministic_flag"],
        "reward_shape": stats["reward_shape"],
        # Standard Metrics 
        "mean_reward": stats["mean_reward"],
        "std_reward": stats["std_reward"],
        "mean_ep_length": stats["mean_len"],
        "reward_success_rate": stats["success_rate"],
        "position_success_rate": stats["pos_success_rate"],
        "crash_rate": stats["crash_rate"],
        "raw_episode_lengths": stats["raw_lengths"], # For Survival Analysis (Kaplan-Meier)
        "raw_outcomes": stats["raw_outcomes"],
        # Diagnostics
        "avg_x_position": float(tracker.rollout_pts_agg[0]['avg']),
        "avg_descent_velocity": float(tracker.rollout_pts_agg[3]['avg']),
        "avg_tilt_angle": float(tracker.rollout_pts_agg[4]['avg']),
        "vertical_stability_index": float(tracker.get_episode_metrics()[3]['std_dev']),
        "horizontal_stability_index": float(tracker.get_episode_metrics()[0]['std_dev'])}
    return final_stats

def run_single_eval_pass(env, model, num_episodes,deterministic_flag = True, tracker=None):
    """Helper to run one evaluation loop (Standard or Shaped)."""
    total_rewards = []
    episode_lengths = []
    crashes = 0
    reward_successes = 0
    position_landings = 0
    
    if tracker: tracker.start_rollout()

    for i in range(num_episodes):
        if tracker: tracker.start_episode()
        obs, _ = env.reset(seed=42+i)
        if tracker: tracker.track_episode(obs)
        
        done = False
        ep_reward = 0
        ep_len = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic_flag)
            ep_len += 1
            obs, reward, terminated, truncated, info = env.step(action)
            if tracker: tracker.track_episode(obs)
            ep_reward += reward
            # -- score based success -- 
            if terminated or truncated:
                if reward <= -100: crashes += 1
                elif reward >= 100: reward_successes += 1
                # -- agent position/terminal observation based success -- 
                term_obs = info.get("terminal_observation", obs)
                if term_obs is not None:
                    is_centered = abs(term_obs[0]) < 0.2
                    is_upright = abs(term_obs[4]) < 0.1
                    legs_down = term_obs[6] > 0.5 and term_obs[7] > 0.5
                    if is_centered and is_upright and legs_down:
                        position_landings += 1
                done = True
        
        if tracker: tracker.end_episode(terminated)
        total_rewards.append(ep_reward)
        episode_lengths.append(ep_len)

    if tracker: tracker.end_rollout()
    
    eval_metrics = {
        "deterministic_flag": deterministic_flag,
        "mean_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "mean_len": float(np.mean(episode_lengths)),
        "success_rate": reward_successes / num_episodes,
        "pos_success_rate": position_landings / num_episodes,
        "crash_rate": crashes / num_episodes,
        "raw_lengths": episode_lengths,
        "raw_outcomes": ["crash" if r <= -100 else "landed" for r in total_rewards]
    }

    return eval_metrics

def evaluate_agent(model, iteration, deterministic= True, reward_code_path=None, num_episodes=10):
    """
    Runs Dual Evaluation:
    1. Standard Env (Ground Truth)
    2. Shaped Env (Perceived Task - only if reward_code_path provided)
    """
    ws = ExperimentWorkspace()
    
    # --- PASS 1: STANDARD EVALUATION (The Objective Truth) ---
    print(f"📊 Eval Pass 1: Standard Environment...")
    std_env = utils.make_base_env() 
    std_tracker = PositionTracker([0, 3, 4], 0, Config.ALGORITHM, Config.ENV_ID)
    std_stats = run_single_eval_pass(std_env, model, num_episodes, deterministic_flag= deterministic, tracker = std_tracker)
    std_stats["reward_shape"]= "Base"
    std_env.close()

    # --- PASS 2: SHAPED EVALUATION (The Agent's Reality) ---
    if reward_code_path:
        print(f"📊 Eval Pass 2: Shaped Environment...")
        shp_env = utils.make_shaped_env(reward_code_path)
        shp_tracker = PositionTracker([0, 3, 4], 0, Config.ALGORITHM, Config.ENV_ID)
        shp_stats = run_single_eval_pass(shp_env, model, num_episodes, deterministic_flag= deterministic, tracker = shp_tracker)
        shp_stats["reward_shape"] = "Shaped"
        shp_env.close()

    # --- MERGE & RETURN ---
    # We prefix keys to distinguish them in the CSV
    json_stats = {
        # Standard Metrics (Keep these as the "main" ones for backward compatibility)
        "mean_reward": std_stats["mean_reward"],
        "std_reward": std_stats["std_reward"],
        "mean_ep_length": std_stats["mean_len"],
        "reward_success_rate": std_stats["success_rate"],
        "position_success_rate": std_stats["pos_success_rate"],
        "crash_rate": std_stats["crash_rate"],
        "raw_episode_lengths": std_stats["raw_lengths"], # For Survival Analysis (Kaplan-Meier)
        "raw_outcomes": std_stats["raw_outcomes"],

        # Shaped Metrics (New)
        "shaped_mean_reward": shp_stats.get("mean_reward", 0.0),
        "shaped_success_rate": shp_stats.get("success_rate", 0.0),

        # Diagnostics (From Standard Pass)
        "diagnostics": {
            "avg_x_position": float(std_tracker.rollout_pts_agg[0]['avg']),
            "avg_descent_velocity": float(std_tracker.rollout_pts_agg[3]['avg']),
            "avg_tilt_angle": float(std_tracker.rollout_pts_agg[4]['avg']),
            "vertical_stability_index": float(std_tracker.get_episode_metrics()[3]['std_dev']),
            "horizontal_stability_index": float(std_tracker.get_episode_metrics()[0]['std_dev'])
        }
    }

    base_stats = summarize_eval(iteration,std_stats,std_tracker)

    shaped_stats = summarize_eval(iteration,shp_stats, shp_tracker)


    return json_stats, base_stats, shaped_stats