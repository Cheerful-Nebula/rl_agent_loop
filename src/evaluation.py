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
                   "policy_behavior": "Deterministic" if stats["deterministic_flag"] else "Stochastic",
                    "reward_shape": stats["reward_shape"],
                    # Standard Metrics 
                    "mean_reward": round(stats["mean_reward"], 4),
                    "median_reward": round(stats["median_reward"], 4),
                    "std_reward": round(stats["std_reward"], 4),
                    "mean_ep_length": round(stats["mean_len"], 4),
                    "reward_success_rate": round(stats["success_rate"], 4),
                    "position_success_rate": round(stats["pos_success_rate"], 4),
                    "crash_rate": round(stats["crash_rate"], 4),
                    "raw_episode_lengths": stats["raw_lengths"],   # For Survival Analysis (Kaplan-Meier)
                    # Diagnostics
                    "avg_x_position": round(float(tracker.rollout_pts_agg[0]['avg']), 4),
                    "avg_descent_velocity": round(float(tracker.rollout_pts_agg[3]['avg']), 4),
                    "avg_tilt_angle": round(float(tracker.rollout_pts_agg[4]['avg']), 4),
                    "vertical_stability_index": round(float(tracker.get_episode_metrics()[3]['std_dev']), 4),
                    "horizontal_stability_index": round(float(tracker.get_episode_metrics()[0]['std_dev']), 4)}
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
        "median_reward": float(np.median(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "mean_len": float(np.mean(episode_lengths)),
        "success_rate": reward_successes / num_episodes,
        "pos_success_rate": position_landings / num_episodes,
        "crash_rate": crashes / num_episodes,
        "raw_lengths": episode_lengths,
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
    print(f"📊 Eval Pass 1: Standard Environment,{"Deterministic Behavior" if deterministic == True else "Stochastic Behavior"} ")
    std_env = utils.make_env() 
    std_tracker = PositionTracker([0, 3, 4], 0, Config.ALGORITHM, Config.ENV_ID)
    std_stats = run_single_eval_pass(std_env, model, num_episodes, deterministic_flag= deterministic, tracker = std_tracker)
    std_stats["reward_shape"]= "Base"
    std_env.close()

    # --- PASS 2: SHAPED EVALUATION (The Agent's Reality) ---
    if reward_code_path:
        print(f"📊 Eval Pass 2: Shaped Environment, {"Deterministic Behavior" if deterministic == True else "Stochastic Behavior"} ")
        shp_env = utils.make_env(reward_code_path)
        shp_tracker = PositionTracker([0, 3, 4], 0, Config.ALGORITHM, Config.ENV_ID)
        shp_stats = run_single_eval_pass(shp_env, model, num_episodes, deterministic_flag= deterministic, tracker = shp_tracker)
        shp_stats["reward_shape"] = "Shaped"
        shp_env.close()

    # --- MERGE & RETURN ---
    # We prefix keys to distinguish them in the CSV
    base_stats = summarize_eval(iteration,std_stats,std_tracker)
    if reward_code_path:
        shaped_stats = summarize_eval(iteration,shp_stats, shp_tracker)
    else: shaped_stats = base_stats.copy()

    return  base_stats, shaped_stats