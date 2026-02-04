# src/evaluation.py

import numpy as np
import gymnasium as gym

def summarize_eval(iteration, stats, tracker):
    """
    Standard summary for the CSV, but passes through the raw logs.
    """
    # ... (Keep your existing summary logic for mean/std/median) ...
    
    summary = {
        "Iteration": iteration,
        "mean_reward": stats["mean_reward"],
        "crash_rate": stats["crash_rate"],
        "success_rate": stats["success_rate"],
        # PASS THROUGH RAW DATA
        "eval_logs": stats["eval_logs"] 
    }
    return summary


def run_single_eval_pass(env, model, num_episodes, deterministic_flag, tracker=None):
    """
    Runs evaluation and captures forensic telemetry for the LLM.
    """
    eval_logs = []  # List to store detailed episode data
    
    # Accumulators for standard metrics
    total_reward = 0
    crashes = 0
    successes = 0
    
    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic_flag)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if tracker: tracker.record_step(obs)

        # --- FORENSIC CAPTURE ---
        # 1. Determine Outcome based on Reward AND Physics
        # (Assuming SparseRewardWrapper returns -100 for crash, 100 for land)
        outcome = "TIMEOUT"
        if done:
            if episode_reward <= -90: 
                outcome = "CRASH"
                crashes += 1
            elif episode_reward >= 90: 
                outcome = "SUCCESS"
                successes += 1
            else:
                outcome = "STALLED" # Rare edge case in sparse envs
        elif truncated:
            outcome = "TIMEOUT"

        # 2. Capture Terminal State (The "Black Box")
        # Standard LunarLander State: [x, y, vx, vy, angle, ang_vel, leg1, leg2]
        terminal_state = obs.tolist() 

        eval_logs.append({
            "episode": i,
            "outcome": outcome,
            "duration": steps,
            "reward": episode_reward,
            "terminal_state": terminal_state
        })
        
        total_reward += episode_reward

    # Return the raw logs AND the summary stats
    return {
        "mean_reward": total_reward / num_episodes,
        "success_rate": successes / num_episodes,
        "crash_rate": crashes / num_episodes,
        "eval_logs": eval_logs, # <--- CRITICAL: Pass this raw data out
        "deterministic_flag": deterministic_flag
    }

