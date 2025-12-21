# ==========================================
# Callbacks for Agentic RL Training/ The Telemetry Layer
# ==========================================

import os
import json
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# ==========================================
# 1. The Supervisor Translator (Quartiles)
# ==========================================
class AgenticObservationTracker(BaseCallback):
    """
    Feeds the LLM Supervisor. 
    Translates raw data into statistical distributions (quartiles, mean, variance).
    """
    def __init__(self, obs_indices: list, save_path: str, verbose=0):
        super(AgenticObservationTracker, self).__init__(verbose)
        self.obs_indices = obs_indices
        self.save_path = save_path
        self.episode_data = {i: [] for i in obs_indices}
        self.outcomes = {"trunc": 0, "term": 0}
        self.rollout_stats = [] 

    def _on_step(self) -> bool:
        # Access the current observation (usually 'new_obs' in PPO)
        obs = self.locals['new_obs']
        
        for i in self.obs_indices:
            self.episode_data[i].append(obs[0, i])
            
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            if "TimeLimit.truncated" in info and info["TimeLimit.truncated"]:
                self.outcomes["trunc"] += 1
            else:
                self.outcomes["term"] += 1
        return True

    def _on_rollout_end(self) -> None:
        summary = {}
        for i in self.obs_indices:
            data = np.array(self.episode_data[i])
            if len(data) > 0:
                summary[f"Obs_{i}"] = {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "p25": float(np.percentile(data, 25)),
                    "p75": float(np.percentile(data, 75)),
                }
        
        # Log to TensorBoard for visual check
        total = self.outcomes["trunc"] + self.outcomes["term"]
        trunc_rate = self.outcomes["trunc"] / total if total > 0 else 0
        self.logger.record("supervisor/truncation_rate", trunc_rate)
        
        self.rollout_stats.append(summary)
        
        # Save to JSON
        file_path = os.path.join(self.save_path, "rollout_summaries.json")
        with open(file_path, "w") as f:
            json.dump(self.rollout_stats, f, indent=4)
        
        # Reset buffers
        self.episode_data = {i: [] for i in self.obs_indices}
        self.outcomes = {"trunc": 0, "term": 0}

# ==========================================
# 2. The Performance Evaluator (Scores/Fuel)
# ==========================================
class ComprehensiveEvalCallback(BaseCallback):
    """
    Feeds the Human Engineer.
    Tracks 'True' score, fuel usage, and strict landing success.
    """
    def __init__(self, threshold_score=200, verbose=0):
        super(ComprehensiveEvalCallback, self).__init__(verbose)
        self.threshold_score = threshold_score
        self.episode_scores = []
        self.current_episode_fuel = 0.0
        self.success_achieved_early = False

    def _on_step(self) -> bool:
        # Fuel Tracking
        action = self.locals['actions'][0]
        if action == 2: self.current_episode_fuel += 0.3
        elif action in [1, 3]: self.current_episode_fuel += 0.03

        # Episode End Logic
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            
            if 'episode' in info:
                true_score = info['episode']['r']
                self.episode_scores.append(true_score)
                self.logger.record("metrics/true_episode_score", true_score)
                
                # Rolling Avg Check
                if len(self.episode_scores) >= 100:
                    rolling_avg = np.mean(self.episode_scores[-100:])
                    if rolling_avg >= self.threshold_score and not self.success_achieved_early:
                        self.success_achieved_early = True
                        print(f"✨ EARLY SUCCESS: Rolling Avg {rolling_avg:.1f} > {self.threshold_score}")

            # Strict Positional Check
            term_obs = info.get("terminal_observation")
            if term_obs is not None:
                is_centered = abs(term_obs[0]) < 0.2
                is_upright = abs(term_obs[4]) < 0.1
                legs_down = term_obs[6] > 0.5 and term_obs[7] > 0.5
                strict_success = int(is_centered and is_upright and legs_down)
                self.logger.record("metrics/strict_position_success", strict_success)

            self.logger.record("metrics/est_fuel_consumed", self.current_episode_fuel)
            self.current_episode_fuel = 0.0
            
        return True