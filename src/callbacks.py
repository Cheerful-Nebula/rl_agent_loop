# ==========================================
# Callbacks for Agentic RL Training/ The Telemetry Layer
# ==========================================

import os
import csv
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

# custom imports
from src.workspace_manager import ExperimentWorkspace


# ==========================================
#  Training Collector
# ==========================================
class MultiEnvEpisodeTracker(BaseCallback):
    """
    Harvests lightweight semantic training statistics and component values.
    Optimized for LLM context windows by dropping raw 8D vectors.
    """
    
    def __init__(self, ws: ExperimentWorkspace, iteration: int, seed_id:int, max_buffer_size: int = 10000):
        super().__init__()
        self.ws = ws
        self.iteration = iteration
        self.max_buffer_size = max_buffer_size
        
        self.episode_buffer: List[Dict[str, Any]] = []
        self.n_envs = 0
        self.rollout_counter = 0
        self.master_csv = self.ws.dirs['telemetry_iteration']/ f"iter{iteration:02d}_seed{seed_id}.csv"
    
    def _on_training_start(self) -> None:
        self.n_envs = self.training_env.num_envs
        print(f"🚀 MultiEnvTracker: {self.n_envs} parallel envs")
    
    def _on_step(self) -> bool:
        """Capture EVERY episode completion across ALL envs"""
        for env_idx in range(self.n_envs):
            if self.locals['dones'][env_idx]:
                info = self.locals['infos'][env_idx]
                if 'episode' in info:
                    term_obs = info.get('terminal_observation', self.locals['new_obs'][env_idx])
                    
                    # === 1. BASE EPISODE DATA ===
                    ep_data = {
                        'rollout_id': self.rollout_counter,
                        'env_id': env_idx,
                        'ep_rew_total': float(info['episode']['r']),
                        'ep_len': int(info['episode']['l']),
                        'timestep_global': int(self.num_timesteps),
                        'terminal_status': self._compute_terminal_status(term_obs, info),
                        'x_pos':term_obs[0],
                        'y_pos':term_obs[1],
                        'x_vel':term_obs[2],
                        'y_vel':term_obs[3],
                        'angle':term_obs[4],
                        'angular_vel':term_obs[5],
                        'leg1_contact':term_obs[6],
                        'leg2_contact':term_obs[7]
                    }
                    
                    # === 2. REWARD COMPONENTS ===
                    components = info.get('reward_components', {})
                    ep_data.update({f'reward_{k.replace(" ", "_").replace("-", "_")}': float(v) 
                                   for k, v in components.items()})
                    
                    self.episode_buffer.append(ep_data)
        return True
    
    def _compute_terminal_status(self, term_obs, info):
        """Semantic tags for terminal state"""
        # Kinematic Stability Checks
        is_centered = abs(term_obs[0]) < 0.2
        is_out_of_bounds = abs(term_obs[0]) > 0.95 or term_obs[1] > 0.95
        is_upright = abs(term_obs[4]) < 0.1 if is_centered else abs(term_obs[4]) < 0.38
        low_velocity = abs(term_obs[2]) < 0.1 and abs(term_obs[3]) < 0.1 

        # Physics Contact Checks
        legs_down = term_obs[6] >= 0.5 and term_obs[7] >= 0.5
        resting_on_one_leg = (term_obs[6] > 0.5) != (term_obs[7] > 0.5)
        below_pad = term_obs[1] < 0.0

        # Truncated (timeout)?
        if info.get('TimeLimit', {}).get('truncated', False):
            return 'landed_off_centered_timeout' if legs_down else 'hover_timeout'
        
        # True termination
        if is_upright and legs_down:
            return 'landed_centered' if is_centered else 'landed_off_centered'
        elif is_upright and low_velocity and resting_on_one_leg and below_pad:
            return "landed_but_slid_into_valley"
        elif is_out_of_bounds:
            return 'out_of_bounds'
        else:
            return 'crashed'
    
    def _on_rollout_end(self) -> None:
        """Append rollout episodes to master CSV + log aggregates"""
        if self.episode_buffer:
            # buffer only contains episodes from this rollout
            # We append to master CSV for incremental collection, crash-proof
            df = pd.DataFrame(self.episode_buffer)
            df.to_csv(self.master_csv, mode='a', header=not self.master_csv.exists(), index=False)
            # Aggregates → progress.csv (Monitor-style)
            scores = df['ep_rew_total'].values
            self.logger.record("rollout/ep_count", len(scores))
            self.logger.record("rollout/ep_score_median", float(np.median(scores)))
            self.logger.record("rollout/ep_score_p90", float(np.percentile(scores, 90)))
        # Clear Buffer for next Rollout
        self.episode_buffer.clear()
        self.rollout_counter += 1
"""     Data analysis scripts for Diagnostician 1 will go below, more streamline to just incoroporate into callback when training ends
    def _on_training_end(self) -> None:
        #Final LLM-ready behavior summary
        if self.master_csv.exists() and self.master_csv.stat().st_size > 0:
            df_all = pd.read_csv(self.master_csv)
            summary = self._compress_behavior(df_all)
            
            summary_path = self.ws.dirs['telemetry'] / f"behavior_iter{self.iteration:03d}.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"✅ Behavior summary: {summary_path} ({len(df_all)} episodes)")
    
    def _compress_behavior(self, df: pd.DataFrame) -> Dict[str, Any]:
        #12 physics-based metrics + status dist (reward-invariant!)
        scores = df['true_score']
        statuses = df['terminal_status']
        
        # Semantic rates 
        status_dist = statuses.value_counts(normalize=True).to_dict()
        
        return {
            "n_episodes": int(len(df)),
            "median_score": float(scores.median()),
            
            # Physics-based (reward invariant!)
            "crash_rate": float(status_dist.get('crashed', 0)),
            "landed_centered_rate": float(status_dist.get('landedcentered', 0)),
            "landed_offcenter_rate": float(status_dist.get('landedoffcentered', 0)),
            "hover_timeout_rate": float(status_dist.get('hovertimeout', 0)),
            "out_of_bounds_rate": float(status_dist.get('outofbounds', 0)),
            
            # Stability metrics
            "wild_oscillation_rate": float((df['terminal_x'].abs() > 0.8).mean()),
            "spin_rate": float((df['terminal_angle_deg'].abs() > 45).mean()),
            "x_stability": float(df['terminal_x'].std()),
            "leg_contact_rate": float(((df['left_leg'] > 0.5) & (df['right_leg'] > 0.5)).mean()),
            
            # Durations
            "avg_crash_length": float(df[statuses == 'crashed']['ep_len'].mean() if 'crashed' in statuses else 0),
            "avg_hover_length": float(df[statuses == 'hovertimeout']['ep_len'].mean() if 'hovertimeout' in statuses else 0),
            
            "terminal_status_dist": {k: float(v) for k, v in status_dist.items()}
        }
"""
# ==========================================
# The Entropy Scheduler (Exploration)
# ==========================================
class EntropyScheduleCallback(BaseCallback):
    """Safe 0.02 -> 0.001 linear entropy schedule for PPO."""
    def __init__(self, initial_ent_coef=0.02, final_ent_coef=0.001, total_timesteps=1e6, verbose=0):
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        # 1. Calculate "progress_remaining" (1.0 starts, 0.0 ends)
        # SB3 tracks num_timesteps internally
        progress = max(0.0, 1.0 - self.num_timesteps / self.total_timesteps)
        
        # 2. Calculate current entropy value (Linear decay example)
        current_ent_coef = self.final_ent_coef + (self.initial_ent_coef - self.final_ent_coef) * progress
        
        # 3. Inject the new value directly into the model
        # For PPO/A2C, this attribute controls the loss calculation
        self.model.ent_coef = current_ent_coef
        
        # Logged to progress_iterXX.csv with other optimization data
        self.logger.record("train/ent_coef", current_ent_coef)
        
        return True


class FourWayEvalCallback(BaseCallback): 
    # Mostly depreciated,replaced by MultiEnvEpisodeTracker
    # Kept incase evaluations during training neccesary in future
    """
    Evaluates the agent on 4 configurations (Base/Shaped x Det/Stoch)
    and logs them in LONG format (Tidy Data) for easier analysis.
    """
    def __init__(
        self,
        eval_env_base,
        eval_env_shaped,
        iteration: int,
        ws: ExperimentWorkspace,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,
        filename: str = "four_way_callback_eval.csv",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env_base = eval_env_base
        self.eval_env_shaped = eval_env_shaped
        self.iteration = iteration
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.path = os.path.join(ws.dirs['telemetry'], filename)
        self._last_eval_step = 0

        # Check if file exists to init headers
        self._file_exists = os.path.exists(self.path)
        
        if not self._file_exists:
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                # LONG FORMAT HEADERS: Matching final_eval.csv style
                writer.writerow([
                    "iteration",
                    "timestep",
                    "reward_shape",       # 'Base' or 'Shaped'
                    "deterministic_flag", # True or False
                    "mean_reward",
                    "std_reward"
                ])

    def _on_step(self) -> bool:
        if (self.num_timesteps - self._last_eval_step) >= self.eval_freq:
            self._last_eval_step = self.num_timesteps

            # Define the 4 configurations to test
            # Tuples: (Environment, Shape Label, Deterministic Flag)
            configs = [
                (self.eval_env_base,   "Base",   True),
                (self.eval_env_base,   "Base",   False),
                (self.eval_env_shaped, "Shaped", True),
                (self.eval_env_shaped, "Shaped", False),
            ]

            results_to_log = []

            # 1. Run Evaluations
            for env, shape_label, det_flag in configs:
                mean_r, std_r = evaluate_policy(
                    self.model,
                    env,
                    n_eval_episodes=self.n_eval_episodes,
                    deterministic=det_flag,
                    render=False,
                )
                
                # Prepare row: Iteration | Step | Shape | Flag | Mean | Std
                results_to_log.append([
                    self.iteration,
                    self.num_timesteps,
                    shape_label,
                    det_flag,
                    mean_r,
                    std_r
                ])

            # 2. Console Logging (Brief Summary)
            if self.verbose > 0:
                # Extracting specific scores for clean printing
                base_det = next(r[4] for r in results_to_log if r[2]=="Base" and r[3])
                shaped_det = next(r[4] for r in results_to_log if r[2]=="Shaped" and r[3])
                print(
                    f"[FourWayEval] step={self.num_timesteps} | "
                    f"Base(Det): {base_det:.1f} | Shaped(Det): {shaped_det:.1f}"
                )

            # 3. Write to CSV (Append Mode)
            with open(self.path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(results_to_log)

        return True

