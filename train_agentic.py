import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from datetime import datetime
# ==========================================
# 1. The Supervisor Translator (Quartiles)
# ==========================================
class AgenticObservationTracker(BaseCallback):
    """
    Feeds the LLM Supervisor. 
    Translates raw data into statistical distributions (quartiles, mean, variance).
    """
    def __init__(self, obs_indices: list, verbose=0):
        super(AgenticObservationTracker, self).__init__(verbose)
        self.obs_indices = obs_indices
        self.episode_data = {i: [] for i in obs_indices}
        self.outcomes = {"trunc": 0, "term": 0}
        # You could save this to a JSON file later for the Supervisor to read
        self.rollout_stats = [] 

    def _on_step(self) -> bool:
        # Access the current observation (for PPO, usually 'new_obs' or 'obs')
        # We need the observation from the *monitor* wrapper if possible, 
        # but 'new_obs' in locals is the raw input to the policy.
        obs = self.locals['new_obs']
        
        # Log the specific indices we care about
        for i in self.obs_indices:
            # obs is (n_envs, n_obs), take env 0
            self.episode_data[i].append(obs[0, i])
            
        # Track Truncation (Time Limit) vs Termination (Crash/Land)
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            if "TimeLimit.truncated" in info and info["TimeLimit.truncated"]:
                self.outcomes["trunc"] += 1
            else:
                self.outcomes["term"] += 1
        return True

    def _on_rollout_end(self) -> None:
        """Called every n_steps (buffer fill) before update."""
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
        
        # Log outcome ratio to TensorBoard for quick visual check
        total = self.outcomes["trunc"] + self.outcomes["term"]
        trunc_rate = self.outcomes["trunc"] / total if total > 0 else 0
        self.logger.record("supervisor/truncation_rate", trunc_rate)
        
        # Save summary to list (or write to JSON here)
        self.rollout_stats.append(summary)
        
        # Reset buffers
        self.episode_data = {i: [] for i in self.obs_indices}
        self.outcomes = {"trunc": 0, "term": 0}

# ==========================================
# 2. The Performance Evaluator (Scores/Fuel)
# ==========================================
class ComprehensiveEvalCallback(BaseCallback):
    """
    Feeds the Human Engineer (You).
    Tracks 'True' score, fuel usage, and strict landing success.
    """
    def __init__(self, threshold_score=200, verbose=0):
        super(ComprehensiveEvalCallback, self).__init__(verbose)
        self.threshold_score = threshold_score
        self.episode_scores = []
        self.current_episode_fuel = 0.0
        self.success_achieved_early = False

    def _on_step(self) -> bool:
        # 1. Track Estimated Fuel
        action = self.locals['actions'][0]
        if action == 2: self.current_episode_fuel += 0.3
        elif action in [1, 3]: self.current_episode_fuel += 0.03

        # 2. End of Episode Logic
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            
            # Get True Reward from Monitor
            if 'episode' in info:
                true_score = info['episode']['r']
                self.episode_scores.append(true_score)
                self.logger.record("metrics/true_episode_score", true_score)
                
                # Check for Early Success (Rolling Avg)
                if len(self.episode_scores) >= 100:
                    rolling_avg = np.mean(self.episode_scores[-100:])
                    if rolling_avg >= self.threshold_score and not self.success_achieved_early:
                        self.success_achieved_early = True
                        print(f"✨ EARLY SUCCESS: Rolling Avg {rolling_avg:.1f} > {self.threshold_score}")

            # Strict Positional Check (The "Perfect Landing" Metric)
            term_obs = info.get("terminal_observation") # Requires specific wrapper or env config
            # If not available, we can sometimes peek at 'new_obs' if done=True, 
            # but 'terminal_observation' is the standard way in new Gym API.
            if term_obs is not None:
                # 0:x, 4:angle, 6,7:legs
                is_centered = abs(term_obs[0]) < 0.2
                is_upright = abs(term_obs[4]) < 0.1
                legs_down = term_obs[6] > 0.5 and term_obs[7] > 0.5
                strict_success = int(is_centered and is_upright and legs_down)
                self.logger.record("metrics/strict_position_success", strict_success)

            # Log Fuel & Reset
            self.logger.record("metrics/est_fuel_consumed", self.current_episode_fuel)
            self.current_episode_fuel = 0.0
            
        return True

# ==========================================
# 3. The Execution Loop
# ==========================================
def run_training():
    # 1. Define the Run ID
    RUN_TAG = "Baseline_Test"
    RUN_ID = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{RUN_TAG}"

    # 2. Define the Base Path
    BASE_DIR = os.path.join("experiments", RUN_ID)

    # 3. Create the Sub-Structure
    DIRS = {
        "tensorboard": os.path.join(BASE_DIR, "telemetry", "tensorboard"),
        "json_logs":   os.path.join(BASE_DIR, "telemetry", "raw"),
        "reasoning":   os.path.join(BASE_DIR, "cognition"),
        "visuals":     os.path.join(BASE_DIR, "artifacts", "visuals"),
        "models":      os.path.join(BASE_DIR, "artifacts", "models"),
    }

    # 4. Make them
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)
    # A. Create Environment (Must Wrap in Monitor!)
    env = gym.make("LunarLander-v3")
    env = Monitor(env) 

    # B. Instantiate Callbacks
    # Track Angle (4) and Legs (6,7) for the Supervisor
    supervisor_callback = AgenticObservationTracker(obs_indices=[4, 6, 7]) 
    
    # Track Performance for You
    metrics_callback = ComprehensiveEvalCallback(threshold_score=200)

    # C. Initialize Model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=DIRS["tensorboard"])

    print(f"🚀 Starting Training: {RUN_TAG}")
    
    # D. THE SECRET SAUCE: Pass list of callbacks
    model.learn(
        total_timesteps=400_000, 
        callback=[supervisor_callback, metrics_callback]
    )

    print("Training Complete.")

if __name__ == "__main__":
    run_training()