import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from datetime import datetime
# Import our custom modules
from config import Config
from utils import AgenticObservationTracker, ComprehensiveEvalCallback
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