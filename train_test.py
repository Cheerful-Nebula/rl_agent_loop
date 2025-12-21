import os
import json
import warnings
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# -- THE NEW CLEAN IMPORTS --
from config import Config
import utils
from wrappers import DynamicRewardWrapper
from callbacks import AgenticObservationTracker, ComprehensiveEvalCallback
from evaluation import evaluate_agent

warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

def run_training_cycle():
    # 1. Setup Run ID & Folders
    iteration = Config.get_iteration()
    RUN_TAG = f"Iter_{iteration:03d}"
    RUN_ID = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{RUN_TAG}"
    BASE_DIR = os.path.join("experiments", RUN_ID)
    
    # Define the "Lab" for this run
    DIRS = {
        "tensorboard": os.path.join(BASE_DIR, "telemetry", "tensorboard"),
        "json_logs":   os.path.join(BASE_DIR, "telemetry", "raw_metrics"),
        "reasoning":   os.path.join(BASE_DIR, "cognition"),
        "generated_code": os.path.join(BASE_DIR, "generated_code"),
        "videos":     os.path.join(BASE_DIR, "artifacts", "visuals", "videos"),
        "plots":       os.path.join(BASE_DIR, "artifacts", "visuals", "plots"),
        "models":      os.path.join(BASE_DIR, "artifacts", "models"),
    }
    for d in DIRS.values(): os.makedirs(d, exist_ok=True)

    # 2. Hardware Scaling (From utils)
    n_envs, device = utils.get_hardware_config()
    ppo_params = utils.get_optimized_ppo_params(n_envs, device)

    # 3. Create Environment (Using the Wrapper from wrappers.py)
    # We use Monitor here to ensure SB3 sees episode stats
    def make_env():
        import gymnasium as gym
        env = gym.make(Config.ENV_ID)
        env = DynamicRewardWrapper(env) # <--- The logic is injected here
        return Monitor(env)

    env = make_vec_env(make_env, n_envs=Config.N_ENVS, vec_env_cls=SubprocVecEnv)

    # 4. Initialize Callbacks (Using callbacks.py)
    supervisor_callback = AgenticObservationTracker(
        obs_indices=[4, 6, 7], 
        save_path=DIRS['json_logs']
    )
    metrics_callback = ComprehensiveEvalCallback(threshold_score=200)

    # 5. Train
    print(f"🚀 [Iter {iteration}] Starting Training on {device}...")
    model = PPO(
        "MlpPolicy",
        env,
        device=ppo_params['device'],
        n_steps=ppo_params['n_steps'],                   
        batch_size=ppo_params['batch_size'],
        tensorboard_log=DIRS['tensorboard'],
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=0
    )
    
    model.learn(
        total_timesteps=Config.TOTAL_TIMESTEPS, 
        callback=[supervisor_callback, metrics_callback]
    )
    
    # 6. Save & Evaluate
    model.save(os.path.join(DIRS['models'], "final_model"))
    
    print("📊 Running Evaluation...")
    stats = evaluate_agent(model, num_episodes=10, run_id=RUN_ID)
    
    # 7. Finalize Data (The Contract with the Agent)
    metrics_payload = {
        "timestamp": datetime.now().isoformat(),
        "run_id": RUN_ID,
        "config": {"total_timesteps": Config.TOTAL_TIMESTEPS, "algorithm": Config.ALGORITHM},
        "performance": stats
    }
    
    with open(Config.METRICS_FILE, "w") as f:
        json.dump(metrics_payload, f, indent=4)
        
    # Archive properly
    utils.save_metrics(iteration)
    print(f"✅ Cycle Complete. Metrics saved to {Config.METRICS_FILE}")

if __name__ == "__main__":
    run_training_cycle()