import json
import os
import torch
class Config:
    # Environment Settings
    ENV_ID = "LunarLander-v3"
    N_ENVS = 32
    
    # Training Settings/ env variable set in benchmark.sh / defaults to 50k
    TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", 50000)) 
    ALGORITHM = "PPO"
    
    # Model / Defaults to 'qwen2.5-coder' if not set
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder")
    

    # Paths
    METRICS_FILE = "metrics.json"
    MODEL_SAVE_PATH = "models/lunar_lander_mvp"
    STATE_FILE = "state.json"  # The source of truth for iteration count
    CODE_DIR = "logs/code_history"
    REASONING_DIR = "logs/reasoning_history"
    METRICS_DIR = "logs/metrics_history"
    SB3_DIR = "logs/sb3_log_history"

    @staticmethod
    def get_iteration():
        """Reads the current iteration from state.json. Defaults to 0 if missing."""
        if not os.path.exists(Config.STATE_FILE):
            return 0
        try:
            with open(Config.STATE_FILE, 'r') as f:
                data = json.load(f)
                return data.get("iteration", 0)
        except:
            return 0

    @staticmethod
    def increment_iteration():
        """Increments the iteration number safely."""
        current = Config.get_iteration()
        new_iter = current + 1
        with open(Config.STATE_FILE, 'w') as f:
            json.dump({"iteration": new_iter}, f)
        return new_iter