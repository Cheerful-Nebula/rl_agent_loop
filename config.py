import json
import os

class Config:
    # Environment Settings
    ENV_ID = "LunarLander-v3"
    
    # Training Settings
    TOTAL_TIMESTEPS = 100000 
    ALGORITHM = "PPO"
    
    # Paths
    METRICS_FILE = "metrics.json"
    MODEL_SAVE_PATH = "models/lunar_lander_mvp"
    STATE_FILE = "state.json"  # NEW: The source of truth

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