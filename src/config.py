import os
import platform

class Config:
    # 1. Experiment Settings
    ENV_ID = "LunarLander-v3"
    N_ENVS = 32 if platform.system() == "Linux" else 4 # Auto-scale based on your benchmarks
    TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", 50000))
    ALGORITHM = "PPO"
    
    # 2. Dynamic Identity (Captured from Bash)
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder")
    CAMPAIGN_TAG = os.getenv("CAMPAIGN_TAG", "Debug_Run")

    # 3. Code Generation Settings
    RETENTION_MEMORY = 3 # How many past iterations to remember in detail