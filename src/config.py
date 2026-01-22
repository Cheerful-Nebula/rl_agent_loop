import os
import platform
from dotenv import load_dotenv

# Load variables from .env file into the environment
load_dotenv()
class Config:
    # 1. Experiment Settings ###################################################################
    ENV_ID = "LunarLander-v3"
    TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", 50000))
    ALGORITHM = "PPO"
    
    # 2. Dynamic Experiment Directory Name ####################################################
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder")
    CAMPAIGN_TAG = os.getenv("CAMPAIGN_TAG", "Debug_Run")

    # 3. Code Generation Settings ###############################################################
    RETENTION_MEMORY = 3 # How many past iterations to remember in detail
    # For Phase 1: Researcher (Diagnosis)
    analyst_options={
        'num_ctx': 16384,      # M4 Max can handle this easily
        'num_predict': 4000,   # Prevent cutoff
        'temperature': 0.65,    # Balance creativity/precision
        'top_p': 0.8,
    }
    # For Phase 2: Formatter (Process Plan)
    formatter_options={
        'num_ctx': 16384,
        'num_predict': 4000,
        'temperature': 0.1,  
    }
    # For Phase 3: Coder (Implementation)
    coder_options={
        'num_ctx': 16384,
        'num_predict': 4000,
        'temperature': 0.1,    # Strict adherence to syntax
        'repeat_penalty': 1.00 # No penalty to allow code structure
    }
    # gpt_oss has 3 thinking levels : low, medium, high
    gpt_think_level = "high"

    # 4. LLM Prompt Templates ###################################################################
    analyst_role = "rl_researcher"
    analyst_task = "diagnose_agent_v02"
    analyst_template = (analyst_role, analyst_task)
 
    formatter_role = "formatter_v01"
    formatter_task = "format_v01"
    formatter_template = (formatter_role, formatter_task)

    code_zero_role = "coder_v03"
    code_zero_task = "initial_shaping_v03"
    code_zero_template = (code_zero_role, code_zero_task)

    code_gen_role = "coder_v03"
    code_gen_task = "implement_plan_v02"
    code_gen_template = (code_gen_role, code_gen_task)

    code_fix_role = "coder_v03"
    code_fix_task = "fix_code"
    code_fix_template = (code_fix_role, code_fix_task)

    # 5. Network Credentials ################################################
    # Saved in .env, raw IPs never see github

    # NETWORK CONFIGS (Loaded securely)
    # The second argument is a default fallback if the .env value is missing
    LINUX_IP = os.getenv("LINUX_IP")
    LINUX_USER = os.getenv("LINUX_USER")
    SSH_KEY_PATH = os.path.expanduser(os.getenv("SSH_KEY_PATH"))
    REMOTE_PROJECT_ROOT = os.getenv("REMOTE_PROJECT_ROOT")
    REMOTE_PYTHON_BIN = os.getenv("REMOTE_PYTHON_BIN")

    # Validation (Optional but recommended)
    @classmethod
    def validate_network_config(cls):
        if not cls.REMOTE_PROJECT_ROOT or not cls.REMOTE_PYTHON_BIN:
            raise ValueError("❌ Missing Remote Configs! Please set REMOTE_PROJECT_ROOT in your .env file.")