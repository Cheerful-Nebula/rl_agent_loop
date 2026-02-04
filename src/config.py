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
        'num_predict': 8000,   # Prevent cutoff
        'temperature': 0.65,    # Balance creativity/precision
        'top_p': 0.8,
    }
    # For Phase 2: Formatter (Process Plan)
    formatter_options={
        'num_ctx': 16384,
        'num_predict': 5000,
        'temperature': 0.2,  
    }
    # For Phase 3: Coder (Implementation)
    def get_coder_options(model_name: str):
        """
        Returns optimal inference parameters based on the specific model architecture.
        """
        model_id = model_name.lower()
        
        # =========================================================
        # GROUP A: REASONING & THINKING MODELS
        # Needs: High Temp (to think), High Output Limit (trace + code)
        # =========================================================
        # Keywords based on your provided list:
        reasoning_keywords = [
            "deepseek-r1", 
            "openthinker", 
            "qwen3:30b-a3b-thinking", 
            "gpt-oss", 
            "cogito"
        ]
        
        if any(k in model_id for k in reasoning_keywords):
            return {
                # MEMORY: 24k fits comfortably in 36GB RAM with 32b weights
                "num_ctx": 24576,        
                
                # OUTPUT: 8k allows ~6k tokens of "Thinking" + 2k code
                "num_predict": 8192,     
                
                # CREATIVITY: 0.6 prevents "thought loops" common in low-temp reasoning
                "temperature": 0.6,      
                "top_p": 0.95,
                
                # STOPS: Standard + Thinking tags to prevent hallucinations
                "stop": ["<|end_of_text|>", "<|user|>", "User:", "</think>"] 
            }
        
        # =========================================================
        # GROUP B: STANDARD CODERS & INSTRUCT MODELS
        # Needs: Low Temp (precision), Moderate Output (code only)
        # =========================================================
        # Covers: granite-code, gemma3, qwen3-coder, exaone, nemotron, devstral, llama3.2-vision
        else:
            return {
                # MEMORY: Standard 16k is sufficient for code-only tasks
                "num_ctx": 16384,        
                
                # OUTPUT: 4k is plenty for just Python code (no thinking trace)
                "num_predict": 4096,     
                
                # PRECISION: 0.1 forces the model to pick the most likely syntax
                "temperature": 0.1,      
                "top_p": 0.5,            # Focus on high-probability tokens
                "repeat_penalty": 1.1,   # Slight penalty to reduce loops
                
                # STOPS: Standard instruction stops
                #"stop": ["<|end_of_text|>", "<|eot_id|>", "User:", "<|file_separator|>"] 
            }
        
    # gpt_oss has 3 thinking levels : low, medium, high
    gpt_think_level = "low"

    # 4. LLM Prompt Templates ###################################################################
    analyst_role = "rl_researcher"
    analyst_task = "diagnose_agent_v03"
    analyst_template = (analyst_role, analyst_task)
 
    formatter_role = "formatter_v02"
    formatter_task = "format_v03"
    formatter_template = (formatter_role, formatter_task)

    formatter_fix_role = "formatter_v02"
    formatter_fix_task = "format_fix"
    formatter_fix_template = (formatter_role, formatter_task)

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

    # NETWORK CONFIGURATION
    LINUX_IP = os.getenv("LINUX_IP", "127.0.0.1")
    LINUX_USER = os.getenv("LINUX_USER", "user")
    ssh_env = os.getenv("SSH_KEY_PATH", "") 
    SSH_KEY_PATH = os.path.expanduser(ssh_env)
    REMOTE_PROJECT_ROOT = os.getenv("REMOTE_PROJECT_ROOT", "")
    REMOTE_PYTHON_BIN = os.getenv("REMOTE_PYTHON_BIN", "")

    # Validation (Optional but recommended)
    @classmethod
    def validate_network_config(cls):
        if not cls.REMOTE_PROJECT_ROOT or not cls.REMOTE_PYTHON_BIN:
            raise ValueError("❌ Missing Remote Configs! Please set REMOTE_PROJECT_ROOT in your .env file.")