import os
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
    
    # 3. LLM Prompt Templates ###################################################################
    diagnose_training_role= "diagnose_training2"
    diagnose_training_task= "diagnose_training"
    diagnose_training_template = (diagnose_training_role,diagnose_training_task)

    strategist_role = "strategist"
    strategist_task = "strategist"
    strategist_template = (strategist_role,strategist_task)

    director_role = ""
    director_task = ""
    director_template = (director_role,director_task)

    dispatcher_role = "formatter_v02"
    dispatcher_task = "format_v03"
    dispatcher_template = (dispatcher_role, dispatcher_task)

    dispatcher_fix_role = "formatter_v02"
    dispatcher_fix_task = "format_fix"
    dispatcher_fix_template = (dispatcher_fix_role, dispatcher_fix_task)

    code_zero_role = "coder_v03"
    code_zero_task = "initial_shaping_v03"
    code_zero_template = (code_zero_role, code_zero_task)

    code_gen_role = "coder_v03"
    code_gen_task = "implement_plan_v02"
    code_gen_template = (code_gen_role, code_gen_task)

    code_fix_role = "coder_v03"
    code_fix_task = "fix_code"
    code_fix_template = (code_fix_role, code_fix_task)

    # 4. Code Generation Settings ###############################################################
    RETENTION_MEMORY = 3 # How many past iterations to remember in detail
    # gpt_oss has 3 thinking levels : low, medium, high
    gpt_think_level = "low"
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
    def get_inference_options(model_name: str, role: str):
        """
        Returns optimal inference parameters based on Model Architecture AND Cognitive Role.
        Target Hardware: Mac Studio/Pro (36GB Unified Memory).
        
        Refactor Update:
        - Splits logic into 'Thinking' (Reasoning) vs 'Standard' (Non-Thinking).
        - Implements specific roles: Diagnostician, Strategist, Research Lead, Coder.
        - Manages Context (KV Cache) to prevent OOM on 36GB RAM.
        """
        model_id = model_name.lower()
        role = role.lower()
        
        # Detect Reasoning Models (DeepSeek-R1, OpenThinker, etc.)
        is_reasoning = any(k in model_id for k in ["deepseek-r1", "thinking", "openthinker", "gpt-oss"])

        # defaults
        options = {
            "num_ctx": 16384,     # Safe default
            "num_predict": 4096,  # Standard output limit
            "stop": ["<|end_of_text|>", "<|user|>", "User:", "Observation:"]
        }

        # ==============================================================================
        # SCENARIO A: THINKING / REASONING MODELS (DeepSeek-R1, etc.)
        # Strategy: Maintain Entropy (0.6+) to prevent Loop Collapse. No Repetition Penalty.
        # ==============================================================================
        if is_reasoning:
            # GLOBAL REASONING DEFAULTS
            options.update({
                "repeat_penalty": 1.0,   # CRITICAL: Do not penalize repetition (breaks CoT loops)
                "num_predict": 8192,     # High limit to allow for <think> blocks
                "top_k": 40,             # Standard sampling window
            })

            if role == 'diagnostician':
                # Goal: Deep trend analysis.
                # We keep temp at 0.6 to allow reasoning flow, but restrict top_p slightly
                # to keep the "conclusion" phase grounded.
                options["temperature"] = 0.6
                options["top_p"] = 0.95
                options["num_ctx"] = 20480 

            elif role == 'strategist':
                # Goal: Structured Diversity.
                # Needs higher entropy to generate novel hypotheses in the CoT.
                options["temperature"] = 0.75 
                options["top_p"] = 0.95
                options["num_ctx"] = 24576   # Max safe context for 36GB Mac

            elif role == 'research_lead':
                # Goal: Disciplined causal comparison.
                # Slightly lower temp to encourage convergence on the most logical path.
                options["temperature"] = 0.6
                options["top_p"] = 0.9
                options["num_ctx"] = 24576

            elif role == 'coder':
                # Goal: Precise code synthesis.
                # WARNING: Do not drop temp to 0.1 for R1 models or they stutter.
                # We rely on the model's internal verification rather than sampling restrictions.
                options["temperature"] = 0.6 
                options["top_p"] = 0.9
                options["num_predict"] = 10000 # Max output for heavy refactoring

        # ==============================================================================
        # SCENARIO B: STANDARD MODELS (Llama 3, Mistral, Command R)
        # Strategy: "Cognitive Thermodynamics" - Use Temperature/Penalties as control levers.
        # ==============================================================================
        else:
            # GLOBAL STANDARD DEFAULTS
            options.update({
                "num_predict": 4096,
            })

            if role == 'diagnostician':
                # Goal: Stable statistical interpretation. 
                # Low temp, fixed seed recommended.
                options["temperature"] = 0.2
                options["top_p"] = 0.9
                options["top_k"] = 40
                options["repeat_penalty"] = 1.05
                options["seed"] = 42  # Deterministic analysis

            elif role == 'strategist':
                # Goal: Diverse, mechanistically distinct strategies.
                # High temp + Presence Penalty to force new topic exploration.
                options["temperature"] = 0.7
                options["top_p"] = 0.95
                options["top_k"] = 60
                options["repeat_penalty"] = 1.1
                options["presence_penalty"] = 0.2 # Forces topic shifting
                options["num_ctx"] = 24576

            elif role == 'research_lead':
                # Goal: Stable, rational selection.
                # Balanced parameters.
                options["temperature"] = 0.3
                options["top_p"] = 0.9
                options["top_k"] = 40
                options["repeat_penalty"] = 1.05
                options["seed"] = 42

            elif role == 'coder':
                # Goal: Deterministic translation.
                # Greedy decoding (very low temp).
                options["temperature"] = 0.1
                options["top_p"] = 0.85
                options["top_k"] = 30
                options["repeat_penalty"] = 1.02 # Slight penalty to prevent line-duplication bugs
                options["seed"] = 42

        return options

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