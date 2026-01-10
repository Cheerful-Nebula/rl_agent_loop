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
    INITIAL_TEMPLATE = """
def calculate_reward(observation, info):
    \"\"\"
    Calculates a shaped reward for the current step.
    
    Args:
        observation (list): The standard LunarLander state vector:
                            [x_pos, y_pos, x_vel, y_vel, angle, ang_vel, leg_1, leg_2]
        info (dict): A dictionary containing derived physics data.
                     Usage: dist = info["raw_physics"]["distance_from_origin"]
                     
                     Available keys in info["raw_physics"]:
                     - "distance_from_origin": (float) Distance from (0,0)
                     - "angular_velocity": (float) Absolute angular velocity
                     - "linear_velocity_mag": (float) Magnitude of linear velocity
                     - "fuel_consumed_this_step": (float) Fuel cost for the last action
    
    Returns:
        float: The calculated reward shaping term.
    \"\"\"
    # ---------------------------------------------------------
    # ITERATION 1: ZERO STATE
    # This is the starting point. The environment is currently 
    # relying solely on the sparse reward (Wrapper).
    # ---------------------------------------------------------
    
    return 0.0
"""

    # 4. Prompt Templates
    analyst_role = "rl_researcher"
    analyst_task = "diagnose_agent_v02"
    analyst_template = (analyst_role, analyst_task)
 
    code_gen_role = "python_coder"
    code_gen_task = "implement_plan"
    code_gen_template = (code_gen_role, code_gen_task)

    code_fix_role = "python_coder"
    code_fix_task = "fix_code"
    code_fix_template = (code_fix_role, code_fix_task)

