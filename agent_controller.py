import json
import os
import re
import shutil
import ollama
from datetime import datetime 
import warnings

# Filter out the specific pkg_resources deprecation warning
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

# Project specific modules
from config import Config
from code_validation import CodeValidator
import prompts 
import utils

MODEL_NAME = Config.LLM_MODEL

# ---------------------------------------------------------
# PROMPT ENGINEERING
# ---------------------------------------------------------

def get_diagnosis_prompt(metrics, current_code, iteration_id):
    stats = metrics['performance']
    diagnostics = metrics.get('diagnostics', {})
    
    # 1. Prepare the Data
    # (Fetch memory logic remains the same)
    short_term_history = utils.get_recent_history(n=3)
    long_term_memory = utils.get_long_term_memory(iteration_id, retention=3)

    # 2. Retrieve the Role (The "Who")
    system_role = prompts.get_role("rl_researcher")

    # 3. Retrieve the Task (The "What")
    # Notice we pass the raw variables into the function.
    # The formatting logic {variable:.2f} is handled inside the Markdown file or here if preferred.
    # To keep the markdown file simple, passing pre-formatted strings or raw floats works.
    user_task = prompts.get_task(
        "diagnose_agent",
        env_id=Config.ENV_ID,
        success_rate=stats['success_rate'],
        crash_rate=stats.get('crash_rate', 0),
        avg_descent=diagnostics.get('avg_descent_velocity', 0),
        avg_tilt=diagnostics.get('avg_tilt_angle', 0),
        avg_x=diagnostics.get('avg_x_position', 0),
        main_eng=diagnostics.get('main_engine_usage', 0),
        side_eng=diagnostics.get('side_engine_usage', 0),
        y_std=diagnostics.get('vertical_stability_index',0),
        x_std=diagnostics.get('horizontal_stability_index',0),
        long_term_memory=long_term_memory,
        short_term_history=short_term_history,
        current_code=current_code if current_code else "# No previous code"
    )

    # Return both so the controller can send them to the LLM
    return system_role, user_task


def get_coding_prompts(plan, current_code):
    """Generates the role and task for the initial code generation."""
    role = prompts.get_role("python_coder")
    task = prompts.get_task(
        "implement_plan", 
        plan=plan, 
        current_code=current_code if current_code else "# No existing code"
    )
    return role, task

def get_fix_prompts(invalid_code, feedback):
    """Generates the role and task for the debugging loop."""
    # We stick with the Python Expert role for fixing
    role = prompts.get_role("python_coder")
    task = prompts.get_task(
        "fix_code", 
        clean_code=invalid_code, 
        feedback=feedback
    )
    return role, task

# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
def run_agentic_improvement():
    # -- Setup
    iteration_id = Config.get_iteration()
    print(f"🔵 AGENT (Iter {iteration_id}): Reading state...")
    
    if not os.path.exists(Config.METRICS_FILE):
        print("❌ No metrics found.")
        return

    metrics = json.loads(utils.load_file(Config.METRICS_FILE))
    current_code = utils.load_file("reward_shaping.py")
    
    # -- Archive Previous Reward Shaping code
    utils.archive_current_code(iteration_id)
    
    # =========================================================
    # PHASE 1: DIAGNOSIS (The "Brain")
    # =========================================================
    print("🔵 AGENT: Phase 1 - Diagnosing & Planning...")
    
    # 1. Get Prompts
    diag_role, diag_task = get_diagnosis_prompt(metrics, current_code, iteration_id)
    
    try:
        # 2. Call LLM
        response = ollama.chat(model=MODEL_NAME, messages=[
            {'role': 'system', 'content': diag_role},
            {'role': 'user', 'content': diag_task}
        ])

        diagnosis_plan = response['message']['content']
        print(f"📝 Plan: {diagnosis_plan}")
        utils.save_reasoning(iteration_id, diagnosis_plan)
        
    except Exception as e:
        print(f"❌ Phase 1 Error: {e}")
        return

    # =========================================================
    # PHASE 2: IMPLEMENTATION (The "Hands")
    # =========================================================
    print("🔵 AGENT: Phase 2 - Writing Code...")
    
    # 1. Get Prompts (Refactored)
    code_role, code_task = get_coding_prompts(diagnosis_plan, current_code)

    try:
        # 2. Call LLM (Initial Write)
        response2 = ollama.chat(model=MODEL_NAME, messages=[
            {'role': 'system', 'content': code_role},
            {'role': 'user', 'content': code_task}
        ])
        
        clean_code = utils.extract_python_code(response2['message']['content'])
        
        # 3. Validation Loop
        validator = CodeValidator(clean_code)

        # STEP A: Static Check (Syntax/Security)
        is_valid, feedback = validator.validate_static()

        # STEP B: Runtime Check (Only if static passed)
        if is_valid:
            is_valid, feedback = validator.validate_runtime()

        attempt_num = 0
        MAX_RETRIES = 20 # Safety limit
        
        while not is_valid and attempt_num < MAX_RETRIES:
            attempt_num += 1
            print(f"⚠️ AGENT: Validation failed (Attempt {attempt_num}). Requesting fix...")
            
            # Archive the broken attempt for later analysis
            utils.archive_current_code(iteration_id, attempt_num=attempt_num) 

            # 4. Get Fix Prompts (Refactored)
            fix_role, fix_task = get_fix_prompts(clean_code, feedback)
            
            # 5. Call LLM (Fix Request)
            response_fix = ollama.chat(model=MODEL_NAME, messages=[
                {'role': 'system', 'content': fix_role},
                {'role': 'user', 'content': fix_task}
            ])
            
            clean_code = utils.extract_python_code(response_fix['message']['content'])
            
            # Re-validate (Static -> Runtime)
            validator = CodeValidator(clean_code)
            is_valid, feedback = validator.validate_static()
            if is_valid:
                is_valid, feedback = validator.validate_runtime()

        # 6. Final Save
        if is_valid:
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = f"# Generated by {MODEL_NAME} (Iter {iteration_id}) on {timestamp_str}\n"
            utils.save_code(header + clean_code, "reward_shaping.py")
            print("✅ AGENT: Code validated and saved.")
        else:
            print("❌ AGENT: Failed to generate valid code after multiple attempts.")

    except Exception as e:
        print(f"❌ Phase 2 Error: {e}")


if __name__ == "__main__":
    run_agentic_improvement()