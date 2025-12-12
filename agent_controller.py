import json
import os
import re
import shutil
import ollama
from datetime import datetime
from config import Config

MODEL_NAME = "qwen2.5-coder" # or "llama3.1"
HISTORY_DIR = "logs/code_history"
REASONING_DIR = "logs/reasoning_history"

def load_file(filepath):
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ""

def archive_current_code(iteration_id):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    source = "reward_shaping.py"
    if not os.path.exists(source): return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reward_shaping_gen{iteration_id:03d}_{timestamp}.py"
    destination = os.path.join(HISTORY_DIR, filename)
    shutil.copy(source, destination)
    print(f"🗂️  Archived code to: {destination}")

def save_reasoning(iteration_id, diagnosis, plan):
    """Saves the diagnosis and plan (Pure Text)."""
    os.makedirs(REASONING_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(REASONING_DIR, f"reasoning_gen{iteration_id:03d}_{timestamp}.md")
    
    with open(filename, "w") as f:
        f.write(f"# Generator Reasoning (Iteration {iteration_id})\n")
        f.write(f"## Model: {MODEL_NAME} | Time: {timestamp}\n\n")
        f.write("### 1. Diagnosis\n")
        f.write(diagnosis + "\n\n")
        f.write("### 2. Implementation Plan\n")
        f.write(plan + "\n")
    
    print(f"🧠 Saved reasoning to: {filename}")

def save_code(code, filepath):
    with open(filepath, 'w') as f:
        f.write(code)
    print(f"💾 Successfully updated {filepath}")

def extract_python_code(llm_response):
    match = re.search(r'```python(.*?)```', llm_response, re.DOTALL)
    if match: return match.group(1).strip()
    match = re.search(r'```(.*?)```', llm_response, re.DOTALL)
    if match: return match.group(1).strip()
    return llm_response

# ---------------------------------------------------------
# PROMPT ENGINEERING
# ---------------------------------------------------------
def get_diagnosis_prompt(metrics, current_code):
    stats = metrics['performance']
    diagnostics = metrics.get('diagnostics', {})
    
    # Extract new metrics
    main_eng = diagnostics.get('main_engine_usage', 0)
    side_eng = diagnostics.get('side_engine_usage', 0)
    avg_x = diagnostics.get('avg_x_position', 0)
    
    return f"""
    You are a Senior RL Researcher. Analyze the agent's performance on '{Config.ENV_ID}'.
    
    METRICS:
    - Success Rate: {stats['success_rate']:.2f} (Target: >0.9)
    - Crash Rate: {stats.get('crash_rate', 0):.2f} (Target: <0.1)
    
    FLIGHT TELEMETRY:
    - Avg Vertical Vel: {diagnostics.get('avg_descent_velocity', 0):.2f} (Target: Negative but controlled)
    - Avg Tilt Angle: {diagnostics.get('avg_tilt_angle', 0):.2f} (Target: near 0.0)
    - Avg X-Position: {avg_x:.2f} (Target: 0.0 is center. +/- 1.0 is edge)
    
    ENGINE TELEMETRY:
    - Main Engine Usage: {main_eng:.2f} (Target: Balanced. If <0.1, it falls. If >0.8, it panics.)
    - Side Engine Usage: {side_eng:.2f} (Target: Low. If >0.5, it is jittering.)

    CURRENT REWARD FUNCTION:
    ```python
    {current_code}
    ```

    TASK:
    1. DIAGNOSE: Why is the agent failing? 
       - If X-Position is far from 0, it is drifting sideways.
       - If Main Engine is low (<0.2) and Velocity is high, it isn't trying to fly.
       - If Side Engines are high, it is wasting energy.
       
    2. PLAN: Propose a specific mathematical adjustment to the reward function to fix this.
    
    OUTPUT FORMAT:
    Provide a clear, concise paragraph explaining the failure and the planned math fix. DO NOT write code yet.
    """

def get_coding_prompt(plan, current_code):
    return f"""
    You are a Python Expert. Implement this plan to fix the RL reward function.

    THE PLAN:
    {plan}

    EXISTING CODE:
    ```python
    {current_code}
    ```

    INSTRUCTIONS:
    - Rewrite the `calculate_reward` function based on the plan.
    - Keep imports and function signature exactly the same.
    - Return ONLY valid Python code inside ```python ``` blocks.
    - Use numpy as np.
    """

# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------

def run_agentic_improvement():
    # 1. Setup
    iteration_id = Config.get_iteration()
    print(f"🔵 AGENT (Iter {iteration_id}): Reading state...")
    
    if not os.path.exists(Config.METRICS_FILE):
        print("❌ No metrics found.")
        return

    metrics = json.loads(load_file(Config.METRICS_FILE))
    current_code = load_file("reward_shaping.py")
    
    # 2. Archive
    archive_current_code(iteration_id)
    
    # 3. PHASE 1: DIAGNOSIS (The "Brain")
    print("🔵 AGENT: Phase 1 - Diagnosing & Planning...")
    diag_prompt = get_diagnosis_prompt(metrics, current_code)
    try:
        response1 = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': diag_prompt}])
        diagnosis_plan = response1['message']['content']
        
        # Save the "Thought" immediately
        save_reasoning(iteration_id, diagnosis=diagnosis_plan, plan=diagnosis_plan)
        
    except Exception as e:
        print(f"❌ Phase 1 Error: {e}")
        return

    # 4. PHASE 2: IMPLEMENTATION (The "Hands")
    print("🔵 AGENT: Phase 2 - Writing Code...")
    code_prompt = get_coding_prompt(diagnosis_plan, current_code)
    try:
        response2 = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': code_prompt}])
        raw_code = response2['message']['content']
        
        clean_code = extract_python_code(raw_code)
        
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"# Generated by {MODEL_NAME} (Iter {iteration_id}) on {timestamp_str}\nimport numpy as np\n\n"
        final_file_content = header + clean_code
        
        save_code(final_file_content, "reward_shaping.py")
        
    except Exception as e:
        print(f"❌ Phase 2 Error: {e}")

if __name__ == "__main__":
    run_agentic_improvement()