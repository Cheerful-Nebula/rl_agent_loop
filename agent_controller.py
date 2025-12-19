import json
import os
import re
import shutil
import ollama
from datetime import datetime 
from config import Config
from code_validation import CodeValidator


MODEL_NAME = Config.LLM_MODEL
HISTORY_DIR = Config.HISTORY_DIR
REASONING_DIR = Config.REASONING_DIR
device = Config.get_device()
# ---------------------------------------------------------
# FILE UTILITIES
# ---------------------------------------------------------
def load_file(filepath):
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ""

def archive_current_code(iteration_id:int, attempt_num:int = None):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    source = "reward_shaping.py"
    if not os.path.exists(source): return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if attempt_num == None:
        filename = f"reward_shaping_gen{iteration_id:03d}_{timestamp}.py"
    else:
        filename = f"reward_shaping_gen{iteration_id:03d}_attempt{attempt_num}_{timestamp}.py"
    destination = os.path.join(HISTORY_DIR, filename)
    shutil.copy(source, destination)
    print(f"🗂️  Archived code to: {destination}")

def save_reasoning(iteration_id, content):
    """Saves the diagnosis and plan (Pure Text)."""
    os.makedirs(REASONING_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(REASONING_DIR, f"reasoning_gen{iteration_id:03d}_{timestamp}.md")
    
    with open(filename, "w") as f:
            f.write(f"# Generator Reasoning (Iteration {iteration_id})\n")
            f.write(f"## Model: {MODEL_NAME} | Time: {timestamp}\n\n")
            f.write("## Analyst Report (Diagnosis & Plan)\n")
            f.write(content + "\n")
    
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
# MEMORY 
# ---------------------------------------------------------
# Detailed short-term memory
def get_recent_history(n=3):
    """
    Retrieves the 'Implementation Plan' from the last N reasoning files.
    """
    if not os.path.exists(REASONING_DIR):
        return "No previous history."
        
    # Get all reasoning files, sorted by name (which includes timestamp/gen ID)
    files = sorted([f for f in os.listdir(REASONING_DIR) if f.endswith('.md')])
    
    # Take the last N
    recent_files = files[-n:]
    history_text = ""
    
    for i, filename in enumerate(recent_files):
        content = load_file(os.path.join(REASONING_DIR, filename))
        # Crude extraction: Get the text after "### 2. Implementation Plan"
        if "### 2. Implementation Plan" in content:
            plan = content.split("### 2. Implementation Plan")[1].strip()
            history_text += f"\n--- HISTORY ENTRY {i+1} ({filename}) ---\n{plan}\n"
            
    return history_text if history_text else "No parseable history found."

# Fuzzy short-term memory
def get_long_term_memory(current_iteration, retention=3):
    """
    Summarizes all iterations older than 'retention'.
    Returns a concise list of 'Lessons Learned'.
    """
    if current_iteration <= retention + 1:
        return "No long-term history yet."
        
    # We want to summarize files from Gen 1 up to (Current - Retention - 1)
    older_files = []
    cutoff = current_iteration - retention
    
    if not os.path.exists(REASONING_DIR): return ""
    
    all_files = sorted([f for f in os.listdir(REASONING_DIR) if f.endswith('.md')])
    
    # Filter for files strictly OLDER than the cutoff
    # Filename format: reasoning_gen005_...
    for f in all_files:
        try:
            gen_num = int(f.split('_gen')[1][:3])
            if gen_num < cutoff:
                older_files.append(f)
        except: continue

    if not older_files:
        return "No older history to summarize."

    print(f"📚 AGENT: Compressing {len(older_files)} old memories into Long-Term Memory...")
    
    # Combine the "Implementation Plan" from these old files
    combined_text = ""
    for f in older_files:
        content = load_file(os.path.join(REASONING_DIR, f))
        if "### 2. Implementation Plan" in content:
            plan = content.split("### 2. Implementation Plan")[1].strip()
            combined_text += f"\n- (Gen {f.split('_gen')[1][:3]}): {plan[:200]}..." # Truncate for token efficiency

    # Ask LLM to distill this
    summary_prompt = f"""
    You are an AI Researcher analyzing your past experiments.
    Here are the short summaries of your old attempts (Generations 1 to {cutoff-1}):
    {combined_text}
    
    TASK:
    Summarize these into a bulleted list of 3-5 "Immutable Lessons". 
    What failed? What worked?
    Example: "- High penalties for tilting caused the agent to freeze."
    """
    
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': summary_prompt}])
        return response['message']['content']
    except:
        return "Could not generate summary."
# ---------------------------------------------------------
# PROMPT ENGINEERING
# ---------------------------------------------------------
def get_diagnosis_prompt(metrics, current_code, iteration_id):
    stats = metrics['performance']
    diagnostics = metrics.get('diagnostics', {})
    
    # Fetch Hierarchical Memory
    short_term_history = get_recent_history(n=3)
    long_term_memory = get_long_term_memory(iteration_id, retention=3)

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

    LONG-TERM MEMORY (Lessons from the distant past):
    {long_term_memory}
    
    SHORT-TERM HISTORY (Recent attempts):
    {short_term_history}

    CURRENT REWARD FUNCTION:
    ```python
    {current_code}
    ```

    TASK:
    1. DIAGNOSE: 
    - Combine the metrics, telemetry, and memory to identify why the agent is underperforming.
    - Why is the agent failing? 
    - Are we repeating past mistakes?
    - Is the reward function misaligned with our goals?
       
    2. PLAN: 
    - Propose a new mathematical adjustment
    
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
    # -- Setup
    iteration_id = Config.get_iteration()
    print(f"🔵 AGENT (Iter {iteration_id}): Reading state...")
    
    if not os.path.exists(Config.METRICS_FILE):
        print("❌ No metrics found.")
        return

    metrics = json.loads(load_file(Config.METRICS_FILE))
    current_code = load_file("reward_shaping.py")
    
    # -- Archive Previous Reward Shaping code
    archive_current_code(iteration_id)
    
    # -- PHASE 1: DIAGNOSIS (The "Brain")
    print("🔵 AGENT: Phase 1 - Diagnosing & Planning...")
    diag_prompt = get_diagnosis_prompt(metrics, current_code, iteration_id)
    try:
        response1 = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': diag_prompt}])
        diagnosis_plan = response1['message']['content']
        
        # Save the "Thought" immediately
        save_reasoning(iteration_id, diagnosis_plan)
        
    except Exception as e:
        print(f"❌ Phase 1 Error: {e}")
        return

    # --- PHASE 2: IMPLEMENTATION ---
    print("🔵 AGENT: Phase 2 - Writing Code...")
    code_prompt = get_coding_prompt(diagnosis_plan, current_code)

    try:
        response2 = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': code_prompt}])
        clean_code = extract_python_code(response2['message']['content'])
        
        validator = CodeValidator(clean_code)
        is_valid, feedback = validator.validate()

        attempt_num = 0
        while not is_valid and attempt_num < 20:  # Added a safety limit to prevent infinite loops
            attempt_num += 1
            print(f"⚠️ AGENT: Validation failed (Attempt {attempt_num}). Requesting fix...")
            archive_current_code(iteration_id, attempt_num=attempt_num) 

            fix_prompt = f"""
            The following code failed validation:
            ```python
            {clean_code}
            ```
            Issue: {feedback}
            
            Please provide a corrected version. 
            - Use only 'numpy' or 'math' for calculations.
            - Do not use forbidden functions (eval, exec, open) or modules (os, sys).
            - Return ONLY valid Python code inside ```python ``` blocks.
            """
            
            response_fix = ollama.chat(model=MODEL_NAME, messages=[{'role': 'user', 'content': fix_prompt}])
            clean_code = extract_python_code(response_fix['message']['content'])
            
            # Re-validate the new code
            validator = CodeValidator(clean_code)
            is_valid, feedback = validator.validate()

        if is_valid:
            # Save the successful code
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = f"# Generated by {MODEL_NAME} (Iter {iteration_id}) on {timestamp_str}\n"
            save_code(header + clean_code, "reward_shaping.py")
            print("✅ AGENT: Code validated and saved.")
        else:
            print("❌ AGENT: Failed to generate valid code after multiple attempts.")

    except Exception as e:
        print(f"❌ Phase 2 Error: {e}")


if __name__ == "__main__":
    run_agentic_improvement()