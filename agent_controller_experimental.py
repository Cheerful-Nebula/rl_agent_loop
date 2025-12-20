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
    
    # 1. Prepare the Data
    # (Fetch memory logic remains the same)
    short_term_history = get_recent_history(n=3)
    long_term_memory = get_long_term_memory(iteration_id, retention=3)

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

    metrics = json.loads(load_file(Config.METRICS_FILE))
    current_code = load_file("reward_shaping.py")
    
    # -- Archive Previous Reward Shaping code
    archive_current_code(iteration_id)
    
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
        save_reasoning(iteration_id, diagnosis_plan)
        
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
        
        clean_code = extract_python_code(response2['message']['content'])
        
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
            archive_current_code(iteration_id, attempt_num=attempt_num) 

            # 4. Get Fix Prompts (Refactored)
            fix_role, fix_task = get_fix_prompts(clean_code, feedback)
            
            # 5. Call LLM (Fix Request)
            response_fix = ollama.chat(model=MODEL_NAME, messages=[
                {'role': 'system', 'content': fix_role},
                {'role': 'user', 'content': fix_task}
            ])
            
            clean_code = extract_python_code(response_fix['message']['content'])
            
            # Re-validate (Static -> Runtime)
            validator = CodeValidator(clean_code)
            is_valid, feedback = validator.validate_static()
            if is_valid:
                is_valid, feedback = validator.validate_runtime()

        # 6. Final Save
        if is_valid:
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