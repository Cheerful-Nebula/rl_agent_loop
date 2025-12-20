import os
import shutil
import re
import json
import ollama
from datetime import datetime
from config import Config

# ---------------------------------------------------------
# FILE OPERATIONS
# ---------------------------------------------------------
def load_file(filepath):
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ""

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
# ARCHIVING & LOGGING
# ---------------------------------------------------------
def archive_current_code(iteration_id:int, attempt_num:int = None):
    code_dir = Config.CODE_DIR
    os.makedirs(code_dir, exist_ok=True)
    source = "reward_shaping.py"
    if not os.path.exists(source): return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if attempt_num is None:
        filename = f"reward_shaping_gen{iteration_id:03d}_{timestamp}.py"
    else:
        filename = f"reward_shaping_gen{iteration_id:03d}_attempt{attempt_num}_{timestamp}.py"
    
    destination = os.path.join(code_dir, filename)
    shutil.copy(source, destination)
    print(f"🗂️  Archived code to: {destination}")

def save_reasoning(iteration_id, content, model_name):
    """Saves the diagnosis and plan (Pure Text)."""
    reasoning_dir = Config.REASONING_DIR
    os.makedirs(reasoning_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(reasoning_dir, f"reasoning_gen{iteration_id:03d}_{timestamp}.md")
    
    with open(filename, "w") as f:
            f.write(f"# Generator Reasoning (Iteration {iteration_id})\n")
            f.write(f"## Model: {model_name} | Time: {timestamp}\n\n")
            f.write("## Analyst Report (Diagnosis & Plan)\n")
            f.write(content + "\n")
    print(f"🧠 Saved reasoning to: {filename}")

def save_metrics(iteration_id, ):
    metric_dir = Config.METRICS_DIR
    os.makedirs(metric_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = os.path.join(metric_dir, f"metrics_gen{iteration_id:03d}_{timestamp_str}.json")
    shutil.copy(Config.METRICS_FILE, archive_path)
# ---------------------------------------------------------
# MEMORY FUNCTIONS
# ---------------------------------------------------------
# Detailed short-term memory
def get_recent_history(n=3):
    """
    Retrieves the 'Implementation Plan' from the last N reasoning files.
    """
    reasoning_dir = Config.REASONING_DIR
    if not os.path.exists(reasoning_dir):
        return "No previous history."
        
    files = sorted([f for f in os.listdir(reasoning_dir) if f.endswith('.md')])
    recent_files = files[-n:]
    history_text = ""
    
    for i, filename in enumerate(recent_files):
        content = load_file(os.path.join(reasoning_dir, filename))
        if "### 2. Implementation Plan" in content:
            plan = content.split("### 2. Implementation Plan")[1].strip()
            history_text += f"\n--- HISTORY ENTRY {i+1} ({filename}) ---\n{plan}\n"
            
    return history_text if history_text else "No parseable history found."

# Fuzzy short-term memory
def get_long_term_memory(current_iteration, model_name, retention=3):
    """
    Summarizes all iterations older than 'retention'.
    Returns a concise list of 'Lessons Learned'.
    """
    reasoning_dir = Config.REASONING_DIR
    if current_iteration <= retention + 1:
        return "No long-term history yet."
    # We want to summarize files from Gen 1 up to (Current - Retention - 1)   
    older_files = []
    cutoff = current_iteration - retention
    
    if not os.path.exists(reasoning_dir): return ""
    
    all_files = sorted([f for f in os.listdir(reasoning_dir) if f.endswith('.md')])

    # Filter for files strictly OLDER than the cutoff
    # Filename format: reasoning_gen005_... 
    for f in all_files:
        try:
            gen_num = int(f.split('_gen')[1][:3])
            if gen_num < cutoff:
                older_files.append(f)
        except: continue

    if not older_files: return "No older history to summarize."

    print(f"📚 AGENT: Compressing {len(older_files)} old memories into Long-Term Memory...")  

    # Combine the "Implementation Plan" from these old files
    combined_text = ""
    for f in older_files:
        content = load_file(os.path.join(reasoning_dir, f))
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
    """
    try:
        response = ollama.chat(model=model_name, messages=[{'role': 'user', 'content': summary_prompt}])
        return response['message']['content']
    except:
        return "Could not generate summary."