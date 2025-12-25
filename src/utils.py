import os
import re
import json
import ollama
import torch
import platform
import importlib.util
import sys
import numpy as np

# ---------------------------------------------------------
# FILE OPERATIONS
# ---------------------------------------------------------
def load_file(filepath):
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ""

def extract_python_code(llm_response):
    """Extracts code block from markdown response."""
    match = re.search(r'```python(.*?)```', llm_response, re.DOTALL)
    if match: return match.group(1).strip()
    match = re.search(r'```(.*?)```', llm_response, re.DOTALL)
    if match: return match.group(1).strip()
    return llm_response

def load_dynamic_module(module_name, file_path):
    """
    Loads a python file from a specific path as a module.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module 
        spec.loader.exec_module(module)
        return module
    raise ImportError(f"Could not load module {module_name} from {file_path}")

# ---------------------------------------------------------
# MEMORY FUNCTIONS (Now Workspace-Aware)
# ---------------------------------------------------------
def get_recent_history(workspace, current_iteration, n=3):
    """
    Retrieves the 'Implementation Plan' from the previous N iterations.
    Uses the workspace to find the files in the correct Iter_XX folders.
    """
    history_text = ""
    start_idx = max(1, current_iteration - n)
    
    for i in range(start_idx, current_iteration):
        # Construct path: experiments/.../Iter_XX/cognition/plan.md
        try:
            plan_path = workspace.get_path("cognition", i, "plan.md")
            if os.path.exists(plan_path):
                content = load_file(plan_path)
                # Just take the whole plan, or split if your prompt format is strict
                history_text += f"\n--- HISTORY ENTRY (Iter {i}) ---\n{content[:500]}...\n"
        except Exception as e:
            continue
            
    return history_text if history_text else "No previous history."

def get_long_term_memory(workspace, current_iteration, model_name, retention=3):
    """
    Summarizes iterations older than 'retention'.
    """
    cutoff = current_iteration - retention
    if cutoff < 1:
        return "No long-term history yet."
    
    older_plans = []
    
    # Iterate from 1 up to the cutoff
    for i in range(1, cutoff):
        try:
            plan_path = workspace.get_path("cognition", i, "plan.md")
            if os.path.exists(plan_path):
                content = load_file(plan_path)
                older_plans.append(f"(Iter {i}): {content[:200]}...")
        except:
            continue

    if not older_plans: 
        return "No older history to summarize."

    # Combine for the LLM
    combined_text = "\n".join(older_plans)
    
    summary_prompt = f"""
    You are an AI Researcher. Here are summaries of your earliest experiments (Iter 1 to {cutoff}):
    {combined_text}
    
    TASK: Summarize these into 3-5 'Immutable Lessons'. What failed? What worked?
    """
    try:
        response = ollama.chat(model=model_name, messages=[{'role': 'user', 'content': summary_prompt}])
        return response['message']['content']
    except:
        return "Could not generate summary."

def summarize_training_log(log_dir):
    """
    Reads the SB3 progress.json and returns a condensed dictionary.
    """
    log_path = os.path.join(log_dir, "progress.json")
    # (Existing logic remains the same, just ensured path join is correct)
    if not os.path.exists(log_path):
        return {"error": "No training log found."}

    data = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except:
        return {"error": "Could not parse log."}

    if not data: return {"error": "Empty log."}

    first = data[0]
    last = data[-1]
    
    return {
        "entropy_start": f"{first.get('train/entropy_loss', 0):.4f}",
        "entropy_end": f"{last.get('train/entropy_loss', 0):.4f}",
        "value_loss_avg": f"{np.mean([d.get('train/value_loss', 0) for d in data]):.4f}",
        "policy_loss_end": f"{last.get('train/policy_gradient_loss', 0):.4f}"
    }

# ---------------------------------------------------------
# HARDWARE
# ---------------------------------------------------------
def get_optimized_ppo_params(n_envs, device_type="auto"):
    # (Keep your existing logic here, it was good)
    TARGET_BUFFER_SIZE = 8192 
    TARGET_NUM_MINIBATCHES = 4 
    n_steps = max(int(TARGET_BUFFER_SIZE // n_envs), 16)
    actual_buffer_size = n_steps * n_envs
    batch_size = int(actual_buffer_size // TARGET_NUM_MINIBATCHES)
    
    if device_type == "auto":
        if torch.cuda.is_available(): device = "cuda"
        elif torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
    else:
        device = device_type

    return {"n_steps": n_steps, "batch_size": batch_size, "device": device}

def get_hardware_config():
    system = platform.system()
    if system == "Linux": return 32, "cuda"
    elif system == "Darwin": return 8, "mps"
    return 4, "cpu"