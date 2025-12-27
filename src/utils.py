import os
import re
import json
import csv
import difflib
import ollama
import torch
import platform
import importlib.util
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
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
# SCIENTIFIC LOGGING & ANALYSIS
# ---------------------------------------------------------
def update_campaign_summary(ws, iteration, metrics):
    """
    Appends a summary row to the master campaign CSV file.
    Acts as the 'Scoreboard' for the entire experiment.
    """
    csv_path = ws.model_root_path / "campaign_summary.csv"
    
    headers = [
            "Iteration", "Timestamp", 
            # Eval Metrics
            "Eval_Reward_Mean", "Eval_Reward_Std", "Eval_Ep_Len_Mean",
            "Reward_Success_Rate", "Position_Success_Rate", "Crash_Rate", 
            # Training Metrics (TensorBoard)
            "Train_Reward_Mean", "Entropy_End", "Value_Loss_Avg", "Policy_Loss_End",
            # Diagnostics
            "Fuel_Efficiency", "Stability_Index", "Generation_Status"
        ]
    
    perf = metrics.get("performance", {})
    diag = perf.get("diagnostics", {})
    train_dyn = metrics.get("training_dynamics", {})
    status = metrics.get("generation_status", "unknown") # Passed from controller
    
    row_data = {
            "Iteration": iteration,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            
            "Eval_Reward_Mean": round(perf.get("mean_reward", 0), 2),
            "Eval_Reward_Std": round(perf.get("std_reward", 0), 2),
            "Eval_Ep_Len_Mean": round(perf.get("mean_ep_length", 0), 2),
            "Reward_Success_Rate": round(perf.get("reward_success_rate", 0), 2),
            "Position_Success_Rate": round(perf.get("position_success_rate", 0), 2),
            "Crash_Rate": round(perf.get("crash_rate", 0), 2),
            
            "Train_Reward_Mean": round(train_dyn.get("raw_train_reward", 0), 2),
            "Entropy_End": round(train_dyn.get("raw_entropy", 0), 4),
            "Value_Loss_Avg": round(train_dyn.get("raw_value_loss", 0), 4),
            "Policy_Loss_End": round(train_dyn.get("raw_policy_loss", 0), 4),
            
            "Fuel_Efficiency": round(diag.get("main_engine_usage", 0), 4),
            "Stability_Index": round(diag.get("vertical_stability_index", 0), 4),
            "Generation_Status": status
        }

    file_exists = csv_path.exists()
    
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)
        
    print(f"📈 Campaign Summary updated: {csv_path}")

def save_readable_context(workspace, iteration, context_dict):
    """
    Converts the raw context dictionary into a beautiful Markdown file 
    for human debugging.
    """
    metrics = context_dict.get('metrics', {})
    training = context_dict.get('training_summary', {})
    memory = context_dict.get('memory_context', {})
    prompts = context_dict.get('prompts', {})

    md_content = f"""# Iteration {iteration} Context Report
**Timestamp:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. The "Eyes" (Incoming Data)
### Performance Metrics
| Metric | Value |
| :--- | :--- |
| Mean Reward | {metrics.get('performance', {}).get('mean_reward', 'N/A')} |
| Reward Success Rate | {metrics.get('performance', {}).get('reward_success_rate', 'N/A')} |
| Position Success Rate | {metrics.get('performance', {}).get('position_success_rate', 'N/A')} |
| Crash Rate | {metrics.get('performance', {}).get('crash_rate', 'N/A')} |

### Training Dynamics
- **Entropy:** {training.get('entropy_start')} -> {training.get('entropy_end')}
- **Value Loss:** {training.get('value_loss_avg')}

## 2. The "Memory"
### Short Term
```text
{memory.get('short', 'No history')}
```
### Long Term
```text
{memory.get('long', 'No history')}
```
## 3. The Prompts Sent
### Diagnosis Prompt
```text
{prompts.get('diagnosis_task', 'N/A')}
```"""
    md_path = workspace.get_path("cognition", iteration, "context_report.md")
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"🧠 Context report saved: {md_path}")

def summarize_training_log(log_dir): 
    """ Reads SB3 progress.json. Returns Strings for LLM, Floats for CSV. """ 
    log_path = os.path.join(log_dir, "progress.json") 
    if not os.path.exists(log_path): return {"error": "No training log."}
    data = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if line.strip(): data.append(json.loads(line))
    except: return {"error": "Parse error."}

    if not data: return {"error": "Empty log."}

    first, last = data[0], data[-1]
    val_loss_avg = np.mean([d.get('train/value_loss', 0) for d in data])

    return {
        # Strings for LLM
        "entropy_start": f"{first.get('train/entropy_loss', 0):.4f}",
        "entropy_end": f"{last.get('train/entropy_loss', 0):.4f}",
        "entropy_trend": "Collapsing" if last.get('train/entropy_loss', 0) < -1.0 else "Stable",
        "value_loss_avg": f"{val_loss_avg:.4f}",
        "policy_loss_end": f"{last.get('train/policy_gradient_loss', 0):.4f}",
        
        # Raw Floats for CSV
        "raw_train_reward": float(last.get('rollout/ep_rew_mean', 0)),
        "raw_entropy": float(last.get('train/entropy_loss', 0)),
        "raw_value_loss": float(val_loss_avg),
        "raw_policy_loss": float(last.get('train/policy_gradient_loss', 0))
    }
def generate_patch(old_code: str, new_code: str, filename: str) -> str:
    """Compares two source strings and returns a Unified Diff."""
    diff = difflib.unified_diff(
        old_code.splitlines(keepends=True), 
        new_code.splitlines(keepends=True), 
        fromfile=f"prev/{filename}", 
        tofile=f"new/{filename}",
        lineterm=""
    )
    return "".join(diff)

def save_diff(old_code, new_code, iteration, attempt, base_dir):
    """Saves a delta patch between attempts to track debugging logic."""
    filename = f"iter{iteration:02d}_attempt_{attempt:02d}.patch"
    filepath = base_dir / filename
    
    diff = difflib.unified_diff(
        old_code.splitlines(keepends=True),
        new_code.splitlines(keepends=True),
        fromfile=f"Code Gen. Attempt {attempt-1}",
        tofile=f"Code Gen. Attempt {attempt}",
        lineterm=""
    )
    
    text = "".join(diff)
    if text:
        with open(filepath, "w") as f:
            f.write(text)
    return filepath
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