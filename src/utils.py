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
from stable_baselines3.common.monitor import Monitor
from typing import Tuple
import pandas as pd
from textwrap import indent

# -- Custom IMPORTS --
from src.wrappers import DynamicRewardWrapper
from src.config import Config
from src.workspace_manager import ExperimentWorkspace
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
def extract_json(llm_response):
    """
    Robustly extracts JSON from an LLM response, handling markdown fences and stray text.
    """
    import json
    import re
    
    # 1. Try finding a markdown block first
    match = re.search(r'```json(.*?)```', llm_response, re.DOTALL)
    if match:
        clean_str = match.group(1).strip()
    else:
        # 2. If no block, try to find the first '{' and last '}'
        start = llm_response.find('{')
        end = llm_response.rfind('}')
        if start != -1 and end != -1:
            clean_str = llm_response[start:end+1]
        else:
            clean_str = llm_response.strip()

    # 3. Validation / Parsing
    try:
        return json.loads(clean_str)
    except json.JSONDecodeError:
        # Optional: Print preview for debugging, but keeping logs clean
        # print(f"⚠️ JSON Parsing Failed. Content preview: {clean_str[:50]}...")
        return None
    
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

def save_cognition_markdown(ws: ExperimentWorkspace,iteration, cognition_list:list[Tuple]):
        # Save the plan for human review
        plan_path = ws.get_path("cognition_markdown", iteration, "cognition_record.md")
        final_content = f"# Cognition prompts and calls: Iteration:{iteration}\n\n"
        for filename, prompt_obj in cognition_list:
            final_content += "*"*25 +f"\n"+ "*"*25 +f"\n" + "*"*25 +f"\n"
            final_content += f"# {filename}"
            final_content += f"\n"+ "*"*25 +f"\n"+ "*"*25 +f"\n" + "*"*25 +f"\n"
            final_content += prompt_obj + f"\n\n"

        with open(plan_path, "w") as f:
            f.write(final_content)
        print(f"📝 Plan saved to {plan_path}") 

def convert_formatter_json_to_markdown(data: dict) -> str:
    """
    Converts the structured Formatter JSON into a clean, human-readable Markdown report.
    Works for schema class: FormatterOutput(BaseModel)
    """
    md_lines = []
    
    # 1. ANALYSIS (The "Why")
    if "analysis" in data and data["analysis"]:
        md_lines.append("## 🔬 Analysis")
        md_lines.append(data["analysis"])
        md_lines.append("")

    # 2. PLAN (The "How" - This goes to the Coder)
    if "plan" in data and data["plan"]:
        md_lines.append("## 🛠️ Refinement Plan")
        md_lines.append(data["plan"])
        md_lines.append("")

    # 3. LESSON (The Long-term Memory)
    if "lesson" in data and data["lesson"]:
        md_lines.append("## 🧠 Immutable Lesson")
        md_lines.append(f"> {data['lesson']}")
        md_lines.append("")
        
    if "hypothesis" in data and data["hypothesis"]:
        md_lines.append("## 🧠 Hypothesis")
        md_lines.append(f"> {data['hypothesis']}")
        md_lines.append("")

    # 4. HYPERPARAMETERS (The Future Config Updates)
    if "hyperparameters" in data and data["hyperparameters"]:
        md_lines.append("## ⚙️ Hyperparameter Recommendations")
        # Handle if it's a string (raw text) or dict
        if isinstance(data["hyperparameters"], dict):
            for k, v in data["hyperparameters"].items():
                md_lines.append(f"- **{k}**: {v}")
        else:
            md_lines.append(str(data["hyperparameters"]))
        md_lines.append("")

    return "\n".join(md_lines)

def plan_json_to_markdown(plan_json: str) -> str:
    """
    Convert the LLM's Reward Function Refinement Plan JSON into Markdown.

    plan_json: JSON string with keys:
      - Analysis (str)
      - Identified Issues (str or list[str])
      - Recommendations (str or list[str])
      - Reward Function Refinement Plan:
          - Modifications: list[{Description, Rationale}]
          - Implementation Steps: str or list[str]
          - Future Work: list[{Hypothesis, Confidence Score, Expected Outcome}]
    """
    plan = json.loads(plan_json)

    analysis = plan.get("Analysis", "").strip()
    issues = plan.get("Identified Issues", [])
    recs = plan.get("Recommendations", [])
    refinement = plan.get("Reward Function Refinement Plan", {}) or {}

    mods = refinement.get("Modifications", [])
    impl_steps = refinement.get("Implementation Steps", [])
    future_work = refinement.get("Future Work", [])

    # Normalize possibly-string fields to lists
    if isinstance(issues, str):
        issues = [s.strip() for s in issues.split("\n") if s.strip()]
    if isinstance(recs, str):
        recs = [s.strip() for s in recs.split("\n") if s.strip()]
    if isinstance(impl_steps, str):
        impl_steps = [s.strip() for s in impl_steps.split("\n") if s.strip()]
    if isinstance(mods, dict):
        mods = [mods]
    if isinstance(future_work, dict):
        future_work = [future_work]

    lines = []

    # Top-level analysis
    if analysis:
        lines.append("## Analysis")
        lines.append("")
        lines.append(analysis)
        lines.append("")

    # Issues
    if issues:
        lines.append("## Identified Issues")
        lines.append("")
        for i, issue in enumerate(issues, 1):
            lines.append(f"{i}. {issue}")
        lines.append("")

    # Recommendations
    if recs:
        lines.append("## Recommendations")
        lines.append("")
        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    # Reward Function Refinement Plan
    if mods or impl_steps or future_work:
        lines.append("## Reward Function Refinement Plan")
        lines.append("")

    # Modifications
    if mods:
        lines.append("### Modifications")
        lines.append("")
        for idx, m in enumerate(mods, 1):
            desc = (m.get("Description") or "").strip()
            rat = (m.get("Rationale") or "").strip()
            lines.append(f"#### Modification {idx}")
            if desc:
                lines.append("")
                lines.append("**Description**")
                lines.append("")
                lines.append(desc)
            if rat:
                lines.append("")
                lines.append("**Rationale**")
                lines.append("")
                lines.append(rat)
            lines.append("")

    # Implementation Steps
    if impl_steps:
        lines.append("### Implementation Steps")
        lines.append("")
        for i, step in enumerate(impl_steps, 1):
            lines.append(f"{i}. {step}")
        lines.append("")

    # Future Work
    if future_work:
        lines.append("### Future Work")
        lines.append("")
        for i, fw in enumerate(future_work, 1):
            hyp = (fw.get("Hypothesis") or "").strip()
            conf = fw.get("Confidence Score") or fw.get("Confidence") or ""
            exp = (fw.get("Expected Outcome") or "").strip()

            lines.append(f"#### Hypothesis {i}")
            if hyp:
                lines.append("")
                lines.append("**Hypothesis**")
                lines.append("")
                lines.append(hyp)
            if conf != "":
                lines.append("")
                lines.append(f"**Confidence Score:** {conf}")
            if exp:
                lines.append("")
                lines.append("**Expected Outcome**")
                lines.append("")
                lines.append(exp)
            lines.append("")

    return "\n".join(lines).rstrip()


# ---------------------------------------------------------
# ENVIRONMENTS
# ---------------------------------------------------------
def make_env(reward_code_path:str | None = None):
    import gymnasium as gym
    env = gym.make(Config.ENV_ID)
    env = DynamicRewardWrapper(env, reward_code_path=reward_code_path) 
    return Monitor(env)
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



def summarize_training_log(ws: ExperimentWorkspace, iteration: int)-> json:
   log_path =  ws.dirs["telemetry_training"] / f"progress_{iteration:02d}.csv"
   df = pd.read_csv(log_path)
   training_summary= {
    "policy_gradient_loss": {"start": df['train/policy_gradient_loss'].iloc[1].round(4),
                             "median": df['train/policy_gradient_loss'].median().round(4),
                             "mean": df['train/policy_gradient_loss'].mean().round(4),
                             "end": df['train/policy_gradient_loss'].iloc[-1].round(4)},
    "approx_kl": {"start": df['train/approx_kl'].iloc[1].round(4),
                  "median": df['train/approx_kl'].median().round(4),
                  "mean": df['train/approx_kl'].mean().round(4),
                  "end": df['train/approx_kl'].iloc[-1].round(4)},
    "loss": {"start": df['train/loss'].iloc[1].round(4),
             "median": df['train/loss'].median().round(4),
             "mean": df['train/loss'].mean().round(4),
             "end": df['train/loss'].iloc[-1].round(4)},
    "explained_variance": {"start": df['train/explained_variance'].iloc[1].round(4),
                           "median": df['train/explained_variance'].median().round(4),
                           "mean": df['train/explained_variance'].mean().round(4),
                           "end": df['train/explained_variance'].iloc[-1].round(4)},
    "clip_range": {"start": df['train/clip_range'].iloc[1].round(4),
                   "median": df['train/clip_range'].median().round(4),
                   "mean": df['train/clip_range'].mean().round(4),
                   "end": df['train/clip_range'].iloc[-1].round(4)},    
    "entropy_loss": {"start": df['train/entropy_loss'].iloc[1].round(4),
                     "median": df['train/entropy_loss'].median().round(4),
                     "mean": df['train/entropy_loss'].mean().round(4),
                     "end": df['train/entropy_loss'].iloc[-1].round(4)},
    "value_loss": {"start": df['train/value_loss'].iloc[1].round(4),
                   "median": df['train/value_loss'].median().round(4),
                   "mean": df['train/value_loss'].mean().round(4),
                   "end": df['train/value_loss'].iloc[-1].round(4)},
    "clip_fraction": {"start": df['train/clip_fraction'].iloc[1].round(4),
                      "median": df['train/clip_fraction'].median().round(4),
                      "mean": df['train/clip_fraction'].mean().round(4),
                      "end": df['train/clip_fraction'].iloc[-1].round(4)},
    "learning_rate": {"start": df['train/learning_rate'].iloc[1],
                      "median": df['train/learning_rate'].median().round(4),
                      "mean": df['train/learning_rate'].mean().round(4),
                      "end": df['train/learning_rate'].iloc[-1]},
    "n_updates": {"total": df['train/n_updates'].iloc[-1]}
    }

   #training_summary_json = json.dumps(training_summary_json, indent=4)

   return training_summary

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
# Agent's Hyperparameters
# ---------------------------------------------------------
def linear_schedule(initial_value: float, final_value: float):
    """
    Linear learning rate schedule.
    :param initial_value: Starting learning rate.
    :param final_value: Ending learning rate.
    :return: schedule that computes current lr based on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0 (end).
        """
        return final_value + (initial_value - final_value) * progress_remaining

    return func

# Usage: Decay from 0.001 to 0.0001
# set when initializing the RL model:
# lr_schedule = linear_schedule(1e-3, 1e-4)
# ---------------------------------------------------------
# MEMORY FUNCTIONS (Now Workspace-Aware)
# ---------------------------------------------------------
def get_recent_history(workspace, current_iteration, n=20):
    """
    Retrieves the 'Implementation Plan' from the previous N iterations.
    Uses the workspace to find the files in the correct Iter_XX folders.
    """
    history_text = ""
    start_idx = max(1, current_iteration - n)
    
    for i in range(start_idx, current_iteration):
        # Construct path: experiments/.../Iter_XX/cognition/plan.md
        try:
            plan_path = workspace.get_path("cognition_lessons", i, "lesson.md")
            if os.path.exists(plan_path):
                content = load_file(plan_path)
                # Just take the whole plan, or split if your prompt format is strict
                history_text += f"\n--- HISTORY ENTRY (Iter {i}) ---\n{content}\n"
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


# ---------------------------------------------------------
# HARDWARE
# ---------------------------------------------------------
def get_optimized_ppo_params(n_envs, device_type="auto"):
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

# -----------------------------------------------------------
# Converting evaluations metrics dictionary to markdown table
# Hoping for improved comprehensision of LLM's analysis
# ------------------------------------------------------------
def performance_telemetry_as_table(stats_list: list[dict]) -> str:
    """
    Converts the list of performance dicts into a Markdown table for better LLM reasoning.
    """
    if not stats_list:
        return "No telemetry data available."

    # To guaruntee the correct order of dicts, so metric values are in correct column
    for stats_dict in stats_list:
        if stats_dict["policy_behavior"] == "Deterministic":
            if stats_dict["reward_shape"] == "Base":
                det_base_stats = stats_dict
            else:
                det_shaped_stats = stats_dict
        else:
            if stats_dict["reward_shape"] == "Base":
                stoch_base_stats = stats_dict
            else:
                stoch_shaped_stats = stats_dict


    # Define the columns we want to compare
    headers = [
        "metric",
        f"Stochastic/Shaped Reward",
        f"Deterministic/Base Reward"
    ]
    
    # Create the header row
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]

    key_list = [
        "mean_reward",
        "median_reward",
        "std_reward",
        "mean_ep_length",
        "reward_success_rate",
        "position_success_rate",
        "crash_rate",
        "avg_x_position",
        "avg_descent_velocity",
        "avg_tilt_angle",
        "vertical_stability_index",
        "horizontal_stability_index"
        ]
    for key in key_list:
        # Pre-calculate/format values to reduce token noise
        row = [
            key,
            f"{stoch_shaped_stats[key]}",
            f"{det_base_stats[key]}"
        ]
        table.append("| " + " | ".join(row) + " |")

    return "\n".join(table)

def training_telemetry_as_table(stats_list: list[dict]) -> str:
    """
    Converts the list of performance dicts into a Markdown table for better LLM reasoning.
    """
    if not stats_list:
        return "No telemetry data available."

     
    # Define the columns we want to compare
    headers = [
        "metric", "start","median","mean","end"
    ]
    
    # Create the header row
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]

    key_list = [
        "policy_gradient_loss",
        "approx_kl",
        "loss",
        "explained_variance",
        "clip_range",
        "entropy_loss",
        "value_loss",
        "clip_fraction",
        "learning_rate"
        ]
    for key in key_list:
        # Pre-calculate/format values to reduce token noise
        row = [
            key,
            f"{stats_list[key]["start"]}",
            f"{stats_list[key]["median"]}",
            f"{stats_list[key]["mean"]}",
            f"{stats_list[key]["end"]}"
        ]
        table.append("| " + " | ".join(row) + " |")

    return "\n".join(table)