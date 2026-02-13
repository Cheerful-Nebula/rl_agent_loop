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
import numpy as np
from scipy import stats

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

    markdown_str = "\n".join(md_lines)

    fallback = []
    for key , value in data.items():
        fallback.append(str(key))
        fallback.append(str(value))
        fallback.append("")

    return markdown_str if not markdown_str.strip() else "\n".join(fallback) 

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
def summarize_training_log_old(ws: ExperimentWorkspace, iteration: int)-> json:
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
# ==============================================================================
# Testing new summarizing training log functions
# ==============================================================================


def compute_trend_consistency(values: np.ndarray) -> str:
    """Characterize trend pattern without absolute thresholds."""
    if len(values) < 3:
        return "insufficient_data"
    
    diffs = np.diff(values)
    
    # Check for monotonicity
    if np.all(diffs >= 0):
        return "monotonic_increasing"
    elif np.all(diffs <= 0):
        return "monotonic_decreasing"
    
    # Check for high volatility
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    volatility_ratio = sign_changes / len(diffs)
    
    if volatility_ratio > 0.4:
        return "noisy"
    elif volatility_ratio > 0.2:
        return "oscillating"
    else:
        return "mostly_monotonic"


#def summarize_training_log(ws: ExperimentWorkspace, iteration: int):
    """
    Summarize training metrics with relational features for generalization.
    
    Provides:
    1. Tertile temporal splits (first/middle/final third means)
    2. Relational features (ratios, change rates)
    3. Coupling indicators (metric interactions)
    4. Trend characterizations (monotonicity, volatility)
    """
 #   log_path = ws.dirs["telemetry_training"] / f"progress_{iteration:02d}.csv"
 #   df = pd.read_csv(log_path)
def summarize_training_log(df):
    # Calculate tertile split indices
    n = len(df)
    first_third_end = n // 3
    second_third_end = 2 * n // 3
    
    # Define metrics to track
    metrics = [
        'train/policy_gradient_loss',
        'train/approx_kl',
        'train/loss',
        'train/explained_variance',
        'train/entropy_loss',
        'train/value_loss',
        'train/clip_fraction'
    ]
    
    training_summary = {}
    raw_data = {}  # Store for relational computations
    
    # Compute tertile statistics for each metric
    for metric in metrics:
        metric_name = metric.replace('train/', '')
        
        values = df[metric].values
        first_third = values[:first_third_end]
        middle_third = values[first_third_end:second_third_end]
        final_third = values[second_third_end:]
        
        training_summary[metric_name] = {
            "first_third_mean": np.mean(first_third).round(4) if len(first_third) > 0 else 0,
            "middle_third_mean": np.mean(middle_third).round(4) if len(middle_third) > 0 else 0,
            "final_third_mean": np.mean(final_third).round(4) if len(final_third) > 0 else 0,
            "overall_median": np.median(values).round(4),
        }
        
        # Store raw data for relational features
        raw_data[metric_name] = {
            'first': first_third,
            'middle': middle_third,
            'final': final_third,
            'all': values
        }
    
    # === RELATIONAL FEATURES (scale-invariant) ===
    
    def safe_ratio(numerator, denominator, default=1.0):
        """Compute ratio with protection against division by zero."""
        if abs(denominator) < 1e-8:
            return default
        return round(numerator / denominator, 4)
    
    def safe_change_rate(start, end):
        """Compute relative change rate."""
        if abs(start) < 1e-8:
            return 0.0
        return round((end - start) / abs(start), 4)
    
    ev_first = training_summary['explained_variance']['first_third_mean']
    ev_middle = training_summary['explained_variance']['middle_third_mean']
    ev_final = training_summary['explained_variance']['final_third_mean']
    
    vl_first = training_summary['value_loss']['first_third_mean']
    vl_final = training_summary['value_loss']['final_third_mean']
    
    ent_first = abs(training_summary['entropy_loss']['first_third_mean'])
    ent_final = abs(training_summary['entropy_loss']['final_third_mean'])
    
    kl_first = training_summary['approx_kl']['first_third_mean']
    kl_middle = training_summary['approx_kl']['middle_third_mean']
    kl_final = training_summary['approx_kl']['final_third_mean']
    
    cf_mean = training_summary['clip_fraction']['overall_median']
    
    training_summary["_relational"] = {
        # Improvement factors (how much did things change?)
        "ev_improvement_factor": safe_ratio(ev_final, max(ev_first, 1e-4)),
        "value_loss_reduction_factor": safe_ratio(vl_first, max(vl_final, 1)),
        "entropy_decay_rate": safe_change_rate(ent_first, ent_final),
        
        # Learning rate comparisons (which component learned faster?)
        "critic_vs_policy_learning_ratio": safe_ratio(
            safe_ratio(vl_first, max(vl_final, 1)),  # critic improvement
            safe_ratio(ent_first, max(ent_final, 1e-4))  # policy convergence
        ),
        
        # Volatility measures
        "kl_coefficient_of_variation": safe_ratio(
            np.std(raw_data['approx_kl']['all']),
            np.mean(raw_data['approx_kl']['all'])
        ) if len(raw_data['approx_kl']['all']) > 1 else 0,
        
        "ev_coefficient_of_variation": safe_ratio(
            np.std(raw_data['explained_variance']['all']),
            np.mean(raw_data['explained_variance']['all'])
        ) if len(raw_data['explained_variance']['all']) > 1 else 0,
    }
    
    # === COUPLING INDICATORS (metric interactions) ===
    
    training_summary["_coupling"] = {
        # Policy update characteristics
        "updates_conservative": (kl_final < kl_first) and (cf_mean < 0.05),
        "updates_aggressive": (kl_middle > kl_first * 1.5) and (cf_mean > 0.1),
        
        # Reward signal patterns
        "late_phase_ev_breakthrough": (ev_final > ev_middle * 2) and (ev_middle < 0.2),
        "early_phase_ev_plateau": (ev_middle > 0.5) and (abs(ev_final - ev_middle) < 0.1),
        
        # Actor-critic synchronization
        "critic_bottleneck": (ev_final < 0.3) and (kl_final < kl_first),
        "weak_gradients_despite_good_critic": (ev_final > 0.6) and (kl_final < 0.01),
        
        # Exploration patterns
        "premature_convergence": (ent_final / ent_first < 0.5) and (ev_final < 0.5),
        "maintained_exploration": (ent_final / ent_first > 0.8),
    }
    
    # === TREND CHARACTERIZATIONS ===
    
    training_summary["_trends"] = {
        "explained_variance_pattern": compute_trend_consistency(raw_data['explained_variance']['all']),
        "value_loss_pattern": compute_trend_consistency(raw_data['value_loss']['all']),
        "approx_kl_pattern": compute_trend_consistency(raw_data['approx_kl']['all']),
        "entropy_pattern": compute_trend_consistency(raw_data['entropy_loss']['all']),
    }
    
    # === METADATA ===
    """
    training_summary["_metadata"] = {
        "n_updates": int(df['train/n_updates'].iloc[-1]),
        "n_steps": len(df),
        "first_third_steps": first_third_end,
        "middle_third_steps": second_third_end - first_third_end,
        "final_third_steps": n - second_third_end,
        "learning_rate": df['train/learning_rate'].iloc[-1].round(6),
        "clip_range": df['train/clip_range'].iloc[-1].round(4),
    }
    """
    return training_summary
# ==============================================================================
# Testing out new training data formating function that goes with new summarize_training_log()
# ==============================================================================
def format_diagnostician_input(training_summary: dict) -> str:
    """
    Format training summary into structured markdown for diagnostician prompt.
    
    Args:
        training_summary: Output from summarize_training_log()
    
    Returns:
        Formatted markdown string with:
        - Raw metrics table
        - Relational features
        - Coupling indicators
        - Trend patterns
    """
    
    # ========================================================================
    # SECTION 1: Raw Metrics Table
    # ========================================================================
    
    metric_order = [
        'policy_gradient_loss',
        'approx_kl',
        'loss',
        'explained_variance',
        'entropy_loss',
        'value_loss',
        'clip_fraction'
    ]
    
    table_lines = [
        "## Training Metrics Table",
        "",
        "| metric | first_third_mean | middle_third_mean | final_third_mean | overall_median |",
        "|--------|------------------|-------------------|------------------|----------------|"
    ]
    
    for metric in metric_order:
        if metric in training_summary:
            data = training_summary[metric]
            line = (
                f"| {metric} | "
                f"{data['first_third_mean']} | "
                f"{data['middle_third_mean']} | "
                f"{data['final_third_mean']} | "
                f"{data['overall_median']} |"
            )
            table_lines.append(line)
    
    raw_table = "\n".join(table_lines)
    
    # ========================================================================
    # SECTION 2: Relational Features
    # ========================================================================
    
    relational = training_summary.get("_relational", {})
    
    relational_section = [
        "",
        "## Derived Features",
        "",
        "### Relational Features (Scale-Invariant)",
        ""
    ]
    
    relational_items = [
        ("EV improvement factor", relational.get("ev_improvement_factor", "N/A")),
        ("Value loss reduction factor", relational.get("value_loss_reduction_factor", "N/A")),
        ("Entropy decay rate", relational.get("entropy_decay_rate", "N/A")),
        ("Critic vs policy learning ratio", relational.get("critic_vs_policy_learning_ratio", "N/A")),
        ("KL coefficient of variation", relational.get("kl_coefficient_of_variation", "N/A")),
        ("EV coefficient of variation", relational.get("ev_coefficient_of_variation", "N/A"))
    ]
    
    for label, value in relational_items:
        relational_section.append(f"- **{label}**: {value}")
    
    relational_text = "\n".join(relational_section)
    
    # ========================================================================
    # SECTION 3: Coupling Indicators
    # ========================================================================
    
    coupling = training_summary.get("_coupling", {})
    
    coupling_section = [
        "",
        "### Coupling Indicators (Diagnostic Patterns)",
        ""
    ]
    
    coupling_items = [
        ("Updates conservative", coupling.get("updates_conservative", False)),
        ("Updates aggressive", coupling.get("updates_aggressive", False)),
        ("Late phase EV breakthrough", coupling.get("late_phase_ev_breakthrough", False)),
        ("Early phase EV plateau", coupling.get("early_phase_ev_plateau", False)),
        ("Critic bottleneck", coupling.get("critic_bottleneck", False)),
        ("Weak gradients despite good critic", coupling.get("weak_gradients_despite_good_critic", False)),
        ("Premature convergence", coupling.get("premature_convergence", False)),
        ("Maintained exploration", coupling.get("maintained_exploration", False))
    ]
    
    for label, value in coupling_items:
        status = "✓ **TRUE**" if value else "✗ false"
        coupling_section.append(f"- {label}: {status}")
    
    coupling_text = "\n".join(coupling_section)
    
    # ========================================================================
    # SECTION 4: Trend Patterns
    # ========================================================================
    
    trends = training_summary.get("_trends", {})
    
    trends_section = [
        "",
        "### Trend Patterns (Stability Analysis)",
        ""
    ]
    
    trend_items = [
        ("Explained variance", trends.get("explained_variance_pattern", "unknown")),
        ("Value loss", trends.get("value_loss_pattern", "unknown")),
        ("Approx KL", trends.get("approx_kl_pattern", "unknown")),
        ("Entropy", trends.get("entropy_pattern", "unknown"))
    ]
    
    for label, pattern in trend_items:
        trends_section.append(f"- **{label}**: `{pattern}`")
    
    trends_text = "\n".join(trends_section)
    
    # ========================================================================
    # SECTION 5: Metadata (Optional Context)
    # ========================================================================
    
    metadata = training_summary.get("_metadata", {})
    
    metadata_section = [
        "",
        "### Training Context",
        "",
        f"- **Total updates**: {metadata.get('n_updates', 'N/A')}",
        f"- **Total steps logged**: {metadata.get('n_steps', 'N/A')}",
        f"- **Steps per phase**: First={metadata.get('first_third_steps', 'N/A')}, "
        f"Middle={metadata.get('middle_third_steps', 'N/A')}, "
        f"Final={metadata.get('final_third_steps', 'N/A')}",
        ""
    ]
    
    metadata_text = "\n".join(metadata_section)
    
    # ========================================================================
    # COMBINE ALL SECTIONS
    # ========================================================================
    
    full_output = "\n".join([
        raw_table,
        relational_text,
        coupling_text,
        trends_text,
        metadata_text
    ])
    
    return full_output
# ==============================================================================
# ==============================================================================
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
        lr = final_value + (initial_value - final_value) * progress_remaining
        print(f"Progress: {progress_remaining:.3f} → LR: {lr:.6f}")
        return lr

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
        "metric", "start","end","median","mean",
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
        "entropy_loss",
        "value_loss",
        "clip_fraction"
        ]
    for key in key_list:
        # LLMs trained a vast amount of data related to loss curves, this specific 
        # implementation of where stablebaselines3 defined entropy_loss as the negative
        # of entropy confuses LLMs when diagnosing training optimization metrics
        if key == "entropy_loss":
            stats_list['entropy'] = {k: -v for k, v in stats_list["entropy_loss"].items()}
            key = "entropy" # Rename for LLM clarity

        row = [
            key,
            f"{stats_list[key]["start"]}",
            f"{stats_list[key]["end"]}",
            f"{stats_list[key]["median"]}",
            f"{stats_list[key]["mean"]}"
        ]
        table.append("| " + " | ".join(row) + " |")

    return "\n".join(table)