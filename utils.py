import os
import shutil
import re
import json
import ollama
from datetime import datetime
import torch
import platform
from config import Config
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
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

def save_metrics(iteration_id ):
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
    
def summarize_training_log(log_dir):
    """
    Reads the SB3 progress.json and returns a condensed dictionary 
    of training dynamics for the LLM.
    """
    log_path = os.path.join(log_dir, "progress.json")
    if not os.path.exists(log_path):
        return {"error": "No training log found."}

    data = []
    try:
        # SB3 JSON logging often writes line-delimited JSON or a standard list
        # We handle the line-delimited case which is common for streaming logs
        with open(log_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except json.JSONDecodeError:
        # Fallback if it was written as a single list object
        try:
            with open(log_path, 'r') as f:
                data = json.load(f)
        except:
            return {"error": "Could not parse training log."}

    if not data:
        return {"error": "Empty training log."}

    # Convert to simple list-of-dicts for easy math
    # We care about: 'train/entropy_loss', 'train/value_loss', 'train/policy_gradient_loss'
    
    first = data[0]
    last = data[-1]
    
    # Calculate Trends
    try:
        entropy_start = first.get('train/entropy_loss', 0)
        entropy_end = last.get('train/entropy_loss', 0)
        entropy_delta = entropy_end - entropy_start # Negative means it decreased (good/normal)
        
        val_loss_avg = sum(d.get('train/value_loss', 0) for d in data) / len(data)
        
        # We format this into a string the LLM can read naturally
        summary = {
            "duration_steps": last.get('time/total_timesteps', 0),
            "entropy_start": f"{entropy_start:.4f}",
            "entropy_end": f"{entropy_end:.4f}",
            "entropy_trend": "Collapsing" if entropy_end < -1.0 else "Stable", # Heuristic
            "value_loss_avg": f"{val_loss_avg:.4f}",
            "policy_loss_end": f"{last.get('train/policy_gradient_loss', 0):.4f}"
        }
        return summary
    except Exception as e:
        return {"error": f"Error summarizing logs: {e}"}

# ---------------------------------------------------------
# Hardware-Aware Model Hyperparameter Tuning
# ---------------------------------------------------------
def get_optimized_ppo_params(n_envs, device_type="auto"):
    """
    Scales PPO parameters to maintain mathematical stability 
    regardless of hardware parallelism (n_envs).
    """
    
    # 1. CONSTANTS (The "Golden Ratio")
    # ---------------------------------
    # Total Buffer: How many frames to collect before updating?
    # 8192 is a "sweet spot" for LunarLander (stable but not too slow).
    TARGET_BUFFER_SIZE = 8192 
    
    # Target Mini-Batches: How many chunks to split the buffer into?
    # 4 chunks = robust updates. 
    TARGET_NUM_MINIBATCHES = 4 

    # 2. CALCULATE DYNAMIC PARAMETERS
    # -------------------------------
    # n_steps: How long does EACH env run?
    # Formula: Target / n_envs
    n_steps = max(int(TARGET_BUFFER_SIZE // n_envs), 16) # Floor at 16 to prevent errors
    
    # Actual Buffer: Recalculate because integer division might be slightly off
    actual_buffer_size = n_steps * n_envs
    
    # batch_size: How big is the chunk sent to the GPU?
    batch_size = int(actual_buffer_size // TARGET_NUM_MINIBATCHES)
    
    # 3. DEVICE SELECTION
    # -------------------
    if device_type == "auto":
        if torch.cuda.is_available():
            device = "cuda"  # NVIDIA (Linux PC)
        elif torch.backends.mps.is_available():
            device = "mps"   # Apple Silicon (Mac)
        else:
            device = "cpu"   # Fallback
    else:
        device = device_type

    print(f"\n--- PPO SCALING REPORT ---")
    print(f"Hardware: {n_envs} Environments | Device: {device}")
    print(f"Scaling : {n_steps} steps/env -> {actual_buffer_size} Total Buffer")
    print(f"Update  : {batch_size} Batch Size ({TARGET_NUM_MINIBATCHES} mini-batches/epoch)")
    print(f"--------------------------\n")

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "device": device
    }

def get_hardware_config():
    system = platform.system()
    if system == "Linux":
        # Based on your benchmark, 32 is the point of stability
        return 32, "cuda" 
    elif system == "Darwin":
        # Based on M4 thermal constraints
        return 8, "mps"
    return 4, "cpu"
# ---------------------------------------------------------
# Callbacks
# ---------------------------------------------------------
# ==========================================
#  The Supervisor Translator (Quartiles)
# ==========================================
class AgenticObservationTracker(BaseCallback):
    """
    Feeds the LLM Supervisor. 
    Translates raw data into statistical distributions (quartiles, mean, variance).
    """
    def __init__(self, obs_indices: list, verbose=0):
        super(AgenticObservationTracker, self).__init__(verbose)
        self.obs_indices = obs_indices
        self.episode_data = {i: [] for i in obs_indices}
        self.outcomes = {"trunc": 0, "term": 0}
        # You could save this to a JSON file later for the Supervisor to read
        self.rollout_stats = [] 

    def _on_step(self) -> bool:
        # Access the current observation (for PPO, usually 'new_obs' or 'obs')
        # We need the observation from the *monitor* wrapper if possible, 
        # but 'new_obs' in locals is the raw input to the policy.
        obs = self.locals['new_obs']
        
        # Log the specific indices we care about
        for i in self.obs_indices:
            # obs is (n_envs, n_obs), take env 0
            self.episode_data[i].append(obs[0, i])
            
        # Track Truncation (Time Limit) vs Termination (Crash/Land)
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            if "TimeLimit.truncated" in info and info["TimeLimit.truncated"]:
                self.outcomes["trunc"] += 1
            else:
                self.outcomes["term"] += 1
        return True

    def _on_rollout_end(self) -> None:
        """Called every n_steps (buffer fill) before update."""
        summary = {}
        for i in self.obs_indices:
            data = np.array(self.episode_data[i])
            if len(data) > 0:
                summary[f"Obs_{i}"] = {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "p25": float(np.percentile(data, 25)),
                    "p75": float(np.percentile(data, 75)),
                }
        
        # Log outcome ratio to TensorBoard for quick visual check
        total = self.outcomes["trunc"] + self.outcomes["term"]
        trunc_rate = self.outcomes["trunc"] / total if total > 0 else 0
        self.logger.record("supervisor/truncation_rate", trunc_rate)
        
        # Save summary to list (or write to JSON here)
        self.rollout_stats.append(summary)
        
        # Reset buffers
        self.episode_data = {i: [] for i in self.obs_indices}
        self.outcomes = {"trunc": 0, "term": 0}

# ==========================================
#  The Performance Evaluator (Scores/Fuel)
# ==========================================
class ComprehensiveEvalCallback(BaseCallback):
    """
    Feeds the Human Engineer (You).
    Tracks 'True' score, fuel usage, and strict landing success.
    """
    def __init__(self, threshold_score=200, verbose=0):
        super(ComprehensiveEvalCallback, self).__init__(verbose)
        self.threshold_score = threshold_score
        self.episode_scores = []
        self.current_episode_fuel = 0.0
        self.success_achieved_early = False

    def _on_step(self) -> bool:
        # 1. Track Estimated Fuel
        action = self.locals['actions'][0]
        if action == 2: self.current_episode_fuel += 0.3
        elif action in [1, 3]: self.current_episode_fuel += 0.03

        # 2. End of Episode Logic
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            
            # Get True Reward from Monitor
            if 'episode' in info:
                true_score = info['episode']['r']
                self.episode_scores.append(true_score)
                self.logger.record("metrics/true_episode_score", true_score)
                
                # Check for Early Success (Rolling Avg)
                if len(self.episode_scores) >= 100:
                    rolling_avg = np.mean(self.episode_scores[-100:])
                    if rolling_avg >= self.threshold_score and not self.success_achieved_early:
                        self.success_achieved_early = True
                        print(f"✨ EARLY SUCCESS: Rolling Avg {rolling_avg:.1f} > {self.threshold_score}")

            # Strict Positional Check (The "Perfect Landing" Metric)
            term_obs = info.get("terminal_observation") # Requires specific wrapper or env config
            # If not available, we can sometimes peek at 'new_obs' if done=True, 
            # but 'terminal_observation' is the standard way in new Gym API.
            if term_obs is not None:
                # 0:x, 4:angle, 6,7:legs
                is_centered = abs(term_obs[0]) < 0.2
                is_upright = abs(term_obs[4]) < 0.1
                legs_down = term_obs[6] > 0.5 and term_obs[7] > 0.5
                strict_success = int(is_centered and is_upright and legs_down)
                self.logger.record("metrics/strict_position_success", strict_success)

            # Log Fuel & Reset
            self.logger.record("metrics/est_fuel_consumed", self.current_episode_fuel)
            self.current_episode_fuel = 0.0
            
        return True