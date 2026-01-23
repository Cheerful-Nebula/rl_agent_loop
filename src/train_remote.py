import argparse
import os
import sys
from src.remote_ops import RemoteManager
from src.config import Config
from src.workspace_manager import ExperimentWorkspace

def run_remote_cycle(iteration):
    # 1. Initialize Local Workspace (Mac side)
    # We need this to find the generated reward file to upload
    ws = ExperimentWorkspace()
    print(f"📡 [Remote-Manager] Initializing Remote Training for Iteration {iteration}")

    # 2. Setup Manager
    manager = RemoteManager(
        Config.LINUX_IP, 
        Config.LINUX_USER, 
        Config.SSH_KEY_PATH, 
        Config.REMOTE_PROJECT_ROOT
    )

    # 3. UPLOAD: Send the generated reward code to Linux
    # Source: experiments/Campaign/Model/generated_code/iterXX_reward.py (Mac)
    local_reward_path = ws.get_path("code", iteration - 1, "reward.py")
    
    # Dest: experiments/Campaign/Model/generated_code/iterXX_reward.py (Linux)
    # We use get_relative_path so we don't accidentally send Mac absolute paths (/Users/...) to Linux
    relative_path = ws.get_relative_path("code", iteration - 1, "reward.py")
    
    print(f"📤 Uploading Reward Function: {local_reward_path}")
    manager.sync_file(str(local_reward_path), str(relative_path))

    # 4. EXECUTE: Trigger Training on Linux
    print(f"🚀 Triggering Remote Training (Iter {iteration})...")
    
    # We capture the Campaign Context from the Mac's shell (set by outer_loop.sh)
    # and package it into a dictionary to inject into the Linux session.
    env_vars = {
        "CAMPAIGN_TAG": os.environ.get("CAMPAIGN_TAG", "Debug_Campaign"),
        "LLM_MODEL": os.environ.get("LLM_MODEL", "Debug_Model"),
        "TOTAL_TIMESTEPS": os.environ.get("TOTAL_TIMESTEPS", "50000")
    }

    # The command is simple because the Manager handles the messy SSH parts.
    # We use -u to force unbuffered output so the stream is real-time.
    cmd = f"{Config.REMOTE_PYTHON_BIN} -u train.py --iteration {iteration}"
    
    # Use our new 'stream_command' to watch the progress on the Mac terminal
    success = manager.stream_command(cmd, env_vars=env_vars)
    
    if not success:
        print("❌ Critical Failure: Remote training crashed.")
        sys.exit(1)

    # 5. DOWNLOAD: Retrieve the Metrics Payload
    # Linux saved it to: experiments/Campaign/Model/telemetry/raw/iterXX_metrics.json
    metrics_rel_path = ws.get_relative_path("telemetry_raw", iteration, "metrics.json")
    local_metrics_dest = ws.get_path("telemetry_raw", iteration, "metrics.json")
    
    print(f"📥 Downloading Metrics: {metrics_rel_path}")
    
    # Ensure the local folder exists before downloading
    local_metrics_dest.parent.mkdir(parents=True, exist_ok=True)
    
    success = manager.retrieve_file(str(metrics_rel_path), str(local_metrics_dest))
    
    if not success:
        print("❌ Failed to retrieve metrics. Controller will crash.")
        sys.exit(1)

    print(f"✅ Remote Cycle {iteration} Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()
    
    run_remote_cycle(args.iteration)