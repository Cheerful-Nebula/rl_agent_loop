import argparse
import os
import sys
from src.workspace_manager import ExperimentWorkspace
from src.remote_ops import RemoteTrainer
from src.config import Config

# --- NETWORK CONFIGURATION  ---
LINUX_IP = Config.LINUX_IP     
LINUX_USER = Config.LINUX_USER
SSH_KEY_PATH = Config.SSH_KEY_PATH
REMOTE_PROJECT_ROOT = Config.REMOTE_PROJECT_ROOT
REMOTE_PYTHON_BIN = Config.REMOTE_PYTHON_BIN 

def run_remote_training_cycle(iteration):
    # 1. Initialize Local (Mac) Workspace
    ws = ExperimentWorkspace()
    print(f"🔗 [Proxy] Initializing Remote Link for Iteration {iteration}")

    trainer = RemoteTrainer(LINUX_IP, LINUX_USER, SSH_KEY_PATH, REMOTE_PROJECT_ROOT)
    
    try:
        trainer.connect()

        # ---------------------------------------------------------
        # STEP 1: PUSH ARTIFACTS (The Reward Function)
        # ---------------------------------------------------------
        # We upload the generated code from the PREVIOUS iteration
        prev_iter = iteration - 1
        
        local_reward_path = ws.get_path("code", prev_iter, "reward.py")
        rel_reward_path = ws.get_relative_path("code", prev_iter, "reward.py")
        
        if os.path.exists(local_reward_path):
            print(f"📤 [Proxy] Sending Reward Function (Iter {prev_iter})...")
            trainer.sync_file(local_reward_path, rel_reward_path)
        else:
            print(f"⚠️ [Proxy] Warning: Reward file not found at {local_reward_path}")

        # ---------------------------------------------------------
        # STEP 2: TRIGGER REMOTE EXECUTION
        # ---------------------------------------------------------
        print(f"🚀 [Proxy] Triggering 'train.py' on Linux GPU...")
        success = trainer.run_training_job(ws, iteration, REMOTE_PYTHON_BIN)
        
        if not success:
            print("❌ [Proxy] Remote Training Failed.")
            sys.exit(1)

        # ---------------------------------------------------------
        # STEP 3: PULL ARTIFACTS (The Telemetry)
        # ---------------------------------------------------------
        print(f"📥 [Proxy] Retrieving Metrics (Iter {iteration})...")
        
        rel_metrics_path = ws.get_relative_path("telemetry_raw", iteration, "metrics.json")
        local_metrics_path = ws.get_path("telemetry_raw", iteration, "metrics.json")
        
        if trainer.retrieve_file(rel_metrics_path, local_metrics_path):
            print("✅ [Proxy] Cycle Complete. Data synced to Mac.")
        else:
            print("❌ [Proxy] Failed to download metrics.")
            sys.exit(1)

    except Exception as e:
        print(f"❌ [Proxy] Error: {e}")
        sys.exit(1)
    finally:
        trainer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()
    
    run_remote_training_cycle(args.iteration)