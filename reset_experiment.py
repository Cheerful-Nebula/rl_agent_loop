import os
import shutil
import sys

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
DIRS_TO_CLEAR = [
    "logs/code_history",
    "logs/metrics_history",
    "logs/reasoning_history",
    "logs/videos",
    "logs/sb3_log_history"
]

FILES_TO_DELETE = [
    "metrics.json",
    "state.json",       # Critical: Resets the loop counter to 1
    "progress_report.png"
]

# The "Clean Slate" source code
BASELINE_SOURCE = "initial_reward_shaping.py"
# The active file we overwrite
TARGET_FILE = "reward_shaping.py"

# ---------------------------------------------------------
# LOGIC
# ---------------------------------------------------------
def clean_directories():
    print("🧹 Cleaning log directories...")
    for folder in DIRS_TO_CLEAR:
        if os.path.exists(folder):
            # We don't delete the folder itself, just the contents
            # This keeps the folder structure intact for the next run
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"   ⚠️ Failed to delete {file_path}. Reason: {e}")
            print(f"   ✅ Cleared contents of {folder}")
        else:
            # If it doesn't exist, create it so it's ready
            os.makedirs(folder, exist_ok=True)
            print(f"   ✨ Created empty folder {folder}")

def delete_bridge_files():
    print("bridge files...")
    for file in FILES_TO_DELETE:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"   ❌ Deleted {file}")
            except Exception as e:
                print(f"   ⚠️ Could not delete {file}: {e}")
        else:
            print(f"   (File {file} not found, skipping)")

def reset_reward_shaping():
    print("🧬 Restoring baseline DNA...")
    
    # Safety Check: Does the baseline exist?
    if not os.path.exists(BASELINE_SOURCE):
        print(f"   🛑 ERROR: Source file '{BASELINE_SOURCE}' not found!")
        print("   Please create this file with your base reward logic first.")
        return

    try:
        shutil.copy(BASELINE_SOURCE, TARGET_FILE)
        print(f"   ✅ Overwrote {TARGET_FILE} with {BASELINE_SOURCE}")
    except Exception as e:
        print(f"   ⚠️ Copy failed: {e}")

def main():
    # 1. Safety Confirmation
    print("\n" + "="*50)
    print("      ⚠️  WARNING: EXPERIMENT RESET  ⚠️")
    print("="*50)
    print("This will PERMANENTLY DELETE all logs, videos, and history.")
    print(f"It will also overwrite '{TARGET_FILE}'.")
    
    confirm = input("\nAre you sure you want to proceed? (yes/no): ").strip().lower()
    
    if confirm == 'yes' or confirm == 'y':
        clean_directories()
        delete_bridge_files()
        reset_reward_shaping()
        print("\n" + "="*50)
        print("   🚀  RESET COMPLETE. READY FOR NEW RUN.  🚀")
        print("="*50 + "\n")
    else:
        print("\n❌ Reset cancelled.")

if __name__ == "__main__":
    main()