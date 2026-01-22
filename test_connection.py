import os
import sys
import time
# - Custom Imports -
from src.config import Config
from src.remote_ops import RemoteTrainer 
################################################################################################################
# This is your "Systems Check." It verifies SSH, file paths, and the Python environment before you run the loop.
################################################################################################################
# --- NETWORK CONFIGURATION  ---
LINUX_IP = Config.LINUX_IP     
LINUX_USER = Config.LINUX_USER
SSH_KEY_PATH = Config.SSH_KEY_PATH
REMOTE_PROJECT_ROOT = Config.REMOTE_PROJECT_ROOT
REMOTE_PYTHON_BIN = Config.REMOTE_PYTHON_BIN 

def run_diagnostics():
    print(f"🩺 Starting Systems Check for {LINUX_USER}@{LINUX_IP}...\n")
    Config.validate_network_config()
    # 1. DIAGNOSTIC: Check if Python can see the key file
    print(f"🔎 Checking Key File: {Config.SSH_KEY_PATH}")
    if os.path.exists(Config.SSH_KEY_PATH):
        print("   ✅ File FOUND.")
    else:
        print("   ❌ File NOT FOUND.")
        return # Stop here if the key is missing
    # 1. INITIALIZE CLIENT
    try:
        trainer = RemoteTrainer(LINUX_IP, LINUX_USER, SSH_KEY_PATH, REMOTE_PROJECT_ROOT)
    except Exception as e:
        print(f"❌ Failed to initialize RemoteTrainer: {e}")
        return

    # 2. TEST CONNECTION
    print("Test 1: SSH Connection...")
    try:
        trainer.connect()
        print("   ✅ Connected successfully.")
    except Exception as e:
        print(f"   ❌ SSH Failed: {e}")
        print("      (Check IP, Username, or if SSH Key is added to authorized_keys)")
        return
    
    # 3. TEST REMOTE DIRECTORY
    print("\nTest 2: Project Directory Existence...")
    # UPDATED: Use run_command instead of client.exec_command
    code, out, err = trainer.run_command(f"ls -d {REMOTE_PROJECT_ROOT}")
    
    if code == 0:
        print(f"   ✅ Found directory: {REMOTE_PROJECT_ROOT}")
    else:
        print(f"   ❌ Directory NOT found: {REMOTE_PROJECT_ROOT}")
        print(f"      (Please 'git clone' the repo to this path on Linux first)")
        trainer.close()
        return

    # 4. TEST PYTHON ENVIRONMENT
    print("\nTest 3: Python Environment...")
    # FIX: We use a math operation to verify execution. 
    # This avoids "quote stripping" errors over SSH.
    cmd = f"{REMOTE_PYTHON_BIN} -c 'import gymnasium; import torch; print(100 + 200)'"
    
    code, out, err = trainer.run_command(cmd)
    
    # We check if the result "300" is in the output
    if code == 0 and "300" in out:
        print(f"   ✅ Python looks good (Torch/Gym found).")
    else:
        print(f"   ❌ Python Environment Error:")
        print(f"      Command: {cmd}")
        print(f"      Exit Code: {code}")
        print(f"      Error Output: {err}")
        print("      (Check if 'REMOTE_PYTHON_BIN' points to the correct Conda env)")
        
    # 5. TEST FILE TRANSFER (Write/Read)
    print("\nTest 4: Write/Read Permissions...")
    test_filename = "connection_test_artifact.txt"
    local_file = f"src/{test_filename}"
    
    # Create dummy local file
    with open(local_file, "w") as f:
        f.write("Hello from Mac")
        
    try:
        # A. Upload
        rel_path = f"src/{test_filename}"
        print(f"   Subtest A: Uploading {rel_path}...")
        trainer.sync_file(os.path.abspath(local_file), rel_path)
        print("      ✅ Upload complete.")
        
        # B. Verify on Remote (UPDATED)
        check_cmd = f"cat {REMOTE_PROJECT_ROOT}/{rel_path}"
        code, out, err = trainer.run_command(check_cmd)
        
        if out == "Hello from Mac":
            print("      ✅ Content verified (Read successful).")
        else:
            print(f"      ❌ Content mismatch: Got '{out}'")
            
        # Cleanup Remote (UPDATED)
        trainer.run_command(f"rm {REMOTE_PROJECT_ROOT}/{rel_path}")
        
    except Exception as e:
        print(f"   ❌ File Transfer Failed: {e}")
    finally:
        # Cleanup Local
        if os.path.exists(local_file):
            os.remove(local_file)
        print("\n🎉 Diagnostics Complete.")

if __name__ == "__main__":
    run_diagnostics()