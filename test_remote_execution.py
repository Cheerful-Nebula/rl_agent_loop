from src.remote_ops import RemoteTrainer
from src.config import Config

# Initialize
trainer = RemoteTrainer(Config.LINUX_IP, Config.LINUX_USER, Config.SSH_KEY_PATH, Config.REMOTE_PROJECT_ROOT)

# Run the mock training
print("Testing Remote Execution...")
# Note: We point to the system python, just to be simple for this test
trainer.run_command(f"{Config.REMOTE_PYTHON_BIN} test_remote_train.py")

# Check if the file was created
print("Checking for metrics...")
code, out, err = trainer.run_command("cat metrics.json")
print(f"Linux says: {out}")