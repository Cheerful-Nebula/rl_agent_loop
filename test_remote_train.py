# test_train.py (On Linux)
import time
import json
import sys

# simulate args
print("🚀 [Mock Train] Starting mock training...")
time.sleep(2) # Pretend to load libraries

print("⚙️ [Mock Train] 'Training' for 5 seconds...")
time.sleep(5) # Pretend to train

# Create dummy metrics
metrics = {
    "iteration": 1,
    "mean_reward": 10.5,
    "status": "success"
}

# Save to file (mimics your real training)
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print("✅ [Mock Train] Metrics saved. Exiting.")