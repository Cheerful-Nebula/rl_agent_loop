import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import glob
from src.config import Config

LOG_DIR = "logs/metrics_history"

def plot_training_history():
    files = sorted(glob.glob(os.path.join(LOG_DIR, "metrics_gen*.json")))
    data = []
    
    for f in files:
        with open(f, 'r') as json_file:
            content = json.load(json_file)
            # Parse filename for Generation ID (metrics_gen005.json)
            gen_id = int(f.split("gen")[1].split(".")[0])
            
            row = {
                "generation": gen_id,
                "mean_reward": content["performance"]["mean_reward"],
                "success_rate": content["performance"]["success_rate"],
                "crash_rate": content["performance"]["crash_rate"]
            }
            data.append(row)
            
    if not data:
        print("No data to plot!")
        return

    df = pd.DataFrame(data)
    df = df.set_index("generation")
    
    # Create Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Reward (Left Axis)
    color = 'tab:blue'
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Mean Reward', color=color)
    ax1.plot(df.index, df['mean_reward'], color=color, marker='o', label='Reward')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Plot Success Rate (Right Axis)
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Success Rate', color=color)
    ax2.plot(df.index, df['success_rate'], color=color, marker='x', linestyle='--', label='Success %')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.1, 1.1)

    plt.title(f"Agentic RL: Optimization Progress with {Config.LLM_MODEL} on {Config.ENV_ID}")
    plt.tight_layout()
    plt.savefig("progress_report.png")
    print("Graph saved to progress_report.png")

if __name__ == "__main__":
    plot_training_history()