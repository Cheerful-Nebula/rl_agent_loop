import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import glob
from src.config import Config

import pandas as pd
import matplotlib.pyplot as plt

from src.workspace_manager import ExperimentWorkspace

ws = ExperimentWorkspace()  # uses env vars for base path


def plot_training_iteration(iteration: int, progress_csv_path: str) -> None:
    """
    Creates a 2-row figure:
      - Top: metrics/true_episode_score + train/entropy_loss
      - Bottom: train/value_loss, train/approx_kl, train/explained_variance
    Saves to artifacts/plots/iterXX_train_curves.png
    """
    df = pd.read_csv(progress_csv_path)

    x = df["time/total_timesteps"]
    reward = df["metrics/true_episode_score"]
    entropy = df["train/entropy_loss"]
    value_loss = df["train/value_loss"]
    approx_kl = df["train/approx_kl"]
    explained_var = df["train/explained_variance"]

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 8),
        sharex=True,
        constrained_layout=True,
    )

    # Panel A: reward + entropy
    color_reward = "tab:blue"
    color_entropy = "tab:orange"

    ax1.plot(x, reward, color=color_reward, label="true_episode_score")
    ax1.set_ylabel("Reward", color=color_reward)
    ax1.tick_params(axis="y", labelcolor=color_reward)

    ax1b = ax1.twinx()
    ax1b.plot(x, entropy, color=color_entropy, label="entropy_loss")
    ax1b.set_ylabel("Entropy loss", color=color_entropy)
    ax1b.tick_params(axis="y", labelcolor=color_entropy)

    ax1.set_title(f"PPO Training Curves – Iteration {iteration}")

    # Panel B: value_loss, approx_kl, explained_variance
    ax2.plot(x, value_loss, label="value_loss")
    ax2.plot(x, approx_kl, label="approx_kl")
    ax2.plot(x, explained_var, label="explained_variance")
    ax2.set_xlabel("time / total_timesteps")
    ax2.set_ylabel("Value / KL / Explained variance")
    ax2.legend(loc="best")

    out_path = ws.get_path("plots", iteration, "train_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_eval_iteration(iteration: int, final_eval_csv_path: str) -> None:
    """
    For a single iteration:
      - Filter final_eval.csv to a single policy_behavior (e.g., 'Stochastic')
      - Compare Base vs Shaped on:
          mean_reward, position_success_rate, mean_ep_length
    Saves to artifacts/plots/iterXX_eval_summary.png
    """
    df = pd.read_csv(final_eval_csv_path)

    # pick your primary behavior here
    behavior = "Stochastic"
    df_it = df[
        (df["Iteration"] == iteration)
        & (df["policy_behavior"] == behavior)
    ]

    # Expect rows for Base and Shaped
    df_it = df_it[df_it["reward_shape"].isin(["Base", "Shaped"])]

    shapes = df_it["reward_shape"].tolist()
    mean_reward = df_it["mean_reward"].tolist()
    pos_success = df_it["position_success_rate"].tolist()
    mean_ep_len = df_it["mean_ep_length"].tolist()

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(12, 4),
        constrained_layout=True,
    )

    # Helper for small bar charts
    x_pos = range(len(shapes))

    # Subplot 1: mean_reward
    ax = axes[0]
    ax.bar(x_pos, mean_reward, color=["tab:gray", "tab:green"])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(shapes, rotation=30)
    ax.set_ylabel("mean_reward")
    ax.set_title(f"Iter {iteration} – mean_reward")

    # Subplot 2: position_success_rate
    ax = axes[1]
    ax.bar(x_pos, pos_success, color=["tab:gray", "tab:blue"])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(shapes, rotation=30)
    ax.set_ylabel("position_success_rate")
    ax.set_title("Position success")

    # Subplot 3: mean_ep_length
    ax = axes[2]
    ax.bar(x_pos, mean_ep_len, color=["tab:gray", "tab:orange"])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(shapes, rotation=30)
    ax.set_ylabel("mean_ep_length")
    ax.set_title("Episode length")

    fig.suptitle(
        f"End-of-Iteration Evaluation – Iter {iteration} ({behavior})",
        fontsize=12,
    )

    out_path = ws.get_path("plots", iteration, "eval_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
