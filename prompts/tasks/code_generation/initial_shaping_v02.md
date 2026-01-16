# Evaluation Task: Initial Reward Shaping (Iteration 1)

You are designing the **baseline reward function** for a PPO agent learning to land on the Moon.
This is the **"Cold Start"** phase. The agent knows nothing. Your code will provide the initial signal to guide it toward the landing pad.

### The "Dual View" Architecture
To prevent confusion between Neural Network inputs (Normalized) and Physics inputs (Raw), the system uses a split view:
1.  **The Agent sees:** Normalized `observation` vector (for the Neural Network).
2.  **You (The Teacher) see:** Raw `info["raw_physics"]` dictionary (for the Reward Function).

**CRITICAL RULE:**
You must calculate the reward using **ONLY** the `info["raw_physics"]` data.
**IGNORE** the `observation` list values, as they are normalized to Viewport coordinates and will break your physics calculations.

### Available Data (`info["raw_physics"]`)
The `info` dictionary contains the raw Box2D measurements from the simulation engine. All units are absolute (Meters, Radians, m/s).

| Key | Description |
| :--- | :--- |
| `x_pos` | Horizontal position. 0.0 is the center (target). |
| `y_pos` | Vertical position. 0.0 is the landing pad (target). |
| `x_vel` | Horizontal velocity (m/s). |
| `y_vel` | Vertical velocity (m/s). Negative is down. |
| `angle` | Tilt angle in Radians. 0.0 is upright. |
| `angle_vel` | Angular velocity (Rotation speed). |
| `leg_left` | 1.0 if touching ground, 0.0 otherwise. |
| `leg_right` | 1.0 if touching ground, 0.0 otherwise. |
| `fuel_consumed_this_step` | Amount of fuel used in the last step (Float). |

### The Goal
The agent must learn to navigate from a random starting point to **(0,0)** and land gently (low velocity) without tipping over.

### Reward Scaling Constraints (CRITICAL)
The environment awards a **Terminal Reward** of **+100** for landing and **-100** for crashing. 
Your shaped reward is added to this signal **every time step**.

* **The Accumulation Trap:** If your function returns large values (e.g., -10.0) every step, the accumulated penalty will reach -5000 by the end of the flight. The agent will learn to **crash intentionally** just to stop the bleeding.
* **Signal-to-Noise Ratio:** Your per-step shaping reward should be small enough that the sum over ~300 steps does not drown out the +100 landing bonus.
* **Recommendation:** Keep your *average* per-step reward roughly between **-1.0 and +1.0**. Rare spikes (e.g., +5.0 for a specific event) are acceptable.

### Task
Write a Python function `calculate_reward(observation, info)` that returns a **float**.
You must use your understanding of physics and Reinforcement Learning to construct a function that biases the agent toward the goal.

**Constraints:**
* **Imports:** You may use `import math` and `import numpy as np` inside the function.
* **Format:** Return **ONLY** the raw Python code. No markdown, no comments outside the code.
* **Signature:**
    ```python
    def calculate_reward(observation, info):
        # ... logic ...
        return reward
    ```