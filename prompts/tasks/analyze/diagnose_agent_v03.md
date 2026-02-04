# Evaluation Task: Single Agent Diagnostic Report

- You are provided with telemetry for a PPO agent on `LunarLander-v3`. 
- **The agent was evaluated in 2 distinct configurations** to isolate "Training Signal" from "Task Success":
    1. **Stochastic Behavior / Shaped Reward:** (The "Teacher") Represents the agent's behavior during training. If this fails, the reward function provides no learning signal.
    2. **Deterministic Behavior / Base Reward:** (The "Exam") Represents the true objective (Sparse +100/-100). If this fails but the Teacher succeeds, the reward function is misaligned (Reward Hacking).
- **Goal:** A controlled descent with minimal tilt in the center (0,0).

**Success Metrics:**
- `reward_success_rate`: % of episodes with Reward >= 100.
- `pos_success_rate`: % of episodes landing legs in contact with ground and within 0.1 radius of center.
- `crash_rate`: % of episodes ending in a crash (Reward <= -100).

**Diagnostic Metrics:**
- `avg_x_position`: Closer to 0 is better.
- `avg_descent_velocity`: Closer to 0 is better (controlled descent).
- `avg_tilt_angle`: Closer to 0 is better.
- `vertical_stability_index`: Lower std dev is better.

## The 2 Diagnostic Views
Analyze these two views comparatively to determine if the failure is due to **Learning** (Agent can't optimize the signal) or **Alignment** (Agent optimizes the signal, but the signal doesn't lead to landing).

### Performance Table
```json
{configuration_json}
```
{performance_table}

{training_table}

```python
{current_code}
```

## Experiment Logs
{short_term_history}

## What You Must Produce

As the Lead RL Researcher, your goal is to synthesize this data into a cohesive narrative and then issue specific directives to your team.

**Part 1: The Diagnosis (Free Thinking)**
Evaluate the efficacy of the current reward signals. Interpret the results out loud. 
- Does the **Stochastic/Shaped** performance indicate the agent is learning *anything*? 
- Does high reward in **Shaped** translate to success in **Base**?
- Identify the failure mode: "No Signal", "Reward Hacking", or "Optimization Failure".

**Part 2: The Directives (The Handoff)**
You must conclude your report by organizing your findings into three strict categories for downstream implementation. Do not mix them.

### 1. The Coding Plan (For the Python Developer)

* **Focus:** The mathematical structure of the reward function.
* **Action:** Explicitly state the mathematical formula(s) needed in the reward function to properly steer the agent toward an optimal policy.
* **CRITICAL REQUIREMENT:** You **MUST** end this section with a single, standalone line in exactly this format:
  **HYPOTHESIS: [A single sentence predicting the specific metric change, e.g., "Increasing tilt penalty will reduce crash rate by 10%."]**

### 2. The Training Configuration (For the Experiment Manager)

* **Focus:** Changes to the PPO Algorithm or Environment settings.
* **Include:** Learning Rate, Entropy Coefficients, Timesteps, Batch Size, or Clip Range.
* **Exclusion:** Do NOT include reward weights here (put those in the Coding Plan).

### 3. The Immutable Lesson

* A single, high-level principle derived from this specific iteration that should be preserved in long-term memory (e.g., "Velocity penalties without distance rewards lead to hovering behavior").