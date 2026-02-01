
# Evaluation Task: Single Agent Diagnostic Report

- You are provided with telemetry for a PPO agent on `LunarLander-v3`. 
- The agent was evaluated 4 times: Based Reward, Deterministic Behavior (i.e. greedy-epsilon)/Base Reward, Stochastic Behavior/ Shaped Reward, Deterministic Behavior/ Shaped Reward, Stochastic Behavior
- The agent was evaluated 4 times with the hopes of having a better lens for deciphering better effectiveness of how well the agent can learn the reward signals of the current reward function but also how well that translates to performing the underlying desired task.
- A controlled descent with minimal tilt in the center (0,0)
- `reward_success_rate` is number of episodes with reward >= 200 divided by total episodes
- `pos_success_rate` is number of episodes that land within a radius of 0.1 of the center divided by total episodes
- `crash_rate` is number of episodes that the lander crashed divided by total number of episodes, where "crash' is assumed when reward is <= -100
- Be sure to analyze the following diagnostic metrics:
    - `avg_x_position`: Average horizontal position over the episode (closer to 0 is better)
    - `avg_descent_velocity`: Average vertical velocity over the episode (closer to 0 is better)
    - `avg_tilt_angle`: Average tilt angle over the episode (closer to 0 is better)
    - `vertical_stability_index`: Standard deviation of vertical position over the episode (lower is better)
    - `horizontal_stability_index`: Standard deviation of horizontal position over the episode (lower is better)

## The 4 Diagnostic Views
These four configurations form a diagnostic matrix that isolates the effects of reward design and policy entropy. Analyze them comparatively to determine what aspects of the agent’s performance arise from shaping, from stochasticity, or from genuine learning. Use differences across configurations to infer failure points, robustness issues, and concrete improvements to PPO training.

## Data Provided
### Agent's Performance Summary: Contrast Deterministic vs. Stochastic Policy crossed with Base vs. Shaped Reward
* Sparse Reward Problem : Agent only receives a signal at terminal state, +100 for Landing or -100 for Crashing

### Performance Table
```json
{configuration_json}

```

{performance_table}

{training_table}

```python
{current_code}

```
# Evaluation Task: Single Agent Diagnostic Report

## What You Must Produce

As the Lead RL Researcher, your goal is to analyze the physics of the failure and derive a mathematical solution.

**Part 1: The Diagnosis (Free Thinking)**
Evaluate the efficacy of the current reward signals. Interpret the results out loud. Identify why the agent is failing (e.g., "The tilt penalty is too weak relative to the velocity reward").

**Part 2: The Directives (The Handoff)**
You must conclude your report by providing a strict mathematical specification.

### 1. The Mathematical Blueprint (The Formula)
* **Constraint:** Do NOT write Python code. Use LaTeX or standard mathematical notation.
* **Constraint:** You must define the Total Reward Function $R_{{total}}$.
* **Variables:** You may ONLY use the following physics variables in your formula:
    * $x, y$ : Horizontal/Vertical Position (Target: 0,0)
    * $v_x, v_y$ : Horizontal/Vertical Velocity
    * $\theta$ : Angle (Radians, 0 is upright)
    * $\omega$ : Angular Velocity
    * $L_1, L_2$ : Leg Contact (Boolean 0 or 1)

* **Example Output:**
    $$R_{{total}} = -2.0 \cdot \sqrt{{x^2 + y^2}} - 0.5 \cdot |v_y| + 100 \cdot (L_1 \land L_2)$$

### 2. The Training Configuration (For the Experiment Manager)
* **Focus:** Changes to the PPO Algorithm or Environment settings (Learning Rate, Timesteps, etc.).
* **Exclusion:** Do NOT include reward weights here.

### 3. The Immutable Lesson
* A single, high-level principle derived from this specific iteration.