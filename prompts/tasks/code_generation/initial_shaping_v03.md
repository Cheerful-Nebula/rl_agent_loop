You are designing an **initial reward shaping function** for training a **PPO agent** in the OpenAI Gymnasium environment **`LunarLander-v3`**.

### Environment context

The agent controls a lunar lander attempting to **safely land at the origin (0,0)** between two flags.

A successful landing has the following qualitative properties:

* The lander approaches the landing zone gradually
* Horizontal drift is minimized
* Vertical descent is slow and controlled near the ground
* The lander remains upright (small angle, low angular velocity)
* Both landing legs make contact with the ground
* Fuel usage is reasonable but *secondary* to safety and stability

Crashes, hard landings, excessive rotation, and uncontrolled speed are undesirable.

### Observation space

The function receives the standard LunarLander state vector:

```
[x_pos, y_pos, x_vel, y_vel, angle, ang_vel, leg_1_contact, leg_2_contact]
```

### Additional provided information

Additional information provided in `info` dictionary:

* `info["action_usage"]`: Index of the action taken (0-3)
* `info["prev_obs]`: Observation Vector from previous `step()` call

### Task

Write **only Python code** that implements a **physics-informed reward shaping function** for *early training*.

This is **Iteration 1**, meaning:

* Prefer **simple, smooth, continuous heuristics**
* Avoid sparse conditionals or phase-based logic
* Do not try to fully solve the task in one reward
* The goal is to gently bias learning toward stability, approach, and control

The function should:

* Encourage proximity to the landing zone
* Penalize excessive linear and angular velocity
* Encourage upright orientation
* Mildly discourage fuel waste
* Reward stable ground contact when it occurs (legs)

### Constraints

* **Output code only** — no explanations, comments, or markdown, only brief code comments allowed
* Use only the provided `observation` and `info`
* Assume the function signature already exists
* Return a single scalar float reward
* Keep reward magnitudes modest and well-scaled for PPO

This is a **starting point**, not a final solution. Favor clarity, physical intuition, and training stability over cleverness.
