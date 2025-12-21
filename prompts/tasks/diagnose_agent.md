Analyze the agent's performance on '{env_id}'.

## Metrics
- Success Rate: {success_rate:.4f} (Target: >0.9)
- Crash Rate: {crash_rate:.4f} (Target: <0.1)

## Flight Telemetry
- Avg Vertical Vel: {avg_descent:.4f} (Target: Negative but controlled)
- Avg Tilt Angle: {avg_tilt:.4f} (Target: near 0.0)
- Avg X-Position: {avg_x:.4f} (Target: 0.0 is center. +/- 1.0 is edge)
- X-Position Standard Deviation: {x_std:.4f}
- Y-Position Standard Deviation: {y_std:.4f}

## Engine Telemetry
- Main Engine Usage: {main_eng:.4f} (Target: Balanced. If <0.1, it falls. If >0.8, it panics.)
- Side Engine Usage: {side_eng:.4f} (Target: Low. If >0.5, it is jittering.)

## Internal Training Dynamics (The Agent's Brain)
- **Exploration (Entropy):** Started at {entropy_start} and ended at {entropy_end}. ({entropy_trend})
  *(Note: Rapid drop means premature convergence. High negative values mean certainty.)*
- **Critic Accuracy (Value Loss):** Average loss was {value_loss_avg}.
  *(Note: High loss means the agent cannot predict the outcome of its actions.)*
- **Actor Stability (Policy Loss):** Ended at {policy_loss_end}.

## Memory
**Long-Term Lessons:**
{long_term_memory}

**Short-Term History:**
{short_term_history}

## Current Reward Function
```python
{current_code}
```

## Task
1. DIAGNOSE: Combine the metrics, telemetry, and memory to identify why the agent is underperforming. Answer:
- Why is the agent failing?
- Are we repeating past mistakes?
- Is the reward function misaligned with our goals?
2. PLAN: Propose a new mathematical adjustment to the reward function.

## Output Format
- Provide a clear, concise paragraph explaining the failure and the planned math fix. 
- DO NOT write code yet.