# Evaluation Task: Single Agent Diagnostic Report

- You are provided with telemetry for a PPO agent on `LunarLander-v3`. 
- The agent was evaluated 4 times: Based Reward, Deterministic Behavior (i.e. greedy-epsilon)/Base Reward, Stochastic Behavior/ Shaped Reward, Deterministic Behavior/ Shaped Reward, Stochastic Behavior
- The agent was evaluate 4 times with the hopes of having a better lens for deciphering better effectiveness of how well the agent can learn the reward signals of the current reward function but also how well that translates to performing the underlying desired task.
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
- Be outspoken about any missing information which is hindering your analysis, you must then give reccomendations on what you need provided to you next time for a complete analysis

## The 4 Diagnostic Views
These four configurations form a diagnostic matrix that isolates the effects of reward design and policy entropy. Analyze them comparatively to determine what aspects of the agent’s performance arise from shaping, from stochasticity, or from genuine learning. Use differences across configurations to infer failure points, robustness issues, and concrete improvements to PPO training.

## Data Provided
{performance_table}
```json
{configuration_json}
```
```python
{current_code}
```
## What You Must Produce

As the head RL researcher, you must evaluate the information given to you to assess the agent's performance of performing the desired task and efficacy of learning the reward signals. You are interpreting the results outloud, providing an analysis on how to overcome the shortcomings, lastly you will return the `Reward Function Refinement Plan` which will be passed directly to our top python developer. Have your plan for the python developer be the very last part of your output

### Reward Function Refinement Plan

- In your Future Work section Generate 3 different hypothesis on behavior that may occur from the new reward function and how the outcome of your hypothesis will inform your next `Reward Function Refinement Plan`. Assign a 'Confidence Score' to each based on how robust you think it will be. Include at least one 'experimental' idea with lower confidence scores but higher potential novelty.


### Your Output Format

Your instructions and plan for the python developer must be last section of your output

