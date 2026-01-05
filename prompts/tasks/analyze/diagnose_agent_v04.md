# Evaluation Task: Single Agent Diagnostic Report

- You are provided with telemetry for a PPO agent on `LunarLander-v3`. 
- The agent was evaluated with: Based Reward, Deterministic Behavior/Base Reward, Stochastic Behavior/ Shaped Reward, Deterministic Behavior/ Shaped Reward, Stochastic Behavior
- The agent was evaluate 4 times with the hopes of having a better lens for deciphering better effectiveness of how well the agent can learn the reward signals of the current reward function but also how well that translates to performing the underlying desired task.
- A controlled descent with minimal tilt in the center (0,0)

## The 4 Diagnostic Views
These four configurations form a diagnostic matrix that isolates the effects of reward design and policy entropy. Analyze them comparatively to determine what aspects of the agent’s performance arise from shaping, from stochasticity, or from genuine learning. Use differences across configurations to infer failure points, robustness issues, and concrete improvements to PPO training.

## Data Provided
```json
{configuration_json}
```

## What You Must Produce

As the head RL researcher, you must evaluate the information given to you to assess the agent's performance of performing the desired task and efficacy of learning the reward signals. You are interpreting the results outloud, providing an analysis on how to overcome the shortcomings, lastly you will return the `Reward Function Refinement Plan` which will be passed directly to our top python developer. Have your plan for the python developer be the very last part of your output and be in JSON

### Reward Function Refinement Plan

- In your Futre Work section Generate 3 different hypothesis on behavior that may occur from the new reward function and how the outcome of your hypothesis will inform your next `Reward Function Refinement Plan`. Assign a 'Confidence Score' to each based on how robust you think it will be. Include at least one 'experimental' idea with lower confidence scores but higher potential novelty.
