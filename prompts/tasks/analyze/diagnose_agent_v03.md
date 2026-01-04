# Evaluation Task: Single Agent Diagnostic Report

You are provided with telemetry for **ONE** agent evaluated under 4 distinct conditions. Your goal is to improve the **General Reward Function** so the agent succeeds in the "Deployment" condition.

## The 4 Diagnostic Views
1. **Deployment (Deterministic_BaseReward):** The Reality Check. This is the only metric that matters for success. If this fails, the agent fails.
2. **Exploration (Stochastic_BaseReward):** Checks if the agent can succeed when allowed to rely on luck/noise. If this is high but Deployment is low, the agent is relying on random chance.
3. **Training Signal (Stochastic_ShapedReward):** What the agent actually feels during training. If this is high but Deployment is low, you are "Reward Hacking" (teaching the wrong objective).
4. **Signal Stability (Deterministic_ShapedReward):** Checks if your shaping terms are noisy.

## Data Provided
[JSON Data Inserted Here]

## Analysis Constraints
1. **Unified Policy Assumption:** You must acknowledge that "Config A" and "Config B" are the SAME agent. Do not say "Agent A did X while Agent B did Y." Say "The agent does X under condition A."
2. **No Conditional Logic:** You are strictly forbidden from suggesting reward functions that check for the configuration name. The reward function must be universal.
3. **Outcome Focus:** Your analysis must focus on why the **Training Signal** (Shaped) is failing to produce a good **Deployment** (Base) result.

## What You Must Produce

### 1. The Sim-to-Real Gap
Compare `Stochastic_ShapedReward` (what it learns) vs `Deterministic_BaseReward` (how it performs).
- Is the agent reward hacking? (High Shaped, Low Base)
- Is the agent relying on noise? (High Stochastic, Low Deterministic)

### 2. Failure Mode Analysis
Based on the **Deployment** telemetry (Deterministic_BaseReward), why is it crashing?
- Velocity too high?
- Tilt instability?
- Hovering without landing?

### 3. Reward Function Refinement Plan
Propose specific mathematical changes to the reward function to fix the Deployment behavior.
- If it crashes due to velocity: Increase velocity penalties.
- If it hovers: Increase position incentives or landing bonuses.
- **CRITICAL:** Do not propose `if/else` logic based on config names.

### 4. PPO Tuning
Suggest hyperparameters (Ent Coefficient, Learning Rate) to bridge the gap between Stochastic and Deterministic performance.