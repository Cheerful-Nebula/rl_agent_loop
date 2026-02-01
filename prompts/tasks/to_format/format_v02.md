Convert the following raw RL researcher analysis into structured JSON format with keys: analysis, plan, lesson, hyperparameters.

**Maintain Knowledge Integrity**
**Do Not Simplify**
- There is vital information which must be preserved
- Downstream tasks rely on the depth and nuance of the approach described in the RL Researcher's Reward Refinement Plan
- Your task is to structure into JSON format with keys: analysis, plan, lesson, hyperparameters 

### Definition of Keys
1. **analysis**: The technical assessment of the PPO agent's performance and training dynamics. Capture the "Why".
2. **plan**: Instructions for the **Python Coder**. This MUST include:
   - Logic changes to the reward function.
   - **Reward Weights/Coefficients** (e.g., `k_height`, `tilt_penalty_weight`).
   - Mathematical formulas.
3. **lesson**: The high-level insight (good or bad) that should be committed to long-term memory.
4. **hyperparameters**: Configuration changes for the **PPO Algorithm** only.
   - **Include:** Learning Rate, Gamma, Batch Size, Entropy Coefficient, Timesteps.
   - **EXCLUDE:** Reward weights (put those in `plan`).

Raw researcher output:
{raw_plan}

Return only the JSON object.
