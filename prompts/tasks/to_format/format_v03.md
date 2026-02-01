Convert the following raw RL researcher analysis into structured JSON format with keys: analysis, plan, lesson, hyperparameters, hypothesis.

**Maintain Knowledge Integrity**
**Do Not Simplify**

### Definition of Keys
1. **analysis**: The technical assessment of the PPO agent's performance. Capture the "Why".
2. **plan**: Instructions for the **Python Coder** (Reward logic, weights, formulas).
4. **hyperparameters**: Configuration changes for PPO (LR, Gamma, Entropy, etc.).
5. **hypothesis**: The specific prediction made in the Coding Plan. 
   - Extract the sentence starting with "Hypothesis:" or "Expected Behavioral Change:".
   - Example: "Increasing tilt penalty will reduce vertical instability."

Raw researcher output:
{raw_plan}

Return only the JSON object.