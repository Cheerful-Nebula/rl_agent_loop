# Evaluation Task: Multi-Configuration PPO Agent Diagnosis

You will receive telemetry describing N different evaluation configurations of the same PPO agent.  
Each configuration contains reward statistics, outcome patterns, stability indices, and rollout analytics.

Your job is to:

1. Compare all configurations.
2. Infer behavioral differences caused by reward shaping, policy randomness, or other configuration settings.
3. Detect PPO-specific training dynamics issues such as entropy collapse, KL oscillation, or reward-hacking.
4. Identify *configuration-specific* and *global* failure modes.
5. Recommend improvements to reward function shaping and PPO hyperparameters.
6. Provide a prioritized action plan.

---

## Data Provided

Below is a JSON list named `configurations`, where each element has fields:

- `"config_id"`: A label such as `"base_stoch"` or `"shaped_det"` or anything else.
- `"meta"`: Any metadata describing the configuration (reward shaping boolean, deterministic flag, etc.).
- `"metrics"`: Reward and success statistics.
- `"diagnostics"`: Stability, tilt, velocity, etc.
- `"training_summary"`: Entropy, KL, losses, etc.

### Configurations (JSON)

```json
{configuration_json}
```
### Training Dynamics

### Memory
**Long-Term Lessons:**
{long_term_memory}

**Short-Term History:**
{short_term_history}

## What You Must Produce

### 1. **Cross-Configuration Comparison**
Identify patterns of improvement or degradation among configurations.  
Discuss differences in:
- reward stability  
- crash behavior  
- horizontal/vertical stability  
- tilt/descent drift  
- outcome distributions  

### 2. **Diagnostic Interpretation**
Explain what behaviors are implied by the telemetry.  
Call out instability, oscillatory control, reward-hacking, edge-case failures, or reward-shaping side-effects.

### 3. **Reward Function Improvements**
Recommend changes such as:
- new shaping terms  
- removal of harmful terms  
- coefficient adjustments  
- penalties/rewards that should be stronger or weaker  

Be precise—your recommendations must map directly to code changes in the reward function.

### 4. **PPO Hyperparameter Adjustments**
Give targeted suggestions about:
- entropy_bonus  
- clip_range  
- learning_rate  
- GAE lambda  
- value loss coefficient  
- number of minibatches  
- batch size stability  

Tie suggestions directly to specific telemetry patterns.

### 5. **Prioritized Action List**
Provide a sorted list of the top interventions with the highest expected benefit.

---

Use only the data provided. Avoid unrelated speculation.  
Produce a concise but technically grounded analysis.