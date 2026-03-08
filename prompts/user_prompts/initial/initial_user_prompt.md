You are designing the **baseline reward shaping function (Iteration 0)** for a PPO agent in `LunarLander-v3`.

### Environment Context
The agent controls a lunar lander attempting to safely land at the origin (0,0). Action Space is Discrete(4): 0 = do nothing, 1 = fire left engine, 2 = fire main engine, 3 = fire right engine.

A successful landing approaches the landing zone gradually, minimizes horizontal drift, descends slowly near the ground, remains upright, and makes contact with both legs.

### Task
Write **only Python code** that implements a physics-informed reward shaping function for early training.
* Prefer simple, smooth, continuous heuristics.
* Do not try to fully solve the task in one reward; just gently bias learning toward stability and approach.
* Penalize excessive linear/angular velocity and encourage upright orientation.

### Output Constraints
* **Output Python code only** inside a markdown block. No conversational text.
* **Return a Tuple:** You MUST return a `Tuple[float, dict]`. 
* **Component Tracking:** Break down your total reward into granular pieces (e.g., distance penalty, velocity penalty) and return them in the dictionary. 
* **Temporal Context:** You have access to `info["current_step"]` if you wish to apply time-based scaling or penalties.