# =========================================================
# FILE: src/evaluation.py
# =========================================================
import numpy as np
from typing import List, Dict, Any

# custom imports
from src import utils

def run_single_eval_pass(env, model, num_episodes, seed_id:int=0, deterministic_flag=True, env_type="Base"):
    """Pure data-harvesting loop. 
    Collects per-step level termporal data:
    Observation Vector, Reward Component Breakdown, Metadata
    """
    temporal_data = [] 
    for i in range(num_episodes):
        obs, _ = env.reset(seed=321+i)
        done = False
        ep_len = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic_flag)
            ep_len += 1
            next_obs, reward, terminated, truncated, info = env.step(action)
            action_val = int(action) if isinstance(action, (np.ndarray, np.generic)) else int(action)

            # --- 1. THE ACTIVE ROW --- Collecting Obs_t with Action_t
            row_active = {
                "seed_id": int(seed_id),
                "reward_type": str(env_type),                                
                "policy_behavior": "Deterministic" if deterministic_flag else "Stochastic",
                "episode": int(i),
                "timestep": int(ep_len),
                "action": action_val,
                "reward": float(reward),
                "x_pos": float(obs[0]),       
                "y_pos": float(obs[1]),
                "x_vel": float(obs[2]),
                "y_vel": float(obs[3]),
                "angle": float(obs[4]),
                "angular_vel": float(obs[5]),
                "leg1_contact": 1 if float(obs[6]) > 0.5 else 0,
                "leg2_contact": 1 if float(obs[7]) > 0.5 else 0,
                "status": "active"
            }
            
            # Merge step-wise reward components to capture the distribution over time
            step_comps = info.get('step_reward_components', {})
            row_active.update({f'step_reward_{k}': float(v) for k, v in step_comps.items()})
            temporal_data.append(row_active)

            # --- 2. THE TERMINAL ROW ---
            if terminated or truncated:
                # Extract observation for BOTH end states
                term_obs = info.get("terminal_observation", obs)

                # Kinematic Stability Checks
                is_centered = abs(term_obs[0]) < 0.2
                is_out_of_bounds = abs(term_obs[0]) >= 0.95 or term_obs[1] >= 0.95
                low_velocity = abs(term_obs[2]) < 0.1 and abs(term_obs[3]) < 0.1 
                
                # Contextual Upright Tolerance
                is_upright = abs(term_obs[4]) < 0.1 if is_centered else abs(term_obs[4]) < 0.38 
                
                # Physics Contact Checks
                legs_down = term_obs[6] >= 0.5 and term_obs[7] >= 0.5
                resting_on_one_leg = (term_obs[6] > 0.5) != (term_obs[7] > 0.5)
                below_pad = term_obs[1] < 0.0
                
                # Routing Logic 
                if terminated:               
                    if is_upright and legs_down:             
                        step_status = "landed_centered" if is_centered else "landed_off_centered"
                    elif is_upright and low_velocity and resting_on_one_leg and below_pad:
                        step_status= "landed_but_slid_into_valley"
                    elif is_out_of_bounds:                       
                        step_status = "out_of_bounds"
                    else:                                        
                        step_status = "crashed"
                
                elif truncated:
                    step_status = "landed_off_centered_timeout" if legs_down else "hover_timeout"


                row_terminal = {
                    "seed_id": int(seed_id),
                    "reward_type": str(env_type),                                
                    "policy_behavior": "Deterministic" if deterministic_flag else "Stochastic",
                    "episode": int(i),
                    "timestep": int(ep_len + 1),
                    "action": -1,  # The numerical terminal flag
                    "reward": 0.0, 
                    "x_pos": float(term_obs[0]),       
                    "y_pos": float(term_obs[1]),
                    "x_vel": float(term_obs[2]),
                    "y_vel": float(term_obs[3]),
                    "angle": float(term_obs[4]),
                    "angular_vel": float(term_obs[5]),
                    "leg1_contact": 1 if float(term_obs[6]) > 0.5 else 0,
                    "leg2_contact": 1 if float(term_obs[7]) > 0.5 else 0,
                    "status": step_status
                }
                
                # Merge the CUMULATIVE episodic reward components into the terminal row
                # Collecting the cumlative sum of each reward component 
                step_comps = info.get('step_reward_components', {})
                row_terminal.update({f'step_reward_{k}': float(v) for k, v in step_comps.items()})
                temporal_data.append(row_terminal)
                
                done = True
                
            obs = next_obs 
                
    return temporal_data

def evaluate_agent(model, seed_id:int = 0, deterministic=True, reward_code_path=None, num_episodes=10) -> List[Dict[str,Any]]:
    """
    Runs Dual Evaluation (Standard and Shaped) and returns raw temporal lists 
    ready to be converted to DataFrames and saved to CSV.
    """
    # --- SHAPED EVALUATION ---
    if reward_code_path:
        print(f"📊 Eval: Shaped Environment, {'Deterministic' if deterministic else 'Stochastic'}")
        shp_env = utils.make_env(reward_code_path)
        shp_temporal_data = run_single_eval_pass(shp_env, model, num_episodes, seed_id, deterministic_flag=deterministic,env_type="Shaped")
        shp_env.close()
        # Return list of data dictionaries to be handled by saving/pandas logic
        return shp_temporal_data
        
    else:
        # --- STANDARD EVALUATION ---
        print(f"📊 Eval: Standard Environment, {'Deterministic' if deterministic else 'Stochastic'}")
        std_env = utils.make_env(reward_code_path)
        std_temporal_data = run_single_eval_pass(std_env, model, num_episodes, seed_id, deterministic_flag=deterministic,env_type="Base")
        std_env.close()
        return std_temporal_data
    


