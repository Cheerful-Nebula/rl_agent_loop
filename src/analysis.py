# Run Command In Terminal to Script so Python can find all modules
# python -m testing_suite.mock_analysis filename

import numpy as np
import json
from collections import defaultdict
from itertools import combinations
import pandas as pd
import numpy as np
import os
import warnings
import sys

#- custom imports
from src.workspace_manager import ExperimentWorkspace
warnings.filterwarnings('ignore')

###########################################################
# For progress.csv
############################################################

def analyze_single_seed_progress(csv_path: str, seed_id: int, window_size: int = 10) -> dict:
    """
    Performs deep dynamical systems analysis on a single PPO training log.
    Designed to run on the Linux compute node.
    Returns a strict dictionary payload for downstream aggregation.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training log not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['train/approx_kl', 'train/value_loss', 'train/policy_gradient_loss']).copy()
    
    if len(df) < window_size:
        return {"seed_id": seed_id, "error": "Insufficient data points for rolling analysis."}

    # 1. Trust Region Integrity
    df['roll_corr_clip_kl'] = df['train/clip_fraction'].rolling(window_size).corr(df['train/approx_kl']).fillna(0)
    
    # 2. Critic Saturation Index
    df['roll_var_rew'] = df['rollout/ep_rew_mean'].rolling(window_size).var()
    df['critic_saturation_index'] = df['train/value_loss'] / (df['roll_var_rew'] + 1e-8)
    
    # 3. Policy Update Efficiency
    df['delta_pg_loss'] = df['train/policy_gradient_loss'].diff()
    df['delta_kl'] = df['train/approx_kl'].diff()
    df['update_efficiency'] = df['delta_pg_loss'] / (df['delta_kl'].abs() + 1e-8)
    
    # 4. Entropy Sacrifice Rate
    df['delta_entropy'] = df['train/entropy_loss'].diff()
    df['entropy_sacrifice_rate'] = df['delta_entropy'] / (df['delta_kl'].abs() + 1e-8)
    
    # Isolate final 25% for converged performance metrics
    final_quartile = df.iloc[-max(1, int(len(df) * 0.25)):]
    max_entropy = np.log(4) # Assuming 4 discrete actions
    final_ent_pct = (-final_quartile['train/entropy_loss'].mean() / max_entropy) * 100

    # Construct strict JSON-serializable payload
    payload = {
        "seed_id": seed_id,
        "timesteps_completed": int(df['time/total_timesteps'].max()),
        "optimization_health": {
            "trust_region": {
                "final_quartile_mean_corr": float(final_quartile['roll_corr_clip_kl'].mean()),
                "is_thrashing_flag": bool(final_quartile['roll_corr_clip_kl'].mean() < 0.3)
            },
            "critic_health": {
                "final_saturation_index": float(final_quartile['critic_saturation_index'].mean()),
                "final_explained_variance": float(final_quartile['train/explained_variance'].mean()),
                "critic_diverged_flag": bool(final_quartile['critic_saturation_index'].mean() > 10.0 and final_quartile['train/explained_variance'].mean() < 0.5)
            },
            "policy_efficiency": {
                "update_efficiency_score": float(final_quartile['update_efficiency'].mean()),
                "is_plateaued_flag": bool(abs(final_quartile['update_efficiency'].mean()) < 0.05)
            },
            "exploration": {
                "final_entropy_retained_pct": float(final_ent_pct),
                "premature_convergence_flag": bool(final_ent_pct < 10.0)
            }
        },
        # Exporting the raw trajectory of the final quartile for the aggregator to compute cross-seed variance
        "trajectories": {
            "critic_saturation_trend": final_quartile['critic_saturation_index'].tolist(),
            "explained_variance_trend": final_quartile['train/explained_variance'].tolist(),
            "reward_trend": final_quartile['rollout/ep_rew_mean'].tolist()
        }
    }
    
    return payload



def aggregate_progress_seeds(seed_payloads: list) -> str:
    """
    Ingests a list of payloads from analyze_single_seed_progress().
    Computes cross-seed robustness, SNR, and trajectory isomorphism.
    Returns the final JSON for the MacBook LLM Diagnostician.
    """
    if not seed_payloads:
        return json.dumps({"error": "No seed payloads provided."})

    num_seeds = len(seed_payloads)
    
    # 1. Extract Final Averages & Arrays
    final_rewards = [np.mean(seed['trajectories']['reward_trend']) for seed in seed_payloads]
    final_csi = [seed['optimization_health']['critic_health']['final_saturation_index'] for seed in seed_payloads]
    
    # Extract the raw trajectories (assumes they are the same length, which they should be for the final quartile)
    reward_trajectories = [seed['trajectories']['reward_trend'] for seed in seed_payloads]
    
    # 2. Compute Cross-Seed Robustness (SNR)
    reward_mean = np.mean(final_rewards)
    reward_std = np.std(final_rewards)
    cross_seed_snr = reward_mean / (reward_std + 1e-5)
    
    # 3. Compute Critic Concordance
    csi_mean = np.mean(final_csi)
    csi_variance = np.var(final_csi)
    all_critics_diverged = all(csi > 10.0 for csi in final_csi)
    
    # 4. Trajectory Isomorphism (Do they learn the same way?)
    # Calculate pairwise correlation between all seed reward trajectories
    correlations = []
    for traj_a, traj_b in combinations(reward_trajectories, 2):
        # Truncate to min length in case of slight timestep mismatches
        min_len = min(len(traj_a), len(traj_b))
        if min_len > 1:
            rho = np.corrcoef(traj_a[:min_len], traj_b[:min_len])[0, 1]
            if not np.isnan(rho):
                correlations.append(rho)
                
    mean_trajectory_corr = np.mean(correlations) if correlations else 0.0

    # 5. Build the Mac-bound LLM Payload
    # We apply the semantic boolean thresholds HERE, at the population level.
    payload = {
        "multi_seed_optimization_health": {
            "population_metrics": {
                "mean_final_reward": round(reward_mean, 2),
                "cross_seed_reward_std": round(reward_std, 2),
                "cross_seed_snr": round(cross_seed_snr, 3)
            },
            "critic_robustness": {
                "mean_critic_saturation_index": round(csi_mean, 2),
                "critic_saturation_variance": round(csi_variance, 2),
                "systemic_critic_divergence_flag": bool(all_critics_diverged)
            },
            "learning_dynamics": {
                "mean_trajectory_isomorphism_rho": round(mean_trajectory_corr, 3),
                "highly_unstable_optimization_landscape_flag": bool(cross_seed_snr < 1.0 and mean_trajectory_corr < 0.3)
            },
            "summary_flags": {
                "is_initialization_sensitive": bool(reward_std > abs(reward_mean) * 0.5), # STD is > 50% of the Mean
                "is_universally_converged": bool(reward_mean > 0 and reward_std < abs(reward_mean) * 0.2)
            }
        }
    }
    
    return json.dumps(payload, indent=2)


def translate_optimization_health(agg_progress_json: dict) -> str:
    """
    Translates the Linux-aggregated progress JSON into a dense, 
    Strategist-ready Markdown diagnostic report.
    """
    opt = agg_progress_json.get("multi_seed_optimization_health", {})
    pop = opt.get("population_metrics", {})
    crit = opt.get("critic_robustness", {})
    dyn = opt.get("learning_dynamics", {})
    flags = opt.get("summary_flags", {})

    # 1. Executive Summary & Convergence Status
    status_header = "🟢 **CONVERGED**" if flags.get("is_universally_converged") else "🔴 **UNSTABLE/FAILED**"
    if flags.get("is_initialization_sensitive"):
        status_header = "🟡 **HIGHLY SENSITIVE TO INITIALIZATION**"

    md = [f"### 1. Optimization Dynamics & Critic Health\n**Status:** {status_header}\n"]

    # 2. Reward Robustness (Scale-Invariant Focus)
    md.append("#### A. Cross-Seed Robustness")
    snr = pop.get('cross_seed_snr', 0)
    md.append(f"- **Signal-to-Noise Ratio (SNR):** `{snr:.2f}`")
    if snr < 1.0:
        md.append("  - *Diagnosis:* The cross-seed variance exceeds the mean reward. The current reward topology is highly chaotic and fails to enforce a universal policy.")
    else:
        md.append("  - *Diagnosis:* Strong cross-seed consistency. The reward landscape is learnable.")

    # 3. Critic Health (Crucial for ARD)
    md.append("\n#### B. Value Network (Critic) Integrity")
    csi = crit.get('mean_critic_saturation_index', 0)
    md.append(f"- **Critic Saturation Index (CSI):** `{csi:.2f}`")
    
    if crit.get('systemic_critic_divergence_flag'):
        md.append("  - *Diagnosis:* **CRITICAL FAILURE.** The Critic has systemically diverged across all seeds (CSI > 10). The value network is entirely saturated by noise.")
        md.append("  - *Action Required:* You must simplify the dense reward terms. The current shaping is introducing severe non-Markovian variance or contradictory gradients.")
    elif csi > 5.0:
        md.append("  - *Diagnosis:* Warning. The Critic is struggling to map the advantage function. Consider reducing the scale of dense penalty components.")
    else:
        md.append("  - *Diagnosis:* Healthy. The Critic is accurately mapping the advantage landscape.")

    # 4. Learning Trajectory Isomorphism
    md.append("\n#### C. Optimization Landscape")
    iso_rho = dyn.get('mean_trajectory_isomorphism_rho', 0)
    md.append(f"- **Trajectory Isomorphism (Pairwise $\\rho$):** `{iso_rho:.3f}`")
    
    if dyn.get('highly_unstable_optimization_landscape_flag'):
        md.append("  - *Diagnosis:* The learning curves are completely uncorrelated across seeds. The reward function has created a jagged landscape with multiple competing local minima.")

    return "\n".join(md)



###########################################################
# For eval.csv
############################################################

def analyze_single_seed_eval(csv_path: str, seed_id: int) -> dict:
    """
    Transforms deterministic evaluation trajectories into rigid-body dynamic metrics.
    Calculates fuel efficiency purely from action distributions, completely independent of reward.
    Returns a strict numerical dictionary for the Linux Aggregator.
    """
    if not os.path.exists(csv_path):
        return {"seed_id": seed_id, "error": f"Eval log not found at {csv_path}"}

    df = pd.read_csv(csv_path)
    if df.empty:
        return {"seed_id": seed_id, "error": "Eval log is empty."}
    
    ep_metrics = []
    
    for ep, ep_df in df.groupby('episode'):
        ep_df = ep_df.sort_values('timestep').reset_index(drop=True)
        ep_len = len(ep_df)
        
        # --- A. Actuator Frequency ---
        chatter_rate = (ep_df['action'].diff() != 0).sum() / max(1, ep_len - 1)
        
        # --- B. Action-Based Fuel Proxy (No Reward Dependency) ---
        noop_count = (ep_df['action'] == 0).sum()
        main_fire_count = (ep_df['action'] == 2).sum()
        side_fire_count = ((ep_df['action'] == 1) | (ep_df['action'] == 3)).sum()
        
        noop_duty = noop_count / max(1, ep_len)
        main_duty = main_fire_count / max(1, ep_len)
        side_duty = side_fire_count / max(1, ep_len)
        
        # --- C. Descent Efficiency (Translation / Fuel Fired) ---
        vert_descent = ep_df['y_pos'].max() - ep_df['y_pos'].min()
        total_fuel_units = main_fire_count + side_fire_count
        efficiency = vert_descent / (total_fuel_units + 1e-5) # Prevent Div/0
        
        # --- D. Attitude Phase-Space Volume ---
        attitude_phase = np.sqrt(ep_df['angle']**2 + ep_df['angular_vel']**2).mean()
        
        # --- E. Lateral Macro-Oscillations (Bounded |x| > 0.2) ---
        ep_df['x_vel_sign'] = np.sign(ep_df['x_vel'].fillna(0))
        sign_changes = ep_df['x_vel_sign'].diff().ne(0) & ep_df['x_vel_sign'].diff().notna()
        macro_oscillations = (sign_changes & (ep_df['x_pos'].abs() > 0.2)).sum()
        
        # Target boolean success metric for correlation
        is_success = 1.0 if 'landed' in str(ep_df['status'].iloc[-1]) else 0.0

        ep_metrics.append({
            'episode': int(ep),
            'status': str(ep_df['status'].iloc[-1]),
            'is_success': is_success,
            'chatter_rate': float(chatter_rate),
            'noop_duty': float(noop_duty),
            'main_duty': float(main_duty),
            'side_duty': float(side_duty),
            'attitude_phase': float(attitude_phase),
            'efficiency': float(efficiency),
            'macro_oscillations': int(macro_oscillations)
        })
        
    metrics_df = pd.DataFrame(ep_metrics)
    
    # Construct the pure data payload for the Aggregator
    payload = {
        "seed_id": seed_id,
        "mean_metrics": {
            "chatter_rate": float(metrics_df['chatter_rate'].mean()),
            "noop_duty": float(metrics_df['noop_duty'].mean()),
            "main_duty": float(metrics_df['main_duty'].mean()),
            "side_duty": float(metrics_df['side_duty'].mean()),
            "attitude_phase": float(metrics_df['attitude_phase'].mean()),
            "efficiency": float(metrics_df['efficiency'].mean()),
            "macro_oscillations": float(metrics_df['macro_oscillations'].mean()),
            "success_rate": float(metrics_df['is_success'].mean())
        },
        "terminal_distribution": {
            str(k): float(v) for k, v in metrics_df['status'].value_counts(normalize=True).to_dict().items()
        },
        # Raw arrays passed to the aggregator for inter-seed variance calculations
        "arrays": {
            "efficiencies": metrics_df['efficiency'].tolist(),
            "chatter_rates": metrics_df['chatter_rate'].tolist(),
            "macro_oscillations": metrics_df['macro_oscillations'].tolist(),
            "attitude_phases": metrics_df['attitude_phase'].tolist(),
            "statuses": metrics_df['status'].tolist()
        }
    }
    
    return payload

def aggregate_eval_seeds(seed_payloads: list) -> str:
    """
    Ingests a list of payloads from analyze_single_seed_eval().
    Computes cross-seed kinematic robustness, mechanical sensitivity, and success variance.
    Returns the final JSON for the MacBook LLM Diagnostician.
    """
    if not seed_payloads:
        return json.dumps({"error": "No eval seed payloads provided."})

    num_seeds = len(seed_payloads)
    
    # 1. Extract Seed-Level Averages
    success_rates = [seed['mean_metrics']['success_rate'] for seed in seed_payloads]
    efficiencies = [seed['mean_metrics']['efficiency'] for seed in seed_payloads]
    chatter_rates = [seed['mean_metrics']['chatter_rate'] for seed in seed_payloads]
    macro_oscillations = [seed['mean_metrics']['macro_oscillations'] for seed in seed_payloads]
    
    # 2. Compute Population Success Metrics
    mean_success = np.mean(success_rates)
    std_success = np.std(success_rates)
    
    # 3. Compute Kinematic Robustness (Coefficient of Variation)
    # CV = std / mean. High CV means the physical behavior changes wildly based on the random seed.
    mean_eff = np.mean(efficiencies)
    cv_eff = np.std(efficiencies) / (mean_eff + 1e-5)
    
    mean_chatter = np.mean(chatter_rates)
    cv_chatter = np.std(chatter_rates) / (mean_chatter + 1e-5)
    
    mean_oscillations = np.mean(macro_oscillations)
    
    # 4. Aggregate Terminal Distribution for Systemic Failure Analysis
    total_episodes = 0
    population_status_counts = {}
    
    # We rebuild the total population distribution by counting occurrences across all seed arrays
    for seed in seed_payloads:
        statuses = seed['arrays']['statuses']
        total_episodes += len(statuses)
        for status in statuses:
            population_status_counts[status] = population_status_counts.get(status, 0) + 1
            
    population_term_dist = {
        str(k): float(v / total_episodes) for k, v in population_status_counts.items()
    }

    # 5. Build the Mac-bound LLM Payload
    # Semantic boolean thresholds are applied here to provide unarguable physical facts to the LLM.
    payload = {
        "multi_seed_evaluation_health": {
            "success_robustness": {
                "population_mean_success_rate": round(mean_success, 3),
                "cross_seed_success_std": round(std_success, 3),
                "is_lottery_ticket_policy_flag": bool(std_success > 0.4) # e.g., one seed is 1.0, others are 0.0
            },
            "kinematic_stability": {
                "population_mean_efficiency": round(mean_eff, 3),
                "efficiency_coefficient_of_variation": round(cv_eff, 3),
                "population_mean_chatter_rate": round(mean_chatter, 3),
                "chatter_coefficient_of_variation": round(cv_chatter, 3),
                "kinematically_sensitive_to_initialization_flag": bool(cv_eff > 0.5 or cv_chatter > 0.5)
            },
            "lateral_control": {
                "population_mean_macro_oscillations": round(mean_oscillations, 2),
                "systemic_lateral_instability_flag": bool(mean_oscillations > 5.0)
            },
            "failure_mode_analysis": {
                "population_terminal_distribution": population_term_dist,
                "stable_but_suboptimal_flag": bool(mean_success < 0.2 and std_success < 0.1), # Fails consistently the exact same way
                "universal_success_flag": bool(min(success_rates) > 0.85) # Every single seed reliably lands
            }
        }
    }
    
    return payload

def translate_behavior_kinematics(agg_eval_json: dict) -> str:
    """
    Translates aggregated evaluation (deterministic) metrics into a physical 
    behavior diagnostic report for the Strategist LLM.
    """
    eval_data = agg_eval_json.get("multi_seed_evaluation_health", {})
    succ = eval_data.get("success_robustness", {})
    kin = eval_data.get("kinematic_stability", {})
    lat = eval_data.get("lateral_control", {})
    fail = eval_data.get("failure_mode_analysis", {})

    md = ["### 2. Kinematic Behavior & Physical Robustness\n"]

    # A. Deterministic Success Rate
    success_rate = succ.get("population_mean_success_rate", 0) * 100
    md.append("#### A. Universal Policy Robustness")
    md.append(f"- **Population Success Rate:** `{success_rate:.1f}%`")
    
    if succ.get("is_lottery_ticket_policy_flag"):
        md.append("  - *Diagnosis:* **LOTTERY TICKET POLICY.** The success rate has extreme cross-seed variance. The reward function fails to enforce a consistent universal law of flight, relying on lucky neural network initializations.")
    elif fail.get("universal_success_flag"):
        md.append("  - *Diagnosis:* Exceptional robustness. The policy flawlessly handles the environment across all seeds.")
    elif fail.get("stable_but_suboptimal_flag"):
        md.append("  - *Diagnosis:* The policy is highly stable but mathematically wrong. It consistently executes the same failure mode across all seeds.")

    # B. Actuator Efficiency & Stability
    md.append("\n#### B. Thermodynamic & Actuator Efficiency")
    md.append(f"- **Mean Descent Efficiency:** `{kin.get('population_mean_efficiency', 0):.3f}`")
    md.append(f"- **Actuator Chatter Rate:** `{kin.get('population_mean_chatter_rate', 0):.3f}`")
    
    if kin.get("kinematically_sensitive_to_initialization_flag"):
        md.append("  - *Diagnosis:* **High Kinematic Sensitivity.** The physical descent strategy (efficiency/chattering) changes wildly depending on the random seed.")
    
    # Check for specific physical failures
    if kin.get('population_mean_chatter_rate', 0) > 0.2:
        md.append("  - *Action Required:* **Severe Actuator Chattering detected.** The agent is rapidly vibrating opposing thrusters. The reward gradient near the decision boundary is too jagged. Consider smoothing penalties or adding a minor action-continuity reward.")

    if lat.get("systemic_lateral_instability_flag"):
        md.append("  - *Action Required:* **Macro-Oscillations detected.** The agent is overcorrecting laterally (drifting left/right). Ensure X-velocity and angle penalties are properly balanced.")

    # C. Terminal Distribution
    md.append("\n#### C. Population Terminal Distribution")
    dist = fail.get("population_terminal_distribution", {})
    for status, pct in sorted(dist.items(), key=lambda x: x[1], reverse=True):
        md.append(f"- `{status}`: {pct * 100:.1f}%")

    return "\n".join(md)

###########################################################
# For train_eps.csv
############################################################
def translate_reward_topology(agg_stochastic_json: dict) -> str:
    """
    Translates aggregated stochastic training metrics into a Reward Topology report.
    Automatically generates a Markdown table for algorithmic credit assignment.
    """
    stoch = agg_stochastic_json.get("multi_seed_stochastic_health", {})
    topo = stoch.get("global_reward_topology", {})
    comps = stoch.get("dynamic_component_analysis", {})
    frag = stoch.get("policy_fragility", {})

    md = ["### 3. Reward Topology & Algorithmic Credit Assignment\n"]

    # A. Global Objective Alignment
    md.append("#### A. Global Objective Alignment (Oracle Test)")
    oracle_rho = topo.get("mean_objective_alignment_rho", 0)
    md.append(f"- **Objective Alignment ($\\rho$):** `{oracle_rho:.3f}`")
    
    if topo.get("topology_is_inverted_flag"):
        md.append("  - *Diagnosis:* **CRITICAL FAILURE.** The total generated reward is NEGATIVELY correlated with successful landings. The agent is receiving mathematically higher returns for crashing/drifting than for landing safely.")
    elif oracle_rho < 0.5:
        md.append("  - *Diagnosis:* Weak alignment. The shaped reward landscape is poorly correlated with the actual goal of landing.")

    if topo.get("survival_hacking_detected_flag"):
        md.append("  - *Action Required:* **Survival Hacking Detected.** The agent is farming points by hovering/delaying the episode. Add a temporal penalty or check for positive constant rewards.")

    # B. Component-Level Credit Assignment (The Dynamic Table)
    md.append("\n#### B. Component-Level Contribution (Algorithmic Credit Assignment)")
    md.append("This table isolates the exact mathematical impact of each component you generated.")

    # Grab the dynamic target name (default to Success if missing)
    target_name = topo.get("target_metric_name", "Task Success")
    md.append(f"| Reward Component | Correlation w/ {target_name} ($\\rho$) | Relative Magnitude | Diagnostic Flag |")
    md.append("|:---|:---:|:---:|:---|")
    
    for comp_name, metrics in comps.items():
        rho = metrics.get('alignment_rho', 0) # Key was renamed here
        mag = metrics.get('relative_magnitude_pct', 0)
        
        flag = "🟢 Optimal"
        if metrics.get('is_traitor_component'):
            flag = "🔴 **TRAITOR COMPONENT** (Invert/Remove)"
        elif metrics.get('is_dead_weight'):
            flag = "🟡 **DEAD WEIGHT** (Scale Too Low)"
        elif rho < 0.2:
            flag = "⚪ Neutral/Noisy"

        md.append(f"| `{comp_name}` | {rho:.3f} | {mag:.1f}% | {flag} |")

    # C. Stochastic Fragility
    md.append("\n#### C. Stochastic Policy Fragility")
    cv = frag.get("mean_intra_rollout_cv", 0)
    ent = frag.get("mean_terminal_entropy_norm", 0)
    md.append(f"- **Intra-Rollout Reward CV:** `{cv:.3f}` (Variance across seeds with *frozen* weights)")
    md.append(f"- **Terminal Mode Entropy:** `{ent:.3f}`")
    
    if frag.get("systemic_policy_collapse_flag"):
        md.append("  - *Diagnosis:* The stochastic policy has completely collapsed into a single pathological failure mode. The KL penalty likely prevented escape from a severe local minimum.")
    elif cv > 0.5:
        md.append("  - *Diagnosis:* The policy is highly fragile. Even with fixed network weights, the agent's performance swings wildly depending on environment initialization.")

    return "\n".join(md)



def aggregate_stochastic_seeds(seed_payloads: list) -> str:
    """
    Ingests payloads from analyze_single_seed_stochastic().
    Dynamically analyzes LLM-generated reward components to find misaligned gradients.
    Returns the final JSON for the MacBook LLM Diagnostician.
    """
    if not seed_payloads:
        return json.dumps({"error": "No stochastic seed payloads provided."})

    num_seeds = len(seed_payloads)
    
    # 1. Extract Global Stats
    oracle_rhos = [seed['objective_alignment_rho'] for seed in seed_payloads]
    hacking_rhos = [seed['survival_hacking_rho'] for seed in seed_payloads]
    entropies = [seed['terminal_entropy_norm'] for seed in seed_payloads]
    cvs = [seed['intra_rollout_cv'] for seed in seed_payloads]
    
    global_success_rate = np.mean([seed['success_rate'] for seed in seed_payloads])
    all_failures = [seed['dominant_failure'] for seed in seed_payloads]
    global_dominant_failure = max(set(all_failures), key=all_failures.count)

    # 2. Determine Global Target Metric
    if 0.0 < global_success_rate < 1.0:
        target_name = "Task Success"
        metric_key = 'succ'
    elif global_success_rate == 1.0:
        target_name = "Impact Softness"
        metric_key = 'imp'
    else: # 0% Success
        target_name = f"Composite Viability ({global_dominant_failure})"
        metric_key = 'composite'
        if global_dominant_failure == 'out_of_bounds': w_sp, w_kin, w_att = 0.7, 0.2, 0.1
        elif global_dominant_failure == 'crashed': w_sp, w_kin, w_att = 0.2, 0.5, 0.3
        elif global_dominant_failure == 'hover_timeout': w_sp, w_kin, w_att = 0.6, 0.2, 0.2
        elif global_dominant_failure == 'landed_but_slid_into_valley': w_sp, w_kin, w_att = 0.1, 0.7, 0.2
        else: w_sp, w_kin, w_att = 0.33, 0.33, 0.34

    # 3. Aggregate Reward Components
    comp_base_corrs = defaultdict(lambda: defaultdict(list))
    comp_means = defaultdict(list)
    
    for seed in seed_payloads:
        for col, corrs in seed.get('reward_components', {}).get('correlations', {}).items():
            for c_type, val in corrs.items():
                comp_base_corrs[col][c_type].append(val)
        for col, mean_val in seed.get('reward_components', {}).get('means', {}).items():
            comp_means[col].append(mean_val)
            
    # Compute component-level diagnostics
    component_diagnostics = {}
    total_abs_magnitude = sum([abs(np.mean(vals)) for vals in comp_means.values()]) + 1e-5
    
    for col in comp_base_corrs.keys():

        mean_val = np.mean(comp_means[col])
        relative_contribution = abs(mean_val) / total_abs_magnitude

        # Resolve the chosen metric
        if metric_key == 'composite':
            mean_sp = np.mean(comp_base_corrs[col]['sp'])
            mean_kin = np.mean(comp_base_corrs[col]['kin'])
            mean_att = np.mean(comp_base_corrs[col]['att'])
            mean_rho = (w_sp * mean_sp) + (w_kin * mean_kin) + (w_att * mean_att)
        else:
            mean_rho = np.mean(comp_base_corrs[col][metric_key])

        component_diagnostics[col] = {
            "alignment_rho": round(mean_rho, 3), 
            "is_traitor_component": bool(mean_rho < -0.2), # Actively penalizing successful behavior
            "relative_magnitude_pct": round(relative_contribution * 100, 1),
            "is_dead_weight": bool(relative_contribution < 0.01) # Contributes < 1% to the gradient
        }

    # Aggregate Terminal Distributions
    population_term_dist = defaultdict(float)
    for seed in seed_payloads:
        for status, pct in seed['terminal_distribution'].items():
            population_term_dist[status] += (pct / num_seeds)

    # 4. Build the Mac-bound LLM Payload
    payload = {
        "multi_seed_stochastic_health": {
            "global_reward_topology": {
                "mean_objective_alignment_rho": round(np.mean(oracle_rhos), 3),
                "topology_is_inverted_flag": bool(np.mean(oracle_rhos) < 0),
                "mean_survival_hacking_rho": round(np.mean(hacking_rhos), 3),
                "survival_hacking_detected_flag": bool(np.mean(hacking_rhos) > 0.6),
                "target_metric_name": target_name # Pass to translation layer
            },
            "dynamic_component_analysis": component_diagnostics,
            "policy_fragility": {
                "mean_intra_rollout_cv": round(np.mean(cvs), 3),
                "mean_terminal_entropy_norm": round(np.mean(entropies), 3),
                "systemic_policy_collapse_flag": bool(np.mean(entropies) < 0.1)
            },
            "population_terminal_distribution": dict(population_term_dist)
        }
    }

    return json.dumps(payload, indent=2)



def analyze_single_seed_stochastic(csv_path: str, seed_id: int, window_size: int = 500) -> dict:
    """
    Transforms episode-level stochastic training logs into metrics of reward topology.
    Derives true objective alignment strictly from semantic terminal states, using 
    failure-conditioned composite proxy metrics for edge cases (0% or 100% success).
    """
    if not os.path.exists(csv_path):
        return {"seed_id": seed_id, "error": f"Train eps log not found at {csv_path}"}

    df = pd.read_csv(csv_path)
    if df.empty:
        return {"seed_id": seed_id, "error": "Train eps log is empty."}

    df = df.sort_values('timestep_global').reset_index(drop=True)
    window = min(window_size, max(1, len(df) // 4))

    # --- A. True Objective Alignment (Global Reward vs. Semantic Success) ---
    # Convert semantic tags to a binary success metric (1.0 for any landing, 0.0 for crash/out_of_bounds)
    df['is_success'] = df['terminal_status'].str.contains('landed', na=False).astype(float)

    # Correlate the LLM's total reward with the binary success metric. 
    # If the LLM's reward function is good, high reward should strongly correlate with 'is_success == 1.0'
    df['objective_alignment_rho'] = df['is_success'].rolling(window).corr(df['ep_rew_total']).fillna(0)

    # --- B. Survival Hacking Index ---
    # Does living longer equal higher LLM reward, regardless of success?
    df['survival_hacking_idx'] = df['ep_len'].rolling(window).corr(df['ep_rew_total']).fillna(0)

    # --- C. Terminal Mode Entropy (Policy Collapse) ---
    terminal_dummies = pd.get_dummies(df['terminal_status']).astype(float)
    rolling_term_dist = terminal_dummies.rolling(window).mean().fillna(1.0 / max(1, len(terminal_dummies.columns)))
    
    rolling_entropy = -(rolling_term_dist * np.log(rolling_term_dist + 1e-9)).sum(axis=1)
    max_term_entropy = np.log(len(terminal_dummies.columns)) if len(terminal_dummies.columns) > 1 else 1.0
    df['terminal_entropy_norm'] = rolling_entropy / max_term_entropy

    # --- D. Intra-Rollout Volatility (Policy Sensitivity) ---
    # Group by the rollout_id (where policy weights are frozen)
    rollout_stats = df.groupby('rollout_id').agg(
        rew_var=('ep_rew_total', 'var'),
        rew_mean=('ep_rew_total', 'mean')
    ).reset_index()
    rollout_stats['reward_cv'] = np.sqrt(rollout_stats['rew_var']) / (np.abs(rollout_stats['rew_mean']) + 1e-5)

    # --- Extract Converged (Late-Stage) Behavior ---
    late_stage_idx = max(1, int(len(df) * 0.2))
    late_stage_eps = df.iloc[-late_stage_idx:]
    
    late_rollout_idx = max(1, int(len(rollout_stats) * 0.2))
    late_rollouts = rollout_stats.iloc[-late_rollout_idx:]

    # Calculate average episode lengths by terminal state
    crashes = late_stage_eps[late_stage_eps['terminal_status'] == 'crashed']
    landings = late_stage_eps[late_stage_eps['is_success'] == 1.0]
    
    avg_crash_len = crashes['ep_len'].mean() if not crashes.empty else 0.0
    avg_landed_len = landings['ep_len'].mean() if not landings.empty else 0.0

    # Clean terminal distribution dict
    term_dist = {str(k): float(v) for k, v in late_stage_eps['terminal_status'].value_counts(normalize=True).to_dict().items()}
    dominant_failure = late_stage_eps['terminal_status'].mode()[0] if not late_stage_eps.empty else 'unknown'
    seed_success_rate = late_stage_eps['is_success'].mean()

    # --- E. DYNAMIC COMPONENT CREDIT ASSIGNMENT ---
    reward_cols = [c for c in df.columns if c.startswith('reward_')]
    component_correlations = {}
    component_means = {}

    if reward_cols:
        # Define all target series
        t_succ = df['is_success']
        t_imp = -np.sqrt(df['x_vel']**2 + df['y_vel']**2)
        t_sp = -np.sqrt(df['x_pos']**2 + df['y_pos']**2)
        t_kin = -np.sqrt(df['x_vel']**2 + df['y_vel']**2)
        t_att = -np.abs(df['angle'])

        for col in reward_cols:
            # Calculate all base correlations
            r_succ = t_succ.corr(df[col]) if df['is_success'].std() > 0 else 0.0
            r_imp = t_imp.corr(df[col])
            r_sp = t_sp.corr(df[col])
            r_kin = t_kin.corr(df[col])
            r_att = t_att.corr(df[col])
            
            component_correlations[col] = {
                'succ': float(r_succ) if not np.isnan(r_succ) else 0.0,
                'imp': float(r_imp) if not np.isnan(r_imp) else 0.0,
                'sp': float(r_sp) if not np.isnan(r_sp) else 0.0,
                'kin': float(r_kin) if not np.isnan(r_kin) else 0.0,
                'att': float(r_att) if not np.isnan(r_att) else 0.0,
            }
            component_means[col] = float(df[col].mean())

    # Construct the exact data contract for the Linux Aggregator
    payload = {
        "seed_id": seed_id,
        "success_rate": float(seed_success_rate),
        "dominant_failure": str(dominant_failure),
        "objective_alignment_rho": float(late_stage_eps['objective_alignment_rho'].mean()),
        "survival_hacking_rho": float(late_stage_eps['survival_hacking_idx'].mean()),
        "terminal_entropy_norm": float(late_stage_eps['terminal_entropy_norm'].mean()),
        "intra_rollout_cv": float(late_rollouts['reward_cv'].mean()),
        "avg_crash_length": float(avg_crash_len),
        "avg_landed_length": float(avg_landed_len),
        "terminal_distribution": term_dist,
        "reward_components": {
            "correlations": component_correlations,
            "means": component_means
        }
    }

    return payload

def generate_metric_payload(iteration:int, num_of_seeds:int =3):
    ws= ExperimentWorkspace(iteration = iteration)
    # List of paths to progress.csv
    path_list1 = [ws.dirs["telemetry_iteration"] / f"progress_iter{iteration:02d}_seed{i}.csv"
                 for i in range(num_of_seeds)]
    # Analyze each progress.csv
    single_progress_results = [analyze_single_seed_progress(path, idx) for idx, path in enumerate(path_list1)]
    # Analyzing progress.csv across seeds
    progress_payload= aggregate_progress_seeds(single_progress_results)

    # List of paths to train_eps.csv
    path_list2 = [ws.dirs["telemetry_iteration"] / f"iter{iteration:02d}_seed{i}_train.csv"
                 for i in range(num_of_seeds)]
    # Analyze each train_eps.csv
    single_train_results = [analyze_single_seed_eval(path, idx) for idx, path in enumerate(path_list2)]
    # Analyzing train_eps.csv across seeds
    train_payload = aggregate_eval_seeds(single_train_results)

    # List of paths to eval.csv
    path_list3 = [ws.dirs["telemetry_iteration"] / f"iter{iteration:02d}_seed{i}_eval.csv"
            for i in range(num_of_seeds)]
    # Analyze each eval.csv
    single_eval_results = [analyze_single_seed_stochastic(path, idx) for idx, path in enumerate(path_list3)]
    # Analyzing eval.csv across seeds
    eval_payload = aggregate_stochastic_seeds(single_eval_results)
 
    metrics = [progress_payload,train_payload,eval_payload]
    save_path = ws.dirs['telemetry_paylods']/ f"iter{iteration:02d}_metric_payload.json"
    with open(save_path, "w") as f:
        json.dump(metrics,f ,indent=4)
    return 

def generate_diagnostic_report(metrics: dict) -> str:
    section_1 = translate_optimization_health(metrics)
    section_2 = translate_reward_topology(metrics)
    section_3 = translate_behavior_kinematics(metrics)
    report = section_1 + f"\n" + section_2 +f"\n"+ section_3
    return report

if __name__ == "__main__":
    
    if len(sys.argv) >1:
        generate_metric_payload(sys.argv[1])
    else:
        print("No argument provided.")
        sys.exit()
    