from . import loader

def build_diagnosis_prompt(metrics, current_code, training_summary, long_term_memory, short_term_history):
    """
    Constructs the system role and user task for Phase 1 (Diagnosis).
    """
    # 1. Fetch the Static Role
    system_role = loader.load_template("roles", "rl_researcher")

    # 2. Extract specific stats (The heavy lifting)
    stats = metrics['performance']
    diagnostics = metrics.get('diagnostics', {})
    
    # 3. Load and Format the Task
    # Note: We pass the raw values. The .md file handles {value:.4f} formatting.
    task_template = loader.load_template("tasks", "diagnose_agent")
    
    user_task = task_template.format(
        env_id="LunarLander-v3", # Or pass in as arg
        # RL Metrics
        reward_success_rate=stats.get('reward_success_rate', 0),
        position_success_rate=stats.get('position_success_rate', 0),
        crash_rate=stats.get('crash_rate', 0),
        avg_descent=diagnostics.get('avg_descent_velocity', 0),
        avg_tilt=diagnostics.get('avg_tilt_angle', 0),
        avg_x=diagnostics.get('avg_x_position', 0),
        main_eng=diagnostics.get('main_engine_usage', 0),
        side_eng=diagnostics.get('side_engine_usage', 0),
        x_std=diagnostics.get('horizontal_stability_index', 0),
        y_std=diagnostics.get('vertical_stability_index', 0),
        # Training Dynamics
        entropy_start=training_summary.get('entropy_start', 'N/A'),
        entropy_end=training_summary.get('entropy_end', 'N/A'),
        entropy_trend=training_summary.get('entropy_trend', 'Unknown'),
        value_loss_avg=training_summary.get('value_loss_avg', 'N/A'),
        policy_loss_end=training_summary.get('policy_loss_end', 'N/A'),
        # Context
        long_term_memory=long_term_memory,
        short_term_history=short_term_history,
        current_code=current_code if current_code else "# No previous code"
    )

    return system_role, user_task

def build_multi_diagnosis_prompt(config_list, current_code, long_term_memory, short_term_history,training_dynamics)-> tuple[str,str]:
    """
    Builds a multi-configuration LLM diagnosis prompt.

    Parameters
    ----------
    config_list : list[dict]
        A list where each element is:
        {
            "config_id": str,
            "meta": {...},
            "metrics": {...},
            "diagnostics": {...},
            "training_summary": {...}
        }
        You can include ANY number of configurations.
    
    current_code : str
        Current reward function code.
    
    long_term_memory : str
        Persistent memory summary.
    
    short_term_history : str
        Dialogue or evaluation context.
    """

    # Load RL Researcher persona
    system_role = loader.load_template("roles", "rl_researcher")

    # Format the JSON blob for the template
    import json
    configuration_json = json.dumps(config_list, indent=4)

    # Load the multi-diagnosis task template
    task_template = loader.load_template("tasks", "diagnose_agent_multi")

    # Build user task
    user_task = task_template.format(
        configuration_json=configuration_json,
        long_term_memory=long_term_memory,
        short_term_history=short_term_history,
        training_dynamics=training_dynamics,
        current_code=current_code if current_code else "# No previous code"
    )

    return system_role, user_task


def build_coding_prompt(plan, current_code):
    """Constructs prompts for Phase 2 (Implementation)."""
    role = loader.load_template("roles", "python_coder")
    
    task_raw = loader.load_template("tasks", "implement_plan")
    task = task_raw.format(
        plan=plan,
        current_code=current_code if current_code else "# No existing code"
    )
    return role, task

def build_fix_prompt(invalid_code, feedback):
    """Constructs prompts for Phase 3: Debugging/fixing."""
    role = loader.load_template("roles", "python_coder")
    
    task_raw = loader.load_template("tasks", "fix_code")
    task = task_raw.format(
        clean_code=invalid_code,
        feedback=feedback
    )
    return role, task