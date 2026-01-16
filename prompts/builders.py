from . import loader
import json
from src.utils import format_telemetry_as_table
def build_diagnosis_prompt(template: tuple[str,str], metrics_json: json, current_code, long_term_memory, short_term_history)-> tuple[str,str]:
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
    system_role = loader.load_template("roles", "analyst", template[0])
    performance_table= format_telemetry_as_table(metrics_json["performance"])
    key_list = [
        "mean_reward",
        "median_reward",
        "std_reward",
        "mean_ep_length",
        "reward_success_rate",
        "position_success_rate",
        "crash_rate",
        "avg_x_position",
        "avg_descent_velocity",
        "avg_tilt_angle",
        "vertical_stability_index",
        "horizontal_stability_index"
        ]
    for stats_dict in metrics_json:
        for key in key_list:
            if key in stats_dict["performance"].keys():
                del stats_dict["performance"][key]
    # Format the JSON blob for the template
    metrics_json_str = json.dumps(metrics_json, indent=4)

    # Load the multi-diagnosis task template
    task_template = loader.load_template("tasks", "analyze",template[1])

    # Build user task
    user_task = task_template.format(
        performance_table = performance_table,
        configuration_json=metrics_json_str,
        long_term_memory=long_term_memory,
        short_term_history=short_term_history,
        current_code=current_code if current_code else "# No previous code"
    )

    return system_role, user_task

def build_initial_shaping_prompt(template: tuple[str,str]):
    """Constructs prompts for generating initial reward shaping function."""
    role = loader.load_template("roles", "coder", template[0])
    
    task = loader.load_template("tasks", "code_generation", template[1])

    return role, task

def build_coding_prompt(template: tuple[str,str],plan, current_code):
    """Constructs prompts for Phase 2 (Implementation)."""
    role = loader.load_template("roles", "coder", template[0])
    
    task_raw = loader.load_template("tasks", "code_generation", template[1])
    task = task_raw.format(
        plan=plan,
        current_code=current_code if current_code else "# No existing code"
    )
    return role, task

def build_fix_prompt(template: tuple[str,str],invalid_code, feedback):
    """Constructs prompts for Phase 3: Debugging/fixing."""
    role = loader.load_template("roles", "coder", template[0])
    
    task_raw = loader.load_template("tasks", "code_fix", template[1])
    task = task_raw.format(
        clean_code=invalid_code,
        feedback=feedback
    )
    return role, task