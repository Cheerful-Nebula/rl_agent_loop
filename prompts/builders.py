from . import loader
import json
from src.utils import performance_telemetry_as_table,training_telemetry_as_table
def build_diagnosis_prompt(template: tuple[str,str], metrics_json: json, current_code, long_term_memory, short_term_history)-> tuple[str,str]:
    """
    Builds a multi-configuration LLM diagnosis prompt.

    Parameters
    ----------
    metrics_json : list[dict]
        A list where each element is:
        {

        }

    
    current_code : str
        Current reward function code.
    
    long_term_memory : str
        Persistent memory summary.
    
    short_term_history : str
        Dialogue or evaluation context.
    """

    # Load RL Researcher persona
    system_role = loader.load_template("roles", "analyst", template[0])
    metrics_json["n_updates"] = metrics_json["training_dynamics"]["n_updates"]

    # Converting data from JSON to formated markdown tables
    performance_table= performance_telemetry_as_table(metrics_json["performance"])
    training_table= training_telemetry_as_table(metrics_json["training_dynamics"])

    # Deleteing unneccesary keys, such as those that were just converted to markdown
    del metrics_json["performance"]
    del metrics_json["training_dynamics"]
    del metrics_json["timestamp"]
    del metrics_json["source_code_path"]
    metrics_json["timesteps"]=metrics_json["config"]["total_timesteps"]
    metrics_json["num_updates"]=metrics_json["n_updates"]["total"]
    del metrics_json["config"]
    del metrics_json["n_updates"]
    # Format the JSON blob for the template
    metrics_json_str = json.dumps(metrics_json, indent=4)

    # Load the multi-diagnosis task template
    task_template = loader.load_template("tasks", "analyze",template[1])

    # Build user task
    user_task = task_template.format(
        performance_table = performance_table,
        training_table=training_table,
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

def build_formatter_prompt(template: tuple[str,str], raw_plan):
    """Constructs prompts for generating initial reward shaping function."""
    role = loader.load_template("roles", "formatter", template[0])
    
    task_raw = loader.load_template("tasks", "to_format", template[1])
    task = task_raw.format(
        raw_plan = raw_plan
    )
    return role, task