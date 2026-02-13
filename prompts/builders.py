from . import loader
import json
from typing import Any
from src.utils import performance_telemetry_as_table,training_telemetry_as_table,format_diagnostician_input


def build_training_diagnosis_prompt(template: tuple[str,str], metrics: dict[Any,Any])-> tuple[str,str]:
    """
    Builds a LLM prompt - diagnosising PPO training optimization
    """

    # Grab system prompt
    system_role = loader.load_template("roles", "analyst", template[0])

    # Converting data to formated markdown tables
    #training_table= training_telemetry_as_table(metrics["training_dynamics"])
    training_table= format_diagnostician_input(metrics)

    # Load the diagnosis task template
    task_template = loader.load_template("tasks", "analyze",template[1])

    # Build user task
    user_task = task_template.format(
        training_table=training_table
    )

    return system_role, user_task

def build_performance_diagnosis_prompt(template: tuple[str,str], metrics: dict[Any,Any])-> tuple[str,str]:
    """
    Builds a LLM prompt - diagnosising PPO training optimization
    """

    # Grab system prompt
    system_role = loader.load_template("roles", "analyst", template[0])

    # Converting data to formated markdown tables
    performance_table= performance_telemetry_as_table(metrics["performance"])

    # Load the diagnosis task template
    task_template = loader.load_template("tasks", "analyze",template[1])

    # Build user task
    user_task = task_template.format(
        performance_table=performance_table
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

def build_formatter_fix_prompt(template: tuple[str,str], raw_plan,json_attempt):
    """Constructs prompts for generating initial reward shaping function."""

    role = loader.load_template("roles", "formatter", template[0])
    task_raw = loader.load_template("tasks", "to_format", template[1])

    task = task_raw.format(
        raw_plan = raw_plan,
        json_attempt=json_attempt
    )
    return role, task

def build_validator_prompt(prev_hypothesis, prev_changes, prev_metrics):
    """
    Builds the prompt for the Phase 0 Validator.
    """
    system_role = """
    You are a skeptical Scientific Reviewer for a Reinforcement Learning experiment.
    Your GOAL is to validate if the Researcher's hypothesis was supported by the data.
    
    DEFAULT POSITION: The hypothesis is REFUTED unless the data proves otherwise clearly.
    
    RULES:
    1. Correlation != Causality. If the reward improved but the targeted metric (e.g., tilt) got worse, it is REFUTED.
    2. Noise Tolerance. Changes of < 5% are statistical noise. Mark as INCONCLUSIVE (False).
    3. Strictness. Do not be "nice". Only mark is_validated=True if the specific mechanism predicted actually happened.
    """
    
    user_task = f"""
    # EXPERIMENT REVIEW
    
    ## The Hypothesis
    "{prev_hypothesis}"
    
    ## The Intervention (Changes Made)
    {prev_changes}
    
    ## The Results (Data)
    {prev_metrics}
    
    # TASK
    Did the metrics support the hypothesis?
    
    # OUTPUT FORMAT (JSON ONLY)
    {{
        "is_validated": boolean,
        "confidence_score": int (1-10),
        "reasoning": "One short sentence explaining why."
    }}
    """
    return system_role, user_task