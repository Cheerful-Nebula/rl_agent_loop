from typing import Tuple
from prompts.loader import load_template
def build_strategist_prompt(template: Tuple[str, str], iteration_number: int, current_reward_code: str, diagnostic_report: str, experiment_ledger: str) -> Tuple[str, str]:
    """Builds the prompt for the Strategist to generate reward shaping proposals."""
    # Load Prompts
    system_role = load_template("system_prompts","strategist",template[0])
    user_template = load_template("user_prompts","strategist",template[1])
    # Format User Prompt
    user_task = user_template.format(
        iteration_number=iteration_number,
        current_reward_code=current_reward_code,
        experiment_ledger=experiment_ledger,
        diagnostic_report=diagnostic_report
    )
    return system_role, user_task

def build_organizer_prompt(template: Tuple[str, str], raw_strategist_output: str) -> Tuple[str, str]:
    """Builds the prompt for the Organizer to sanitize and format Strategist proposals."""
    system_role = load_template("system_prompts","organizer",template[0])
    user_template = load_template("user_prompts","organizer",template[1])
    user_task = user_template.format(
        raw_strategist_output=raw_strategist_output
    )
    return system_role, user_task

def build_lead_prompt(template: Tuple[str, str], iteration: int, experiment_ledger: str, strategist_proposals_markdown: str) -> Tuple[str, str]:
    """Builds the prompt for the Research Lead to make an executive decision."""
    system_role = load_template("system_prompts","research_lead",template[0])
    user_template = load_template("user_prompts","research_lead",template[1])
    user_task = user_template.format(
        iteration=iteration,
        experiment_ledger=experiment_ledger,
        strategist_proposals_markdown=strategist_proposals_markdown
    )
    return system_role, user_task

def build_dispatcher_prompt(template: Tuple[str, str], research_lead_decision: str) -> Tuple[str, str]:
    """Builds the prompt for the Dispatcher to route payloads to Coder and Validator."""
    system_role = load_template("system_prompts","dispatcher",template[0])
    user_template = load_template("user_prompts","dispatcher",template[1])
    user_task = user_template.format(
        research_lead_decision=research_lead_decision
    )
    return system_role, user_task

def build_coder_prompt(template: Tuple[str, str], current_reward_code: str, coder_payload_from_dispatcher: str) -> Tuple[str, str]:
    """Builds the prompt for the Coder to implement the reward function update."""
    system_role = load_template("system_prompts","coder",template[0])
    user_template = load_template("user_prompts","coder",template[1])
    user_task = user_template.format(
        current_reward_code=current_reward_code,
        coder_payload_from_dispatcher=coder_payload_from_dispatcher
    )
    return system_role, user_task

def build_validator_prompt(template: Tuple[str, str], previous_iteration_number: int, validator_payload_from_dispatcher: str, new_diagnostic_report: str) -> Tuple[str, str]:
    """Builds the prompt for the Validator to evaluate empirical results against the hypothesis."""
    system_role = load_template("system_prompts","validator",template[0])
    user_template = load_template("user_prompts","validator",template[1])
    user_task = user_template.format(
        previous_iteration_number=previous_iteration_number,
        validator_payload_from_dispatcher=validator_payload_from_dispatcher,
        new_diagnostic_report=new_diagnostic_report
    )
    return system_role, user_task