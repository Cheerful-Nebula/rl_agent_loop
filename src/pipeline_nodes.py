import time
import re
from datetime import timedelta 

# custom imports
import prompts  
from src.cognitive_node import CognitiveNode
from src.config import Config
from src.code_validation import CodeValidator
from src.utils import extract_python_code

def generate_proposals(brain: CognitiveNode, iteration: int, diagnostic_report: str, current_code: str, expt_ledger: str) -> str:
    """Generate Proposals of Reward Function Modifications based on metrics and history."""
    start_time = time.perf_counter()

    # Build Prompts (Now passing the explicitly required expt_ledger)
    sys_prompt, user_prompt = prompts.build_strategist_prompt(
        template=Config.strategist_template,
        iteration_number=iteration,
        current_reward_code=current_code,
        diagnostic_report=diagnostic_report,
        experiment_ledger=expt_ledger
    )
    
    print("Generating Proposals")
    proposals = brain.chat(
        phase_name='strategist',
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        options=Config.think_options
    )

    elapsed_time = time.perf_counter() - start_time
    print(f"Proposal Generation took: {timedelta(seconds=elapsed_time)}")
    return proposals

def organize_proposals(brain: CognitiveNode, proposals: str) -> str:
    """Sanitizes and formats the raw output from the Strategist."""
    start_time = time.perf_counter()
    
    # Build Prompt (Fixed copy-paste bug, now calling build_organizer_prompt)
    sys_prompt, user_prompt = prompts.build_organizer_prompt(
        template=Config.organizer_template,
        raw_strategist_output=proposals
    )
    
    print("Organizing Proposals")
    proposal_report = brain.chat(
        phase_name='organizer',
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        options=Config.think_options
    )
    
    elapsed_time = time.perf_counter() - start_time
    print(f"Organizing Proposals took: {timedelta(seconds=elapsed_time)}")
    return proposal_report

def choose_proposal(brain: CognitiveNode, iteration: int, proposal_report: str, expt_ledger: str) -> str:
    """Evaluates proposals against the experiment ledger to choose the most viable path."""
    start_time = time.perf_counter()

    # Build Prompt (Removed unused diagnostic_report, added required iteration)
    sys_prompt, user_prompt = prompts.build_lead_prompt(
        template=Config.lead_template,
        iteration=iteration,
        experiment_ledger=expt_ledger,
        strategist_proposals_markdown=proposal_report
    )
    
    print("Choosing Proposal")
    chosen_proposal = brain.chat(
        phase_name='research_lead',
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        options=Config.think_options
    )
    
    elapsed_time = time.perf_counter() - start_time
    print(f"Choosing Proposal took: {timedelta(seconds=elapsed_time)}")
    return chosen_proposal

def generate_work_order(brain: CognitiveNode, chosen_proposal: str) -> str:
    """Routes the chosen proposal into strict payload formats for downstream nodes."""
    start_time = time.perf_counter()
    
    # Build Prompt
    sys_prompt, user_prompt = prompts.build_dispatcher_prompt(
        template=Config.dispatcher_template,
        research_lead_decision=chosen_proposal
    )
    
    print("Generating Work Order")
    work_order = brain.chat(
        phase_name='dispatcher',
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        options=Config.think_options
    )
    
    elapsed_time = time.perf_counter() - start_time
    print(f"Work Order Generation took: {timedelta(seconds=elapsed_time)}")
    return work_order


def generate_code(brain: CognitiveNode, coder_payload: str, current_code: str, max_retries: int = 3) -> str:
    """Translates work orders into executable code with an iterative validation loop."""
    start_time = time.perf_counter()
    
    sys_prompt, user_prompt = prompts.build_coder_prompt(
        template=Config.coder_template,
        current_reward_code=current_code,
        coder_payload_from_dispatcher=coder_payload
    )

    attempt = 0
    generated_code = ""

    while attempt < max_retries:
        print(f"Generating Code (Attempt {attempt + 1}/{max_retries})")
        
        raw_response = brain.chat(
            phase_name='coder',
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            options=Config.think_options
        )

        generated_code = extract_python_code(raw_response)
        
        validator = CodeValidator(generated_code)
        is_valid, validation_result = validator.validate()

        if is_valid:
            print("✅ Code Validation Passed!")
            break
            
        print(f"⚠️ Validation Failed: {validation_result}")
        
        user_prompt += (
            f"\n\nYour previous attempt failed validation with the following error:\n"
            f"ERROR: {validation_result}\n"
            f"Please analyze the error, fix the code, and output ONLY the corrected Python code inside a markdown block."
        )
        attempt += 1

    if attempt == max_retries:
        print(f"🚨 Warning: Max retries ({max_retries}) reached. The resulting code may be unstable.")

    elapsed_time = time.perf_counter() - start_time
    print(f"Coder Generation & Validation took: {timedelta(seconds=elapsed_time)}")
    
    return generated_code

def generate_ledger_entry(brain: CognitiveNode, iteration: int, val_payload: str, diagnostic_report: str) -> str:
    """Validator: Peer Reviews Hypothesis against Diagnostic Report to create a ledger entry."""
    start_time = time.perf_counter()

    sys_prompt, user_prompt = prompts.build_validator_prompt(
        template=Config.validator_template,
        previous_iteration_number=iteration,
        validator_payload_from_dispatcher=val_payload,
        new_diagnostic_report=diagnostic_report
    )
    
    print("Generating Ledger Entry (Peer Review)")
    ledger_entry = brain.chat(
        phase_name='validator',
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        options=Config.think_options
    )

    elapsed_time = time.perf_counter() - start_time
    print(f"Peer Review took: {timedelta(seconds=elapsed_time)}")
    
    return ledger_entry