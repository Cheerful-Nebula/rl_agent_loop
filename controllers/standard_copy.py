import argparse
import ollama
from datetime import datetime, timedelta 
import warnings
import time
import os
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

# -- PROJECT IMPORTS --
import prompts  
from src.workspace_manager import ExperimentWorkspace
from src.code_validation_v03 import CodeValidator
from src import utils
from src.config import Config
from src.llm_utils import *
from src.cognitive_node import CognitiveNode
from src.ledger import ExperimentLedger  # <--- NEW IMPORT

MODEL_NAME = Config.LLM_MODEL
MAX_RETRIES = 5 
# Define your chosen roster here (Copy-pasted from our chat)
if MODEL_NAME == "gemma3:27b": # Deep Reasoning Team
    # Team 1: Deep Logic
    ACTIVE_ROSTER = {
    "architect": "gemma3:27b",
    "formatter": "nemotron-3-nano:30b",
    "coder":    "qwen3-coder:30b",
    "validator": "qwen3-coder:30b"
    }
elif MODEL_NAME == "openthinker:32b": # Maximum Safety Team
    # Team 2: Maximum Safety
    ACTIVE_ROSTER ={
        "architect": "openthinker:32b",   # Balanced, risk-averse planning
        "formatter": "nemotron-3-nano:30b", # Reliable data entry
        "coder":     "qwen3-coder:30b",   # Fast, competent coding
        "validator": "llama3.2-vision:11b" # The "Gatekeeper" - Refutes 80% of bad ideas
        }
elif MODEL_NAME == "nemotron-3-nano:30b": # Maximum Throughput Team
    # Team 3: Speed Run (The Stable Loop)
    ACTIVE_ROSTER={
        "architect": "nemotron-3-nano:30b", # Fast, structured planning
        "formatter": "nemotron-3-nano:30b", # (Self-formatting)
        "coder":     "qwen3-coder:30b",      # Fast execution
        "validator": "openthinker:32b"    # Nuanced verdicts (keeps loop moving)
    }
else:
    # Fallback / Default
    ACTIVE_ROSTER = {
        "architect": Config.LLM_MODEL,
        "formatter": Config.LLM_MODEL,
        "coder":     Config.LLM_MODEL,
        "validator": Config.LLM_MODEL
    }
def run_agentic_improvement(iteration):
    # For catching excution time at the end 
    start_time = time.perf_counter()
    
    # 1. Initialize Workspace, Brain, and Memory
    ws = ExperimentWorkspace()
    brain = CognitiveNode(iteration=iteration, workspace=ws, model=MODEL_NAME)
    brain_memory = ExperimentLedger(ws.model_root_path) # Initialize Experiment Ledger
    
    if iteration == 1:
        upsert_model_metadata_row(ws.containers["model_metadata"], MODEL_NAME)
    
    print(f"🔵 AGENT (Iter {iteration}): Active in {ws.model_root_path}")

    # 2. Load Context
    # A. Metrics from the run just finished (Iteration N)
    metrics = ws.load_metrics(iteration)
    if not metrics:
        print(f"❌ No metrics found for Iteration {iteration}. Cannot proceed.")
        return

    # B. The code that produced those metrics (Previous Iteration)
    prev_code_path = ws.get_path("code", iteration - 1, "reward.py")
    with open(prev_code_path, "r") as f:
        current_code = f.read()

    # =========================================================
    # PHASE 0: HYPOTHESIS VALIDATION (The Causal Check)
    # =========================================================
    # We validate the experiment that produced the CURRENT metrics.
    # Logic: Standard(Iter N-1) created Experiment N. Train(Iter N) produced Metrics N.
    # So we validate Experiment N using Metrics N.
    
    if iteration > 1:
        # Get the most recent experiment from the ledger (should be Exp ID = iteration)
        last_exp = brain_memory.get_last_experiment()
        
        # Guard clause: Ensure we have an experiment to validate (Bypasses Iter 1 / Baseline issues)
        if last_exp:
            print(f"🔍 Validating Hypothesis from Experiment {last_exp['id']}...")
            
            # Format metrics for the validator (Using same table utility as Analyst)
            # We select the first entry in 'performance' which typically contains the summary stats
            formatted_metrics = utils.performance_telemetry_as_table(metrics.get('performance', []))
            
            val_role, val_task = prompts.build_validator_prompt(
                prev_hypothesis=last_exp.get('hypothesis', 'No Hypothesis Recorded'),
                prev_changes=last_exp.get('config_changes', {}),
                prev_metrics=formatted_metrics
            )
            
            # Call LLM (Fast, JSON Mode)
            validation_result = brain.chat(
                phase_name='validation',
                system_prompt=val_role,
                user_prompt=val_task,
                parse_json=True,
                options={"temperature": 0.1,'num_ctx': 8182,'num_predict': 4096},
                model_override= ACTIVE_ROSTER["validator"] 
            )
            
            # Fallback for parsing failures
            if not validation_result:
                validation_result = {"is_validated": False, "reasoning": "JSON Parsing Failed"}

            # Update Ledger with the verdict
            brain_memory.update_validation(
                iteration=last_exp['id'], 
                metrics=metrics.get('performance', [{}])[0], 
                validation_result=validation_result
            )
        else:
            print(f"ℹ️ Iteration {iteration}: No history in Ledger to validate (Likely Baseline).")

    # =========================================================
    # PHASE 1: DIAGNOSIS & COGNITION SNAPSHOT
    # =========================================================
    
    # D. Long/Short Term Memory
    # SWAP: Use Ledger Table instead of raw text history
    short_term_history = brain_memory.get_context_for_llm(limit=5)
    
    if iteration != 1:
        long_term_memory = utils.get_long_term_memory(ws, iteration, MODEL_NAME)
    else:
        long_term_memory = "1st Iteration, No Previous History"

    # Build Prompt using our Prompt Builder
    # Note: metrics_json is modified in-place by build_diagnosis_prompt, so we pass a copy if needed
    # but here it's fine. utils.performance_telemetry_as_table inside the builder handles the 
    # filtering to "Stochastic/Shaped" and "Deterministic/Base".
    diag_role, diag_task = prompts.build_diagnosis_prompt(
        Config.analyst_template,
        metrics_json=metrics, 
        current_code=current_code,
        long_term_memory=long_term_memory,
        short_term_history=short_term_history # <--- Injected Validated History
    )
    
    plan_raw = brain.chat(phase_name='diagnosing',
                          system_prompt=diag_role, 
                          user_prompt=diag_task, 
                          parse_json= False,
                          options=Config.analyst_options)

    # =========================================================
    # PHASE 2: Convert Raw Analysis to Structured Output
    # =========================================================
    
    # Build Prompt using our Prompt Builder and the NEW v03 Template
    format_role, format_task = prompts.build_formatter_prompt(Config.formatter_template, plan_raw)

    plan_formatted = brain.chat(phase_name='formatting',
                                system_prompt=format_role,
                                user_prompt=format_task,
                                parse_json=True,
                                options=Config.formatter_options,
                                model_override=ACTIVE_ROSTER["formatter"])

    coder_input_plan = plan_raw # Default fallback
    lesson = None

    if plan_formatted is None:
        format_attempt = 0
        # Check for both the object and the critical 'plan' key
        while (plan_formatted is None or not plan_formatted.get('plan')) and format_attempt < MAX_RETRIES:
            format_attempt +=1
            format_fix_role, format_fix_task = prompts.build_formatter_fix_prompt(Config.formatter_fix_template, plan_raw, json_attempt = plan_formatted)

            plan_formatted = brain.chat(phase_name='formatting_fix',
                                        system_prompt=format_fix_role,
                                        user_prompt=format_fix_task,
                                        parse_json=True,
                                        options=Config.formatter_options,
                                        model_override=ACTIVE_ROSTER["formatter"])
        if plan_formatted is None:
            print("⚠️ Formatting returned None. Falling back to raw diagnosis.")
            print("Parsing failed, No Lesson saved")
        else:
            # Grab the 'plan' key
            coder_input_plan = plan_formatted.get('plan', plan_raw)
            
            # Save lesson (Legacy support, though Ledger is now primary)
            lesson = plan_formatted.get('lesson', None)
            lesson_path = ws.get_path("cognition_lessons", iteration, "lesson.md")
            if lesson:
                with open(lesson_path, "w") as f:
                    f.write(lesson)
                print(f"📝 Lesson saved to {lesson_path}")

    else:
        # Grab the 'plan' key
        coder_input_plan = plan_formatted.get('plan', plan_raw)
        
        # Save lesson (Legacy support, though Ledger is now primary)
        lesson = plan_formatted.get('lesson', None)
        lesson_path = ws.get_path("cognition_lessons", iteration, "lesson.md")
        if lesson:
            with open(lesson_path, "w") as f:
                f.write(lesson)
            print(f"📝 Lesson saved to {lesson_path}")

        # =========================================================
        # PHASE 4: LOG INTENT (Open New Experiment)
        # =========================================================
        # We are about to generate code for the NEXT iteration (Iteration + 1).
        # We must log this intent now so it can be validated in the next run.
        
        next_iter_id = iteration + 1
        hypothesis = plan_formatted.get('hypothesis', plan_formatted.get('lesson', 'Optimization'))
        config_changes = plan_formatted.get('plan', {}) # The coding plan
        
        brain_memory.start_experiment(
            iteration=next_iter_id,
            hypothesis=hypothesis,
            config_changes=config_changes
        )

    # =========================================================
    # PHASE 3: IMPLEMENTATION (WITH SAFETY NET)
    # =========================================================
    
    code_role, code_task = prompts.build_coding_prompt(Config.code_gen_template, coder_input_plan, current_code)

    # Initialize Delta Debugging Trackers / Validation Loop
    previous_attempt_code = current_code
    attempt_num = 0
    is_valid = False

    # Initial Code Generation Attempt
    code_iter_response = brain.chat(phase_name='code_iteration',
                            system_prompt=code_role,
                            user_prompt=code_task,
                            parse_json=False,
                            options=Config.get_coder_options(ACTIVE_ROSTER["coder"]),
                            model_override=ACTIVE_ROSTER["coder"])
    
    clean_code = f"import numpy as np\nimport math\n" 
    # ### Safety Check 1 (Prevent Crash on Initial Generation) ###
    if code_iter_response:
        clean_code += utils.extract_python_code(code_iter_response)
        validator = CodeValidator(clean_code)
        is_valid, feedback = validator.validate_static()
        if is_valid: 
            is_valid, feedback = validator.validate_runtime()
    else:
        print("⚠️ Initial generation failed (Empty Response). Pushing to fix loop...")
        is_valid = False
        feedback = "The model failed to generate any code (Empty Response)."
        # We leave clean_code as just imports so the validator fails immediately below if checked, 
        # but we already set is_valid=False so we go straight to the while loop.

    # --- RETRY LOOP ---
    while not is_valid and attempt_num < MAX_RETRIES:
        attempt_num += 1
        print(f"⚠️ Validation failed (Attempt {attempt_num}). Feedback: {feedback}")
        
        fail_dir = ws.dirs["failed_code"]
        fail_filename = f"fail_{attempt_num:02d}.py"
        fail_path = ws.get_path("failed_code", iteration, fail_filename)
        
        if attempt_num == 1:
            with open(fail_path, "w") as f:
                f.write(f"# Error: {feedback}\n")
                f.write(clean_code)
        else: 
            utils.save_diff(previous_attempt_code, clean_code, iteration, attempt_num, fail_dir)
            with open(fail_path, "w") as f:
                f.write(f"# Error: {feedback}\n")
                f.write(clean_code)
            
        previous_attempt_code = clean_code 
        print(f"🔧 Fixing Code...")
        
        fix_role, fix_task = prompts.build_fix_prompt(Config.code_fix_template,clean_code, feedback)
        
        code_fix_response = brain.chat(phase_name='code_fix',
                                       system_prompt=fix_role,
                                       user_prompt=fix_task,
                                       parse_json=False,
                                       options=Config.get_coder_options(ACTIVE_ROSTER["coder"]),
                                       model_override=ACTIVE_ROSTER["coder"])
        
        # ### Safety Check 2 (Prevent Crash inside Retry Loop) ###
        if code_fix_response is None:
             print(f"⚠️ Attempt {attempt_num} failed: Model returned None. Retrying...")
             # Skip extraction and let the loop spin again. 
             # We rely on 'feedback' remaining the same (or you could update it)
             continue 

        clean_code = f"import numpy as np\nimport math\n" 
        clean_code += utils.extract_python_code(code_fix_response)
        validator = CodeValidator(clean_code)
        is_valid, feedback = validator.validate_static()

        if is_valid:
            is_valid, feedback = validator.validate_runtime()
                

    # ---------------------------------------------------------
    # FINAL SAVE & SAFETY NET
    # ---------------------------------------------------------
    save_path = ws.get_path("code", iteration, "reward.py")
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if is_valid:
        # SUCCESS: Save the new evolution
        header = f"# Generated by {MODEL_NAME} (Iter {iteration}) on {timestamp_str}\n"
        final_content = header + clean_code
        print(f"✅ Code validated and saved.")
        
        # [EVOLUTION]: Save the Diff vs Previous Iteration
        patch_path = ws.get_path("code", iteration, "changes.patch")
        with open(patch_path, "w") as f:
            f.write(utils.generate_patch(current_code, clean_code, "reward.py"))
                
    else:
        # FAILURE: Engage Safety Net
        print(f"🪂 ENGAGING SAFETY NET: Reverting to Iteration {iteration-1}")
        
        header = f"# GENERATION STATUS: FALLBACK (Failed {MAX_RETRIES} attempts)\n"
        header += f"# CLONED FROM: Iteration {iteration-1} | DATE: {timestamp_str}\n"
        final_content = header + current_code 

    with open(save_path, "w") as f:
        f.write(final_content)

    brain.save_report()
    
    elapsed_time = time.perf_counter() - start_time
    print(f"Execution took: {timedelta(seconds=elapsed_time)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()
    
    run_agentic_improvement(args.iteration)