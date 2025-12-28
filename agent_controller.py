import argparse
import ollama
from datetime import datetime 
import warnings
import os
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

# -- PROJECT IMPORTS --
import prompts  
from src.workspace_manager import ExperimentWorkspace
from src.code_validation import CodeValidator
from src import utils
from src.config import Config

MODEL_NAME = Config.LLM_MODEL
MAX_RETRIES = 5 

def run_agentic_improvement(iteration):
    # 1. Initialize Workspace
    ws = ExperimentWorkspace()
    print(f"🔵 AGENT (Iter {iteration}): Active in {ws.model_root_path}")
    
    # 2. Load Context (The "Eyes" of the Agent)
    # A. Metrics from the run just finished
    metrics = ws.load_metrics(iteration)
    if not metrics:
        print(f"❌ No metrics found for Iteration {iteration}. Cannot proceed.")
        return

    # B. The code that produced those metrics (Previous Iteration)
    if iteration == 1:
        prev_code_path = "seed_reward.py"
    else:
        prev_code_path = ws.get_path("code", iteration - 1, "reward.py")
    
    with open(prev_code_path, "r") as f:
        current_code = f.read()

    # C. Training Dynamics (Tensorboard Summary)
    tb_dir = ws.dirs["tensorboard"]
    training_summary = utils.summarize_training_log(str(tb_dir))

    # D. Long/Short Term Memory
    short_term_history = utils.get_recent_history(ws, iteration)
    long_term_memory = utils.get_long_term_memory(ws, iteration, MODEL_NAME)


    # =========================================================
    # PHASE 1: DIAGNOSIS & COGNITION SNAPSHOT
    # =========================================================
    print("🔵 AGENT: Phase 1 - Diagnosing & Planning...")
    
    # Build Prompt using our new Clean Builder
    diag_role, diag_task = prompts.build_diagnosis_prompt(
        metrics=metrics,
        current_code=current_code,
        training_summary=training_summary,
        long_term_memory=long_term_memory,
        short_term_history=short_term_history
    )
    # This captures exactly what the LLM saw before it made decisions
    cognition_snapshot = {
        "timestamp": datetime.now().isoformat(),
        "iteration": iteration,
        "input_context": {
            "metrics": metrics,
            "training_summary": training_summary,
            "memory_context": {"short": short_term_history, "long": long_term_memory}
        },
        "prompts": {
            "diagnosis_roles": diag_role,
            "diagnosis_task": diag_task,
        }
    }
    

    try:
        response = ollama.chat(model=MODEL_NAME, messages=[
            {'role': 'system', 'content': diag_role},
            {'role': 'user', 'content': diag_task}
        ])
        diagnosis_plan = response['message']['content']
        
        # Save the plan for human review
        plan_path = ws.get_path("cognition", iteration, "plan.md")
        with open(plan_path, "w") as f:
            f.write(diagnosis_plan)
        print(f"📝 Plan saved to {plan_path}")
            
    except Exception as e:
        print(f"❌ Phase 1 Error: {e}")
        return

    # =========================================================
    # PHASE 2: IMPLEMENTATION (WITH SAFETY NET)
    # =========================================================
    print("🔵 AGENT: Phase 2 - Writing Code...")
    
    code_role, code_task = prompts.build_coding_prompt(diagnosis_plan, current_code)

    # adding prompts to cognition snapshot for visability, analysis and debugging later
    cognition_snapshot["prompts"].update({
        "code_roles": code_role, 
        "code_task": code_task
    })

    # Initialize Delta Debugging Trackers / Validation Loop
    previous_attempt_code = current_code
    attempt_num = 0
    is_valid = False

    # Initial Code Generation Attempt
    try:
        response2 = ollama.chat(model=MODEL_NAME, messages=[
            {'role': 'system', 'content': code_role},
            {'role': 'user', 'content': code_task}
        ])
        clean_code = utils.extract_python_code(response2['message']['content'])
        
        validator = CodeValidator(clean_code)
        is_valid, feedback = validator.validate_static()
        if is_valid: 
            is_valid, feedback = validator.validate_runtime()

    except Exception as e:
        # Catch network/parsing errors, treat as invalid to trigger loop
        feedback = f"Generation Error: {str(e)}"
        is_valid = False
        print(f"❌ Phase 2 Initial Error: {e}")

    # --- RETRY LOOP ---
    # This runs if validation failed OR if an exception occurred above
    while not is_valid and attempt_num < MAX_RETRIES:
        attempt_num += 1
        print(f"⚠️ Validation failed (Attempt {attempt_num}). Feedback: {feedback}")
        
        # Save to the 'failed_code' directory defined in Workspace
        fail_dir = ws.dirs["failed_code"]
        fail_filename = f"fail_{attempt_num:02d}.py"
        fail_path = ws.get_path("failed_code", iteration, fail_filename)
        
        if attempt_num == 1:
            with open(fail_path, "w") as f:
                f.write(f"# Error: {feedback}\n")
                f.write(clean_code)
        else: 
            # Subsequent failures: Save diffs for Debugging Delta Analysis
            utils.save_diff(previous_attempt_code, clean_code, iteration, attempt_num, fail_dir)
            with open(fail_path, "w") as f:
                f.write(f"# Error: {feedback}\n")
                f.write(clean_code)
            
        previous_attempt_code = clean_code 
        print(f"🔧 Fixing Code...")
        
        fix_role, fix_task = prompts.build_fix_prompt(clean_code, feedback)
        try:
            fix_response = ollama.chat(model=MODEL_NAME, messages=[
                {'role': 'system', 'content': fix_role},
                {'role': 'user', 'content': fix_task}
            ])
            
            # Log prompts for debugging
            cognition_snapshot["prompts"].update({
                f"fix_roles_{attempt_num}": fix_role, 
                f"fix_task_{attempt_num}": fix_task,
                f"fix_response_{attempt_num}": fix_response['message']['content']
            })
            
            clean_code = utils.extract_python_code(fix_response['message']['content'])
            validator = CodeValidator(clean_code)
            is_valid, feedback = validator.validate_static()
            if is_valid:
                is_valid, feedback = validator.validate_runtime()
                
        except Exception as e:
            feedback = str(e)
            print(f"❌ Phase 2 Retry Error: {e}")

    # ---------------------------------------------------------
    # FINAL SAVE & SAFETY NET
    # ---------------------------------------------------------
    save_path = ws.get_path("code", iteration, "reward.py")
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    generation_status = "success"
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
        generation_status = "fallback"
        
        header = f"# GENERATION STATUS: FALLBACK (Failed {MAX_RETRIES} attempts)\n"
        header += f"# CLONED FROM: Iteration {iteration-1} | DATE: {timestamp_str}\n"
        final_content = header + current_code # Re-save the old code

    # Write the file (Good or Bad, we always write something)
    with open(save_path, "w") as f:
        f.write(final_content)

    # ---------------------------------------------------------
    # 5. UPDATE SCOREBOARD
    # ---------------------------------------------------------
    # Inject status into metrics for the CSV
    metrics["generation_status"] = generation_status
    utils.update_campaign_summary(ws, iteration, metrics)
    ws.save_metrics(iteration, cognition_snapshot) # Saves as JSON in telemetry/raw
    utils.save_readable_context(ws, iteration, cognition_snapshot['input_context']) # . Save Markdown (For Humans/Debugging) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()
    
    run_agentic_improvement(args.iteration)