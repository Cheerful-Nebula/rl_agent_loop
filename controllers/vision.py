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
from src.llm_utils import *


MODEL_NAME = Config.LLM_MODEL
MAX_RETRIES = 5 



def run_agentic_improvement(iteration):
    # 1. Initialize Workspace
    ws = ExperimentWorkspace()
    cognition_json_path = ws.get_path("cognition_json", iteration, "cognition_record.json")
    cognition_iter = init_cognition_iteration(iteration=iteration, model_name=MODEL_NAME)
    if iteration == 1:
        upsert_model_metadata_row(ws.containers["model_metadata"], MODEL_NAME)
    
    print(f"🔵 AGENT (Iter {iteration}): Active in {ws.model_root_path}")
    # 2. Load Context
    # A. Metrics from the run just finished
    metrics = ws.load_metrics(iteration)
    if not metrics:
        print(f"❌ No metrics found for Iteration {iteration}. Cannot proceed.")
        return

    # B. The code that produced those metrics (Previous Iteration)
    if iteration == 1:
        print("🌱 Loading Virtual Seed (Template) for Iteration 1 analysis.")
        current_code = Config.INITIAL_TEMPLATE
    else:
        prev_code_path = ws.get_path("code", iteration - 1, "reward.py")
        with open(prev_code_path, "r") as f:
            current_code = f.read()


    # C. Training Dynamics (Tensorboard Summary)
    logger_dir = ws.dirs["telemetry_raw"]

    # D. Long/Short Term Memory
    short_term_history = utils.get_recent_history(ws, iteration)
    long_term_memory = utils.get_long_term_memory(ws, iteration, MODEL_NAME)


    # =========================================================
    # PHASE 1: DIAGNOSIS & COGNITION SNAPSHOT
    # =========================================================
    print("🔵 AGENT: Phase 1 - Diagnosing & Planning...")
    
    # Build Prompt using our Prompt Builder
    diag_role, diag_task = prompts.build_diagnosis_prompt(
        Config.analyst_template,
        config_list=metrics,
        current_code=current_code,
        long_term_memory=long_term_memory,
        short_term_history=short_term_history,
    )
    
    try:
        response = ollama.chat(
            model=MODEL_NAME, 
            messages=[
                {'role': 'system', 'content': diag_role},
                {'role': 'user', 'content': diag_task},
                {"images": ws.dirs["plots"] / f"iter{iteration:02d}_training_curves.png"}],
                options = Config.analyst_options)
        
        diagnosis_plan = response['message']['content']
 
        
 
    except Exception as e:
        print(f"❌ Phase 1 Error: {e}")
        return

    # Saving input prompts and responses as easy to read Markdown documents for later
    cognition_list = []
    cognition_list.append((f"LLM Input: `diag_role` using {Config.analyst_template[0]}.md",diag_role))
    cognition_list.append((f"LLM Input: `diag_task` using {Config.analyst_template[1]}.md",diag_task))
    cognition_list.append(("LLM Output: plan",response['message']['content']))

    # Save ChatResponse Data to CSV
    append_chatresponse_row(
        csv_path =ws.containers['cognition_csv'], 
        model_name=MODEL_NAME, 
        response=response,
        run_id = f"Iter_{iteration:02d}_diag", 
        iteration=iteration, 
        phase="Phase_1", 
        prompt_type="diagnosis",
        prompt_template_roles=f"{Config.analyst_template[0]}.md",
        prompt_template_tasks=f"{Config.analyst_template[1]}.md",
        cognition_path=cognition_json_path)

    # Saves full prompts/responses of LLM to JSON, One JSON per iteration
    add_cognition_call(
        cognition_iter=cognition_iter,
        response=response,
        run_id=f"Iter_{iteration:02d}_diag",
        phase="diagnosis",
        system_role=diag_role,
        user_task=diag_task,
        options=Config.analyst_options,
        prompt_template_roles=f"{Config.analyst_template[0]}.md",
        prompt_template_tasks=f"{Config.analyst_template[1]}.md",)
    # =========================================================
    # PHASE 2: IMPLEMENTATION (WITH SAFETY NET)
    # =========================================================
    print("🔵 AGENT: Phase 2 - Writing Code...")
    
    code_role, code_task = prompts.build_coding_prompt(Config.code_gen_template, diagnosis_plan, current_code)

    # Initialize Delta Debugging Trackers / Validation Loop
    previous_attempt_code = current_code
    attempt_num = 0
    is_valid = False

    # Initial Code Generation Attempt
    try:
        response2 = ollama.chat(
            model=MODEL_NAME,
            messages=[
            {'role': 'system', 'content': code_role},
            {'role': 'user', 'content': code_task}],
            options = Config.coder_options)
        
        # Save ChatResponse Data to CSV
        append_chatresponse_row(
            csv_path =ws.containers['cognition_csv'], 
            model_name=MODEL_NAME, 
            response=response2,
            run_id = f"Iter_{iteration:02d}_code", 
            iteration=iteration, 
            phase="Phase_2", 
            prompt_type="code_generation",
            prompt_template_roles=f"{Config.code_gen_template[0]}.md",
            prompt_template_tasks=f"{Config.code_gen_template[1]}.md",
            cognition_path=cognition_json_path)

        # Saves full prompts/responses of LLM to JSON, One JSON per iteration
        add_cognition_call(
            cognition_iter=cognition_iter,
            response=response2,
            run_id=f"Iter_{iteration:02d}_code",
            phase="code_generation",
            system_role=code_role,
            user_task=code_task,
            options=Config.coder_options,
            prompt_template_roles=f"{Config.code_gen_template[0]}.md",
            prompt_template_tasks=f"{Config.code_gen_template[1]}.md")

        # Saving input prompts and responses as easy to read Markdown documents for later
        cognition_list.append((f"LLM Input: `code_role` from {Config.code_gen_template[0]}.md", code_role))
        cognition_list.append((f"LLM Input: `code_task` from {Config.code_gen_template[1]}.md", code_task))
        cognition_list.append(("LLM Output: `code_response`", response2['message']['content']))

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
        
        fix_role, fix_task = prompts.build_fix_prompt(Config.code_fix_template,clean_code, feedback)
        try:
            response3 = ollama.chat(
                model=MODEL_NAME,
                messages=[
                {'role': 'system', 'content': fix_role},
                {'role': 'user', 'content': fix_task}],
                options = Config.coder_options)

            # Save ChatResponse Data to CSV
            append_chatresponse_row(
                csv_path =ws.containers['cognition_csv'], 
                model_name=MODEL_NAME, 
                response=response3,
                run_id = f"Iter_{iteration:02d}_code_fix{attempt_num:02d}", 
                iteration=iteration, 
                phase="Phase_2", 
                prompt_type="code_fix",
                prompt_template_roles=f"{Config.code_fix_template[0]}.md",
                prompt_template_tasks=f"{Config.code_fix_template[1]}.md",
                cognition_path=cognition_json_path)

            # Saves full prompts/responses of LLM to JSON, One JSON per iteration
            add_cognition_call(
                cognition_iter=cognition_iter,
                response=response3,
                run_id=f"Iter_{iteration:02d}_code_fix{attempt_num:02d}",
                phase="code_fix",
                system_role=fix_role,
                user_task=fix_task,
                options=None, #diag_options,
                prompt_template_roles=f"{Config.code_fix_template[0]}.md",
                prompt_template_tasks=f"{Config.code_fix_template[1]}.md",)
            # Saving input prompts and responses as easy to read Markdown documents for later
            cognition_list.append((f"LLM Input: `fix_role_attempt{attempt_num:02d}` using {Config.code_fix_template[0]}.md", fix_role))
            cognition_list.append((f"LLM Input: `fix_task_attempt{attempt_num:02d}` using {Config.code_fix_template[1]}.md", fix_task))    
            cognition_list.append(("LLM Output: `fix_response`", response3['message']['content']))

            clean_code = utils.extract_python_code(response3['message']['content'])
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
    # Save cognition history in easy read markdown documents 
    utils.save_cognition_markdown(ws,iteration, cognition_list)
    # Save the per-iteration JSON of LLM cognition
    save_cognition_iteration(cognition_iter, cognition_json_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()
    
    run_agentic_improvement(args.iteration)