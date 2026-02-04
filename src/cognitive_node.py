import time
import ollama
from typing import Optional, Dict, Any, Union
from src import utils, llm_utils
from src.config import Config
from src.workspace_manager import ExperimentWorkspace


class CognitiveNode:
    def __init__(self, iteration: int, workspace: ExperimentWorkspace, model: str = Config.LLM_MODEL):
        """
        The 'Brain' of the operation. Handles the dirty work of:
        1. Reliable API communication (Retries, Backoffs)
        2. Automatic Logging (JSON, CSV, Markdown)
        3. Basic Response Parsing
        """
        self.iteration = iteration
        self.ws = workspace
        self.model = model
        self.max_retries = 3 
        
        # --- Memory & Logging Setup ---
        # 1. JSON History (Full context replay)
        self.cognition_iter = llm_utils.init_cognition_iteration(iteration, model)
        self.json_path = self.ws.get_path("cognition_json", iteration, "cognition_record.json")
        
        # 2. CSV Telemetry (Stats/Metrics)
        # We use .get() to be safe if the container key is missing
        self.csv_container = self.ws.containers.get("cognition_csv", None)
        
        # 3. Markdown Report Buffer (Human readable summary)
        self.markdown_buffer = []

    def chat(self, 
             phase_name: str, 
             system_prompt: str, 
             user_prompt: str, 
             parse_json: bool = False,
             options: Optional[Dict] = None,
             model_override: Optional[str] = None) -> Union[str, Dict, None]:
        """
        Executes a thought step: Call LLM -> Validate Response -> Log Everything.
        
        Args:
            phase_name: Label for logging (e.g., "Diagnosis", "Coding_Fix_01")
            system_prompt: The fully constructed system instructions.
            user_prompt: The fully constructed user task/context.
            parse_json: If True, attempts to extract and parse JSON from output.
            options: Ollama options dict (temperature, etc.)

        Returns: 
            - Raw string content (if parse_json=False)
            - Parsed Dict (if parse_json=True and success)
            - None (if API failed or JSON parsing failed)
        """
        print(f"🔵 AGENT (Iter {self.iteration}): Phase '{phase_name}' Model {model_override if model_override else self.model}")


        # 1. Execute Robust API Call
        response = self._robust_api_call(system_prompt, user_prompt, options, model_override)
        
        if not response:
            print(f"   💀 Critical Failure in {phase_name}: No response received.")
            return None

        # 2. Parse Content (If requested)
        content = response['message']['content']
        parsed_result = None
        log_content = content # Default to raw text for the Markdown log

        if parse_json:
            parsed_result = utils.extract_json(content)
            if parsed_result:
                # If parsing succeeded, make the log pretty
                log_content = utils.convert_formatter_json_to_markdown(parsed_result)
                print(f"   ✅ JSON Parsed successfully.")
            else:
                # If parsing failed, we treat this as a failure of the "Thought"
                print(f"   ⚠️ JSON Parsing failed for {phase_name}.")
                log_content = f"## Parsing Failed\nRaw Output:\n{content}"
                parsed_result = None 

        # 3. Centralized Logging (The "Exhaust")
        # We log regardless of success/fail to keep a record of the attempt
        self._log_interaction(
            phase_name=phase_name,
            response=response,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            log_content_for_md=log_content,
            options=options,
            model_override=model_override if model_override else self.model
        )

        # Return the parsed dict if JSON was requested, otherwise raw text
        return parsed_result if parse_json else content

    def save_report(self):
        """Flushes the internal markdown buffer to the workspace file."""
        if self.markdown_buffer:
            utils.save_cognition_markdown(self.ws, self.iteration, self.markdown_buffer)

    def _robust_api_call(self, system_prompt: str, user_prompt: str, options: Dict,model_override: Optional[str] = None) -> Optional[Any]:
        """Internal loop handling retries, timeouts, and empty responses."""
        # Check who is speaking, needed for MOE runs
        active_model = model_override if model_override else self.model
        # print(f"Model listed as 'active_model' inside of '_robust_api_call {active_model}'") # For debugging
        for attempt in range(1, self.max_retries + 1):
            try:
                response = ollama.chat(
                    model=active_model,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ],
                    options=options
                )
                
                # Validation: Check for empty content
                if not response or 'message' not in response or not response['message']['content'].strip():
                    print(f"   ⚠️ Attempt {attempt}: Received empty response from {active_model}. Retrying...")
                    time.sleep(1)
                    continue
                    
                return response

            except Exception as e:
                print(f"   ⚠️ Attempt {attempt} Error: {e}")
                # Exponential Backoff strategy: Wait 2s, 4s, 8s...
                time.sleep(2 ** attempt)
        
        return None

    def _log_interaction(self, phase_name, response, system_prompt, user_prompt, log_content_for_md, options,model_override: Optional[str] = None):
        """Encapsulates all side-effect logging to clean up the main logic."""
        run_id = f"Iter_{self.iteration:02d}_{phase_name}"
        active_model = model_override if model_override else self.model
        # print(f"Model listed as 'active_model' inside of '_log_iteration' {active_model}'") # For debugging
        # A. JSON History (Detailed Replay Data)
        llm_utils.add_cognition_call(
            cognition_iter=self.cognition_iter,
            response=response,
            run_id=run_id,
            phase=phase_name,
            system_role=system_prompt,
            user_task=user_prompt,
            options=options,
            model_override= active_model
        )
        # Checkpoint the JSON immediately in case of crash later
        llm_utils.save_cognition_iteration(self.cognition_iter, self.json_path)

        # B. CSV Telemetry (High-level Metrics)
        if self.csv_container:
            llm_utils.append_chatresponse_row(
                csv_path=self.csv_container,
                model_name=active_model,
                response=response,
                run_id=run_id,
                iteration=self.iteration,
                phase=phase_name,
                prompt_type=phase_name,
                cognition_path=self.json_path
            )

        # C. Markdown Buffer (Human Readable)
        # We append the prompts AND the result for context
        self.markdown_buffer.append((f"Phase: {phase_name} [System] {active_model}", system_prompt))
        self.markdown_buffer.append((f"Phase: {phase_name} [User] {active_model}", user_prompt))
        if response.message.thinking is not None and response.message.thinking != "":
            self.markdown_buffer.append((f"Phase: {phase_name} [Thinking Trace] {active_model}", response.message.thinking ))
        self.markdown_buffer.append((f"Phase: {phase_name} [Output] {active_model}", log_content_for_md))