import os
from pathlib import Path

# Define the base directory for prompts relative to this file
PROMPT_DIR = Path(__file__).parent

def _load_file(subcategory: str, filename: str) -> str:
    """
    Helper to read text from prompts/subcategory/filename.md
    
    Args:
        subcategory: 'roles' or 'tasks'
        filename: name of the file without extension
    """
    # The '/' operator here works exactly like os.path.join()
    # It handles Windows/Linux path separators automatically.
    path = PROMPT_DIR / subcategory / f"{filename}.md"
    
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def get_role(role_name: str) -> str:
    """
    Loads a static persona from prompts/roles/.
    Usage: system_msg = prompts.get_role('rl_expert')
    """
    return _load_file("roles", role_name)

def get_task(task_name: str, **kwargs) -> str:
    """
    Loads a dynamic task from prompts/tasks/ and injects variables.
    Usage: user_msg = prompts.get_task('generate_code', plan=my_plan)
    """
    raw_text = _load_file("tasks", task_name)
    try:
        # This fills in {plan}, {error}, etc. using the arguments you provide
        return raw_text.format(**kwargs)
    except KeyError as e:
        # Helpful error if you forget to pass a variable needed by the prompt
        raise ValueError(f"Missing argument for prompt '{task_name}': {e}")