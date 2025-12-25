import os
from pathlib import Path

# Define the base directory relative to THIS file
PROMPT_DIR = Path(__file__).parent

def load_template(category: str, filename: str) -> str:
    """
    Reads text from prompts/{category}/{filename}.md
    """
    path = PROMPT_DIR / category / f"{filename}.md"
    
    if not path.exists():
        raise FileNotFoundError(f"❌ Prompt template not found: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()