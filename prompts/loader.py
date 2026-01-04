import os
from pathlib import Path

# Define the base directory relative to THIS file
PROMPT_DIR = Path(__file__).parent

def load_template(category: str, subcategory: str,filename: str) -> str:
    """
    Reads text from prompts/{category}/{subcategory}/{filename}.md
    """
    path = PROMPT_DIR / category / subcategory/ f"{filename}.md"
    
    if not path.exists():
        raise FileNotFoundError(f"❌ Prompt template not found: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()