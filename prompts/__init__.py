# This allows: from prompts import build_diagnosis_prompt
from .builders import build_diagnosis_prompt
from .builders import build_coding_prompt
from .builders import build_fix_prompt

# This allows: from prompts import load_template (if needed)
from .loader import load_template