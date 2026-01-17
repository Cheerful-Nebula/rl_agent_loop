# This allows: from prompts import build_diagnosis_prompt
from .builders import build_diagnosis_prompt
from .builders import build_coding_prompt
from .builders import build_fix_prompt
from .builders import build_initial_shaping_prompt
from .builders import build_formatter_prompt


# This allows: from prompts import load_template (if needed)
from .loader import load_template