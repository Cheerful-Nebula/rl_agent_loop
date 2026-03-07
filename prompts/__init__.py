# This allows: from prompts import build_diagnosis_prompt
from .builders import build_strategist_prompt
from .builders import build_organizer_prompt
from .builders import build_lead_prompt
from .builders import build_dispatcher_prompt
from .builders import build_coder_prompt
from .builders import build_validator_prompt

# This allows: from prompts import load_template (if needed)
from .loader import load_template
