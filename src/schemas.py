from pydantic import BaseModel, Field
from typing import Literal
from pydantic import BaseModel
from typing import Optional, Dict, Union, Any
class VlmDiagnosis(BaseModel):
    run_id: str
    iteration: int
    collapse_likelihood: float = Field(ge=0.0, le=1.0)
    learning_progress: Literal["stalled", "improving", "unstable"]
    suspected_issue: Literal[
        "entropy_collapse",
        "lr_too_high",
        "lr_too_low",
        "reward_misaligned",
        "exploration_issue",
        "credit_assignment",
        "other",
    ]
    suggested_actions: list[str]
    evidence_markdown: str

class FormatterOutput(BaseModel):
    """
    Defines the contract for the Formatter LLM output.
    """
    analysis: str
    plan: str
    lesson: str
    hyperparameters: Union[str, Dict[str, Any]] # Can be a text summary or strict dict