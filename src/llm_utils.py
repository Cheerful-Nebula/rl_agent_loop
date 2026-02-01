import os
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import ollama
from ollama import ChatResponse

# -- Custom Imports -- 
from src.config import Config
# This module is dedicated to collecting,storing,viewing,setting the LLM meta-data and cognition data

def get_thinking_flag(MODEL_NAME : str) -> bool|str:
    if MODEL_NAME.startswith(('deepseek-r1','qwen3','magistral')):
        thinking_flag = True
    elif MODEL_NAME.startswith('gpt_oss'):
        thinking_flag = Config.gpt_think_level
    else: thinking_flag = False
    return thinking_flag
##########################################################################################
# for on-demand Markdown view from per-iteration cognition JSON for qualitative review.
##########################################################################################
def cognition_json_to_markdown(json_path: str, md_path: str) -> None:
    """
    Read one iteration's cognition JSON and write a human-readable Markdown summary.

    - Groups by call (diagnosis, code, fixes).
    - Renders newlines properly for prompts and responses.
    """
    data: Dict[str, Any] = json.loads(Path(json_path).read_text(encoding="utf-8"))

    iteration = data.get("iteration", "UNKNOWN")
    model_name = data.get("model_name", "UNKNOWN")
    calls = data.get("calls", [])

    lines = []
    lines.append(f"# Iteration {iteration} – Cognition Log\n")
    lines.append(f"- Model: `{model_name}`\n")
    lines.append(f"- Number of LLM calls: {len(calls)}\n")

    for idx, call in enumerate(calls, start=1):
        run_id = call.get("run_id", f"call_{idx}")
        phase = call.get("phase", "unknown")
        role_system = call.get("role_system", "")
        role_user = call.get("role_user", "")
        response = call.get("response_content", "")
        options = call.get("options", {}) or {}
        created_at = (call.get("timestamps") or {}).get("created_at", "")

        lines.append("\n---\n")
        lines.append(f"## {idx}. {phase} ({run_id})\n")
        if created_at:
            lines.append(f"- **Time**: {created_at}\n")
        if options:
            lines.append(
                f"- **Options**: "
                f"T={options.get('temperature')}, "
                f"top_p={options.get('top_p')}, "
                f"top_k={options.get('top_k')}, "
                f"num_ctx={options.get('num_ctx')}, "
                f"num_predict={options.get('num_predict')}\n"
            )

        # System prompt
        if role_system:
            lines.append("\n### System prompt\n")
            lines.append("```markdown\n")
            lines.append(role_system)
            lines.append("\n```\n")

        # User prompt
        if role_user:
            lines.append("\n### User task\n")
            lines.append("```markdown\n")
            lines.append(role_user)
            lines.append("\n```\n")

        # Response
        if response:
            lines.append("\n### LLM response\n")
            lines.append("```markdown\n")
            lines.append(response)
            lines.append("\n```\n")

    Path(md_path).write_text("\n".join(lines), encoding="utf-8")


#################################################################################
# functions for collecting ollama.chat() per-call data
# Saves to CSV file, One CSV file per LLM , One row per ollama.chat() call
################################################################################

def _to_plain(obj: Any) -> Any:
    """Convert Pydantic-style objects to plain Python structures."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def extract_chat_metrics(
    model_name: str,
    response: ChatResponse,
    *,
    run_id: Optional[str] = None,
    iteration: Optional[int] = None,
    phase: Optional[str] = None,
    prompt_type: Optional[str] = None,
    prompt_template_roles: Optional[str] = None,
    prompt_template_tasks: Optional[str] = None,
    cognition_path: Optional[str] = None,
    prompt_preview_len: int = 160,
    response_preview_len: int = 160,
) -> Dict[str, Any]:
    """
    Pull out the most useful info from a ChatResponse for logging to CSV.

    - Assumes Ollama Python library's ChatResponse structure.
    - Keeps scalars and short previews only (full text lives in cognition JSON/MD).
    """
    r = _to_plain(response)  # ChatResponse -> dict-like

    message = r.get("message") or {}
    options = r.get("options") or {}

    created_at = r.get("created_at")
    total_duration = r.get("total_duration")          # ns, if present
    load_duration = r.get("load_duration")            # ns, if present
    prompt_eval_count = r.get("prompt_eval_count")
    prompt_eval_duration = r.get("prompt_eval_duration")
    eval_count = r.get("eval_count")
    eval_duration = r.get("eval_duration")

    content = message.get("content") or ""
    role = message.get("role")
    tool_calls = message.get("tool_calls") or []

    # Thinking-capable models expose a "thinking" field on the message. [web:96][web:99]
    thinking = message.get("thinking")
    has_thinking = thinking is not None and thinking != ""

    # Short previews for CSV readability.
    prompt_preview = None
    # In your loop, the "user task" prompt is what you pass as the user message.
    # Here we only see the *response* message; prompt preview comes from caller if needed.
    # For now, we treat "content" as response_preview and leave prompt_preview to be
    # supplied as an extra column from the caller (or derive from cognition JSON later).
    response_preview = content[:response_preview_len] if content else None
    thinking_preview = thinking[:prompt_preview_len] if isinstance(thinking, str) else None

    # Derive prompt_type default from phase if not explicitly given.
    if prompt_type is None:
        prompt_type = phase

    row: Dict[str, Any] = {
        # Identifiers / context
        "run_id": run_id,
        "iteration": iteration,
        "phase": phase,
        "prompt_type": prompt_type,
        "model_name": model_name,
        "cognition_path": cognition_path,

        # Prompt template references (filenames or relative paths)
        "prompt_template_roles": prompt_template_roles,
        "prompt_template_tasks": prompt_template_tasks,

        # Core message info
        "role": role,  # usually "assistant"
        "response_preview": response_preview,
        "has_tool_calls": bool(tool_calls),
        "num_tool_calls": len(tool_calls),

        # Thinking / reasoning traces
        "has_thinking": has_thinking,
        "thinking_preview": thinking_preview,

        # Options actually used (temperature, top_p, etc.) [web:117][web:53]
        "opt_temperature": options.get("temperature"),
        "opt_top_p": options.get("top_p"),
        "opt_top_k": options.get("top_k"),
        "opt_repeat_penalty": options.get("repeat_penalty"),
        "opt_num_ctx": options.get("num_ctx"),
        "opt_num_predict": options.get("num_predict"),
        "opt_think": options.get("think"),

        # Low-level timing & token stats if available [web:35][web:111]
        "created_at": created_at,
        "total_duration_ns": total_duration,
        "load_duration_ns": load_duration,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration_ns": prompt_eval_duration,
        "eval_count": eval_count,
        "eval_duration_ns": eval_duration,
    }

    return row


def append_chatresponse_row(
    csv_path: str,
    model_name: str,
    response: ChatResponse,
    *,
    run_id: Optional[str] = None,
    iteration: Optional[int] = None,
    phase: Optional[str] = None,
    prompt_type: Optional[str] = None,
    prompt_template_roles: Optional[str] = None,
    prompt_template_tasks: Optional[str] = None,
    cognition_path: Optional[str] = None,
) -> None:
    """
    Append one row of ChatResponse metrics to `csv_path`.

    Arguments beyond `model_name` and `response` let you connect this usage
    record to your wider experiment structure:

    - run_id: e.g. "Iter_003_diag", "Iter_003_code_fix_01"
    - iteration: integer iteration index
    - phase: "diagnosis" | "initial_code" | "fix_code" | ...
    - prompt_type: optional; defaults to phase
    - prompt_template_roles / tasks: filenames of the Markdown templates
    - cognition_path: path to the per-iteration cognition JSON
    """
    row = extract_chat_metrics(
        model_name=model_name,
        response=response,
        run_id=run_id,
        iteration=iteration,
        phase=phase,
        prompt_type=prompt_type,
        prompt_template_roles=prompt_template_roles,
        prompt_template_tasks=prompt_template_tasks,
        cognition_path=cognition_path,
    )

    fieldnames = [
        # IDs / experiment context
        "run_id",
        "iteration",
        "phase",
        "prompt_type",
        "model_name",
        "cognition_path",

        # Prompt template references
        "prompt_template_roles",
        "prompt_template_tasks",

        # Message info
        "role",
        "response_preview",
        "has_tool_calls",
        "num_tool_calls",

        # Thinking / reasoning
        "has_thinking",
        "thinking_preview",

        # Options
        "opt_temperature",
        "opt_top_p",
        "opt_top_k",
        "opt_repeat_penalty",
        "opt_num_ctx",
        "opt_num_predict",
        "opt_think",

        # Timing / token stats
        "created_at",
        "total_duration_ns",
        "load_duration_ns",
        "prompt_eval_count",
        "prompt_eval_duration_ns",
        "eval_count",
        "eval_duration_ns",
    ]

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        safe_row = {k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames}
        writer.writerow(safe_row)

#########################################################################################
# Per-iteration cognition tracking helpers
#########################################################################################
# ---------------------- Example use below ------------------------------------
# At start of iteration
# cognition_iter = init_cognition_iteration(iteration=iteration, model_name=MODELNAME)
# cognition_path = ws.getpath_cognition(iteration, "cognition.json")  # or similar

# Phase 1: Diagnosis
# diag_role, diag_task = prompts.builddiagnosisprompt(
#     metrics=metrics,
#     currentcode=currentcode,
#     trainingsummary=trainingsummary,
#     longtermmemory=longtermmemory,
#     shorttermhistory=shorttermhistory,
# )
# diag_options = {"temperature": 0.7, "top_p": 0.9, "think": True}
# 
# diag_response: ChatResponse = ollama.chat(
#     model=MODELNAME,
#     messages=[
#         {"role": "system", "content": diag_role},
#         {"role": "user", "content": diag_task},
#     ],
#     options=diag_options,
# )
# 
# run_id_diag = f"Iter_{iteration:03d}_diag"
# 
# add_cognition_call(
#     cognition_iter=cognition_iter,
#     response=diag_response,
#     run_id=run_id_diag,
#     phase="diagnosis",
#     system_role=diag_role,
#     user_task=diag_task,
#     options=diag_options,
#     prompt_template_roles="roles/rl_researcher.md",
#     prompt_template_tasks="tasks/diagnose_agent.md",
# )

# ... later, Phase 2 initial code, Phase 2 fix loops: call add_cognition_call(...)
# with appropriate run_id (Iter_XXX_code_0, Iter_XXX_code_fix_01...), phase, roles/tasks, options.

# At end of iteration
# save_cognition_iteration(cognition_iter, cognition_path)

#########################################################################################

def _to_plain(obj: Any) -> Any:
    """Convert Pydantic-style objects to plain Python structures."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def init_cognition_iteration(
    iteration: int,
    model_name: str,
) -> Dict[str, Any]:
    """
    Initialize the in-memory structure for one iteration's cognition log.
    """
    return {
        "iteration": iteration,
        "model_name": model_name,
        "calls": [],  # list[Dict]
    }


def add_cognition_call(
    cognition_iter: Dict[str, Any],
    *,
    response: ChatResponse,
    run_id: str,
    phase: str,
    system_role: str,
    user_task: str,
    options: Optional[Dict[str, Any]] = None,
    prompt_template_roles: Optional[str] = None,
    prompt_template_tasks: Optional[str] = None,
) -> None:
    """
    Append one LLM call record (including thinking trace) to the iteration log.

    - `system_role` and `user_task` are the rendered prompt strings you passed.
    - `options` should mirror what you passed to `ollama.chat` (temperature, think, etc.).
    - `prompt_template_roles` / `prompt_template_tasks` are template filenames/paths.
    """
    data = _to_plain(response)
    message: Dict[str, Any] = data.get("message") or {}

    content: str = message.get("content") or ""
    thinking: Optional[str] = message.get("thinking")  # reasoning trace, if enabled 
    tool_calls: List[Dict[str, Any]] = message.get("tool_calls") or []

    created_at = data.get("created_at")
    total_duration = data.get("total_duration")
    load_duration = data.get("load_duration")
    prompt_eval_count = data.get("prompt_eval_count")
    prompt_eval_duration = data.get("prompt_eval_duration")
    eval_count = data.get("eval_count")
    eval_duration = data.get("eval_duration")

    opts = options or {}

    call_record: Dict[str, Any] = {
        "run_id": run_id,
        "phase": phase,
        "model_name": cognition_iter.get("model_name"),

        # Prompt templates and rendered text
        "prompt_template_roles": prompt_template_roles,
        "prompt_template_tasks": prompt_template_tasks,
        "role_system": system_role,
        "role_user": user_task,

        # Response and reasoning
        "response_role": message.get("role"),
        "response_content": content,
        "thinking_trace": thinking,
        "has_thinking": thinking is not None and thinking != "",
        "tool_calls": tool_calls,

        # Options / generation settings
        "options": {
            "temperature":       opts.get("temperature"),
            "top_p":             opts.get("top_p"),
            "top_k":             opts.get("top_k"),
            "repeat_penalty":    opts.get("repeat_penalty"),
            "num_ctx":           opts.get("num_ctx"),
            "num_predict":       opts.get("num_predict"),
            "think":             opts.get("think"),
        },

        # Timing / eval stats (if provided by Ollama) [web:35][web:3]
        "runtime": {
            "created_at":             created_at,
            "total_duration_ns":      total_duration,
            "load_duration_ns":       load_duration,
            "prompt_eval_count":      prompt_eval_count,
            "prompt_eval_duration_ns": prompt_eval_duration,
            "eval_count":             eval_count,
            "eval_duration_ns":       eval_duration,
        },
    }

    cognition_iter["calls"].append(call_record)


def save_cognition_iteration(
    cognition_iter: Dict[str, Any],
    path: str,
) -> None:
    """
    Persist one iteration's cognition log (with thinking traces) to JSON.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cognition_iter, ensure_ascii=False, indent=2), encoding="utf-8")

######################################################################################################

########################################################
# Collecting LLM metadata or usage data, saved to a CSV
#######################################################
# -- collecting meta-data, run once per LLM 
# usage: 
# upsert_model_metadata_row("model_metadata.csv", "llama3.2:8b-instruct-q5_K_M")
# upsert_model_metadata_row("model_metadata.csv", "qwen2.5:7b-instruct-q4_0")

def _safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _parse_name_tag(full: str):
    # full looks like "llama3.2:8b-instruct-q5_K_M" or "qwen2.5:14b"
    if ":" in full:
        name, tag = full.split(":", 1)
    else:
        name, tag = full, ""
    return name, tag


def _derive_fields_from_tag(tag: str) -> Dict[str, Optional[str]]:
    # very heuristic, but good enough for later analysis
    parts = tag.split("-") if tag else []
    size = None
    role = None
    quant = None

    for p in parts:
        if p.lower().endswith("b"):
            size = p
        elif p.lower() in {"base", "instruct", "chat"}:
            role = p
        elif p.lower().startswith("q"):
            quant = p

    return {
        "size_tag": size,
        "role": role,
        "quant_scheme": quant,
    }


def _derive_capabilities(details: Dict[str, Any]) -> Dict[str, Any]:
    families = details.get("families") or []
    if isinstance(families, str):
        families = [families]

    fam_lower = [f.lower() for f in families]

    is_multimodal = "clip" in fam_lower or "vision" in fam_lower
    base_family = None
    for cand in ["llama", "qwen", "mistral", "gemma", "phi", "mixtral"]:
        if cand in fam_lower:
            base_family = cand
            break

    return {
        "families": ",".join(families),
        "base_family": base_family,
        "is_multimodal": is_multimodal,
    }


def _parse_parameters_from_modelfile(modelfile: str) -> Dict[str, Optional[float]]:
    """
    Very simple PARAMETER parser from a Modelfile.
    Looks for lines like: PARAMETER temperature 0.7
    """
    params = {
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "repeat_penalty": None,
        "num_ctx": None,
        "num_predict": None,
        "num_batch": None,
        "num_gpu": None,
    }

    if not modelfile:
        return params

    for line in modelfile.splitlines():
        line = line.strip()
        if not line.upper().startswith("PARAMETER"):
            continue
        parts = line.split()
        # PARAMETER <name> <value>
        if len(parts) < 3:
            continue
        key = parts[1]
        val = parts[2]
        if key in params:
            try:
                params[key] = float(val)
            except ValueError:
                # leave as None if it is not numeric
                params[key] = None

    return params


def upsert_model_metadata_row(csv_path: str, model_name: str) -> None:
    """
    Fetch Ollama metadata for `model_name` and upsert a row into `csv_path`.

    - If the CSV doesn't exist, it is created with a fixed column schema.
    - If a row with the same model_name already exists, it is updated in-place.
    - Otherwise, a new row is appended.
    """
    # 1. Get model info from Ollama
    info = ollama.show(model_name)  # returns a dict [web:49][web:55]

    full_model = info.get("model", model_name)
    details = info.get("details", {}) or {}
    modelfile_text = info.get("modelfile") or ""

    name, tag = _parse_name_tag(full_model)
    tag_fields = _derive_fields_from_tag(tag)
    caps = _derive_capabilities(details)
    params = _parse_parameters_from_modelfile(modelfile_text)

    row = {
        "model_name": name,
        "model_full": full_model,
        "tag": tag,
        "size_tag": tag_fields["size_tag"],
        "role": tag_fields["role"],
        "quant_scheme": tag_fields["quant_scheme"],
        "families": caps["families"],
        "base_family": caps["base_family"],
        "is_multimodal": caps["is_multimodal"],
        "license": _safe_get(details, "license"),
        "author": _safe_get(details, "author"),
        "modified_at": _safe_get(info, "modified_at"),
        # Parameters from Modelfile PARAMETER lines
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "top_k": params["top_k"],
        "repeat_penalty": params["repeat_penalty"],
        "num_ctx": params["num_ctx"],
        "num_predict": params["num_predict"],
        "num_batch": params["num_batch"],
        "num_gpu": params["num_gpu"],
        # Raw modelfile for deeper analysis (optional)
        "modelfile": modelfile_text,
    }

    # 2. Ensure consistent column order
    fieldnames = [
        "model_name",
        "model_full",
        "tag",
        "size_tag",
        "role",
        "quant_scheme",
        "families",
        "base_family",
        "is_multimodal",
        "license",
        "author",
        "modified_at",
        "temperature",
        "top_p",
        "top_k",
        "repeat_penalty",
        "num_ctx",
        "num_predict",
        "num_batch",
        "num_gpu",
        "modelfile",
    ]

    # 3. Read existing CSV (if any), upsert, and write back
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames or []
            # if schema changed, still keep old columns; add new ones as needed
            merged_fields = list(dict.fromkeys(existing_fields + fieldnames))
            fieldnames = merged_fields
            for r in reader:
                rows.append(r)

    # Upsert by model_full (or fallback to model_name)
    key_field = "model_full" if "model_full" in fieldnames else "model_name"
    updated = False
    for r in rows:
        if r.get(key_field) == row[key_field]:
            r.update({k: ("" if v is None else v) for k, v in row.items()})
            updated = True
            break

    if not updated:
        rows.append({k: ("" if v is None else v) for k, v in row.items()})

    # 4. Write back
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

