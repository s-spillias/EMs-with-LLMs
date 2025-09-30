#!/usr/bin/env python3
import json
import os
import time
import math
import random
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import partial

# Force-flush all prints in this module (mirrors make_model.py behavior)
print = partial(print, flush=True)

# External dependencies from your project
from scripts.search import search_engine, rag_query
from scripts.ask_AI import ask_ai

# -----------------------------------------------------------------------------
# Configurable timeouts & retries (via env vars)
# -----------------------------------------------------------------------------
def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default

SEARCH_TIMEOUT = _get_env_float("GET_PARAMS_SEARCH_TIMEOUT", 45.0)      # seconds per search call
LLM_TIMEOUT    = _get_env_float("GET_PARAMS_LLM_TIMEOUT", 90.0)         # seconds per ask_ai call
MAX_RETRIES    = _get_env_int("GET_PARAMS_MAX_RETRIES", 3)              # attempts per parameter
PER_PARAM_BUDGET = _get_env_float("GET_PARAMS_PER_PARAM_BUDGET", 180.0) # hard wall-clock cap per parameter (s)

# A small jitter to avoid thundering herds when many workers start together
STARTUP_JITTER_MAX = _get_env_float("GET_PARAMS_STARTUP_JITTER_MAX", 0.0)

# Thread pool used to enforce timeouts around blocking calls
_EXECUTOR = ThreadPoolExecutor(max_workers=16)

# --- POSIX signal-based timeout wrapper for get_params.py ---

import signal
import threading
from concurrent.futures import TimeoutError as FuturesTimeoutError

class _TimeoutAlarm(Exception):
    pass

def _call_with_timeout(func, timeout: float, *args, **kwargs):
    """
    Enforce a hard wall-clock timeout using SIGALRM (Unix only).
    - Works inside multiprocessing workers (no threads/processes spawned).
    - Must be called from the *main thread* of the current process.
    - If the timeout elapses, raises FuturesTimeoutError and the underlying
      function is interrupted by the signal.
    """
    if timeout is None or timeout <= 0:
        return func(*args, **kwargs)

    # Ensure we're in the main thread (POSIX signals only hit the main thread)
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(
            "Signal-based timeout requires being called from the main thread. "
            "Consider the subprocess-based fallback for non-main-thread contexts."
        )

    def _alarm_handler(signum, frame):
        raise _TimeoutAlarm(f"call exceeded {timeout}s")

    # Save previous handlers/timers
    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        # setitimer gives sub-second precision vs. alarm()
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, float(timeout))
        try:
            return func(*args, **kwargs)
        finally:
            # Always cancel alarm
            signal.setitimer(signal.ITIMER_REAL, 0.0)
    except _TimeoutAlarm:
        raise FuturesTimeoutError(f"call exceeded {timeout}s")
    finally:
        # Restore previous handler
        try:
            signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            pass

# -----------------------------------------------------------------------------
# Numeric helpers
# -----------------------------------------------------------------------------
def _to_number_or_none(v):
    """Coerce to float if possible; return None otherwise."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s.lower() in ("", "na", "n/a", "null", "none"):
        return None
    try:
        return float(s)
    except Exception:
        return None

def _normalize_bounds(lb, ub) -> Tuple[Optional[float], Optional[float], str]:
    """
    Ensure bounds are ordered (lb <= ub) and not identical.
    Returns (lb, ub, note).
    """
    note = ""
    lb_n = _to_number_or_none(lb)
    ub_n = _to_number_or_none(ub)
    if lb_n is None and ub_n is None:
        return None, None, note
    if lb_n is None:
        return None, ub_n, note
    if ub_n is None:
        return lb_n, None, note
    if lb_n > ub_n:
        lb_n, ub_n = ub_n, lb_n
        note = "Swapped bounds (lower_bound > upper_bound)."
    if lb_n == ub_n:
        eps = max(1e-12, abs(lb_n) * 1e-9)
        ub_n = lb_n + eps
        if note:
            note += " "
        note += "Adjusted equal bounds by epsilon."
    return lb_n, ub_n, note

# -----------------------------------------------------------------------------
# Model resolution (from population_metadata.rag_choice)
# -----------------------------------------------------------------------------
def _resolve_model_name_from_rag_choice(rag_choice: Optional[str]) -> str:
    """
    Convert population_metadata.json 'rag_choice' into the model_name expected by ask_AI.ask_ai.
    """
    default_model = "claude-3-5-sonnet-20241022"
    if not rag_choice:
        return default_model
    s = str(rag_choice).strip()
    # Allow local/dynamic models
    if s.startswith("ollama:") or s.startswith("ollama_"):
        return s
    # Back-compat aliases
    key = s.lower().replace(" ", "_").replace("/", "_")
    legacy_mapping = {
        "anthropic_sonnet": "claude-3-5-sonnet-20241022",
        "anthropic_haiku": "claude-3-5-haiku-20241022",
        "groq": "llama-3.3-70b-versatile",
        "gemini": "gemini-2.5-pro",
        "bedrock": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    }
    return legacy_mapping.get(key, s)

# -----------------------------------------------------------------------------
# LLM extraction (with timeout guards)
# -----------------------------------------------------------------------------
def extract_values_from_text(
    search_results: str,
    parameter: str,
    description: str,
    model_name: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Use LLM to extract values from search results with a configurable model.
    Returns (parsed_values or None, raw_llm_response).
    Expected JSON fields in response:
    - min, max, value (floats)
    - relevant_text (string)
    - citations_used (list[int])
    """
    is_log_param = parameter.startswith("log_")
    is_logit_param = parameter.startswith("logit_")

    if is_log_param:
        conversion_instruction = f"""
IMPORTANT: The parameter '{parameter}' is on a LOG SCALE.
- If you find a raw value (e.g., 9.0 years), convert it to log scale: log(9.0) = 2.197
- If you find multiple raw values, convert each to log scale before calculating min/max/average
- Return the LOG-TRANSFORMED values in your JSON response
"""
    elif is_logit_param:
        conversion_instruction = f"""
IMPORTANT: The parameter '{parameter}' is on a LOGIT SCALE.
- If you find proportions/percentages (e.g., 0.3 or 30%), convert to logit scale: logit(0.3) = -0.847
- Convert percentages to proportions first (30% = 0.3), then apply logit transformation
- If you find multiple proportions, convert each to logit scale before calculating min/max/average
- Return the LOGIT-TRANSFORMED values in your JSON response
"""
    else:
        conversion_instruction = """
The parameter is on a LINEAR SCALE. Use the raw values as found in the literature.
"""

    prompt = f"""
You are given N numbered source extracts. Each section is prefixed like: [1] SOURCE_URL:<url> then TEXT:<content>.
TASK
- Based on these sources, extract numerical values for:
  Parameter: {parameter}
  Description: {description}
- Carefully read only the numbered sources below. Use citations to support any value you extract.
SCALE
{conversion_instruction}
OUTPUT (JSON ONLY)
Return a single JSON object with:
  "min": <float>,
  "max": <float>,
  "value": <float>, // if multiple values are found, use the mean of the converted values
  "relevant_text": "<short quotes you used, include the [index] markers inline>",
  "citations_used": [<int>, ...] // the list of [indices] of sources actually used for the values
If no relevant values are found, respond with exactly the string: No relevant values found
SOURCES
{search_results}
Example JSON:
{{"min": 1.2, "max": 3.4, "value": 2.3, "relevant_text": "…as reported by [2] and [3]…", "citations_used": [2,3]}}
"""

    raw_response = ""
    try:
        # Run ask_ai with a hard timeout
        raw_response = _call_with_timeout(ask_ai, LLM_TIMEOUT, prompt, model_name=model_name)
        print(f"DEBUG: LLM response for {parameter}: {raw_response}")
        if "No relevant values found" in raw_response:
            return None, raw_response

        # Extract JSON payload (first '{' to last '}')
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None, raw_response
        json_str = raw_response[start : end + 1]
        values = json.loads(json_str)
        if not all(k in values for k in ["min", "max", "value"]):
            return None, raw_response

        result: Dict[str, Any] = {}
        for k in ["min", "max", "value"]:
            result[k] = float(values[k])
        if "relevant_text" in values:
            result["relevant_text"] = str(values["relevant_text"])

        if "citations_used" in values and isinstance(values["citations_used"], list):
            norm_idx = []
            seen = set()
            for v in values["citations_used"]:
                try:
                    i = int(v)
                    if i not in seen:
                        seen.add(i)
                        norm_idx.append(i)
                except Exception:
                    continue
            result["citations_used"] = norm_idx

        return result, raw_response

    except FuturesTimeoutError:
        msg = f"TIMEOUT: ask_ai exceeded {LLM_TIMEOUT}s for parameter {parameter}"
        print(msg)
        return None, msg
    except Exception as e:
        print(f"DEBUG: Error extracting values for {parameter}: {e}")
        return None, raw_response

# -----------------------------------------------------------------------------
# Citation helpers
# -----------------------------------------------------------------------------
def flatten_list(nested_list):
    """Flatten a deeply nested list of strings."""
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened

def _infer_citations_from_relevant_text(relevant_text: str, sources: List[Dict[str, str]]) -> List[int]:
    """
    Fallback: If the model didn't return citations_used, try to infer indices by:
    1) Reading explicit [n] markers in relevant_text
    2) Substring matching sentences against source content
    Returns a list of 1-based indices.
    """
    indices: List[int] = []
    if not relevant_text:
        return indices
    # 1) Explicit [n] markers
    try:
        import re
        for m in re.findall(r"\[(\d+)\]", relevant_text):
            i = int(m)
            if 1 <= i <= len(sources) and i not in indices:
                indices.append(i)
    except Exception:
        pass
    # 2) Light substring matching on longer snippets
    if not indices:
        snippets = [s.strip() for s in relevant_text.split(".") if len(s.strip()) >= 40]
        for i, src in enumerate(sources, start=1):
            text = (src.get("content") or "")[:100000]  # cap for speed
            for snip in snippets:
                if snip and snip in text:
                    if i not in indices:
                        indices.append(i)
                    break
    return indices

# -----------------------------------------------------------------------------
# One-parameter workflow (with timeout guards around search)
# -----------------------------------------------------------------------------
def process_parameter(
    parameter: str,
    description: str,
    enhanced_semantic_description: str,
    import_type: str,
    cpp_content: str,
    population_dir: str,
    doc_store: Optional[str],
    model_name: str,
    max_retries: int = MAX_RETRIES
) -> Dict[str, Any]:
    """
    Process a single parameter using RAG with Web Search + LLM extraction.

    This version:
      - Attempts *all* search engines (random order) regardless of whether
        earlier engines returned sources.
      - Aggregates sources from all engines and passes the combined numbered
        extracts to the LLM.
    """
    sources: List[Dict[str, str]] = []
    started = time.monotonic()

    # Single query for this parameter
    search_query = enhanced_semantic_description or description or parameter
    print(f"Searching for {parameter} with query: {search_query}")

    # Build engine list; include RAG only if a doc store is available
    engines = ["semantic_scholar", "serper"]
    if doc_store is not None:
        engines.insert(1, "rag")  # keep rough preference but we'll shuffle next

    # Randomize order (helps when many workers run in parallel)
    random.shuffle(engines)
    print(f"Engine order for {parameter}: {engines}")

    # Track seen URLs to avoid duplicate citations across engines
    seen_urls: set = set()

    # Attempt *all* engines, each with up to `max_retries`, within the global budget
    for engine in engines:
        for attempt in range(max_retries):
            # Budget guard: stop attempts once per-parameter wall-clock budget is exceeded
            elapsed = time.monotonic() - started
            if elapsed >= PER_PARAM_BUDGET:
                print(f"Budget exceeded ({PER_PARAM_BUDGET}s) for {parameter}; stopping searches.")
                break

            try:
                if engine == "rag":
                    def _rag():
                        return rag_query(search_query, doc_store, population_dir)
                    search_results, citations = _call_with_timeout(_rag, SEARCH_TIMEOUT)
                else:
                    def _se():
                        return search_engine(search_query, engine=engine, directory=doc_store)
                    search_results, citations = _call_with_timeout(_se, SEARCH_TIMEOUT)

                if not search_results:
                    print(f"No search results for {parameter} (engine={engine}, attempt={attempt + 1})")
                    # Try the same engine again (up to max_retries)
                    continue
                print(f"Search results found for {parameter} (engine={engine}, attempt={attempt + 1})")

                # Aggregate results from this engine; de-duplicate by URL
                for content, url in zip(search_results, citations):
                    url_norm = (url or "").strip()
                    if url_norm and url_norm not in seen_urls:
                        sources.append({"content": content, "citation": url})
                        seen_urls.add(url_norm)

                # Successful call for this engine; move on to the next engine
                break

            except FuturesTimeoutError:
                print(f"TIMEOUT: search (engine={engine}) exceeded {SEARCH_TIMEOUT}s for {parameter}")
                # Backoff a little, then retry this engine
                time.sleep(1.0 + attempt)
                continue
            except Exception as e:
                print(f"Error searching {parameter} (engine={engine}, attempt={attempt + 1}): {e}")
                # crude 429 hint backoff (short to avoid stalling)
                if "429" in str(e):
                    time.sleep(3 * (attempt + 1))
                continue
        # Next engine (we do NOT stop early even if we already have sources)

    if not sources:
        # No sources at all; return minimal
        return {
            "parameter": parameter,
            "citations": [],
            "all_citations": [],
            "source": "literature",
            "raw_llm_response": "NO_SOURCES"
        }

    # Build numbered RAG info (1-based indices) from *all* gathered sources
    numbered_chunks: List[str] = []
    for idx, s in enumerate(sources, start=1):
        url = s.get("citation", "")
        text = s.get("content", "")
        numbered_chunks.append(f"[{idx}] SOURCE_URL:{url}\nTEXT:\n{text}\n")
    rag_info = "\n\n".join(numbered_chunks)

    print(f"DEBUG: Aggregated {len(sources)} total source chunks for {parameter}:")
    for i, s in enumerate(sources, start=1):
        print(f" Source [{i}]: {s.get('citation','')}")

    # LLM extraction on the combined sources
    values, raw_llm_response = extract_values_from_text(
        rag_info, parameter, description, model_name=model_name
    )

    out: Dict[str, Any] = {
        "parameter": parameter,
        "source": "literature",
        "raw_llm_response": raw_llm_response,
        # only relevant citations here
        "citations": [],
        # full list separately for QA (all unique citations we gathered)
        "all_citations": [s.get("citation", "") for s in sources]
    }

    if values:
        v_min = float(values["min"])
        v_max = float(values["max"])
        v_val = float(values["value"])
        if v_min > v_max:
            v_min, v_max = v_max, v_min

        out["found_value"] = v_val
        out["found_lower_bound"] = v_min
        out["found_upper_bound"] = v_max
        if "relevant_text" in values:
            out["relevant_text"] = values["relevant_text"]

        # Map LLM-reported indices to URLs; fallback heuristic if missing
        used_indices: List[int] = values.get("citations_used", []) if isinstance(values, dict) else []
        if not used_indices:
            used_indices = _infer_citations_from_relevant_text(out.get("relevant_text", ""), sources)

        relevant_citations: List[str] = []
        for i in used_indices:
            if isinstance(i, int) and 1 <= i <= len(sources):
                url = sources[i - 1].get("citation")
                if url and url not in relevant_citations:
                    relevant_citations.append(url)

        out["citations"] = relevant_citations

    return out


# -----------------------------------------------------------------------------
# Metadata store helpers
# -----------------------------------------------------------------------------
def _load_existing_metadata(path: str) -> Dict[str, Any]:
    """
    Load existing parameters_metadata.json in a tolerant way.
    Accepted shapes on disk:
    A) {"parameters": { "<parameter>": {...}, ... }, "meta": {...}}
    B) {"parameters": [ {...}, {...}, ... ], "meta": {...}} # each item must include "parameter"
    We normalize to:
    {"parameters": { "<parameter>": { ...full merged record... } }, "meta": {...}}
    """
    if not os.path.exists(path):
        return {"parameters": {}, "meta": {}}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        meta = data.get("meta", {}) if isinstance(data, dict) else {}
        params_raw = data.get("parameters", {}) if isinstance(data, dict) else {}
        params_dict: Dict[str, Any] = {}
        if isinstance(params_raw, dict):
            params_dict = params_raw
        elif isinstance(params_raw, list):
            for item in params_raw:
                if isinstance(item, dict):
                    pname = item.get("parameter")
                    if pname:
                        params_dict[pname] = item
        else:
            params_dict = {}
        return {"parameters": params_dict, "meta": meta}
    except Exception:
        return {"parameters": {}, "meta": {}}

def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write JSON atomically to avoid partial reads by other processes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{random.randint(0, 1_000_000)}"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=4)
    os.replace(tmp, path)

# -----------------------------------------------------------------------------
# Core: compute final fields for parameters.json
# -----------------------------------------------------------------------------
def _finalize_value_and_bounds_for_runtime(
    base_param: Dict[str, Any],
    found_lb: Optional[float],
    found_ub: Optional[float],
    found_val: Optional[float]
) -> Dict[str, Any]:
    """
    Build the minimal record to write back to parameters.json.
    Rule:
    - If found_* bounds exist, they OVERWRITE runtime bounds (after normalization).
    - If only one found bound is present, combine with the other existing bound then normalize.
    - If no found bounds, keep existing and normalize if needed.
    - 'value' becomes found_value if available; otherwise keep existing value.
    """
    runtime = dict(base_param)

    # VALUE
    if found_val is not None:
        runtime["value"] = found_val

    # BOUNDS
    curr_lb = _to_number_or_none(runtime.get("lower_bound"))
    curr_ub = _to_number_or_none(runtime.get("upper_bound"))
    cand_lb = found_lb if found_lb is not None else curr_lb
    cand_ub = found_ub if found_ub is not None else curr_ub
    norm_lb, norm_ub, _ = _normalize_bounds(cand_lb, cand_ub)

    if norm_lb is not None:
        runtime["lower_bound"] = norm_lb
    else:
        runtime.pop("lower_bound", None)
    if norm_ub is not None:
        runtime["upper_bound"] = norm_ub
    else:
        runtime.pop("upper_bound", None)

    # Strip metadata-ish fields
    for k in list(runtime.keys()):
        if k.startswith("found_") or k.startswith("llm_") or k.startswith("literature_") or k in {
            "citations", "all_citations", "raw_llm_response", "processed"
        }:
            runtime.pop(k, None)
    return runtime

# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
def get_params(directory_path):
    """
    Process parameters:
    - Read parameters.json (created initially by make_model.json)
    - For literature-sourced parameters not yet processed (per metadata), perform search+extraction
    - Write minimal updates back to parameters.json (value/lower_bound/upper_bound only)
    - Write full provenance and flags to parameters_metadata.json

    Naming:
    * llm_value / llm_lower_bound / llm_upper_bound = initial values from parameters.json
    * found_value / found_lower_bound / found_upper_bound = values extracted from literature
    * processed is stored ONLY in parameters_metadata.json
    """
    if STARTUP_JITTER_MAX > 0:
        jitter = random.uniform(0, STARTUP_JITTER_MAX)
        print(f"DEBUG: startup jitter {jitter:.2f}s for {directory_path}")
        time.sleep(jitter)

    print(f"DEBUG: get_params() called with directory_path: {directory_path}")
    # Resolve population context
    population_dir = os.path.dirname(directory_path)
    print(f"DEBUG: population_dir: {population_dir}")
    meta_file = os.path.join(population_dir, "population_metadata.json")
    print(f"DEBUG: meta_file path: {meta_file}")

    rag_choice = None
    doc_store_dir = None
    print("DEBUG: Checking if meta_file exists...")
    if os.path.exists(meta_file):
        print("DEBUG: meta_file exists, attempting to read...")
        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
            print("DEBUG: Successfully read meta_file")
            rag_choice = meta.get("rag_choice")
            doc_store_dir = meta.get("doc_store_dir", None)
            print(f"DEBUG: rag_choice: {rag_choice}, doc_store_dir: {doc_store_dir}")
        except Exception as e:
            print(f"Warning: could not read population_metadata.json: {e}")
    else:
        print("DEBUG: meta_file does not exist")

    print("DEBUG: About to resolve model name...")
    model_name = _resolve_model_name_from_rag_choice(rag_choice)
    print(f"[get_params] Using model from rag_choice: '{rag_choice}' -> '{model_name}'")

    doc_store = None if doc_store_dir is None else doc_store_dir
    if doc_store is None:
        print("[get_params] doc_store_dir is None - RAG disabled")
    else:
        print(f"[get_params] Using doc_store_dir: '{doc_store_dir}'")

    # File paths
    params_file = os.path.join(directory_path, "parameters.json")
    metadata_file = os.path.join(directory_path, "parameters_metadata.json")

    # Load artifacts
    with open(params_file, "r") as f:
        params_data = json.load(f)
    with open(os.path.join(directory_path, "model.cpp"), "r") as f:
        cpp_str = f.read()

    metadata_store = _load_existing_metadata(metadata_file)
    metadata_params: Dict[str, Any] = metadata_store.get("parameters", {})
    metadata_meta: Dict[str, Any] = metadata_store.get("meta", {})

    now_iso = datetime.utcnow().isoformat() + "Z"
    params_out: List[Dict[str, Any]] = []

    # Process each parameter
    for param in params_data.get("parameters", []):
        pname = param.get("parameter")
        source = (param.get("source") or "").lower()
        import_type = param.get("import_type", "PARAMETER")

        already_processed = bool(metadata_params.get(pname, {}).get("processed"))
        should_process = ("literature" in source) and (import_type == "PARAMETER") and (not already_processed)

        # Capture initial values for metadata
        original_value = param.get("value")
        original_lb = param.get("lower_bound")
        original_ub = param.get("upper_bound")

        if should_process:
            print(f"Processing literature parameter: {pname}")
            started_param = time.monotonic()

            processed = process_parameter(
                pname,
                param.get("description", ""),
                param.get("enhanced_semantic_description", "") or param.get("description", ""),
                import_type,
                cpp_str,
                population_dir,
                doc_store,
                model_name=model_name,
                max_retries=MAX_RETRIES
            )

            # Extract found_* for runtime and metadata
            found_val = processed.get("found_value")
            found_lb = processed.get("found_lower_bound")
            found_ub = processed.get("found_upper_bound")

            # --- Write minimal fields back to parameters.json (using found_* bounds) ---
            runtime_record = _finalize_value_and_bounds_for_runtime(
                base_param=param,
                found_lb=found_lb,
                found_ub=found_ub,
                found_val=found_val
            )
            params_out.append(runtime_record)

            # --- Write full metadata ---
            value_within_bounds = None
            lb_num = _to_number_or_none(runtime_record.get("lower_bound"))
            ub_num = _to_number_or_none(runtime_record.get("upper_bound"))
            val_num = _to_number_or_none(runtime_record.get("value"))
            if val_num is not None and (lb_num is not None or ub_num is not None):
                ok_lb = (lb_num is None) or (val_num >= lb_num)
                ok_ub = (ub_num is None) or (val_num <= ub_num)
                value_within_bounds = bool(ok_lb and ok_ub)

            meta_entry = {
                "parameter": pname,
                "source": "literature",
                "processed": True,  # ONLY here
                "processed_at": now_iso,
                "rag_choice": rag_choice,
                "model_name": model_name,

                # *** IMPORTANT: only relevant citations here ***
                "citations": processed.get("citations", []),

                # Keep all searched citations for traceability
                "all_citations": processed.get("all_citations", []),
                "raw_llm_response": processed.get("raw_llm_response", ""),

                # Initials from parameters.json BEFORE enrichment
                "llm_value": original_value,
                "llm_lower_bound": original_lb,
                "llm_upper_bound": original_ub,

                # Literature extraction results
                "found_value": found_val,
                "found_lower_bound": found_lb,
                "found_upper_bound": found_ub,

                # Relevant text from literature
                "relevant_text": processed.get("relevant_text", ""),

                # Simple QA flag
                "value_within_bounds": value_within_bounds,
                "elapsed_seconds": round(time.monotonic() - started_param, 3),
            }
            existing_meta = metadata_params.get(pname, {})
            merged_meta = {**existing_meta, **meta_entry}  # meta_entry keys override, others preserved
            metadata_params[pname] = merged_meta

        else:
            # Not processed this round; make sure no stray metadata-ish fields remain in parameters.json
            param_copy = dict(param)
            for k in list(param_copy.keys()):
                if k.startswith("found_") or k.startswith("llm_") or k.startswith("literature_") or k in {
                    "citations", "all_citations", "raw_llm_response", "processed"
                }:
                    param_copy.pop(k, None)
            params_out.append(param_copy)

    # Save minimal parameters.json (atomic)
    _atomic_write_json(params_file, {"parameters": params_out})

    # Save metadata (append/merge semantics preserved) (atomic)
    metadata_meta.update({
        "last_updated": now_iso,
        "population_dir": population_dir,
        "doc_store_dir": doc_store_dir,
        "search_timeout_s": SEARCH_TIMEOUT,
        "llm_timeout_s": LLM_TIMEOUT,
        "max_retries": MAX_RETRIES,
        "per_param_budget_s": PER_PARAM_BUDGET,
    })
    metadata_params_list = list(metadata_params.values())
    metadata_store = {
        "parameters": metadata_params_list,
        "meta": metadata_meta,
    }
    _atomic_write_json(metadata_file, metadata_store)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
    else:
        directory_path = "BRANCH_0/Version_0000"  # Default path
    get_params(directory_path)
