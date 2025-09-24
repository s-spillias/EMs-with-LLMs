import json
import os
import logging
import pandas as pd
import time
from scripts.search import search_engine, rag_query
from scripts.ask_AI import ask_ai

# -----------------------------------------------------------------------------
# Helpers for numeric coercion & bounds
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


def _normalize_bounds(lb, ub):
    """
    Ensure bounds are in correct order and not identical.
    Returns (lb, ub, note) where note explains any normalization.
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
        note = "Swapped lower/upper bounds as lower_bound > upper_bound."

    if lb_n == ub_n:
        # Add a tiny epsilon to create an open interval
        eps = max(1e-12, abs(lb_n) * 1e-9)
        ub_n = lb_n + eps
        if note:
            note += " "
        note += "Adjusted equal bounds by epsilon to avoid zero-width interval."

    return lb_n, ub_n, note


# -----------------------------------------------------------------------------
# Resolve model_name from population_metadata.rag_choice
#   - Passes ollama:* and ollama_* through as-is (ask_AI supports them).
#   - Maps common aliases used elsewhere in your stack to actual keys in ask_AI.Config.MODELS.
# -----------------------------------------------------------------------------

def _resolve_model_name_from_rag_choice(rag_choice: str | None) -> str:
    """
    Convert population_metadata.json 'rag_choice' into an actual model_name expected by ask_AI.ask_ai.
    Since we now use actual model names in genetic_algorithm.py, this mostly passes through the value.
    - If 'ollama:' or 'ollama_' prefix is used, pass through (dynamic models supported).
    - For backward compatibility, still handle some old aliases.
    Fallback is a solid general model.
    """
    default_model = "claude-3-5-sonnet-20241022"
    if not rag_choice:
        return default_model

    s = str(rag_choice).strip()

    # Ollama local models: let ask_AI register them dynamically
    if s.startswith("ollama:") or s.startswith("ollama_"):
        return s

    # For backward compatibility with old aliases (in case old configs still exist)
    key = s.lower().replace(" ", "_").replace("/", "_")
    legacy_mapping = {
        "anthropic_sonnet": "claude-3-5-sonnet-20241022",
        "anthropic_haiku":  "claude-3-5-haiku-20241022",
        "groq":             "llama-3.3-70b-versatile",
        "gemini":           "gemini-2.5-pro",
        "bedrock":          "anthropic.claude-3-5-sonnet-20240620-v1:0",
    }

    # Check if it's a legacy alias, otherwise pass through as-is (assuming it's a valid model name)
    return legacy_mapping.get(key, s)


# -----------------------------------------------------------------------------
# LLM extraction
# -----------------------------------------------------------------------------

def extract_values_from_text(search_results, parameter, description, model_name: str):
    """Use LLM to extract values from search results with a configurable model."""
    prompt = f"""
    Based on the following search results, extract numerical values for:
    {description}

    Search Results:
    {search_results}

    Please analyze these results and provide a JSON object with 'min', 'max', and 'value' for this parameter.
    If multiple values are found, 'value' should be the average.
    If no relevant values are found, respond with 'No relevant values found'.
    Note the units that the model expects. Please convert the values to be the appropriate units for the model.

    Example response format:
    {{"min": 1.2, "max": 3.4, "value": 2.3}}
    """
    try:
        # NOTE: ask_AI.ask_ai now expects model_name=..., not ai_model=...
        response = ask_ai(prompt, model_name=model_name)
        if "No relevant values found" in response:
            return None
        # Try to parse the JSON response
        try:
            # Find JSON-like string in the response
            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                values = json.loads(json_match.group(0))
                if all(k in values for k in ['min', 'max', 'value']):
                    # Convert values to float
                    values = {k: float(values[k]) for k in values}
                    return values
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing values: {str(e)}")
            return None
    except Exception as e:
        print(f"Error extracting values: {str(e)}")
        return None


def flatten_list(nested_list):
    """Flatten a deeply nested list of strings."""
    flattened = []
    for item in nested_list:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))  # Recursively flatten sublist
        else:
            flattened.append(item)
    return flattened


def process_parameter(
    parameter: str,
    description: str,
    enhanced_semantic_description: str,
    import_type: str,
    cpp_content: str,
    population_dir: str,
    doc_store: str,
    model_name: str,
    max_retries: int = 3
):
    """Process a single parameter using RAG with Web Search + LLM extraction."""
    for attempt in range(max_retries):
        # Use the enhanced semantic description for broader, higher-quality results
        search_query = enhanced_semantic_description or description or parameter
        print(f"Attempt {attempt + 1} for {parameter}: {search_query}")
        all_search_results = []
        all_citations = []
        try:
            if doc_store is None:
                # Only use web search engines when doc_store is disabled
                search_engines = ["semantic_scholar", "serper"]
            else:
                # Use all search engines including local RAG when doc_store is available
                search_engines = ["semantic_scholar", "rag", "serper"]
            
            engine = search_engines[attempt % len(search_engines)]
            if engine == "rag":
                search_results, citations = rag_query(search_query, doc_store, population_dir)
            else:
                search_results, citations = search_engine(search_query, engine=engine, directory=doc_store)
            if not search_results:
                print(f"No search results found for {parameter} in attempt {attempt + 1}")
                continue
            print(f"Search results found for {parameter}")
            all_search_results.append(search_results)
            all_citations.append(citations)
        except Exception as e:
            print(f"Error processing {parameter} in attempt {attempt + 1}: {str(e)}")
            # Add delay if we hit rate limits
            if "429" in str(e):
                time.sleep(5 * (attempt + 1))

    # Use LLM to extract values from aggregated search results
    flat_search_results = flatten_list(all_search_results)
    flat_citations = flatten_list(all_citations)
    rag_info = "\n".join(flat_search_results)
    print("*" * 50 + '\n\n' + rag_info + '\n\n' + "*" * 50)
    values = extract_values_from_text(rag_info, parameter, description, model_name=model_name)

    output = {
        "parameter": parameter,
        "citations": flat_citations,
        "source": "literature",
        "processed": True
    }

    # Add found values if they exist with 'found_' prefix
    if values:
        try:
            v_min = float(values["min"])
            v_max = float(values["max"])
            v_val = float(values["value"])
            # Normalize found min/max in case sources are reversed
            if v_min > v_max:
                v_min, v_max = v_max, v_min
        except Exception:
            v_min = values.get("min")
            v_max = values.get("max")
            v_val = values.get("value")

        output["found_value"] = v_val
        output["found_min"] = v_min
        output["found_max"] = v_max
        print(f"Found values for {parameter}: {values}")

    print(f"Successfully processed {parameter}")
    return output


def _apply_and_validate_bounds(param):
    """
    Merge and validate bounds on a single param record.
    - Preserve existing lower/upper if present.
    - Add literature_* bounds from found_min/found_max.
    - Fill missing lower/upper from found_min/found_max when available.
    - Validate order and value containment; annotate non-destructively.
    """
    # Existing primary bounds
    lb = _to_number_or_none(param.get("lower_bound"))
    ub = _to_number_or_none(param.get("upper_bound"))

    # Literature-derived bounds from the extraction step
    found_min = _to_number_or_none(param.get("found_min"))
    found_max = _to_number_or_none(param.get("found_max"))
    if found_min is not None:
        param["literature_lower_bound"] = found_min
    if found_max is not None:
        param["literature_upper_bound"] = found_max

    # If primary bounds missing, use literature-derived ones
    if lb is None and found_min is not None:
        lb = found_min
    if ub is None and found_max is not None:
        ub = found_max

    # Normalize and record any note
    lb, ub, norm_note = _normalize_bounds(lb, ub)
    if lb is not None:
        param["lower_bound"] = lb
    if ub is not None:
        param["upper_bound"] = ub

    # Value containment check (non-destructive)
    val = _to_number_or_none(param.get("value"))
    value_within = True
    bounds_note = ""

    if val is not None:
        if lb is not None and val < lb:
            value_within = False
            suggested = lb
            bounds_note = f"value ({val}) < lower_bound ({lb}); suggested to raise to {suggested}"
            param["value_suggested"] = suggested
        if ub is not None and val > ub:
            value_within = False
            suggested = ub
            note_piece = f"value ({val}) > upper_bound ({ub}); suggested to lower to {suggested}"
            bounds_note = (bounds_note + "; " + note_piece) if bounds_note else note_piece

    # Attach normalization note if any
    if norm_note:
        bounds_note = (bounds_note + "; " + norm_note) if bounds_note else norm_note

    # Flag
    param["value_within_bounds"] = value_within
    if bounds_note:
        param["bounds_note"] = bounds_note


def get_params(directory_path):
    """
    Read parameters.json and process literature parameters.
    LLM engine is selected ONLY from population_metadata.json -> 'rag_choice'.
    """
    # Extract population_dir from directory_path
    population_dir = os.path.dirname(directory_path)

    # Resolve model_name and doc_store_dir from population_metadata.json
    meta_file = os.path.join(population_dir, "population_metadata.json")
    rag_choice = None
    doc_store_dir = None # Default fallback
    if os.path.exists(meta_file):
        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
                rag_choice = meta.get("rag_choice")
                doc_store_dir = meta.get("doc_store_dir", None)  # Use default if not found
        except Exception as e:
            print(f"Warning: could not read population_metadata.json: {e}")

    model_name = _resolve_model_name_from_rag_choice(rag_choice)
    print(f"[get_params] Using model from rag_choice: '{rag_choice}' -> '{model_name}'")
    
    if doc_store_dir is None:
        print(f"[get_params] doc_store_dir is None - RAG feature disabled")
        doc_store = None
    else:
        print(f"[get_params] Using doc_store_dir from metadata: '{doc_store_dir}'")
        doc_store = doc_store_dir

    # Read existing parameters.json
    params_file = os.path.join(directory_path, "parameters.json")
    with open(params_file, 'r') as file:
        params_data = json.load(file)

    # Read model.cpp for context (passed through to search/LLM)
    with open(os.path.join(directory_path, 'model.cpp'), 'r') as file:
        cpp_str = file.read()

    # Process parameters from literature
    processed_outputs = []
    for param in params_data["parameters"]:
        # Only process PARAMETERs from literature that are not yet processed
        import_type = param.get("import_type", "PARAMETER")
        if param.get("source") == "literature" and import_type == "PARAMETER" and not param.get("processed", False):
            print(f"Processing literature parameter: {param['parameter']}")
            processed_result = process_parameter(
                param["parameter"],
                param.get("description", ""),
                param.get("enhanced_semantic_description", "") or param.get("description", ""),
                import_type,
                cpp_str,
                population_dir,
                doc_store,  # This can be None - process_parameter will handle it
                model_name=model_name
            )
            if processed_result:
                # Update with new fields while preserving original values
                param.update(processed_result)
        else:
            # Mark as processed (either already processed or not eligible)
            param["processed"] = True

        # Apply merge/validation of bounds for every parameter
        _apply_and_validate_bounds(param)

        processed_outputs.append(param)

    # Save the updated parameters
    output_file = os.path.join(directory_path, "parameters.json")
    with open(output_file, 'w') as file:
        json.dump({"parameters": processed_outputs}, file, indent=4)

    print("Parameter processing completed.")
    return processed_outputs


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
    else:
        directory_path = "BRANCH_0/Version_0000"  # Default path
    get_params(directory_path)
