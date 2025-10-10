import os
import json
import re
import shutil
from datetime import datetime
from scripts.ask_AI import ask_ai
from dotenv import load_dotenv
load_dotenv()
# Define the ecological characteristics scorecard based on NPZ model equations
ECOLOGICAL_CHARACTERISTICS = {
    # -----------------------------
    # dN/dt (Nutrient equation): 3 subcomponents → each = 1/3 ≈ 0.333
    # -----------------------------
    "nutrient_equation_uptake": {
        "description": "In dN/dt: Nutrient uptake by phytoplankton (Michaelis-Menten or alternates).",
        "weight": 0.333
    },
    "nutrient_equation_recycling": {
        "description": "In dN/dt: Nutrient recycling from zooplankton (predation losses, excretion).",
        "weight": 0.333
    },
    "nutrient_equation_mixing": {
        "description": "In dN/dt: Environmental mixing term (entrainment/dilution).",
        "weight": 0.333
    },

    # -----------------------------
    # dP/dt (Phytoplankton equation): 4 subcomponents → each = 0.25
    # -----------------------------
    "phytoplankton_equation_growth": {
        "description": "In dP/dt: Growth via nutrient + light limitation (Michaelis-Menten, Droop, f(I)).",
        "weight": 0.25
    },
    "phytoplankton_equation_grazing_loss": {
        "description": "In dP/dt: Loss to zooplankton grazing (Ivlev/Holling/threshold/acclimation).",
        "weight": 0.25
    },
    "phytoplankton_equation_mortality": {
        "description": "In dP/dt: Non-grazing mortality (linear or quadratic).",
        "weight": 0.25
    },
    "phytoplankton_equation_mixing": {
        "description": "In dP/dt: Physical loss via mixing/entrainment.",
        "weight": 0.25
    },

    # -----------------------------
    # dZ/dt (Zooplankton equation): 2 subcomponents → each = 0.5
    # -----------------------------
    "zooplankton_equation_growth": {
        "description": "In dZ/dt: Growth through grazing on phytoplankton (with assimilation efficiency).",
        "weight": 0.5
    },
    "zooplankton_equation_mortality": {
        "description": "In dZ/dt: Zooplankton mortality (linear or density-dependent).",
        "weight": 0.5
    }
}


def _extract_json_from_text(response_text: str):
    """
    Try to extract a JSON object from an LLM response.

    Strategy:
      1) Prefer fenced ```json ... ``` blocks.
      2) Else, find the first {...} that looks like JSON.
    Returns a Python dict or None.
    """
    if not isinstance(response_text, str) or not response_text.strip():
        return None

    # 1) Fenced json block
    m = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass  # fall through

    # 2) First top-level JSON-looking block
    m = re.search(r"(\{.*\})", response_text, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Try a mild cleanup pass (remove trailing commas, normalize whitespace)
            cleaned = re.sub(r",\s*}", "}", candidate)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return None
    return None


def read_ground_truth():
    """Read the ground truth NPZ model file."""
    ground_truth_path = "Data/NPZ_example/NPZ_model.py"
    with open(ground_truth_path, 'r') as f:
        return f.read()

def evaluate_model(model_content):
    """
    Use LLM to evaluate ecological characteristics of a model against a TRUTH NPZ
    and award an ordinal score per characteristic:
      3 = TRUTH_MATCH
      2 = ALTERNATE
      1 = SIMILAR_NOT_LISTED
      0 = NOT_PRESENT_OR_INCORRECT
    Returns a dict with qualitative description, per-characteristic results, and audit fields.
    """
    # --- Load TRUTH NPZ source so the LLM can compare 1:1 ---
    try:
        truth_content = read_ground_truth()
    except Exception:
        truth_content = ""

    # Construct the canonical characteristics list shown to the LLM
    characteristics_text = "\n".join([
        f"- {name}: {details['description']} (weight: {details['weight']})"
        for name, details in ECOLOGICAL_CHARACTERISTICS.items()
    ])

    # Curated alternates from Franks (2002): Tables 1-4 (light, uptake, grazing, mortality)
    # (We provide examples; LLM should match function families, not variable names.)
    alternates_text = """
Catalog of alternate formulations (examples, non-exhaustive):

1) Phytoplankton response to irradiance f(I):
   - Linear: f(I) = a * I
   - Saturating hyperbolae / exponentials / tanh:
       I/(I + I_s),  1 - exp(-I/I0),  tanh(I/I0)
   - Photo-inhibiting forms (increase then decline at high I)

2) Nutrient uptake g(N):
   - Michaelis-Menten: V_max * N/(k + N)
   - Liebig minimum limitation: growth = min( light_limit , nutrient_limit )
   - Droop (cell quota): internal quota Q with dQ/dt = uptake - use; growth ∝ (1 - Q0/Q)

3) Zooplankton grazing h(P):
   - Linear or bilinear with saturation at R_m
   - Saturating with threshold P0: R_m * (P - P0)/(λ + P - P0)
   - Holling-/Ivlev-type saturating: R_m * [1 - exp(-A P)] ; variants with threshold
   - Acclimating forms: near-linear at high P due to grazing acclimation

4) Loss/closure terms i(P), j(Z):
   - Linear mortality: ω * P  (for phytoplankton),  ε * Z (for zooplankton)
   - Quadratic (density-dependent) mortality: m * P^2,  μ * Z^2
   - Saturating density-dependence for zooplankton: ε Z^2 / (b + Z)
"""

    # Build scoring rubric
    rubric_text = """
Scoring rubric per characteristic (choose exactly one category):
- 3 = TRUTH_MATCH
    The mathematical structure is equivalent to the TRUTH model (modulo variable names,
    syntax, factor grouping, and coefficient naming). Quote the exact snippet that matches.
- 2 = ALTERNATE
    The implementation matches one of the alternates enumerated above,
    even if not identical to TRUTH. Name the family (e.g., "Michaelis-Menten uptake",
    "Ivlev grazing with threshold", "linear mortality", "Droop quota").
- 1 = SIMILAR_NOT_LISTED
    The implementation plays the same ecological role and is mathematically similar
    (e.g., another saturating curve or plausible closure) but is not represented in TRUTH
    or alternates list.
- 0 = NOT_PRESENT_OR_INCORRECT
    The ecological component is missing or cannot be identified.

Important:
• Always justify the category selection and reference the concrete term(s) or code lines.
• Accept differences in variable names, code organization, and equivalent algebra.
• If multiple terms exist for the same component, grade the best-matching one.
"""

    prompt = f"""You are assessing whether a C++ NPZ model implements canonical ecological components
when compared to a TRUTH NPZ model ("human model") and to alternate formulations.

TRUTH NPZ (ground truth reference):
{truth_content}

Candidate model to evaluate (C++):
{model_content}

Canonical components to check (by equation dN/dt, dP/dt, dZ/dt):
{characteristics_text}

Relevant alternates from the literature:
{alternates_text}

{rubric_text}

Additionally, identify any EXTRA ECOLOGICAL COMPONENTS present in the candidate that are NOT present in the TRUTH NPZ.
Definition and guidance:
- Consider an "extra component" as a distinct ecological process, state variable, or source/sink term (e.g., added detritus pool, temperature/Q10 modifier on rates, extra mortality/closure terms, explicit exudation, DOM remineralization, etc.) that does not exist in the TRUTH equations.
- Parameter renaming, algebraic regrouping, or purely notational changes are NOT extra components.
- If components are merged/split relative to TRUTH, only count them as "extra" if a truly new process/term is introduced, not merely a refactor.
- Briefly list each extra component and its role so a human can verify.

OUTPUT STRICTLY AS JSON with this schema (types shown as choices/labels; your actual output must be valid JSON without comments):
{{
  "qualitative_description": "Overall narrative comparing the candidate to TRUTH and literature alternates",
  "extra_components_count": 0 | 1 | 2 | 3 | ...,
  "extra_components_description": "Short list-style description naming each extra component and its role (or empty if none)",
  "characteristic_scores": {{
    "characteristic_name": {{
      "score": 0 | 1 | 2 | 3,
      "category": "TRUTH_MATCH" | "ALTERNATE" | "SIMILAR_NOT_LISTED" | "NOT_PRESENT_OR_INCORRECT",
      "matched_form": "e.g., Michaelis-Menten uptake / Ivlev grazing / linear mortality / Droop quota / (or empty)",
      "explanation": "Short rationale quoting the exact term(s) or code line(s)"
    }}
  }},
  "notes": "any caveats or ambiguities"
}}
"""

    # Query the LLM
    response = ask_ai(prompt, 'openrouter:gpt-5-mini')

    # --- Robust JSON extraction/cleanup (unchanged) ---
    try:
        import re, json
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                print("Error: Could not find JSON content in response")
                return None

        json_str = json_str.strip()
        json_str = re.sub(r'\s+', ' ', json_str)
        json_str = re.sub(r'},+', '},', json_str)
        json_str = re.sub(r'"},+', '"},', json_str)
        json_str = re.sub(r'(\d+|\d+\.\d+)\s*"', r'\1,"', json_str)
        json_str = re.sub(r'",\s*}', '"}', json_str)
        json_str = re.sub(r'},\s*}', '}}', json_str)

        try:
            evaluation = _extract_json_from_text(response)
            if not isinstance(evaluation, dict):
                print("Error: Could not parse JSON from LLM response.")
                return None
            return evaluation
        except json.JSONDecodeError as e:
            print(f"Error parsing cleaned JSON: {str(e)}")
            print("Cleaned JSON string:")
            print(json_str)
            return None
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error parsing JSON: {str(e)}")
        return None


def calculate_total_score(characteristic_scores):
    """Calculate weighted total score from individual characteristic scores."""
    total_score = 0.0
    for char_name, details in characteristic_scores.items():
        if char_name in ECOLOGICAL_CHARACTERISTICS:
            weight = ECOLOGICAL_CHARACTERISTICS[char_name]["weight"]
            score = details["score"]
            total_score += weight * score
    return total_score

def update_individual_metadata(individual_dir, evaluation):
    """Update the individual's metadata.json with ecological assessment."""
    metadata_path = os.path.join(individual_dir, "metadata.json")
    population_dir = os.path.dirname(individual_dir)
    individual_name = os.path.basename(individual_dir)
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        metadata = {}
        
    # Get generation number and objective value from population metadata
    try:
        with open(os.path.join(population_dir, "population_metadata.json"), 'r') as f:
            pop_metadata = json.load(f)
            # Get generation number
            if "generations" in pop_metadata:
                generation = len(pop_metadata["generations"])
                metadata["generation"] = generation
            
            # Get objective value from current_best_performers
            if "current_best_performers" in pop_metadata:
                for performer in pop_metadata["current_best_performers"]:
                    if performer["individual"] == individual_name:
                        objective_value = performer["objective_value"]
                        break
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        generation = None
        objective_value = None
    
    # Add ecological assessment
    metadata["ecological_assessment"] = {
        "qualitative_description": evaluation["qualitative_description"],
        "characteristic_scores": evaluation["characteristic_scores"],
        "total_score": calculate_total_score(evaluation["characteristic_scores"]),
        "characteristics_present": [
            char_name for char_name, details in evaluation["characteristic_scores"].items()
            if details["score"] > 0
        ]
    }
    
    # Add objective value if available
    if objective_value is not None:
        metadata["ecological_assessment"]["objective_value"] = objective_value
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def update_population_metadata(population_dir, generation):
    """
    Update population_metadata.json by reading 'scores.json' from each
    current best performer and storing an 'ecological_score' on that entry.

    Prefers 'aggregate_scores.final_score' if present; otherwise uses
    'aggregate_scores.normalized_total'. Writes back only if a change occurred.
    """
    metadata_path = os.path.join(population_dir, "population_metadata.json")
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Note: population metadata not found or invalid at {metadata_path}")
        return

    if "current_best_performers" not in metadata:
        return

    updated = False

    for performer in metadata["current_best_performers"]:
        indiv = performer.get("individual")
        if not indiv:
            continue

        scores_path = os.path.join(population_dir, indiv, "scores.json")
        try:
            with open(scores_path, 'r') as f:
                scores = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue

        agg = scores.get("aggregate_scores", {})
        eco_score = agg.get("final_score", agg.get("normalized_total"))
        if isinstance(eco_score, (int, float)):
            # Write if absent or changed
            if performer.get("ecological_score") != eco_score:
                performer["ecological_score"] = eco_score
                updated = True

    if updated:
        print(f"Updating ecological scores in population metadata for generation {generation}")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

def evaluate_individual(individual_dir, overwrite: bool = False):
    """
    Evaluate ecological characteristics of an individual model and save results to scores.json.
    If overwrite=False (default), skip re-evaluation when a valid scores.json exists.
    If overwrite=True, back up existing scores.json and recompute fresh scores.
    """
    scores_path = os.path.join(individual_dir, "scores.json")

    if not overwrite:
        # Skip re-evaluation if scores.json already exists and parses
        try:
            with open(scores_path, 'r') as f:
                existing = json.load(f)
            print(f"Individual in {individual_dir} has already been evaluated (scores.json found), skipping...")
            return existing
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    else:
        # If overwriting and a scores.json exists, back it up
        if os.path.exists(scores_path):
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_path = f"{scores_path}.bak-{ts}"
            try:
                shutil.copy2(scores_path, backup_path)
                print(f"[overwrite] Backed up existing scores.json -> {backup_path}")
            except Exception as e:
                print(f"[overwrite] Warning: could not back up existing scores.json: {e}")

    # Read the C++ model
    model_path = os.path.join(individual_dir, "model.cpp")
    try:
        with open(model_path, 'r') as f:
            model_content = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find model.cpp in {individual_dir}")
        return None

    # Evaluate via LLM
    evaluation = evaluate_model(model_content)
    if not evaluation:
        print(f"Error: LLM evaluation failed for {individual_dir}")
        return None

    # Persist to scores.json
    scores_payload = write_scores_file(individual_dir, evaluation)

    # Update population metadata ecological_score from scores.json
    population_dir = os.path.dirname(individual_dir)
    try:
        if "POPULATION_" in os.path.basename(population_dir):
            generation = int(os.path.basename(population_dir).split('_')[1])
        else:
            generation = scores_payload.get("generation")
        update_population_metadata(population_dir, generation)
    except (ValueError, IndexError):
        print(f"Note: Not updating population metadata for {individual_dir} (not in a population directory)")

    return scores_payload

def write_scores_file(individual_dir, evaluation):
    """
    Write ecological assessment to a new file 'scores.json' inside the individual's directory.
    The file contains the per-characteristic scores plus aggregate scores and handy metadata.

    Structure:
    {
      "individual": "...",
      "generation": int | null,
      "qualitative_description": "...",
      "characteristic_scores": { ... },   # returned by evaluate_model(...)
      "aggregate_scores": {
        "raw_total": float,               # sum(weight * score), unnormalized
        "normalized_total": float,        # raw_total / (sum(weights) * 3.0) -> 0..1
        "final_score": float              # if provided by evaluation; else normalized_total
      },
      "objective_value": float | null,    # if found in population metadata
      ... optional passthroughs from evaluation (audit, etc.)
    }
    """
    scores_path = os.path.join(individual_dir, "scores.json")
    population_dir = os.path.dirname(individual_dir)
    individual_name = os.path.basename(individual_dir)

    # # Optional context from population metadata: generation & objective_value
    # generation = None
    # objective_value = None
    # pop_meta_path = os.path.join(population_dir, "population_metadata.json")
    # try:
    #     with open(pop_meta_path, 'r') as f:
    #         pop_metadata = json.load(f)
    #     if "generations" in pop_metadata:
    #         generation = len(pop_metadata["generations"])
    #     for performer in pop_metadata.get("current_best_performers", []):
    #         if performer.get("individual") == individual_name:
    #             objective_value = performer.get("objective_value")
    #             break
    # except (FileNotFoundError, json.JSONDecodeError, KeyError):
    #     pass

    # Aggregate scores
    char_scores = evaluation.get("characteristic_scores", {}) or {}
    raw_total = float(calculate_total_score(char_scores))  # unnormalized
    # Normalize to 0..1 using current weights and max per-component score = 3
    sum_weights = sum(d["weight"] for d in ECOLOGICAL_CHARACTERISTICS.values())
    normalized_total = raw_total / (sum_weights * 3.0) if sum_weights > 0 else 0.0

    # Prefer explicit final_score if the evaluator provides one; else normalized_total
    main_score = float(evaluation.get("final_score", normalized_total))

    # Build payload
    scores_payload = {
        "individual": individual_name,
        # "generation": generation,
        "qualitative_description": evaluation.get("qualitative_description", ""),
        "characteristic_scores": char_scores,
        "aggregate_scores": {
            "raw_total": raw_total,
            "normalized_total": normalized_total,
            "final_score": main_score
        }
    }
    # if objective_value is not None:
    #     scores_payload["objective_value"] = objective_value
    # Optional: persist extra component fields if provided by the evaluator
    if "extra_components_count" in evaluation:
        scores_payload["extra_components_count"] = evaluation["extra_components_count"]
    if "extra_components_description" in evaluation:
        scores_payload["extra_components_description"] = evaluation["extra_components_description"]
    # Pass-through optional fields if present
    for key in ("audit", "extra_components", "missing_components",
                "mass_balance_assessment", "penalty_total", "fidelity_score"):
        if key in evaluation:
            scores_payload[key] = evaluation[key]

    # Write scores.json
    with open(scores_path, 'w') as f:
        json.dump(scores_payload, f, indent=2)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate ecological characteristics for one individual.")
    parser.add_argument("-i", "--individual", required=True,
                        help="Path to the individual directory (contains model.cpp).")
    parser.add_argument("-f", "--overwrite", action="store_true",
                        help="Overwrite existing scores.json if present.")
    args = parser.parse_args()

    result = evaluate_individual(args.individual, overwrite=args.overwrite)
    if result:
        print("Evaluation complete. Check metadata files for results.")