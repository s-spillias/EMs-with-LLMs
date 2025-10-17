import os
from scripts_analysis.form_utils import (
    run_npz_workflow,
    load_params_with_mapping,
    EPS,
    NPZ_COMPONENT_SPECS,
)

# =============================================================================
# TRUTH REFERENCE (VISIBLE, DO NOT EDIT)
# =============================================================================
TRUTH_PARAMS = {
    "a": 0.2, "b": 0.2, "c": 0.4, "eN": 0.03,
    "k": 0.05, "q": 0.075, "r": 0.10, "s": 0.04,
    "N0_mix": 0.6,
    "alpha": 0.25, "beta": 0.33, "gamma": 0.5,
    "lambda_": 0.6, "mu_P": 0.035,
}

TRUTH_EQUATIONS = r"""
# Nutrient–Phytoplankton–Zooplankton (TRUTH) functional forms (Do Not Edit)

uptake  = (N / (eN + N + EPS)) * (a / (b + c*P + EPS)) * P
grazing = lambda_ * (P^2) / (mu_P^2 + P^2 + EPS) * Z

dP/dt = uptake - grazing - r*P - (s + k)*P
dZ/dt = alpha*grazing - q*Z
dN/dt = -uptake + r*P + beta*grazing + gamma*q*Z + k*(N0_mix - N)

Derived TRUTH flux components:
P_mort  = r*P + (s + k)*P
Z_growth= alpha*grazing
Z_mort  = q*Z
reminZ  = gamma*q*Z
reminP  = r*P
reminG  = beta*grazing
mixN    = k*(N0_mix - N)
dN      = -uptake + reminP + reminG + reminZ + mixN
"""

# =============================================================================
# LEMMA PLUGIN (LLM edits ONLY inside this block)
# =============================================================================
# Map symbol names -> keys in parameters_metadata.json
# IMPORTANT: Values are read from `optimized_value`.
# STRICT: no defaults injected; fix mapping if a key is missing.
PARAM_MAP = {
    # Example keys; verify/adjust to match parameters_metadata.json in each workspace
    "a": "a", "b": "b", "c": "c", "eN": "eN",
    "k": "k", "q": "q", "r": "r", "s": "s",
    "N0_mix": "N0_mix",
    "alpha": "alpha", "beta": "beta", "gamma": "gamma",
    "lambda_": "lambda", "mu_P": "mu_P",
}

def lemma_fluxes(t, N, P, Z, p):
    """
    Implement LEMMA (TMB) functional forms with parameters loaded from parameters_metadata.json.
    You MUST return at least:
        {'uptake': ..., 'grazing': ...}
    You MAY return additional components (e.g., 'P_mort', 'mixN'); the framework will align
    the TRUTH∪LEMMA union and impute missing counterparts for ISD.

    IMPORTANT:
    - Access parameters strictly via p['...'] that are mapped by PARAM_MAP to `optimized_value`.
    - Do NOT rely on defaults; fix mapping if keys are missing.
    - Do NOT edit the TRUTH reference block above.
    """
    # --- LLM: Replace the example below with the exact LEMMA forms from model.cpp ---
    # uptake  = (N / (p["eN"] + N + EPS)) * (p["a"] / (p["b"] + p["c"]*P + EPS)) * P
    # grazing = p["lambda_"] * (P**2) / (p["mu_P"]**2 + P**2 + EPS) * Z
    # return {"uptake": uptake, "grazing": grazing}

    raise NotImplementedError("Insert LEMMA functional forms and return at least {'uptake', 'grazing'}.")

# =============================================================================
# End LEMMA PLUGIN
# =============================================================================

def main():
    # STRICT source: read from parameters_metadata.json (optimized_value)
    params = load_params_with_mapping("parameters_metadata.json", PARAM_MAP, defaults=None)

    # Surface TRUTH components & TRUTH reference for the LLM (read-only)
    print("TRUTH_COMPONENTS:", list(NPZ_COMPONENT_SPECS.keys()))
    print("TRUTH_PARAMS_REFERENCE:", TRUTH_PARAMS)
    print("TRUTH_EQUATIONS_REFERENCE_BEGIN")
    print(TRUTH_EQUATIONS.strip())
    print("TRUTH_EQUATIONS_REFERENCE_END")

    notes = "NPZ comparison via plugin (LLM edits: PARAM_MAP + lemma_fluxes; params from parameters_metadata.json/optimized_value)"
    run_npz_workflow(lemma_fluxes, params, notes=notes)

if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()