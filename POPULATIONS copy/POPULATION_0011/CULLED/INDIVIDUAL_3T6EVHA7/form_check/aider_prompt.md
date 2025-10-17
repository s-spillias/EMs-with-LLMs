You are assisting with an NPZ model comparison workflow.
**Goal**: Edit `plot_forms_NPZ.py` so that its LEMMA model section exactly reflects the
functional forms and optimized parameter values of the TMB model defined by:
- C++ model: `model.cpp` (in this folder)
- Parameters: `parameters.json` (in this folder; may contain a `found_value` mapping)
- Truth reference file: `Data/NPZ_example/NPZ_model.py`
### Required changes in `plot_forms_NPZ.py`
1) **LEMMA parameter block**
- Replace hard-coded values with optimized values from `parameters.json`.
- Preserve variable names used in the script (mu_max, K_N, gmax, ...).
- Add any required parameters present in this model but missing in the script; ignore unused ones.
2) **LEMMA RHS function(s)**
- Update `rhs_tmb_like` so equations match `model.cpp` functional forms.
- If a process is absent, set its corresponding flux series to `None`.
3) **Machine-readable overall score**
- Print a single line: OVERALL_ISD_SUM: <float> (sum across FLUXES absent-as-zero).
4) **Scope**
- Do not modify Truth; only adjust LEMMA side.
- Save edits directly to `plot_forms_NPZ.py`.
**Model ID**: `CULLED_INDIVIDUAL_3T6EVHA7`

### Figure saving & summary output (REQUIRED)
- **Do not display** figures (`plt.show()` must NOT be called). Use a non-interactive backend.
- Save every figure to a subfolder **./figs** (create if missing). Respect environment variables:
- `SAVE_PLOTS_DIR` (default: "figs")
- `SAVE_PLOTS_FORMAT` (default: "png")
- Use descriptive filenames prefixed with the model id "CULLED_INDIVIDUAL_3T6EVHA7" and a short chart name.
- Save at 300 DPI (`plt.savefig(..., dpi=300, bbox_inches="tight")`), then `plt.close("all")`.
- Emit **one** machine-readable summary line near the end:
`SUMMARY_JSON: { "overall_isd_sum": <float>, "n_states": <int or null>, "n_fluxes_compared": <int or null>, "notes": "<optional>" }`
- Also emit the score line exactly as:
`OVERALL_ISD_SUM: <float>`

Please make minimal, surgical edits and SAVE the file.