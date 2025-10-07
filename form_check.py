#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch LEMMA analysis orchestrator using `make_script`

Changes in this version:
- Save plots to files (no GUI display):
  * Uses MPLBACKEND=Agg for subprocess
  * Provides SAVE_PLOTS_DIR (default: 'figs') and SAVE_PLOTS_FORMAT (default: 'png')
  * Prompt instructs plot script to save figures into ./figs and never call plt.show()

- Create spreadsheet with summary:
  * Writes Results/batch_summary.xlsx (configurable via --excel-out)
  * Columns include: timestamp, model_name (relative path), model_type, overall_isd_sum,
    objective (from model_report.json highest integer iteration), plus keys parsed from
    a single-line machine-readable summary emitted by the plot script:
      SUMMARY_JSON: { ... }

- Parse objective from model_report.json:
  * Selects entry at highest integer key in 'iterations' and reads its 'objective'
"""

import argparse
import csv
import datetime as dt
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# NEW: pandas for Excel writing
import pandas as pd  # Requires openpyxl engine for .xlsx I/O

FLOAT_RE = re.compile(r"\[+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?")

def debug(msg: str):
    print(f"[batch] {msg}")

def sanitize_rel_path(path: Path) -> str:
    rel = str(path)
    return re.sub(r"[^A-Za-z0-9._\-]+", "_", rel).strip("_")

def find_model_pairs(root: Path):
    pairs = []
    for dirpath, _dirnames, filenames in os.walk(root):
        filenames = set(filenames)
        if "model.cpp" in filenames and "parameters.json" in filenames:
            d = Path(dirpath)
            pairs.append({
                "dir": d,
                "model_cpp": d / "model.cpp",
                "opt_json": d / "parameters.json",
            })
    return sorted(pairs, key=lambda x: str(x["dir"]))

def write_prompt(
    workspace: Path,
    rel_model_dir: Path,
    model_cpp: Path,
    opt_json: Path,
    truth_rel: Path,
    plot_script_name: str,
    model_type: str
) -> str:
    """
    Generate dynamic prompt instructions and also REQUIRE the plot script to:
      * Save all figures into ./figs (or env SAVE_PLOTS_DIR), format env SAVE_PLOTS_FORMAT (default png)
      * Never call plt.show(); close figures after saving
      * Emit a single SUMMARY_JSON: {...} line with high-level quantitative results
      * Emit OVERALL_ISD_SUM: <float> line
    """
    model_id = sanitize_rel_path(rel_model_dir)

    # Shared figure/summary requirements appended to both NPZ and COTS prompts
    figure_and_summary_clause = f"""
### Figure saving & summary output (REQUIRED)
- **Do not display** figures (`plt.show()` must NOT be called). Use a non-interactive backend.
- Save every figure to a subfolder **./figs** (create if missing). Respect environment variables:
  - `SAVE_PLOTS_DIR` (default: "figs")
  - `SAVE_PLOTS_FORMAT` (default: "png")
- Use descriptive filenames prefixed with the model id "{model_id}" and a short chart name.
- Save at 300 DPI where reasonable (e.g., `plt.savefig(..., dpi=300, bbox_inches="tight")`), then `plt.close("all")`.
- Emit **one** machine-readable summary line near the end:
  `SUMMARY_JSON: {{ "overall_isd_sum": <float>, "n_states": <int or null>, "n_fluxes_compared": <int or null>, "notes": "<optional>" }}`
  (Add any other relevant top-level numeric metrics from your analysis.)
- Also emit the score line exactly as:
  `OVERALL_ISD_SUM: <float>`
"""

    if model_type == "NPZ":
        prompt = f"""You are assisting with an NPZ model comparison workflow.
**Goal**: Edit `{plot_script_name}` so that its LEMMA model section exactly reflects the
functional forms and optimized parameter values of the TMB model defined by:
- C++ model: `{model_cpp.name}` (in this folder)
- Parameters: `{opt_json.name}` (in this folder; may contain a `found_value` mapping)
- Truth reference file for naming/context: `{truth_rel}`

### Required changes in `{plot_script_name}`
1) **LEMMA parameter block**
- Replace hard-coded values with the optimized values from `{opt_json.name}`.
- Preserve variable names used in the script (e.g., `mu_max`, `K_N`, `gmax`, `h`, `K_P`, `e_g`, `m_Z`, `m_P`, `d_N`,
  and optional `log_sigma_*`). If `{opt_json.name}` uses a `found_value` dict or similar, map names accordingly.
- Add any required parameters present in this model but missing in the script; ignore unused ones.

2) **LEMMA RHS function(s)**
- Update `rhs_tmb_like` (or create a clearly named alternative used consistently) so equations match `model.cpp`
  functional forms (uptake, grazing, mortalities, remin, etc.).
- If a process is absent in the C++ LEMMA, set its corresponding flux series in the analysis section to `None`
  so the plotting/metrics annotate its absence (consistent with script conventions).

3) **Machine-readable overall score**
- After existing prints at the end, print a single line:
  OVERALL_ISD_SUM: <float>
  where `<float>` is the **sum of the ISD values** across the **FLUXES (absent-as-zero)** comparison
  (the table titled "Integrated squared difference — FLUXES (absent-as-zero)").

4) **Scope**
- Do not modify the Truth model. Only adjust the LEMMA side.
- Save edits directly to `{plot_script_name}`.

**Model ID**: `{model_id}`

{figure_and_summary_clause}

Please make minimal, surgical edits and SAVE the file."""
    else:  # COTS
        prompt = f"""You are assisting with a CoTS (Crown of Thorns Starfish) model comparison workflow.
**Goal**: Edit `{plot_script_name}` so that its LEMMA model section reflects the TMB model defined by:
- C++ model: `{model_cpp.name}` (in this folder)
- Parameters: `{opt_json.name}` (in this folder)
- Truth reference file for context: `{truth_rel}`

### Required changes in `{plot_script_name}`
1) **LEMMA parameter block**
- Replace hard-coded values with the parameter values from `{opt_json.name}` (parameters.json).
- Ensure all parameters required by the TMB model are present and correctly assigned.

2) **LEMMA RHS function(s)**
- Update the ODE system so equations match the functional forms in `{model_cpp.name}`.
- Include SST anomaly term. If no SST data is available, use a constant SST anomaly for this comparison.

3) **Machine-readable overall score**
- After existing prints at the end, print a single line:
  OVERALL_ISD_SUM: <float>
  where `<float>` is the sum of ISD values across FLUXES (absent-as-zero) comparison.

4) **Scope**
- Do not modify the Truth model. Only adjust the LEMMA side.
- Save edits directly to `{plot_script_name}`.

**Model ID**: `{model_id}`

{figure_and_summary_clause}

Please make minimal, surgical edits and SAVE the file."""

    # Write prompt to workspace for auditability
    (workspace / "aider_prompt.md").write_text(prompt, encoding="utf-8")
    # Log which model type was used
    debug(f"Using model-type: {model_type} (prompt adapted accordingly)")
    return prompt

def parse_overall_isd(stdout: str):
    for line in stdout.splitlines():
        if line.strip().startswith("OVERALL_ISD_SUM:"):
            try:
                return float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return None

# NEW: Parse a one-line JSON summary: SUMMARY_JSON: {...}
def parse_summary_json(stdout: str):
    for line in stdout.splitlines():
        if line.strip().startswith("SUMMARY_JSON:"):
            payload = line.split(":", 1)[1].strip()
            try:
                return json.loads(payload)
            except Exception:
                return None
    return None

def import_make_script(module_and_func=None, file_path=None, func_name="make_script"):
    if module_and_func:
        mod, fn = module_and_func.split(":") if ":" in module_and_func else (module_and_func, func_name)
        make_mod = importlib.import_module(mod)
        return getattr(make_mod, fn)
    if file_path:
        spec = importlib.util.spec_from_file_location("make_script_mod", file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import from file: {file_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, func_name)
    raise ValueError("Provide --make-script-import or --make-script-file")

# UPDATED: allow env injection (Agg backend etc.)
def run_plot(plot_name: str, cwd: Path, python_cmd: str, env: dict):
    try:
        res = subprocess.run([python_cmd, plot_name], cwd=str(cwd), capture_output=True, text=True, env=env)
        return res.returncode, res.stdout, res.stderr
    except FileNotFoundError:
        return 127, "", f"Python not found: {python_cmd}"

def load_parameters(parameters_file):
    """Load parameters from a parameters.json file with retry mechanism to strip comments."""
    try:
        with open(parameters_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        # Retry by stripping comment lines starting with '#' or '//'
        try:
            cleaned_lines = []
            with open(parameters_file, 'r', encoding='utf-8') as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith('#') or stripped.startswith('//'):
                        continue
                    cleaned_lines.append(line)
            cleaned_content = ''.join(cleaned_lines)
            return json.loads(cleaned_content)
        except Exception as e:
            print(f"Error loading parameters file after retry: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading parameters file: {e}")
        sys.exit(1)

# NEW: Read objective from <model_source_dir>/model_report.json
def read_objective_from_report(model_source_dir: Path):
    """
    Expecting a JSON structure like:
    {
      "iterations": {
        "0": { "objective": ... },
        "1": { "objective": ... },
        ...
      },
      ...
    }
    We pick the entry at the highest integer key in 'iterations' and return its 'objective' (float).
    """
    report = model_source_dir / "model_report.json"
    if not report.exists():
        return None
    try:
        with report.open("r", encoding="utf-8") as f:
            data = json.load(f)
        iters = data.get("iterations", {})
        if not isinstance(iters, dict) or not iters:
            return None
        # Keys may be strings; convert to ints where possible
        max_k = None
        for k in iters.keys():
            try:
                i = int(k)
                if (max_k is None) or (i > max_k):
                    max_k = i
            except Exception:
                continue
        if max_k is None:
            return None
        last = iters.get(str(max_k), {})
        # objective may be nested, but we assume top-level 'objective' exists
        obj = last.get("objective", None)
        if isinstance(obj, (int, float)):
            return float(obj)
        # Sometimes objective could be under other keys; try fallback common patterns
        if isinstance(last, dict):
            # try e.g., last.get('report', {}).get('objective')
            inner = last.get("report") or last.get("metrics") or {}
            if isinstance(inner, dict):
                val = inner.get("objective")
                if isinstance(val, (int, float)):
                    return float(val)
        return None
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(description="Batch LEMMA functional-form analysis using make_script()")
    ap.add_argument("--parameters", default="parameters.json", help="Path to parameters.json")
    ap.add_argument("--plot-script", default="plot_forms.py", help="Base analysis script")
    ap.add_argument("--python-cmd", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model-type", default="NPZ", choices=["NPZ", "COTS"])
    ap.add_argument("--make-script-import", default=None)
    ap.add_argument("--make-script-file", default=None)
    ap.add_argument("--make-script-func", default="make_script")

    # NEW: figure/output options
    ap.add_argument("--fig-dir", default="figs", help="Where to save generated figures inside each workspace")
    ap.add_argument("--fig-format", default="png", choices=["png", "pdf", "svg"], help="Figure file format")
    ap.add_argument("--excel-out", default="Results/batch_summary.xlsx", help="Path to output Excel spreadsheet")

    args = ap.parse_args()

    params_data = load_parameters(args.parameters)
    config = {p.get("parameter"): p.get("value") for p in params_data.get("parameters", [])}

    # Root inference unchanged
    if "root" in config:
        root = Path(config["root"])
    else:
        # Infer from --parameters path
        root = Path(args.parameters).parent.parent  # e.g., POPULATIONS/POPULATION_0008

    plot_src = Path(args.plot_script)
    truth_src = Path(config.get("npz_truth", "Data/NPZ_example/NPZ_model.py")) if args.model_type == "NPZ" else Path(config.get("cots_truth", "FROM_JACOB/CoTSmodel_v4.cpp"))
    truth_params = Path(config.get("cots_params", "FROM_JACOB/ControlFile.R"))

    runs_root = Path("Results")
    runs_root.mkdir(parents=True, exist_ok=True)

    temperature = float(config.get("temperature", 0.1))
    llm_choice = config.get("llm_choice", "anthropic_sonnet")

    if not plot_src.exists():
        print(f"ERROR: plot script not found: {plot_src}")
        sys.exit(2)

    pairs = find_model_pairs(root)
    if not pairs:
        print(f"No (model.cpp, parameters.json) pairs found under {root}")
        sys.exit(0)

    debug(f"Found {len(pairs)} model pairs under {root}")

    try:
        make_script = import_make_script(args.make_script_import, args.make_script_file, args.make_script_func)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    results = []
    ts = dt.datetime.now().isoformat(timespec="seconds")

    # NEW: Aggregate all summary keys to shape the Excel columns
    all_summary_keys = set()

    for i, p in enumerate(pairs, 1):
        rel = p["dir"].relative_to(root)
        safe = sanitize_rel_path(rel)
        work = runs_root / safe
        work.mkdir(parents=True, exist_ok=True)
        debug(f"[{i}/{len(pairs)}] Workspace: {work}")

        # Copy scripts and inputs into workspace
        shutil.copy2(plot_src, work / plot_src.name)
        shutil.copy2(p["model_cpp"], work / p["model_cpp"].name)
        shutil.copy2(p["opt_json"], work / p["opt_json"].name)

        if args.model_type == "NPZ" and truth_src.exists():
            dest_truth = work / "Data/NPZ_example/NPZ_model.py"
            dest_truth.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(truth_src, dest_truth)
        elif args.model_type == "COTS" and truth_src.exists() and truth_params.exists():
            dest_truth = work / "FROM_JACOB/CoTSmodel_v4.cpp"
            dest_truth.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(truth_src, dest_truth)
            dest_params = work / "FROM_JACOB/ControlFile.R"
            dest_params.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(truth_params, dest_params)

        truth_rel_path = "Data/NPZ_example/NPZ_model.py" if args.model_type == "NPZ" else "FROM_JACOB/CoTSmodel_v4.cpp"

        prompt_text = write_prompt(
            work, rel, Path(p["model_cpp"].name), Path(p["opt_json"].name),
            Path(truth_rel_path), plot_src.name, args.model_type
        )

        if args.dry_run:
            debug("Dry-run: skipping make_script and plot execution")
            continue

        filenames = [plot_src.name]
        read_files = [p["model_cpp"].name, p["opt_json"].name]
        if args.model_type == "NPZ":
            read_files.append("Data/NPZ_example/NPZ_model.py")
        else:
            read_files.extend(["FROM_JACOB/CoTSmodel_v4.cpp", "FROM_JACOB/ControlFile.R"])

        cwd_prev = os.getcwd()
        os.chdir(str(work))
        try:
            _coder = make_script(
                filenames=filenames,
                read_files=read_files,
                prompt=prompt_text,
                temperature=temperature,
                llm_choice=llm_choice
            )
        finally:
            os.chdir(cwd_prev)

        # NEW: Prepare environment for non-interactive plotting and saving
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"  # force non-interactive
        env["SAVE_PLOTS_DIR"] = args.fig_dir
        env["SAVE_PLOTS_FORMAT"] = args.fig_format
        env["NO_SHOW"] = "1"

        rc, out, err = run_plot(plot_src.name, cwd=work, python_cmd=args.python_cmd, env=env)

        (work / "stdout.txt").write_text(out, encoding="utf-8")
        (work / "stderr.txt").write_text(err or "", encoding="utf-8")

        overall = parse_overall_isd(out)
        summary = parse_summary_json(out) or {}

        # Track keys to build Excel columns
        for k in summary.keys():
            all_summary_keys.add(k)

        # Objective from source directory's model_report.json
        objective = read_objective_from_report(p["dir"])

        results.append({
            "timestamp": ts,
            "model_rel": str(rel),
            "workspace": str(work),
            "return_code": rc,
            "overall_isd_sum": overall,
            "objective_value": objective,
            "model_type": args.model_type,
            "summary": summary
        })

        if overall is not None:
            debug(f"OVERALL_ISD_SUM = {overall:.6g}")
        else:
            debug("WARNING: Could not parse OVERALL_ISD_SUM")

    # Write CSV / JSON (enriched)
    out_csv = runs_root / "batch_results.csv"
    out_json = runs_root / "batch_results.json"

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # CSV: keep original columns and add objective + model_type + summary_json
        w.writerow(["timestamp", "model_rel", "workspace", "return_code", "overall_isd_sum", "objective", "model_type", "summary_json"])
        for r in results:
            w.writerow([
                r["timestamp"], r["model_rel"], r["workspace"], r["return_code"],
                r["overall_isd_sum"], r["objective"], r["model_type"], json.dumps(r.get("summary") or {}, ensure_ascii=False)
            ])

    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "results": results}, f, indent=2)

    # Build Excel-friendly rows by flattening `summary` keys
    excel_rows = []
    for r in results:
        row = {
            "timestamp": r["timestamp"],
            "model_name": r["model_rel"],  # friendlier column name
            "model_type": r["model_type"],
            "workspace": r["workspace"],
            "return_code": r["return_code"],
            "overall_isd_sum": r["overall_isd_sum"],
            "objective": r["objective"],
        }
        for k in all_summary_keys:
            row[k] = (r.get("summary") or {}).get(k, None)
        excel_rows.append(row)

    # DataFrame -> Excel
    df = pd.DataFrame(excel_rows)
    excel_out = Path(args.excel_out)
    excel_out.parent.mkdir(parents=True, exist_ok=True)
    # Important: engine openpyxl for .xlsx
    df.to_excel(excel_out, index=False, engine="openpyxl")

    ok = [r for r in results if isinstance(r["overall_isd_sum"], (int, float))]
    fail = [r for r in results if r["overall_isd_sum"] is None]

    print(f"\nCompleted {len(results)} model(s). Parsed scores: {len(ok)}; Missing: {len(fail)}")
    if ok:
        best = sorted(ok, key=lambda r: r["overall_isd_sum"])[:5]
        print("Lowest OVERALL_ISD_SUM (top 5):")
        for r in best:
            print(f" {r['model_rel']}: {r['overall_isd_sum']}")

if __name__ == "__main__":
    main()
