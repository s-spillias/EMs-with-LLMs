#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch LEMMA analysis orchestrator with:
- Dynamic plot script selection (NPZ vs COTS) per model
- Ability to run by passing only --population (defaults for others)
- Workspace isolation, figure saving, and Excel summary

Enhancements:
- Per-individual workspaces live INSIDE each individual's original directory
  (configurable via --per-individual-subdir, default: "form_check")
- Append progress to a CSV and rebuild the Excel synthesis AFTER EACH ITERATION
  so progress is trackable and the run can resume with --resume
- NEW: Vendor form_utils.py into each workspace as scripts_analysis/form_utils.py so plot scripts can import it
"""

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Use the repository's parser helpers (keep as-is if you already have scripts_analysis)
from scripts_analysis.form_utils import parse_overall_isd, parse_summary_json  # noqa: E402

# Existing helper to generate the model-editing prompt
from scripts.make_model import make_script  # noqa: E402

# Excel synthesis
import pandas as pd  # Requires openpyxl for .xlsx I/O

FLOAT_RE = re.compile(r"\[\+\-]?\d+(?:\.\d+)?(?:[eE][\+\-]?\d+)?")

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

# --- Normalization helper: strip trailing per-individual workspace subdir ---
from pathlib import Path

def normalize_rel(rel: Path, subdir_name: str) -> Path:
    """
    If rel ends with the workspace subdir (e.g., 'form_check'), return the parent;
    otherwise return rel unchanged. Used to prevent duplicate model_rel values.
    """
    return rel.parent if rel.name == subdir_name else rel

def write_prompt(
    workspace: Path,
    rel_model_dir: Path,
    model_cpp: Path,
    opt_json: Path,
    truth_rel: Path,
    plot_script_name: str,
    model_type: str
) -> str:
    """Writes aider_prompt.md with instructions."""
    model_id = sanitize_rel_path(rel_model_dir)
    figure_and_summary_clause = f"""
### Figure saving & summary output (REQUIRED)
- **Do not display** figures (`plt.show()` must NOT be called). Use a non-interactive backend.
- Save every figure to a subfolder **./figs** (create if missing). Respect environment variables:
- `SAVE_PLOTS_DIR` (default: "figs")
- `SAVE_PLOTS_FORMAT` (default: "png")
- Use descriptive filenames prefixed with the model id "{model_id}" and a short chart name.
- Save at 300 DPI (`plt.savefig(..., dpi=300, bbox_inches="tight")`), then `plt.close("all")`.
- Emit **one** machine-readable summary line near the end:
`SUMMARY_JSON: {{ "overall_isd_sum": <float>, "n_states": <int or null>, "n_fluxes_compared": <int or null>, "notes": "<optional>" }}`
- Also emit the score line exactly as:
`OVERALL_ISD_SUM: <float>`
"""
    if model_type == "NPZ":
        prompt = f"""You are assisting with an NPZ model comparison workflow.
**Goal**: Edit `{plot_script_name}` so that its LEMMA model section exactly reflects the
functional forms and optimized parameter values of the TMB model defined by:
- C++ model: `{model_cpp.name}` (in this folder)
- Parameters: `{opt_json.name}` (in this folder; may contain a `found_value` mapping)
- Truth reference file: `{truth_rel}`
### Required changes in `{plot_script_name}`
1) **LEMMA parameter block**
- Replace hard-coded values with optimized values from `{opt_json.name}`.
- Preserve variable names used in the script (mu_max, K_N, gmax, ...).
- Add any required parameters present in this model but missing in the script; ignore unused ones.
2) **LEMMA RHS function(s)**
- Update `rhs_tmb_like` so equations match `{model_cpp.name}` functional forms.
- If a process is absent, set its corresponding flux series to `None`.
3) **Machine-readable overall score**
- Print a single line: OVERALL_ISD_SUM: <float> (sum across FLUXES absent-as-zero).
4) **Scope**
- Do not modify Truth; only adjust LEMMA side.
- Save edits directly to `{plot_script_name}`.
**Model ID**: `{model_id}`
{figure_and_summary_clause}
Please make minimal, surgical edits and SAVE the file."""
    else:
        prompt = f"""You are assisting with a CoTS (Crown of Thorns Starfish) model comparison workflow.
**Goal**: Edit `{plot_script_name}` so that its LEMMA model reflects the TMB model defined by:
- C++ model: `{model_cpp.name}`
- Parameters: `{opt_json.name}`
- Truth reference file: `{truth_rel}`
### Required changes in `{plot_script_name}`
1) **LEMMA parameter block**
- Replace hard-coded values with values from `{opt_json.name}`.
2) **LEMMA RHS function(s)**
- Update equations to match `{model_cpp.name}` functional forms.
- Include SST anomaly term; use a constant if no SST series is available.
3) **Machine-readable overall score**
- Print OVERALL_ISD_SUM: <float> (sum across FLUXES absent-as-zero).
4) **Scope**
- Do not modify Truth; only adjust LEMMA side.
- Save edits directly to `{plot_script_name}`.
**Model ID**: `{model_id}`
{figure_and_summary_clause}
Please make minimal, surgical edits and SAVE the file."""
    (workspace / "aider_prompt.md").write_text(prompt, encoding="utf-8")
    debug(f"Using model-type: {model_type} (prompt adapted accordingly)")
    return prompt

def run_plot(plot_name: str, cwd: Path, python_cmd: str, env: dict):
    try:
        res = subprocess.run([python_cmd, plot_name], cwd=str(cwd), capture_output=True, text=True, env=env)
        return res.returncode, res.stdout, res.stderr
    except FileNotFoundError:
        return 127, "", f"Python not found: {python_cmd}"

def load_parameters(parameters_file):
    """Load parameters from parameters.json with retry mechanism to strip comments."""
    try:
        with open(parameters_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
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

import re
import json
from pathlib import Path

def read_objective_from_report(model_source_dir: Path):
    """
    Reads <model_source_dir>/model_report.json and returns the objective value from
    the latest iteration. Robust to:
      - key names 'objective_value' (preferred) or 'objective'
      - numbers stored as float or string
      - nested dicts like 'report' or 'metrics'

    Returns float or None.
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

        # Find the highest integer key in 'iterations'
        numeric_keys = []
        for k in iters.keys():
            # Accept plain integer strings like "4"
            if str(k).isdigit():
                numeric_keys.append(int(k))
            else:
                # Fallback: strip non-digits, try to parse
                stripped = re.sub(r"[^0-9]", "", str(k))
                if stripped.isdigit():
                    numeric_keys.append(int(stripped))

        if not numeric_keys:
            return None

        latest_k = str(max(numeric_keys))
        last = iters.get(latest_k, {}) or {}

        # Preferred key names (top-level first, then nested)
        candidate_keys = ("objective_value", "objective")

        # Helper: coerce to float if possible
        def _as_float(x):
            if isinstance(x, (int, float)):
                return float(x)
            if isinstance(x, str):
                try:
                    return float(x)
                except Exception:
                    return None
            return None

        # Try top-level
        for key in candidate_keys:
            val = _as_float(last.get(key, None))
            if val is not None:
                return val

        # Try nested dicts: 'report', then 'metrics'
        for nested in ("report", "metrics"):
            obj = last.get(nested) or {}
            if isinstance(obj, dict):
                for key in candidate_keys:
                    val = _as_float(obj.get(key, None))
                    if val is not None:
                        return val

        return None
    except Exception:
        return None

# ----------- New helpers for dynamic behavior -----------
def detect_model_type(population_dir: Path) -> str:
    """
    Determine model type by checking the 'report_file' field in population_metadata.json.
    Returns "NPZ" if "NPZ" is in the report file path, otherwise returns "COTS".
    """
    metadata_path = population_dir / "population_metadata.json"
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        report_file = metadata.get("report_file", "")
        if "NPZ" in report_file.upper():
            return "NPZ"
        else:
            return "COTS"
    except Exception as e:
        debug(f"Error reading population metadata: {e}")
        # Default to NPZ if metadata can't be read
        return "NPZ"

def _first_existing(path_like: Path, hints: list[Path]) -> Path | None:
    candidates = []
    if path_like is not None:
        candidates.append(path_like)
        if not path_like.is_absolute():
            candidates.append(Path(__file__).parent / path_like.name)
            candidates.append(Path.cwd() / path_like.name)
    candidates.extend(hints)
    for p in candidates:
        if p and Path(p).exists():
            return Path(p)
    return None

def resolve_plot_script(chosen_type: str,
                        override_script: str | None,
                        plot_npz: str = "plot_forms_NPZ.py",
                        plot_cots: str = "plot_forms_COTS.py") -> Path:
    """Pick the appropriate plot script; allow explicit override."""
    if override_script:
        p = _first_existing(Path(override_script), [])
        if p is None:
            raise FileNotFoundError(f"--plot-script not found: {override_script}")
        return p
    if chosen_type == "NPZ":
        p = _first_existing(Path(plot_npz), [
            Path(__file__).parent / "plot_forms_NPZ.py",
            Path.cwd() / "plot_forms_NPZ.py",
        ])
    else:
        p = _first_existing(Path(plot_cots), [
            Path(__file__).parent / "plot_forms_COTS.py",
            Path.cwd() / "plot_forms_COTS.py",
        ])
    if p is None:
        raise FileNotFoundError(
            f"Auto-selected plot script for {chosen_type} not found. "
            f"Tried '{plot_npz if chosen_type=='NPZ' else plot_cots}' in script dir and CWD. "
            f"Pass --plot-script to override."
        )
    return p

# ----------- NEW: locate & vendor form_utils.py into workspace -----------
def locate_form_utils() -> Path | None:
    """
    Try typical locations for form_utils.py in your repo and return the first that exists.
    """
    candidates = [
        Path(__file__).parent / "scripts_analysis/form_utils.py",
        Path.cwd() / "scripts_analysis/form_utils.py",
        Path(__file__).parent / "scripts_analysis/form_utils.py",
        Path.cwd() / "scripts_analysis/form_utils.py",
        Path(__file__).parent / "form_utils.py",
        Path.cwd() / "form_utils.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def vendor_form_utils_into_workspace(work: Path):
    """
    Copy form_utils.py as work/scripts_analysis/form_utils.py and ensure work/scripts_analysis/__init__.py exists.
    """
    src = locate_form_utils()
    scripts_dir = work / "scripts_analysis"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "__init__.py").write_text("", encoding="utf-8")
    if src is not None:
        shutil.copy2(src, scripts_dir / "form_utils.py")
        debug(f"Vendored form_utils.py into {scripts_dir}")
    else:
        debug("WARNING: Could not find form_utils.py to vendor into workspace. Plot script imports may fail.")

# ----------- incremental logging & Excel synthesis -----------
CSV_COLUMNS = ["timestamp", "model_rel", "workspace", "return_code",
               "overall_isd_sum", "objective", "model_type", "summary_json"]

def append_result_to_csv(csv_path: Path, row: dict):
    """Append one row to the CSV, creating it with header if it doesn't exist."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if is_new:
            w.writeheader()
        clean = {k: row.get(k, None) for k in CSV_COLUMNS}
        w.writerow(clean)

def rebuild_excel_from_csv(csv_path: Path, excel_out: Path):
    """
    Rebuild the flattened Excel workbook from the CSV. The CSV stores a 'summary_json'
    column that we expand into columns. This is idempotent and safe to call often.
    """
    if not csv_path.exists():
        return
    df_base = pd.read_csv(csv_path, dtype=str)
    def _parse_js(x):
        try:
            return json.loads(x) if isinstance(x, str) and x.strip() else {}
        except Exception:
            return {}
    summary_dicts = df_base.get("summary_json", pd.Series([], dtype=str)).apply(_parse_js)
    if not summary_dicts.empty:
        df_summary = pd.json_normalize(summary_dicts)
    else:
        df_summary = pd.DataFrame()

    keep_cols = ["timestamp", "model_rel", "model_type", "workspace",
                 "return_code", "overall_isd_sum", "objective"]
    df_out = df_base.copy()
    if "overall_isd_sum" in df_out.columns:
        df_out["overall_isd_sum"] = pd.to_numeric(df_out["overall_isd_sum"], errors="coerce")
    if "return_code" in df_out.columns:
        df_out["return_code"] = pd.to_numeric(df_out["return_code"], errors="coerce")
    if "objective" in df_out.columns:
        df_out["objective"] = pd.to_numeric(df_out["objective"], errors="coerce")

    if not df_summary.empty:
        df_summary.index = df_out.index
        df_out = pd.concat([df_out, df_summary], axis=1)

    if "summary_json" in df_out.columns:
        df_out = df_out.drop(columns=["summary_json"])

    known = [c for c in keep_cols if c in df_out.columns]
    rest = [c for c in df_out.columns if c not in known]
    df_out = df_out[known + rest]

    excel_out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_excel(excel_out, index=False, engine="openpyxl")

def main():
    ap = argparse.ArgumentParser(description="Batch LEMMA functional-form analysis using make_script()")
    ap.add_argument("--population", default=None,
                    help="Path to POPULATIONS/POPULATION_xxxx (root of model subfolders)")
    ap.add_argument("--parameters", default=None, help="Path to parameters.json (optional)")
    ap.add_argument("--plot-script", default=None, help="Explicit plot script to use for ALL models (override auto)")
    ap.add_argument("--plot-npz", default="plot_forms_NPZ.py", help="Path/name to NPZ plot script for auto selection")
    ap.add_argument("--plot-cots", default="plot_forms_COTS.py", help="Path/name to COTS plot script for auto selection")
    ap.add_argument("--python-cmd", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model-type", default=None, choices=["NPZ", "COTS"],
                    help="Force model-type for all models (skip detection)")

    # Figure/output options
    ap.add_argument("--fig-dir", default="figs", help="Where to save generated figures inside each workspace")
    ap.add_argument("--fig-format", default="png", choices=["png", "pdf", "svg"], help="Figure file format")

    # Synthesis/progress tracking
    ap.add_argument("--excel-out", default="Results/batch_summary.xlsx", help="Path to output Excel spreadsheet")
    ap.add_argument("--results-csv", default="Results/batch_results.csv",
                    help="Append-only CSV used for progress tracking and resume")
    ap.add_argument("--resume", action="store_true",
                    help="Skip individuals that already have entries in --results-csv")
    ap.add_argument("--per-individual-subdir", default="form_check",
                    help="Subfolder created under each individual's directory to store workspace and outputs")

    args = ap.parse_args()

    # Determine root/population
    config = {}
    if args.population:
        root = Path(args.population)
        if not root.exists():
            populations_root = Path("POPULATIONS") / args.population
            if populations_root.exists():
                root = populations_root
            else:
                print(f"ERROR: --population not found: {root} or {populations_root}")
                sys.exit(2)
        debug(f"Running for population root: {root}")
    elif args.parameters:
        params_data = load_parameters(args.parameters)
        config = {p.get("parameter"): p.get("value") for p in params_data.get("parameters", [])}
        root = Path(config["root"]) if "root" in config else Path(args.parameters).parent.parent
        debug(f"Running from parameters: root={root}")
    else:
        root = Path.cwd()
        debug(f"No --population/--parameters given; defaulting root to CWD: {root}")

    # Truth references
    truth_src_npz = Path(config.get("npz_truth", "Data/NPZ_example/NPZ_model.py"))
    truth_src_cots = Path(config.get("cots_truth", "FROM_JACOB/CoTSmodel_v4.cpp"))
    truth_params_cots = Path(config.get("cots_params", "FROM_JACOB/ControlFile.R"))

    runs_root = Path("Results")
    runs_root.mkdir(parents=True, exist_ok=True)

    temperature = float(config.get("temperature", 0.1))
    llm_choice = config.get("llm_choice", "anthropic_sonnet")

    # Discover model pairs, then EXCLUDE any directory that is within per-individual workspaces
    discovered = find_model_pairs(root)
    pairs = [p for p in discovered if args.per_individual_subdir not in p["dir"].parts]

    if not pairs:
        print(f"No (model.cpp, parameters.json) pairs found under {root}")
        sys.exit(0)

    debug(f"Found {len(pairs)} model pairs under {root} (excluded '{args.per_individual_subdir}' workspaces)")
    if not pairs:
        print(f"No (model.cpp, parameters.json) pairs found under {root}")
        sys.exit(0)
    debug(f"Found {len(pairs)} model pairs under {root}")

    processed = set()
    processed = set()
    results_csv = Path(args.results_csv)
    if args.resume and results_csv.exists():
        try:
            prev_df = pd.read_csv(results_csv, dtype=str)
            if "model_rel" in prev_df.columns:
                for s in prev_df["model_rel"].dropna().tolist():
                    rel_p = Path(s)
                    # Normalize: strip trailing '/form_check' if present
                    rel_p = normalize_rel(rel_p, args.per_individual_subdir)
                    processed.add(str(rel_p))
            debug(f"Resume enabled: will skip {len(processed)} previously logged individuals (normalized)")
        except Exception as e:
            debug(f"Resume enabled but failed to read CSV ({e}); proceeding without skipping")

    ts = dt.datetime.now().isoformat(timespec="seconds")
    run_count = 0
    ok_count = 0
    fail_count = 0

    for i, p in enumerate(pairs, 1):
        rel = p["dir"].relative_to(root)
        # Normalize to avoid trailing '/form_check'
        rel_norm = normalize_rel(rel, args.per_individual_subdir)
        safe = sanitize_rel_path(rel_norm)

        if args.resume and str(rel_norm) in processed:
            debug(f"[{i}/{len(pairs)}] SKIP (resume): {rel_norm}")
            continue

        # Per-individual workspace inside the individual's folder
        work = (root / rel_norm) / args.per_individual_subdir
        work.mkdir(parents=True, exist_ok=True)
        debug(f"[{i}/{len(pairs)}] Workspace: {work}")

        # Detect model type dynamically unless forced
        chosen_type = args.model_type or detect_model_type(root)
        debug(f"Detected model-type={chosen_type} for {rel} (override={args.model_type is not None})")

        # Resolve appropriate plot script
        try:
            plot_src = resolve_plot_script(
                chosen_type,
                override_script=args.plot_script,
                plot_npz=args.plot_npz,
                plot_cots=args.plot_cots,
            )
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(2)

        # Copy plot script and model inputs into workspace
        shutil.copy2(plot_src, work / plot_src.name)
        shutil.copy2(p["model_cpp"], work / p["model_cpp"].name)
        shutil.copy2(p["opt_json"], work / p["opt_json"].name)

        # NEW: vendor form_utils.py into workspace at work/scripts_analysis/form_utils.py
        vendor_form_utils_into_workspace(work)

        # Copy Truth files into expected relative locations (if present)
        if chosen_type == "NPZ" and truth_src_npz.exists():
            dest_truth = work / "Data/NPZ_example/NPZ_model.py"
            dest_truth.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(truth_src_npz, dest_truth)
            truth_rel_path = "Data/NPZ_example/NPZ_model.py"
        elif chosen_type == "COTS":
            truth_rel_path = "FROM_JACOB/CoTSmodel_v4.cpp"
            if truth_src_cots.exists():
                dest_truth = work / truth_rel_path
                dest_truth.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(truth_src_cots, dest_truth)
            if truth_params_cots.exists():
                dest_params = work / "FROM_JACOB/ControlFile.R"
                dest_params.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(truth_params_cots, dest_params)
        else:
            truth_rel_path = "Data/NPZ_example/NPZ_model.py"

        # Prepare prompt
        prompt_text = write_prompt(
            work,
            rel,
            Path(p["model_cpp"].name),
            Path(p["opt_json"].name),
            Path(truth_rel_path),
            plot_src.name,
            chosen_type
        )

        if args.dry_run:
            debug("Dry-run: skipping make_script and plot execution")
            continue

        filenames = [plot_src.name]
        read_files = [p["model_cpp"].name, p["opt_json"].name]
        if chosen_type == "NPZ":
            read_files.append("Data/NPZ_example/NPZ_model.py")
        else:
            read_files.extend(["FROM_JACOB/CoTSmodel_v4.cpp", "FROM_JACOB/ControlFile.R"])

        # Run make_script inside the workspace
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

        # Prepare environment for non-interactive plotting and saving
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env["SAVE_PLOTS_DIR"] = args.fig_dir
        env["SAVE_PLOTS_FORMAT"] = args.fig_format
        env["NO_SHOW"] = "1"
        env["MODEL_ID"] = safe  # prefix output figures with a model-specific id

        # Run the plot script
        rc, out, err = run_plot(plot_src.name, cwd=work, python_cmd=args.python_cmd, env=env)
        (work / "stdout.txt").write_text(out, encoding="utf-8")
        (work / "stderr.txt").write_text(err or "", encoding="utf-8")

        overall = parse_overall_isd(out)
        summary = parse_summary_json(out) or {}

        # --- objective (always set before writing 'row') ---
        # Read from the individual's directory (parent of workspace), not from form_check
        model_dir_for_objective = (root / rel_norm)
        objective = read_objective_from_report(model_dir_for_objective)

        # Optional: quick audit of where we're reading from
        debug(f"Objective source: {(model_dir_for_objective / 'model_report.json')}")


        run_count += 1
        if isinstance(overall, (int, float)):
            ok_count += 1
            debug(f"OVERALL_ISD_SUM = {overall:.6g}")
        else:
            fail_count += 1
            debug("WARNING: Could not parse OVERALL_ISD_SUM")

        # Append progress & rebuild Excel after EACH individual
        row = {
            "timestamp": ts,
            "model_rel": str(rel_norm),   # normalized (no trailing '/form_check')
            "workspace": str(work),
            "return_code": rc,
            "overall_isd_sum": overall,
            "objective": objective,
            "model_type": chosen_type,
            "summary_json": json.dumps(summary, ensure_ascii=False),
        }
        append_result_to_csv(results_csv, row)
        rebuild_excel_from_csv(results_csv, Path(args.excel_out))

    print(f"\nCompleted {run_count} model(s) in this session. Parsed scores: {ok_count}; Missing: {fail_count}")
    try:
        df_all = pd.read_csv(results_csv)
        df_this = df_all[df_all["timestamp"] == ts].copy()
        df_this["overall_isd_sum"] = pd.to_numeric(df_this["overall_isd_sum"], errors="coerce")
        ok = df_this.dropna(subset=["overall_isd_sum"])
        if not ok.empty:
            best = ok.sort_values("overall_isd_sum", ascending=True).head(5)
            print("Lowest OVERALL_ISD_SUM (this session, top 5):")
            for _, r in best.iterrows():
                print(f" {r['model_rel']}: {r['overall_isd_sum']}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
