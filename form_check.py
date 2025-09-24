#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch LEMMA analysis orchestrator using `make_script`
----------------------------------------------------

For each (model.cpp, optimized_parameters.json) pair under POPULATIONS/POPULATION_0003/,
this script:

  1) Creates a per-model workspace under `.aider_runs/<relative_path_sanitized>/`
  2) Copies in:
        - plot_forms.py (editable target)
        - model.cpp, optimized_parameters.json (read-only context)
        - Data/NPZ_example/NPZ_model.py (read-only context)
  3) Generates a tailored prompt that asks `make_script` to:
        - Pull functions/parameters from model.cpp + optimized_parameters.json
        - Update LEMMA parameter block and RHS in plot_forms.py accordingly
        - Print:  OVERALL_ISD_SUM: <float>   (sum of ISD across FLUXES absent-as-zero)
  4) Calls your make_script(...) programmatically
  5) Runs the edited plot_forms.py, parses OVERALL_ISD_SUM (or sums the FLUXES absent-as-zero ISD table)

Outputs:
  - .aider_runs/<...>/stdout.txt, stderr.txt
  - .aider_runs/batch_results.csv
  - .aider_runs/batch_results.json
Author: M365 Copilot
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
from textwrap import dedent

FLOAT_RE = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

def debug(msg: str):
    print(f"[batch] {msg}")

def sanitize_rel_path(path: Path) -> str:
    rel = str(path)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", rel).strip("_")

def find_model_pairs(root: Path):
    pairs = []
    for dirpath, _dirnames, filenames in os.walk(root):
        filenames = set(filenames)
        if "model.cpp" in filenames and "optimized_parameters.json" in filenames:
            d = Path(dirpath)
            pairs.append(
                {
                    "dir": d,
                    "model_cpp": d / "model.cpp",
                    "opt_json": d / "optimized_parameters.json",
                }
            )
    return sorted(pairs, key=lambda x: str(x["dir"]))

def write_prompt(workspace: Path, rel_model_dir: Path, model_cpp: Path, opt_json: Path,
                 truth_npz_rel: Path, plot_script_name: str = "plot_forms.py") -> str:
    model_id = sanitize_rel_path(rel_model_dir)
    prompt = f"""
You are assisting with an NPZ model comparison workflow.

**Goal**: Edit `{plot_script_name}` so that its LEMMA model section exactly reflects the
functional forms and optimized parameter values of the TMB model defined by:

- C++ model: `{model_cpp.name}` (in this folder)
- Parameters: `{opt_json.name}` (in this folder; may contain a `found_value` mapping)
- Truth reference file for naming/context: `{truth_npz_rel}`

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
     (i.e., the same lines shown in the table titled
     "Integrated squared difference — FLUXES (absent-as-zero)").

4) **Scope**
   - Do not modify the Truth model. Only adjust the LEMMA side.
   - Save edits directly to `{plot_script_name}`.

**Model ID**: `{model_id}`

Please make minimal, surgical edits and SAVE the file.
"""
    # Also drop a copy in the workspace for auditability
    (workspace / "aider_prompt.md").write_text(prompt, encoding="utf-8")
    return prompt

def parse_overall_isd(stdout: str):
    # Prefer explicit line
    for line in stdout.splitlines():
        if line.strip().startswith("OVERALL_ISD_SUM:"):
            try:
                return float(line.split(":", 1)[1].strip())
            except Exception:
                pass
    # Fallback: sum ISD from the "FLUXES (absent-as-zero)" table
    lines = stdout.splitlines()
    in_flux_zero = False
    total_isd = 0.0
    saw_any = False
    header_seen = False
    for line in lines:
        if "Integrated squared difference — FLUXES (absent-as-zero)" in line:
            in_flux_zero = True
            continue
        if in_flux_zero:
            if not line.strip():
                if saw_any:
                    break
                else:
                    continue
            if not header_seen and line.strip().startswith("Component"):
                header_seen = True
                continue
            # After header: first numeric column on each row is ISD
            nums = FLOAT_RE.findall(line)
            if nums:
                try:
                    total_isd += float(nums[0])
                    saw_any = True
                except Exception:
                    pass
            else:
                if saw_any:
                    break
    return total_isd if saw_any else None

def import_make_script(module_and_func=None, file_path=None, func_name="make_script"):
    """
    Load make_script either from a module path 'pkg.mod:func' or from a file.
    Returns a callable: make_script(filenames, read_files, prompt, temperature=..., llm_choice=...)
    """
    if module_and_func:
        if ":" in module_and_func:
            mod, fn = module_and_func.split(":", 1)
        else:
            mod, fn = module_and_func, func_name
        make_mod = importlib.import_module(mod)
        ms = getattr(make_mod, fn)
        return ms

    if file_path:
        spec = importlib.util.spec_from_file_location("make_script_mod", file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import from file: {file_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, func_name)

    raise ValueError("You must provide --make-script-import or --make-script-file")

def run_plot(plot_name: str, cwd: Path, python_cmd: str):
    try:
        res = subprocess.run([python_cmd, plot_name], cwd=str(cwd), capture_output=True, text=True)
        return res.returncode, res.stdout, res.stderr
    except FileNotFoundError:
        return 127, "", f"Python not found: {python_cmd}"

def main():
    ap = argparse.ArgumentParser(description="Batch LEMMA functional-form analysis using make_script()")
    ap.add_argument("--root", default="POPULATIONS/POPULATION_0003", help="Root directory to scan for model pairs")
    ap.add_argument("--plot-script", default="plot_forms.py", help="Path to the base analysis script (Truth vs LEMMA)")
    ap.add_argument("--npz-truth", default="Data/NPZ_example/NPZ_model.py", help="Path to NPZ truth file (context)")
    ap.add_argument("--runs-dir", default=".aider_runs", help="Workspaces root for per-model runs")

    # How to import your make_script
    ap.add_argument("--make-script-import", default=None,
                    help="Import path like 'mypkg.mymod:make_script'")
    ap.add_argument("--make-script-file", default=None,
                    help="Path to a .py file that defines make_script")
    ap.add_argument("--make-script-func", default="make_script",
                    help="Function name in the module/file (default: make_script)")

    # LLM settings forwarded to make_script
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--llm-choice", default="anthropic_sonnet")

    # Execution
    ap.add_argument("--python-cmd", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true", help="Prepare workspaces only; do not call make_script or run plot")

    args = ap.parse_args()

    root = Path(args.root)
    plot_src = Path(args.plot_script)
    truth_src = Path(args.npz_truth)
    runs_root = Path(args.runs_dir)

    if not plot_src.exists():
        print(f"ERROR: plot script not found: {plot_src}")
        sys.exit(2)

    pairs = find_model_pairs(root)
    if not pairs:
        print(f"No (model.cpp, optimized_parameters.json) pairs found under {root}")
        sys.exit(0)

    debug(f"Found {len(pairs)} model pairs under {root}")

    # Load make_script
    make_script = import_make_script(
        module_and_func=args.make_script_import,
        file_path=args.make_script_file,
        func_name=args.make_script_func,
    )

    results = []
    ts = dt.datetime.now().isoformat(timespec="seconds")

    for i, p in enumerate(pairs, 1):
        rel = p["dir"].relative_to(root)
        safe = sanitize_rel_path(rel)
        work = runs_root / safe
        work.mkdir(parents=True, exist_ok=True)
        debug(f"[{i}/{len(pairs)}] Workspace: {work}")

        # Copy base script (editable)
        work_plot = work / plot_src.name
        shutil.copy2(plot_src, work_plot)

        # Copy model files (context)
        work_model_cpp = work / p["model_cpp"].name
        work_opt_json  = work / p["opt_json"].name
        shutil.copy2(p["model_cpp"], work_model_cpp)
        shutil.copy2(p["opt_json"],  work_opt_json)

        # Copy truth file (context) to the same relative path
        if truth_src.exists():
            dest_truth = work / Path(args.npz_truth)
            dest_truth.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(truth_src, dest_truth)
        else:
            dest_truth = None
        # Build prompt text
        prompt_text = write_prompt(
            workspace=work,
            rel_model_dir=rel,
            model_cpp=Path(p["model_cpp"].name),
            opt_json=Path(p["opt_json"].name),
            truth_npz_rel=Path(args.npz_truth),
            plot_script_name=plot_src.name,
        )

        if args.dry_run:
            debug("Dry-run: skipping make_script and plot execution")
            continue

        # Call your make_script with files to edit + read-only context
        filenames = [work_plot.name]  # editable
        read_files = [work_model_cpp.name, work_opt_json.name]
        if dest_truth is not None:
            read_files.append(str(Path(args.npz_truth)))  # preserve subpath

        # Change CWD into workspace so the file names are correct relative references
        cwd_prev = os.getcwd()
        os.chdir(str(work))
        try:
            _coder = make_script(
                filenames=filenames,
                read_files=read_files,
                prompt=prompt_text,
                temperature=args.temperature,
                llm_choice=args.llm_choice,
            )
        finally:
            os.chdir(cwd_prev)

        # Run the plot script in the workspace and parse output
        rc, out, err = run_plot(plot_src.name, cwd=work, python_cmd=args.python_cmd)
        (work / "stdout.txt").write_text(out, encoding="utf-8")
        (work / "stderr.txt").write_text(err or "", encoding="utf-8")
        if rc != 0:
            print(f"ERROR running {plot_src.name} in {work} (rc={rc})\nSTDERR:\n{err}")

        overall = parse_overall_isd(out)
        results.append({
            "model_rel": str(rel),
            "workspace": str(work),
            "return_code": rc,
            "overall_isd_sum": overall,
        })
        if overall is not None:
            debug(f"OVERALL_ISD_SUM = {overall:.6g}")
        else:
            debug("WARNING: Could not parse OVERALL_ISD_SUM (explicit or fallback)")

    # Save aggregate results
    out_csv = runs_root / "batch_results.csv"
    out_json = runs_root / "batch_results.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "model_rel", "workspace", "return_code", "overall_isd_sum"])
        for r in results:
            w.writerow([ts, r["model_rel"], r["workspace"], r["return_code"], r["overall_isd_sum"]])

    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "results": results}, f, indent=2)

    # Summary to console
    ok = [r for r in results if isinstance(r["overall_isd_sum"], (int, float))]
    fail = [r for r in results if r["overall_isd_sum"] is None]
    print(f"\nCompleted {len(results)} model(s). Parsed scores: {len(ok)}; Missing: {len(fail)}")
    if ok:
        best = sorted(ok, key=lambda r: r["overall_isd_sum"])[:5]
        print("Lowest OVERALL_ISD_SUM (top 5):")
        for r in best:
            print(f"  {r['model_rel']}: {r['overall_isd_sum']}")

if __name__ == "__main__":
    main()
