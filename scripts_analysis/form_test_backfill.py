#!/usr/bin/env python3
# backfill_objectives.py
import json
import re
from pathlib import Path
import pandas as pd

def read_objective_from_report(model_dir: Path):
    report = model_dir / "model_report.json"
    if not report.exists():
        return None
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
        iters = data.get("iterations", {})
        if not isinstance(iters, dict) or not iters:
            return None
        numeric_keys = []
        for k in iters.keys():
            if str(k).isdigit():
                numeric_keys.append(int(k))
            else:
                stripped = re.sub(r"[^0-9]", "", str(k))
                if stripped.isdigit():
                    numeric_keys.append(int(stripped))
        if not numeric_keys:
            return None
        latest_k = str(max(numeric_keys))
        last = iters.get(latest_k, {}) or {}
        def _as_float(x):
            if isinstance(x, (int, float)):
                return float(x)
            if isinstance(x, str):
                try:
                    return float(x)
                except Exception:
                    return None
            return None
        for key in ("objective_value", "objective"):
            v = _as_float(last.get(key))
            if v is not None:
                return v
        for nested in ("report", "metrics"):
            obj = last.get(nested) or {}
            if isinstance(obj, dict):
                for key in ("objective_value", "objective"):
                    v = _as_float(obj.get(key))
                    if v is not None:
                        return v
        return None
    except Exception:
        return None

if __name__ == "__main__":
    csv_path = Path("Results/batch_results.csv")  # adjust if needed
    excel_out = Path("Results/batch_summary.xlsx")
    df = pd.read_csv(csv_path)
    updated = 0
    for i, row in df.iterrows():
        if pd.isna(row.get("objective")) or str(row.get("objective")).strip() == "":
            # model_rel is relative to the population root; workspace is absolute path to the per-individual subdir
            # The model_report.json sits one level above the workspace (the individual's directory)
            ws = Path(str(row.get("workspace")))
            model_dir = ws.parent if (ws.name == "form_check") else ws
            obj = read_objective_from_report(model_dir)
            if obj is not None:
                df.at[i, "objective"] = obj
                updated += 1
    df.to_csv(csv_path, index=False)
    # Rebuild Excel
    # Expand SUMMARY_JSON fields into columns
    def _parse_js(x):
        try:
            return json.loads(x) if isinstance(x, str) and x.strip() else {}
        except Exception:
            return {}
    df_base = pd.read_csv(csv_path, dtype=str)
    summary_dicts = df_base.get("summary_json", pd.Series([], dtype=str)).apply(_parse_js)
    df_summary = pd.json_normalize(summary_dicts) if not summary_dicts.empty else pd.DataFrame()
    keep_cols = ["timestamp", "model_rel", "model_type", "workspace", "return_code", "overall_isd_sum", "objective"]
    df_out = df_base.copy()
    for col in ["overall_isd_sum", "return_code", "objective"]:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce")
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
    print(f"Backfilled objectives in {updated} row(s). Wrote updated CSV and Excel.")