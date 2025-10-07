
#!/usr/bin/env python3

import os
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

# Figure saving helper
def _savefig(fig, name, save_dir=None, save_fmt="png", model_id="model"):
    if save_dir is None:
        save_dir = os.environ.get("SAVE_PLOTS_DIR", "figs")
    save_fmt = os.environ.get("SAVE_PLOTS_FORMAT", save_fmt)
    model_id = os.environ.get("MODEL_ID", model_id)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out = Path(save_dir) / f"{model_id}_{name}.{save_fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ISD metrics
def isd_discrete(t, f, g):
    return float(np.trapz((f - g) ** 2, t))

def rmse_discrete(t, f, g):
    T = float(t[-1] - t[0]) if len(t) > 1 else 1.0
    return float(np.sqrt(isd_discrete(t, f, g) / (T if T > 0 else 1.0)))

def rel_l2_discrete(t, f_truth, f_model, eps=1e-12):
    denom = float(np.trapz(f_truth ** 2, t))
    return float(isd_discrete(t, f_truth, f_model) / (denom + eps))

def average_rel_L2(rows):
    return float(np.mean([r[3] for r in rows])) if rows else float("nan")

# Series handling
def _series_or_zero(truth_series, lemma_series, zero_if_absent=True):
    if lemma_series is None:
        return truth_series, (np.zeros_like(truth_series) if zero_if_absent else None)
    return truth_series, lemma_series

# ISD table computation and printing
def compute_isd_table(t, pairs, title, absent_as_zero=True):
    rows = []
    for name, f_truth, f_lemma in pairs:
        f_truth, f_lemma = _series_or_zero(f_truth, f_lemma, absent_as_zero)
        if f_lemma is None:
            continue
        v_isd = isd_discrete(t, f_truth, f_lemma)
        v_rmse = rmse_discrete(t, f_truth, f_lemma)
        v_rel = rel_l2_discrete(t, f_truth, f_lemma)
        rows.append((name, v_isd, v_rmse, v_rel))
    print(f"\\n{title}{'' if absent_as_zero else ' (present-only)'}")
    print("{:<40s} {:>14s} {:>14s} {:>14s}".format("Component", "ISD", "RMSE", "Rel L2"))
    for name, a, b, c in rows:
        print(f"{name:<40s} {a:14.6g} {b:14.6g} {c:14.6g}")
    return rows

# Plotting helpers
def _plot_state(ax, title, truth_series, lemma_series, ylabel=None, to_percent=False, K=None):
    color_truth = "#1f77b4"; color_lemma = "#ff7f0e"
    y_truth = truth_series if not to_percent else (100.0 * truth_series / K)
    ax.plot(np.arange(len(y_truth)), y_truth, label="Truth", color=color_truth, lw=2)
    if lemma_series is not None:
        y_lemma = lemma_series if not to_percent else (100.0 * lemma_series / K)
        ax.plot(np.arange(len(y_lemma)), y_lemma, label="LEMMA", color=color_lemma, lw=2, ls="--")
    else:
        ax.text(0.02, 0.88, "LEMMA: absent", transform=ax.transAxes,
                fontsize=10, color=color_lemma,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_lemma))
    ax.set_title(title)
    ax.set_xlabel("Time")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend(frameon=True)

def _plot_facet(ax, title, t, s_truth, s_lemma):
    color_truth = "#1f77b4"; color_lemma = "#ff7f0e"
    if s_truth is not None:
        ax.plot(t, s_truth, color=color_truth, lw=2, label="Truth")
    else:
        ax.text(0.02, 0.9, "Truth: not available", transform=ax.transAxes,
                fontsize=10, color=color_truth,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_truth))
    if s_lemma is not None:
        ax.plot(t, s_lemma, color=color_lemma, lw=2, ls="--", label="LEMMA")
    else:
        ax.text(0.02, 0.75, "LEMMA: absent", transform=ax.transAxes,
                fontsize=10, color=color_lemma,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_lemma))
    ax.set_title(title); ax.set_xlabel("Time"); ax.axhline(0, color="k", lw=0.6, alpha=0.4)

# Summary output
def print_summary(overall_isd_sum, score_states, score_flux_zero, score_flux_present,
                  n_states, n_fluxes, notes=""):
    print(f"OVERALL_ISD_SUM: {overall_isd_sum}")
    summary_json = {
        "overall_isd_sum": overall_isd_sum,
        "avg_relL2_states": score_states,
        "avg_relL2_fluxes_zeroed": score_flux_zero,
        "avg_relL2_fluxes_present": score_flux_present,
        "n_states": n_states,
        "n_fluxes_compared": n_fluxes,
        "notes": notes
    }
    print("SUMMARY_JSON: " + json.dumps(summary_json))
