#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COTS (Crown-of-Thorns Starfish) model comparison:
- Truth: Discrete-year simulation based on CoTSmodel_v4.cpp functional forms
- LEMMA: ***INTENTIONALLY ABSENT*** (placeholder for make_script injection)

Batch-friendly:
- Uses shared helpers from analysis_utils.py
- Saves plots, prints OVERALL_ISD_SUM and SUMMARY_JSON
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scripts_analysis.form_utils import _savefig, compute_isd_table, average_rel_L2, _plot_state, _plot_facet, print_summary

# ================================================================
# Environment
# ================================================================
EPS = 1e-12
SAVE_DIR = os.environ.get("SAVE_PLOTS_DIR", "figs")
SAVE_FMT = os.environ.get("SAVE_PLOTS_FORMAT", "png")
MODEL_ID = os.environ.get("MODEL_ID", "model")

# ================================================================
# Truth model parameters (from CoTSmodel_v4.cpp & ControlFile)
# ================================================================
Years = np.arange(2000, 2026)
NumYrs = len(Years)
K = 3000.0; rf = 0.5; rm = 0.1
p2f = 10.0; p2m = 8.0; h = 0.5; R0 = 1.0
Imm_CoTS = 1.0; sigCoTS = 0.7
CoTS_init = 0.3; p1f = 0.15; ptil = 0.5; p1m = 0.06
Mcots = 2.5; lam = 0.0
Cf_init = 0.16; Cm_init = 0.12
switchSlope = 5
Eta_f = 2.0; Eta_m = 1.0
M_SST50_f = 34; M_SST50_m = 32
SST0_f = 26; SST0_m = 27
SST_sig_f = 2.0; SST_sig_m = 4.0
SST_CONST = 27.0
Imm_res_yrs = [2009, 2011, 2012]
immigration_devs = np.array([1.5, 1.6, 0.7])
Imm_res = np.zeros(NumYrs)
for dev, yr in zip(immigration_devs, Imm_res_yrs):
    idx = yr - Years[0]
    if 0 <= idx < NumYrs:
        Imm_res[idx] = dev

# ================================================================
# Helper functions for Truth model
# ================================================================
def sst_gauss(sst, sst0, sig):
    return np.exp(-((sst - sst0) ** 2) / (2.0 * (sig ** 2) + EPS))

def bleaching_logistic(stock, eta, sst, sst50):
    return stock * (1.0 / (1.0 + np.exp(-eta * (sst - sst50))))

def prey_switch(Cf, K, switchSlope):
    rho = np.exp(-switchSlope * (Cf / (K + EPS)))
    f = (1.0 - ptil) + ptil * rho
    return rho, f

def predation_loss(stock, rho_choice, p1, N12, p2):
    return stock * rho_choice * p1 * (N12 / (1.0 + np.exp(-(N12) / (p2 + EPS))))

def age_mortality(Mcots, lam, age_idx):
    return Mcots + lam / (1.0 + age_idx)

def bh_params(Mcots, R0, h):
    Kots_sp = R0 * (np.exp(-2 * Mcots) / (1.0 + np.exp(-Mcots))) + R0 * np.exp(-Mcots)
    SPR0    = np.exp(-2 * Mcots) / (1.0 + np.exp(-Mcots))
    beta    = Kots_sp * ((1.0 - h) / (5.0 * h - 1.0))
    alpha   = (beta + Kots_sp) / (SPR0 + EPS)
    return alpha, beta, Kots_sp

# ================================================================
# Truth simulation
# ================================================================
def simulate_truth():
    Cf = np.zeros(NumYrs); Cm = np.zeros(NumYrs)
    N0 = np.zeros(NumYrs); N1 = np.zeros(NumYrs); N2p = np.zeros(NumYrs)
    Cf[0] = Cf_init * K; Cm[0] = Cm_init * K
    N0[0] = CoTS_init * np.exp(2.0 * Mcots)
    N1[0] = CoTS_init * np.exp(1.0 * Mcots)
    N2p[0] = CoTS_init

    Cf_growth = np.zeros(NumYrs); Cm_growth = np.zeros(NumYrs)
    Qf = np.zeros(NumYrs); Qm = np.zeros(NumYrs)
    M_ble_f = np.zeros(NumYrs); M_ble_m = np.zeros(NumYrs)
    Rcots = np.zeros(NumYrs)

    alpha, beta, Kots_sp = bh_params(Mcots, R0, h)

    for k in range(NumYrs - 1):
        sst = SST_CONST
        rho, f = prey_switch(Cf[k], K, switchSlope)
        N12 = N1[k] + N2p[k]

        rho_SST_F = sst_gauss(sst, SST0_f, SST_sig_f)
        rho_SST_M = sst_gauss(sst, SST0_m, SST_sig_m)
        Cf_growth[k] = Cf[k] * (rho_SST_F * rf * (1.0 - (Cf[k] + Cm[k]) / (K + EPS)))
        Cm_growth[k] = Cm[k] * (rho_SST_M * rm * (1.0 - (Cf[k] + Cm[k]) / (K + EPS)))

        Qf[k] = predation_loss(Cf[k], (1.0 - rho), p1f, N12, p2f)
        Qm[k] = predation_loss(Cm[k], rho, p1m, N12, p2m)
        Qf[k] = min(Qf[k], max(Cf[k] - 1e-3, 0.0))
        Qm[k] = min(Qm[k], max(Cm[k] - 1e-3, 0.0))

        M_ble_f[k] = bleaching_logistic(Cf[k], Eta_f, sst, M_SST50_f)
        M_ble_m[k] = bleaching_logistic(Cm[k], Eta_m, sst, M_SST50_m)
        M_ble_f[k] = min(M_ble_f[k], max(Cf[k] - 1e-3, 0.0))
        M_ble_m[k] = min(M_ble_m[k], max(Cm[k] - 1e-3, 0.0))

        Cf[k+1] = Cf[k] * (1.0 + rho_SST_F * rf * (1.0 - (Cf[k] + Cm[k]) / (K + EPS))) - Qf[k] - M_ble_f[k]
        Cm[k+1] = Cm[k] * (1.0 + rho_SST_M * rm * (1.0 - (Cf[k] + Cm[k]) / (K + EPS))) - Qm[k] - M_ble_m[k]
        Cf[k+1] = max(Cf[k+1], 0.0); Cm[k+1] = max(Cm[k+1], 0.0)

        M_age0 = age_mortality(Mcots, lam, 0)
        M_age1 = age_mortality(Mcots, lam, 1)
        M_age2 = age_mortality(Mcots, lam, 2)
        N1[k+1] = N0[k] * np.exp(-1.0 * M_age0)
        N2p[k+1] = N1[k] * np.exp(-f * M_age1) + N2p[k] * np.exp(-f * M_age2)

        Rcots[k+1] = (alpha * ((N2p[k+1] / (Kots_sp + EPS)))) / (beta + (N2p[k+1] / (Kots_sp + EPS)) + EPS)
        N0[k+1] = (Rcots[k+1] + Imm_CoTS) * np.exp(Imm_res[k+1] + (sigCoTS ** 2) / 2.0)

    N_total = N0 + N1 + N2p
    return {
        "Years": Years, "Cf": Cf, "Cm": Cm,
        "N0": N0, "N1": N1, "N2p": N2p, "N_total": N_total,
        "Cf_growth": Cf_growth, "Cm_growth": Cm_growth,
        "Qf": Qf, "Qm": Qm, "M_ble_f": M_ble_f, "M_ble_m": M_ble_m,
        "Rcots": Rcots
    }

# ================================================================
# LEMMA placeholder
# ================================================================
# def simulate_lemma():
#     pass

# ================================================================
# Main
# ================================================================
def main():
    truth = simulate_truth()
    try:
        lemma = simulate_lemma()
    except NameError:
        lemma = {k: None for k in truth.keys()}
        lemma["Years"] = truth["Years"]

    # Plot states
    plt.style.use("seaborn-v0_8-whitegrid")
    fig1, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=False)
    _plot_state(axes[0], "Fast coral (Cf)", truth["Cf"], lemma["Cf"], ylabel="Cover (% of K)", to_percent=True, K=K)
    _plot_state(axes[1], "Slow coral (Cm)", truth["Cm"], lemma["Cm"], ylabel="Cover (% of K)", to_percent=True, K=K)
    _plot_state(axes[2], "CoTS (age 2+)", truth["N2p"], lemma["N2p"])
    fig1.tight_layout()
    _savefig(fig1, "cots_states", SAVE_DIR, SAVE_FMT, MODEL_ID)

    # Flux facets
    facets = [
        ("Cf growth", truth["Cf_growth"], lemma["Cf_growth"]),
        ("Cm growth", truth["Cm_growth"], lemma["Cm_growth"]),
        ("Predation Qf", truth["Qf"], lemma["Qf"]),
        ("Predation Qm", truth["Qm"], lemma["Qm"]),
        ("Bleaching fast", truth["M_ble_f"], lemma["M_ble_f"]),
        ("Bleaching slow", truth["M_ble_m"], lemma["M_ble_m"]),
        ("CoTS recruitment", truth["Rcots"], lemma["Rcots"]),
        ("CoTS total abundance", truth["N_total"], lemma["N_total"]),
    ]
    nrows, ncols = 2, 4
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(18, 7), sharex=False)
    axes2 = axes2.ravel()
    for ax, (title, s_truth, s_lemma) in zip(axes2, facets):
        _plot_facet(ax, title, truth["Years"], s_truth, s_lemma)
    fig2.tight_layout(rect=[0, 0, 1, 0.98])
    _savefig(fig2, "cots_flux_facets", SAVE_DIR, SAVE_FMT, MODEL_ID)

    # ISD tables
    state_pairs = [
        ("Fast coral Cf", truth["Cf"], lemma["Cf"]),
        ("Slow coral Cm", truth["Cm"], lemma["Cm"]),
        ("CoTS age 2+", truth["N2p"], lemma["N2p"]),
        ("CoTS total", truth["N_total"], lemma["N_total"]),
    ]
    state_results = compute_isd_table(truth["Years"], state_pairs, "ISD — STATES", absent_as_zero=True)
    flux_results_zero = compute_isd_table(truth["Years"], facets, "ISD — FLUXES (absent-as-zero)", absent_as_zero=True)
    flux_results_present = compute_isd_table(truth["Years"], facets, "ISD — FLUXES (present-only)", absent_as_zero=False)

    score_states = average_rel_L2(state_results)
    score_flux_zero = average_rel_L2(flux_results_zero)
    score_flux_present = average_rel_L2(flux_results_present)
    overall_isd_sum = float(np.sum([r[1] for r in flux_results_zero]))

    # Summary output
    print_summary(overall_isd_sum, score_states, score_flux_zero, score_flux_present,
                  len(state_pairs), len(facets), notes="COTS comparison: LEMMA placeholder.")

    # ISD bar chart
    try:
        fig3 = plt.figure(figsize=(12, 4))
        names = [r[0] for r in flux_results_zero]
        vals = [r[1] for r in flux_results_zero]
        x = np.arange(len(names))
        plt.bar(x, vals, color="#8da0cb")
        plt.xticks(x, names, rotation=30, ha="right")
        plt.ylabel(r"ISD $\int (f-g)^2\,dt$")
        plt.title("Integrated Squared Difference — Fluxes (absent-as-zero)")
        plt.tight_layout()
        _savefig(fig3, "cots_flux_isd_bar", SAVE_DIR, SAVE_FMT, MODEL_ID)
    except Exception:
        pass

if __name__ == "__main__":
    main()