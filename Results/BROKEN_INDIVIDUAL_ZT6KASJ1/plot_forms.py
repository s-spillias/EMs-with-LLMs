#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare NPZ models with flux breakdowns; annotate absent processes
- Model A: Your richer Truth NPZ (self-shading, fractionated recycling, mixing)
- Model B: LEMMA NPZ with your optimized parameters (found_values)

Outputs:
  Fig 1: NPZ dynamics (N, P, Z) — 3 facets, two curves (Truth vs LEMMA)
  Fig 2: Ecological components/fluxes — 2x5 facets, two curves per facet where present
          Absent processes are annotated instead of plotted as zeros.

Author: Scott Spillias — helper script by Copilot
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
SAVE_FIGS = False
FIG1_PATH = "npz_dynamics.png"
FIG2_PATH = "npz_flux_components.png"

# ---------------------------------------------
# 1) PARAMETERS
# ---------------------------------------------
# ---- 1A) Truth NPZ parameters (from NPZ_model.py) ----
a       = 0.2     # m^-1 day^-1 (light/self-shading modulator)
b       = 0.2     # m^-1
c       = 0.4     # m^2 g^-1 C^-1
eN      = 0.03    # g C m^-3  (Monod half-sat for N; renamed to avoid math.e)
k       = 0.05    # day^-1    (mixing/restoring rate)
q       = 0.075   # day^-1    (Z mortality)
r       = 0.10    # day^-1    (P mortality that also recycles to N)
s       = 0.04    # day^-1    (P loss, e.g., sinking)
N0_mix  = 0.6     # g C m^-3  (restoration target for N)
alpha   = 0.25    # assimilation efficiency to Z growth
beta    = 0.33    # fraction of grazing returning to N
gamma   = 0.5     # fraction of Z mortality returning to N
lambda_ = 0.6     # day^-1    (grazing max)
mu_P    = 0.035   # g C m^-3  (grazing half-sat)

# ---- 1B) TMB model parameters from parameters.json ----
r_cots_max   = 1.0    # year^-1; Max per-capita growth rate of adult-equivalent COTS
m_cots       = 0.5    # year^-1; Background adult mortality rate of COTS
c_cots_density = 0.5  # (m^2 ind^-1) year^-1; Self-limitation coefficient
e_cots_imm   = 0.3    # dimensionless; Conversion from larval immigration to adult density
A_crit       = 0.1    # individuals m^-2; Allee threshold scale
K_prey       = 0.1    # proportion; Half-saturation for prey availability

beta_sst_cots = 1.0   # Celsius^-1; SST effect slope on recruitment
f_sst_lo     = 1.0    # dimensionless; Lower bound SST multiplier
f_sst_hi     = 1.5    # dimensionless; Upper bound SST multiplier

attack       = 5.0    # year^-1 ind^-1 m^2; Attack rate in predation
handling     = 0.2    # year; Handling time in predation
pref_fast    = 0.7    # dimensionless; Preference for fast coral
pref_slow    = 0.3    # dimensionless; Preference for slow coral
holling_q    = 1.5    # dimensionless; Functional response exponent

r_fast       = 0.6    # year^-1; Fast coral intrinsic growth
r_slow       = 0.3    # year^-1; Slow coral intrinsic growth
m_fast       = 0.15   # year^-1; Fast coral background mortality
m_slow       = 0.08   # year^-1; Slow coral background mortality
K_tot        = 0.6    # proportion; Total coral carrying capacity

beta_bleach_fast = 0.6  # per Celsius; Fast coral bleaching sensitivity
beta_bleach_slow = 0.4  # per Celsius; Slow coral bleaching sensitivity
tau_bleach   = 1.0    # Celsius; SST threshold for bleaching

# Observation error SDs
sigma_cots_log = 0.2    # SD on log scale for COTS
sigma_fast_logit = 0.3  # SD on logit scale for fast coral
sigma_slow_logit = 0.3  # SD on logit scale for slow coral

# ---------------------------------------------
# 2) INITIAL CONDITIONS & TIME GRID
# ---------------------------------------------
N0 = 0.4   # g C m^-3
P0 = 0.1   # g C m^-3
Z0 = 0.05  # g C m^-3
y0 = [N0, P0, Z0]

t_start, t_end = 0.0, 100.0
n_eval = 400
t_eval = np.linspace(t_start, t_end, n_eval)

EPS = 1e-12

# ---------------------------------------------
# 3) ODE RIGHT-HAND SIDES
# ---------------------------------------------
def rhs_python(t, y):
    """
    Your Truth NPZ model:
      uptake = (N/(eN+N)) * (a/(b+cP)) * P
      grazing = lambda * (P^2/(mu^2 + P^2)) * Z
      dP/dt = uptake - grazing - rP - (s+k)P
      dZ/dt = alpha*grazing - qZ
      dN/dt = -uptake + rP + beta*grazing + gamma*qZ + k(N0_mix - N)
    """
    N, P, Z = y
    N = max(N, 0.0); P = max(P, 0.0); Z = max(Z, 0.0)

    uptake  = (N/(eN + N + EPS)) * (a/(b + c*P + EPS)) * P
    grazing = lambda_ * (P**2)/(mu_P**2 + P**2 + EPS) * Z
    dPdt = uptake - grazing - r*P - (s + k)*P
    dZdt = alpha * grazing - q * Z
    dNdt = -uptake + r*P + beta*grazing + gamma*q*Z + k*(N0_mix - N)
    return [dNdt, dPdt, dZdt]

def rhs_tmb_like(t, y):
    """
    TMB COTS-coral model:
      f_sst = f_sst_lo + (f_sst_hi - f_sst_lo) * invlogit(beta_sst_cots * sst_anom)
      f_prey = prey_avail / (K_prey + prey_avail)
      f_allee = A / (A + A_crit)
      r_eff = r_cots_max * f_prey * f_sst * f_allee - m_cots - c_cots_density * A
      
      Predation uses multi-prey Holling with prey switching
    """
    N, P, Z = y  # Here N=nutrients, P=phyto, Z=zoo map to coral/COTS variables
    N = max(N, 0.0); P = max(P, 0.0); Z = max(Z, 0.0)
    
    # SST effect on COTS recruitment
    sst_effect = f_sst_lo + (f_sst_hi - f_sst_lo) * (1.0 / (1.0 + np.exp(-beta_sst_cots * sst_anom)))
    
    # Prey availability and Allee effect
    prey_avail = pref_fast * P + pref_slow * N  # Using N,P as coral types
    f_prey = prey_avail / (K_prey + prey_avail + EPS)
    f_allee = Z / (Z + A_crit + EPS)  # Z as COTS density
    
    # COTS population dynamics
    r_eff = r_cots_max * f_prey * sst_effect * f_allee - m_cots - c_cots_density * Z
    
    # Multi-prey Holling predation
    V = pref_fast * (P**holling_q) + pref_slow * (N**holling_q) + EPS
    cons_per_pred = attack * V / (1.0 + attack * handling * V + EPS)
    share_P = (pref_fast * (P**holling_q)) / V
    share_N = (pref_slow * (N**holling_q)) / V
    
    # Final derivatives
    dPdt = r_fast * P * (1.0 - (P + N)/K_tot) - Z * cons_per_pred * share_P - m_fast * P
    dNdt = r_slow * N * (1.0 - (P + N)/K_tot) - Z * cons_per_pred * share_N - m_slow * N
    dZdt = Z * (r_eff + e_cots_imm)  # Immigration added to growth
    return [dNdt, dPdt, dZdt]

# ---------------------------------------------
# 4) INTEGRATE BOTH MODELS
# ---------------------------------------------
sol_py = solve_ivp(rhs_python,  (t_start, t_end), y0, t_eval=t_eval, method="RK45",
                   rtol=1e-7, atol=1e-9)
sol_tm = solve_ivp(rhs_tmb_like,(t_start, t_end), y0, t_eval=t_eval, method="RK45",
                   rtol=1e-7, atol=1e-9)

if not sol_py.success or not sol_tm.success:
    raise RuntimeError("Integration failed. Try adjusting tolerances or time span.")

# ---------------------------------------------
# 5) DYNAMICS (STATE TRAJECTORIES)
# ---------------------------------------------
def _pos(arr): 
    """Clamp tiny negatives from numerics."""
    return np.maximum(arr, 0.0)

t    = sol_py.t  # shared grid
N_py, P_py, Z_py = _pos(sol_py.y[0]), _pos(sol_py.y[1]), _pos(sol_py.y[2])
N_tm, P_tm, Z_tm = _pos(sol_tm.y[0]), _pos(sol_tm.y[1]), _pos(sol_tm.y[2])

# ---------------------------------------------
# 6) ECOLOGICAL COMPONENTS (FLUXES)
# ---------------------------------------------
# ---- Truth model fluxes ----
uptake_py   = (N_py/(eN + N_py + EPS)) * (a/(b + c*P_py + EPS)) * P_py
grazing_py  = lambda_ * (P_py**2)/(mu_P**2 + P_py**2 + EPS) * Z_py
P_mort_py   = r*P_py + (s + k)*P_py        # total non-grazing loss of P
Z_growth_py = alpha * grazing_py           # assimilated grazing to Z
Z_mort_py   = q * Z_py
reminZ_py   = gamma * q * Z_py             # fraction of Z mortality to N
reminP_py   = r * P_py                     # P mortality recycled to N
reminG_py   = beta * grazing_py            # fraction of grazing to N
mixN_py     = k * (N0_mix - N_py)          # mixing to N
# Net nutrient tendency (for context)
dN_py       = -uptake_py + reminP_py + reminG_py + reminZ_py + mixN_py

# ---- LEMMA model fluxes ----
uptake_tm   = mu_max * N_tm/(K_N + N_tm + EPS) * P_tm
grazing_tm  = gmax   * (P_tm**h)/(K_P**h + P_tm**h + EPS) * Z_tm
P_mort_tm   = m_P * P_tm
Z_growth_tm = e_g * grazing_tm
Z_mort_tm   = m_Z * Z_tm
reminZ_tm   = d_N * Z_tm

# Fluxes not present in LEMMA model: set to None (so we annotate instead of plotting zeros)
reminP_tm   = None
reminG_tm   = None
mixN_tm     = None
dN_tm       = -uptake_tm + reminZ_tm

# ---------------------------------------------
# 7) PLOT — FIGURE 1: NPZ DYNAMICS (3 FACETS)
# ---------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
labels = [r"N (nutrients)", r"P (phytoplankton)", r"Z (zooplankton)"]
color_py = "#1f77b4"  # solid
color_tm = "#ff7f0e"  # dashed

# N
axes1[0].plot(t, N_py, color=color_py, lw=2, label="Truth model")
axes1[0].plot(t, N_tm, color=color_tm, lw=2, ls="--", label="LEMMA (optimized)")
axes1[0].set_title(labels[0])
axes1[0].set_ylabel(r"Concentration (g C m$^{-3}$)")

# P
axes1[1].plot(t, P_py, color=color_py, lw=2)
axes1[1].plot(t, P_tm, color=color_tm, lw=2, ls="--")
axes1[1].set_title(labels[1])

# Z
axes1[2].plot(t, Z_py, color=color_py, lw=2)
axes1[2].plot(t, Z_tm, color=color_tm, lw=2, ls="--")
axes1[2].set_title(labels[2])

for ax in axes1:
    ax.set_xlabel("Time (days)")

axes1[0].legend(loc="best", frameon=True)
fig1.suptitle("NPZ Time Series: Truth vs LEMMA (optimized)", y=1.05, fontsize=14)
fig1.tight_layout()

if SAVE_FIGS:
    fig1.savefig(FIG1_PATH, dpi=200, bbox_inches="tight")

plt.show()

# ---------------------------------------------
# 8) PLOT — FIGURE 2: ECOLOGICAL COMPONENTS (2x5 FACETS, WITH ANNOTATIONS)
# ---------------------------------------------
def plot_facet(ax, title, t, series_py, series_tm,
               color_py="#1f77b4", color_tm="#ff7f0e",
               annotate_absence=True):
    """Plot a single facet; annotate when a series is absent (None)."""
    handles = []
    labels_ = []

    if series_py is not None:
        line_py, = ax.plot(t, series_py, color=color_py, lw=2)
        handles.append(line_py); labels_.append("Truth model")
    else:
        if annotate_absence:
            ax.text(0.02, 0.90, "Not present in Truth model",
                    transform=ax.transAxes, fontsize=10,
                    color=color_py, ha="left", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_py, alpha=0.8))

    if series_tm is not None:
        line_tm, = ax.plot(t, series_tm, color=color_tm, lw=2, ls="--")
        handles.append(line_tm); labels_.append("LEMMA (optimized)")
    else:
        if annotate_absence:
            ax.text(0.02, 0.78, "Not present in LEMMA model",
                    transform=ax.transAxes, fontsize=10,
                    color=color_tm, ha="left", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_tm, alpha=0.8))

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(r"Rate (g C m$^{-3}$ day$^{-1}$)")
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)

    return handles, labels_

# Facets: title, python_series, tmb_series
facets = [
    ("Uptake to P",                     uptake_py,   uptake_tm),
    ("Grazing loss of P",               grazing_py,  grazing_tm),
    ("P non-grazing mortality (total)", P_mort_py,   P_mort_tm),
    ("Z growth (assimilated)",          Z_growth_py, Z_growth_tm),
    ("Z mortality",                     Z_mort_py,   Z_mort_tm),
    ("Remin to N from Z",               reminZ_py,   reminZ_tm),
    ("Recycling to N from P mort",      reminP_py,   reminP_tm),  # LEMMA: absent
    ("Recycling to N from grazing",     reminG_py,   reminG_tm),  # LEMMA: absent
    ("Mixing to N (k·(N0−N))",          mixN_py,     mixN_tm),    # LEMMA: absent
    ("Net nutrient tendency dN/dt",     dN_py,       dN_tm),
]

nrows, ncols = 2, 5
fig2, axes2 = plt.subplots(nrows, ncols, figsize=(18, 6), sharex=True)
axes2 = axes2.ravel()

# Build a global legend only from actually plotted lines
global_handles, global_labels = [], []
have_py, have_tm = False, False

for ax, (title, s_py, s_tm) in zip(axes2, facets):
    handles, labels_ = plot_facet(ax, title, t, s_py, s_tm,
                                  color_py=color_py, color_tm=color_tm,
                                  annotate_absence=True)
    for h, lbl in zip(handles, labels_):
        if lbl.startswith("Truth") and not have_py:
            global_handles.append(h); global_labels.append(lbl); have_py = True
        if lbl.startswith("TMB") and not have_tm:
            global_handles.append(h); global_labels.append(lbl); have_tm = True
if global_handles:
    fig2.legend(global_handles, global_labels, loc="upper center", ncol=2, frameon=True)

fig2.suptitle("Ecological Components: Truth vs LEMMA (optimized parameters)", y=1.04, fontsize=14)
fig2.tight_layout(rect=[0, 0, 1, 0.98])

if SAVE_FIGS:
    fig2.savefig(FIG2_PATH, dpi=200, bbox_inches="tight")

plt.show()

# ---------------------------------------------
# 9) OPTIONAL: PRINT PARAMETER SNAPSHOT
# ---------------------------------------------
def print_params():
    print("\n--- Truth model params ---")
    print(f"a={a}, b={b}, c={c}, eN={eN}, k={k}, q={q}, r={r}, s={s}, N0_mix={N0_mix}")
    print(f"alpha={alpha}, beta={beta}, gamma={gamma}, lambda_={lambda_}, mu_P={mu_P}")
    print("\n--- LEMMA (optimized) params ---")
    print(f"mu_max={mu_max}, K_N={K_N}, gmax={gmax}, h={h}, K_P={K_P}")
    print(f"e_g={e_g}, m_Z={m_Z}, m_P={m_P}, d_N={d_N}")
    print(f"log_sigma_N={log_sigma_N}, log_sigma_P={log_sigma_P}, log_sigma_Z={log_sigma_Z}")
    print("\n--- Initial conditions & integration ---")
    print(f"N0={N0}, P0={P0}, Z0={Z0}, t=[{t_start}, {t_end}], n_eval={n_eval}\n")

# ==== Quantify flux differences (append to your script) ====
import numpy as np
import matplotlib.pyplot as plt

def integrate(t, y):  # simple trapezoidal integral
    return np.trapz(y, t)

summary = [
    ("Uptake to P",                     integrate(t, uptake_py),   integrate(t, uptake_tm)),
    ("Grazing loss of P",               integrate(t, grazing_py),  integrate(t, grazing_tm)),
    ("P non-grazing mortality (total)", integrate(t, P_mort_py),   integrate(t, P_mort_tm)),
    ("Z growth (assimilated)",          integrate(t, Z_growth_py), integrate(t, Z_growth_tm)),
    ("Z mortality",                     integrate(t, Z_mort_py),   integrate(t, Z_mort_tm)),
    ("Remin to N from Z",               integrate(t, reminZ_py),   integrate(t, reminZ_tm)),
    ("Recycling to N from P mort",      integrate(t, reminP_py),   0.0),  # absent in LEMMA
    ("Recycling to N from grazing",     integrate(t, reminG_py),   0.0),  # absent in LEMMA
    ("Mixing to N (k·(N0−N))",          integrate(t, mixN_py),     0.0),  # absent in LEMMA
    ("Net nutrient tendency dN/dt",     integrate(t, dN_py),       integrate(t, dN_tm)),
]

print("\nIntegrated fluxes over the simulation window (g C m^-3):")
print("{:<35s} {:>14s} {:>14s} {:>10s}".format("Component", "Truth", "LEMMA", "Ratio P/T"))
for name, v_py, v_tm in summary:
    ratio = (v_py / v_tm) if abs(v_tm) > 1e-12 else np.nan
    print(f"{name:<35s} {v_py:14.4f} {v_tm:14.4f} {ratio:10.3f}")

# Bar chart for quick visual comparison
labels = [row[0] for row in summary]
py_vals = [row[1] for row in summary]
tm_vals = [row[2] for row in summary]

x = np.arange(len(labels))
w = 0.38
plt.figure(figsize=(14, 5))
plt.bar(x - w/2, py_vals, width=w, label="Truth", color="#1f77b4")
plt.bar(x + w/2, tm_vals, width=w, label="LEMMA", color="#ff7f0e")
plt.xticks(x, labels, rotation=35, ha="right")
plt.ylabel(r"Integrated flux (g C m$^{-3}$)")
plt.title("Integrated Ecological Components — Truth vs LEMMA")
plt.legend()
plt.tight_layout()
plt.show()

# ==== Integrated squared difference (ISD) between corresponding functions ====

def _as_array_or_zero(f_truth, f_model, absent_as_zero=True):
    """
    Returns (f_truth, f_model_array or None) with shape matching truth.
    If model series is None and absent_as_zero=True, returns zeros_like(truth).
    If model series is None and absent_as_zero=False, returns None (to be skipped).
    """
    if f_model is None:
        if absent_as_zero:
            return f_truth, np.zeros_like(f_truth)
        else:
            return f_truth, None
    return f_truth, f_model

def isd(t, f, g):
    """Integrated squared difference ∫ (f-g)^2 dt using the trapezoidal rule."""
    return np.trapz((f - g) ** 2, t)

def rmse(t, f, g):
    """Root-mean-squared error over [t0,t1]."""
    T = float(t[-1] - t[0])
    T = T if T > 0 else 1.0
    return np.sqrt(isd(t, f, g) / T)

def l2_relative(t, f_truth, f_model, eps=1e-12):
    """
    Relative L2 error: ISD / ∫ f_truth^2 dt.
    (Sometimes called normalized L2; robust to scaling.)
    """
    denom = np.trapz(f_truth ** 2, t)
    return isd(t, f_truth, f_model) / (denom + eps)

def _print_isd_table(title, rows):
    print(f"\n{title}")
    print("{:<38s} {:>14s} {:>14s} {:>14s}".format("Component", "ISD", "RMSE", "Rel L2"))
    for name, isd_val, rmse_val, rel_val in rows:
        print(f"{name:<38s} {isd_val:14.6g} {rmse_val:14.6g} {rel_val:14.6g}")

def compute_and_report_isd(t, pairs, title, absent_as_zero=True):
    """
    pairs: list of (name, truth_series, model_series or None)
    absent_as_zero: if True, treat None model series as zeros; else skip those pairs.
    """
    results = []
    for name, f_truth, f_model in pairs:
        f_truth, f_model = _as_array_or_zero(f_truth, f_model, absent_as_zero=absent_as_zero)
        if f_model is None:
            # Skipped because the model lacks this process and absent_as_zero=False
            continue
        val_isd = isd(t, f_truth, f_model)
        val_rmse = rmse(t, f_truth, f_model)
        val_rel  = l2_relative(t, f_truth, f_model)
        results.append((name, val_isd, val_rmse, val_rel))
    _print_isd_table(title + ("" if absent_as_zero else " (present-only)"), results)
    return results

# --------- Build the comparison pairs ---------
state_pairs = [
    ("N (nutrients)",      N_py, N_tm),
    ("P (phytoplankton)",  P_py, P_tm),
    ("Z (zooplankton)",    Z_py, Z_tm),
]

flux_pairs = [
    ("Uptake to P",                        uptake_py,   uptake_tm),
    ("Grazing loss of P",                  grazing_py,  grazing_tm),
    ("P non-grazing mortality (total)",    P_mort_py,   P_mort_tm),
    ("Z growth (assimilated)",             Z_growth_py, Z_growth_tm),
    ("Z mortality",                        Z_mort_py,   Z_mort_tm),
    ("Remin to N from Z",                  reminZ_py,   reminZ_tm),
    ("Recycling to N from P mort",         reminP_py,   reminP_tm),  # None in LEMMA
    ("Recycling to N from grazing",        reminG_py,   reminG_tm),  # None in LEMMA
    ("Mixing to N (k·(N0−N))",             mixN_py,     mixN_tm),     # None in LEMMA
    ("Net nutrient tendency dN/dt",        dN_py,       dN_tm),
]

# --------- Run reports ---------
# 1) States
_ = compute_and_report_isd(t, state_pairs,
                           title="Integrated squared difference — STATE VARIABLES (Truth vs LEMMA)",
                           absent_as_zero=True)

# 2a) Fluxes — treat absent processes in LEMMA as zero-valued functions
flux_results_zero = compute_and_report_isd(t, flux_pairs,
                           title="Integrated squared difference — FLUXES (absent-as-zero)",
                           absent_as_zero=True)

# 2b) Fluxes — present-only comparison (skip absent processes)
flux_results_present = compute_and_report_isd(t, flux_pairs,
                           title="Integrated squared difference — FLUXES",
                           absent_as_zero=False)

# ---- Optional: a single scalar score (average relative L2) ----
def average_relative_L2(results):
    """Average of relative L2 across the provided components."""
    if not results:
        return np.nan
    return float(np.mean([r[3] for r in results]))

score_states     = average_relative_L2(_)
score_flux_zero  = average_relative_L2(flux_results_zero)
score_flux_pres  = average_relative_L2(flux_results_present)

print("\n----- Summary scalar scores (lower is better) -----")
print(f"Avg relative L2 — States           : {score_states:.6g}")
print(f"Avg relative L2 — Fluxes (zeroed)  : {score_flux_zero:.6g}")
print(f"Avg relative L2 — Fluxes (present) : {score_flux_pres:.6g}")

# ---- Optional: quick bar plot of ISD for visual comparison ----
try:
    plt.figure(figsize=(12, 4))
    names = [r[0] for r in flux_results_zero]
    vals  = [r[1] for r in flux_results_zero]  # ISD
    x = np.arange(len(names))
    plt.bar(x, vals, color="#8da0cb")
    plt.xticks(x, names, rotation=35, ha="right")
    plt.ylabel(r"ISD $\int (f-g)^2\,dt$")
    plt.title("Integrated Squared Difference (Fluxes, absent-as-zero)")
    plt.tight_layout()
    plt.show()
except Exception:
    pass

if __name__ == "__main__":
    print_params()
