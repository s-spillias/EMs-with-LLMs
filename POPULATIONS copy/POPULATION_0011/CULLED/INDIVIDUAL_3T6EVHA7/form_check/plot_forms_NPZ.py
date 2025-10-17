
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scripts_analysis.form_utils import _savefig, compute_isd_table, average_rel_L2, _plot_state, _plot_facet, print_summary

EPS = 1e-12
SAVE_DIR = os.environ.get("SAVE_PLOTS_DIR", "figs")
SAVE_FMT = os.environ.get("SAVE_PLOTS_FORMAT", "png")
MODEL_ID = os.environ.get("MODEL_ID", "model")

# Truth NPZ parameters
a = 0.2; b = 0.2; c = 0.4; eN = 0.03
k = 0.05; q = 0.075; r = 0.10; s = 0.04
N0_mix = 0.6
alpha = 0.25; beta = 0.33; gamma = 0.5
lambda_ = 0.6; mu_P = 0.035

# Initial conditions & time grid
N0 = 0.4; P0 = 0.1; Z0 = 0.05
y0 = [N0, P0, Z0]
t_start, t_end = 0.0, 100.0
n_eval = 400
t_eval = np.linspace(t_start, t_end, n_eval)

def rhs_truth(t, y):
    N, P, Z = y
    N = max(N, 0.0); P = max(P, 0.0); Z = max(Z, 0.0)
    uptake = (N/(eN + N + EPS)) * (a/(b + c*P + EPS)) * P
    grazing = lambda_ * (P**2)/(mu_P**2 + P**2 + EPS) * Z
    dPdt = uptake - grazing - r*P - (s + k)*P
    dZdt = alpha * grazing - q * Z
    dNdt = -uptake + r*P + beta*grazing + gamma*q*Z + k*(N0_mix - N)
    return [dNdt, dPdt, dZdt]

def simulate_truth():
    sol = solve_ivp(rhs_truth, (t_start, t_end), y0, t_eval=t_eval, method="RK45", rtol=1e-7, atol=1e-9)
    if not sol.success:
        raise RuntimeError("Integration failed for Truth model.")
    t = sol.t
    N, P, Z = np.maximum(sol.y[0], 0.0), np.maximum(sol.y[1], 0.0), np.maximum(sol.y[2], 0.0)
    uptake = (N/(eN + N + EPS)) * (a/(b + c*P + EPS)) * P
    grazing = lambda_ * (P**2)/(mu_P**2 + P**2 + EPS) * Z
    P_mort = r*P + (s + k)*P
    Z_growth = alpha * grazing
    Z_mort = q * Z
    reminZ = gamma * q * Z
    reminP = r * P
    reminG = beta * grazing
    mixN = k * (N0_mix - N)
    dN = -uptake + reminP + reminG + reminZ + mixN
    return {
        "t": t, "N": N, "P": P, "Z": Z,
        "uptake": uptake, "grazing": grazing,
        "P_mort": P_mort, "Z_growth": Z_growth, "Z_mort": Z_mort,
        "reminZ": reminZ, "reminP": reminP, "reminG": reminG, "mixN": mixN,
        "dN": dN
    }

# LEMMA parameters from parameters.json
mu_max = 0.8      # Maximum phytoplankton growth rate
K_N = 0.08        # Half-saturation for nutrient uptake
hN = 2.0          # Hill exponent for nutrient limitation
I0 = 1.0          # Surface irradiance
K_I = 0.5         # Half-saturation for light
k_Ishade = 1.0    # Self-shading coefficient
k_Ibg = 0.5       # Background light attenuation
q10_mu = 2.0      # Q10 for phytoplankton growth
q10_g = 4.31      # Q10 for zooplankton grazing
g_max = 0.9       # Maximum grazing rate
K_g = 0.5         # Half-saturation for grazing
h_exp = 2.0       # Grazing response exponent
c_BD = 0.2        # Predator interference
e_Z = 0.3         # Zooplankton assimilation efficiency
mP1 = 0.05        # Linear P mortality
mP2 = 0.1         # Quadratic P mortality
mZ1 = 0.05        # Linear Z mortality
mZ2 = 0.1         # Quadratic Z mortality
rP_N = 0.8        # P remineralization fraction
rZ_N = 0.8        # Z remineralization fraction
y_PN = 1.0        # Yield coefficient
k_mix = 0.05      # Mixing rate
N_deep = 0.3      # Deep nutrient concentration
T_C = 15.0        # Ambient temperature
T_ref = 15.0      # Reference temperature

def rhs_lemma(t, y):
    N, P, Z = y
    N = max(N, 0.0); P = max(P, 0.0); Z = max(Z, 0.0)
    
    # Temperature scaling
    theta_mu = q10_mu**((T_C - T_ref)/10.0)
    theta_g = q10_g**((T_C - T_ref)/10.0)
    
    # Nutrient limitation (Hill)
    fN = (N**hN) / (K_N**hN + N**hN + EPS)
    
    # Light limitation with self-shading
    I_eff = I0 * np.exp(-(k_Ibg + k_Ishade * P))
    fI = I_eff / (K_I + I_eff + EPS)
    
    # Growth with co-limitation
    f_lim = min(fN, fI)  # Liebig's law
    P_growth = mu_max * theta_mu * f_lim * P
    
    # Grazing with interference
    G_fun = P**h_exp / (K_g**h_exp + P**h_exp + EPS)
    interference = 1.0 + c_BD * Z
    Z_grazing = g_max * theta_g * (G_fun / interference) * Z
    
    # Growth and mortality fluxes
    Z_growth = e_Z * Z_grazing
    P_losses = mP1 * P + mP2 * P * P
    Z_losses = mZ1 * Z + mZ2 * Z * Z
    
    # Nutrient cycling
    N_uptake = P_growth / y_PN
    N_remin = rP_N * P_losses + rZ_N * Z_losses + (1.0 - e_Z) * Z_grazing
    
    # System ODEs
    dNdt = k_mix * (N_deep - N) - N_uptake + N_remin
    dPdt = P_growth - Z_grazing - P_losses
    dZdt = Z_growth - Z_losses
    
    return [dNdt, dPdt, dZdt]

def simulate_lemma():
    sol = solve_ivp(rhs_lemma, (t_start, t_end), y0, t_eval=t_eval, 
                    method="RK45", rtol=1e-7, atol=1e-9)
    if not sol.success:
        raise RuntimeError("Integration failed for LEMMA model.")
    
    t = sol.t
    N, P, Z = np.maximum(sol.y[0], 0.0), np.maximum(sol.y[1], 0.0), np.maximum(sol.y[2], 0.0)
    
    # Compute all fluxes for comparison
    theta_mu = q10_mu**((T_C - T_ref)/10.0)
    theta_g = q10_g**((T_C - T_ref)/10.0)
    
    fN = (N**hN) / (K_N**hN + N**hN + EPS)
    I_eff = I0 * np.exp(-(k_Ibg + k_Ishade * P))
    fI = I_eff / (K_I + I_eff + EPS)
    f_lim = np.minimum(fN, fI)
    
    P_growth = mu_max * theta_mu * f_lim * P
    G_fun = P**h_exp / (K_g**h_exp + P**h_exp + EPS)
    interference = 1.0 + c_BD * Z
    
    uptake = P_growth / y_PN
    grazing = g_max * theta_g * (G_fun / interference) * Z
    P_mort = mP1 * P + mP2 * P * P
    Z_growth = e_Z * grazing
    Z_mort = mZ1 * Z + mZ2 * Z * Z
    
    reminZ = rZ_N * Z_mort
    reminP = rP_N * P_mort
    reminG = (1.0 - e_Z) * grazing
    mixN = k_mix * (N_deep - N)
    
    dN = -uptake + reminP + reminG + reminZ + mixN
    
    return {
        "t": t, "N": N, "P": P, "Z": Z,
        "uptake": uptake, "grazing": grazing,
        "P_mort": P_mort, "Z_growth": Z_growth, "Z_mort": Z_mort,
        "reminZ": reminZ, "reminP": reminP, "reminG": reminG, "mixN": mixN,
        "dN": dN
    }

def main():
    truth = simulate_truth()
    try:
        lemma = simulate_lemma()
    except NameError:
        lemma = {k: None for k in truth.keys()}
        lemma["t"] = truth["t"]

    # Plot states
    plt.style.use("seaborn-v0_8-whitegrid")
    fig1, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    _plot_state(axes[0], "Nutrients (N)", truth["N"], lemma["N"], ylabel="g C m$^{-3}$")
    _plot_state(axes[1], "Phytoplankton (P)", truth["P"], lemma["P"])
    _plot_state(axes[2], "Zooplankton (Z)", truth["Z"], lemma["Z"])
    fig1.tight_layout()
    _savefig(fig1, "npz_states", SAVE_DIR, SAVE_FMT, MODEL_ID)

    # Flux facets
    facets = [
        ("Uptake", truth["uptake"], lemma["uptake"]),
        ("Grazing", truth["grazing"], lemma["grazing"]),
        ("P mortality", truth["P_mort"], lemma["P_mort"]),
        ("Z growth", truth["Z_growth"], lemma["Z_growth"]),
        ("Z mortality", truth["Z_mort"], lemma["Z_mort"]),
        ("Remin Z", truth["reminZ"], lemma["reminZ"]),
        ("Remin P", truth["reminP"], lemma["reminP"]),
        ("Remin G", truth["reminG"], lemma["reminG"]),
        ("Mixing N", truth["mixN"], lemma["mixN"]),
        ("dN/dt", truth["dN"], lemma["dN"]),
    ]
    nrows, ncols = 2, 5
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(18, 6), sharex=True)
    axes2 = axes2.ravel()
    for ax, (title, s_truth, s_lemma) in zip(axes2, facets):
        _plot_facet(ax, title, truth["t"], s_truth, s_lemma)
    fig2.tight_layout(rect=[0, 0, 1, 0.98])
    _savefig(fig2, "npz_flux_facets", SAVE_DIR, SAVE_FMT, MODEL_ID)

    # ISD tables
    state_pairs = [
        ("N", truth["N"], lemma["N"]),
        ("P", truth["P"], lemma["P"]),
        ("Z", truth["Z"], lemma["Z"]),
    ]
    state_results = compute_isd_table(truth["t"], state_pairs, "ISD — STATES", absent_as_zero=True)
    flux_results_zero = compute_isd_table(truth["t"], facets, "ISD — FLUXES (absent-as-zero)", absent_as_zero=True)
    flux_results_present = compute_isd_table(truth["t"], facets, "ISD — FLUXES (present-only)", absent_as_zero=False)

    score_states = average_rel_L2(state_results)
    score_flux_zero = average_rel_L2(flux_results_zero)
    score_flux_present = average_rel_L2(flux_results_present)
    overall_isd_sum = float(np.sum([r[1] for r in flux_results_zero]))

    # Summary output
    print_summary(overall_isd_sum, score_states, score_flux_zero, score_flux_present,
                  len(state_pairs), len(facets), notes="NPZ comparison: TMB-like LEMMA implementation")
    
    # Machine readable summary
    print(f"OVERALL_ISD_SUM: {overall_isd_sum}")
    summary = {
        "overall_isd_sum": overall_isd_sum,
        "n_states": len(state_pairs),
        "n_fluxes_compared": len(facets),
        "notes": "TMB-like LEMMA implementation"
    }
    print(f"SUMMARY_JSON: {json.dumps(summary)}")

    # ISD bar chart
    try:
        fig3 = plt.figure(figsize=(12, 4))
        names = [r[0] for r in flux_results_zero]
        vals = [r[1] for r in flux_results_zero]
        x = np.arange(len(names))
        plt.bar(x, vals, color="#8da0cb")
        plt.xticks(x, names, rotation=30, ha="right")
        plt.ylabel(r"ISD $\\int (f-g)^2 dt$")
        plt.title("Integrated Squared Difference — Fluxes (absent-as-zero)")
        plt.tight_layout()
        _savefig(fig3, "npz_flux_isd_bar", SAVE_DIR, SAVE_FMT, MODEL_ID)
    except Exception:
        pass

if __name__ == "__main__":
    main()
