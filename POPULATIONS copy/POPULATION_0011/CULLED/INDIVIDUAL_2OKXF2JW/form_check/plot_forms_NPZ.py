
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

# LEMMA parameters from TMB model
def get_lemma_params():
    # Environmental forcing parameters
    I0 = 1.0        # Mean light
    I_amp = 0.2     # Seasonal amplitude
    K_I = 0.5       # Light half-saturation
    phi_I = 0.0     # Light phase
    T_ref = 15.0    # Reference temperature
    T_amp = 2.0     # Temperature amplitude
    phi_T = 0.0     # Temperature phase
    Q10 = 1.8       # P growth temperature sensitivity
    Q10_g = 2.0     # Z grazing temperature sensitivity
    
    # Growth and limitation
    mu_max = 0.8    # Maximum P growth rate
    K_N = 0.05      # Nutrient half-saturation
    y_P = 1.0       # P yield per nutrient
    
    # Grazing
    g_max = 0.6     # Maximum grazing rate
    K_P = 0.1       # Grazing half-saturation
    h = 1.5         # Holling response order
    beta = 0.7      # Grazing efficiency
    
    # Mortality and losses
    mPl = 0.05      # Linear P mortality
    mPq = 0.05      # Quadratic P mortality
    mzl = 0.05      # Linear Z mortality  
    mZq = 0.05      # Quadratic Z mortality
    ex_z = 0.05     # Z excretion rate
    
    # Remineralization and mixing
    rho_P = 0.8     # P mortality to N
    rho_Z = 0.8     # Z mortality to N
    k_mix = 0.05    # Mixing rate
    N_ext = 0.5     # External nutrient
    
    return locals()

def rhs_lemma(t, y, p):
    N, P, Z = y
    N = max(N, 0.0); P = max(P, 0.0); Z = max(Z, 0.0)
    
    # Environmental forcing
    It = p['I0'] * (1.0 + p['I_amp'] * np.sin(2*np.pi * (t/365.0) + p['phi_I']))
    It = max(It, 0.0)
    LI = It / (p['K_I'] + It + EPS)
    
    Tt = p['T_ref'] + p['T_amp'] * np.sin(2*np.pi * (t/365.0) + p['phi_T'])
    Theta_T_P = p['Q10']**((Tt - p['T_ref'])/10.0)
    Theta_T_G = p['Q10_g']**((Tt - p['T_ref'])/10.0)
    
    # Resource limitation and growth
    LN = N / (p['K_N'] + N + EPS)
    rP = p['mu_max'] * Theta_T_P * LN * LI
    G_P = rP * P
    U_N = G_P / (p['y_P'] + EPS)
    
    # Grazing
    P_pow_h = P**p['h']
    K_P_pow_h = p['K_P']**p['h']
    g_fun = p['g_max'] * Theta_T_G * P_pow_h / (K_P_pow_h + P_pow_h + EPS)
    G = g_fun * Z
    
    # Partitioning and losses
    Z_gain = p['beta'] * G
    Z_excr = p['ex_z'] * Z
    P_mort = p['mPl'] * P + p['mPq'] * P * P
    Z_mort = p['mzl'] * Z + p['mZq'] * Z * Z
    
    # Remineralization and mixing
    N_remin = (1.0 - p['beta']) * G + p['rho_P'] * P_mort + p['rho_Z'] * Z_mort
    N_mix = p['k_mix'] * (p['N_ext'] - N)
    
    # State derivatives
    dPdt = G_P - G - P_mort
    dZdt = Z_gain - Z_excr - Z_mort
    dNdt = -U_N + N_remin + N_mix
    
    return [dNdt, dPdt, dZdt]

def simulate_lemma():
    params = get_lemma_params()
    sol = solve_ivp(rhs_lemma, (t_start, t_end), y0, t_eval=t_eval, 
                    args=(params,), method="RK45", rtol=1e-7, atol=1e-9)
    if not sol.success:
        raise RuntimeError("Integration failed for LEMMA model.")
    
    t = sol.t
    N = np.maximum(sol.y[0], 0.0)
    P = np.maximum(sol.y[1], 0.0)
    Z = np.maximum(sol.y[2], 0.0)
    
    # Compute all fluxes using the same equations as in rhs_lemma
    It = params['I0'] * (1.0 + params['I_amp'] * np.sin(2*np.pi * (t/365.0) + params['phi_I']))
    It = np.maximum(It, 0.0)
    LI = It / (params['K_I'] + It + EPS)
    
    Tt = params['T_ref'] + params['T_amp'] * np.sin(2*np.pi * (t/365.0) + params['phi_T'])
    Theta_T_P = params['Q10']**((Tt - params['T_ref'])/10.0)
    Theta_T_G = params['Q10_g']**((Tt - params['T_ref'])/10.0)
    
    LN = N / (params['K_N'] + N + EPS)
    rP = params['mu_max'] * Theta_T_P * LN * LI
    uptake = rP * P
    U_N = uptake / (params['y_P'] + EPS)
    
    P_pow_h = P**params['h']
    K_P_pow_h = params['K_P']**params['h']
    g_fun = params['g_max'] * Theta_T_G * P_pow_h / (K_P_pow_h + P_pow_h + EPS)
    grazing = g_fun * Z
    
    Z_growth = params['beta'] * grazing
    Z_mort = params['mzl'] * Z + params['mZq'] * Z * Z
    P_mort = params['mPl'] * P + params['mPq'] * P * P
    
    reminZ = params['rho_Z'] * Z_mort
    reminP = params['rho_P'] * P_mort
    reminG = (1.0 - params['beta']) * grazing
    mixN = params['k_mix'] * (params['N_ext'] - N)
    
    dN = -U_N + reminP + reminG + reminZ + mixN
    
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
    # Machine-readable outputs
    print(f"OVERALL_ISD_SUM: {overall_isd_sum}")
    summary = {
        "overall_isd_sum": overall_isd_sum,
        "n_states": len(state_pairs),
        "n_fluxes_compared": len(facets),
        "notes": "NPZ comparison: TMB-like LEMMA implementation"
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
    # Use non-interactive backend
    plt.switch_backend('Agg')
    
    # Create figs directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    main()
    plt.close('all')
