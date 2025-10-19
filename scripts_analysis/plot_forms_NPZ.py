# Compare TRUTH (NPZ_model.py) vs LEMMA (model.cpp) functional components
# Faceted overlays + "Missing from LEMMA | TRUTH" annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# ------------------ TRUTH params (from NPZ_model.py) ------------------
a = 0.2
b = 0.2
c = 0.4
e = 0.03
k = 0.05
q = 0.075
r = 0.10
s = 0.04
N0 = 0.6
alpha = 0.25
beta = 0.33
gamma = 0.5
lambda_ = 0.6
mu_hsat = 0.035

# Initial conditions (used to hold other states constant when needed)
N_init = 0.4
P_init = 0.1
Z_init = 0.05

# --------------- LEMMA params (mapped to TRUTH for form-compare) ---------------
mu_max = a / b          # baseline phyto specific rate
K_N    = e              # nutrient half-saturation
f_T    = 1.0            # hold temperature neutral
f_I    = 1.0            # hold light neutral
g_max  = lambda_
K_G    = mu_hsat
h      = 2.0
e_Z    = alpha
m_P    = r + s
r_P    = (r / (r + s)) if (r + s) > 0 else 0.0
m_Z    = q
gamma_Z= 0.0            # keep quadratic Z mortality off for core compare
r_Z    = 0.0            # TRUTH has no Z->N mortality; keep 0 in core compare
ex_Z   = gamma * q
k_mix  = k
N_star = N0

# --- Detritus pathway parameters for LEMMA-only facets (diagnostic) ---
# We set k_rem>0 so Rem/Snk panels are visible; adjust as needed.
k_rem  = 0.05           # d^-1 (diagnostic; TRUTH has no D)
k_sink = 0.0            # d^-1

# ------------------ TRUTH components ------------------
def uptake_truth(N, P):
    # U = (N/(e+N)) * (a/(b+cP)) * P
    return (N / (e + N)) * (a / (b + c * P)) * P

def grazing_loss_truth(P, Z):
    # G = lambda * P^2 / (mu_hsat^2 + P^2) * Z
    return (lambda_ * P**2 / (mu_hsat**2 + P**2)) * Z

def recycling_truth(N, P, Z):
    # to N: r*P + beta*G + gamma*q*Z   (no detritus pool)
    G = grazing_loss_truth(P, Z)
    return r * P + beta * G + gamma * q * Z

def mixing_truth_N(N):
    return k * (N0 - N)

def growth_truth_P(N, P):
    return uptake_truth(N, P)

def mortality_truth_P(P):
    return (r + s) * P

def mixing_truth_P(P):
    # magnitude of P-mixing loss term in TRUTH: k * P
    return k * P
def growth_truth_Z(P, Z):
    return alpha * grazing_loss_truth(P, Z)

def mortality_truth_Z(Z):
    return q * Z
# ------------------ LEMMA components (model.cpp forms) ------------------
def uptake_lemma(N, P):
    # U = mu * P, mu = mu_max * f_T * f_I * f_N
    fN = N / (K_N + N)
    mu = mu_max * f_T * f_I * fN
    return mu * P

def holl(P):
    return (P**h) / (K_G**h + P**h)

def grazing_loss_lemma(P, Z):
    # G = g_max * f_T * Holl(P) * Z
    g_rate = g_max * f_T * holl(P)
    return g_rate * Z

def mortality_lemma_P(P):
    return m_P * P

def growth_lemma_Z(P, Z):
    return e_Z * grazing_loss_lemma(P, Z)

def mortality_lemma_Z(Z):
    return m_Z * Z + gamma_Z * Z**2

def recycling_lemma(N, P, Z, D=0.0):
    RpN = r_P * m_P * P
    RzN = r_Z * m_Z * Z
    Ex  = ex_Z * Z
    Rem = k_rem * D
    return RpN + RzN + Ex + Rem

def mixing_lemma_N(N):
    return k_mix * (N_star - N)

def growth_lemma_P(N, P):
    return uptake_lemma(N, P)

# ------------------ LEMMA-only flux helpers (Detritus & Z->N) ------------------
def lemma_z_to_n_recycling(Z):
    # RzN = r_Z * m_Z * Z  (TRUTH: none)
    return r_Z * m_Z * Z

def lemma_detritus_from_unassimilated(P, Z):
    # RgD = (1 - e_Z) * G
    G = grazing_loss_lemma(P, Z)
    return (1.0 - e_Z) * G

def lemma_detritus_from_p_mort(P):
    # RpD = (1 - r_P) * m_P * P
    return (1.0 - r_P) * m_P * P

def lemma_detritus_from_z_mort(Z):
    # RzD = (1 - r_Z) * m_Z * Z
    return (1.0 - r_Z) * m_Z * Z

def lemma_D_star(P, Z):
    # Diagnostic steady-state D*: D* = (RgD + RpD + RzD) / (k_rem + k_sink)
    RgD = lemma_detritus_from_unassimilated(P, Z)
    RpD = lemma_detritus_from_p_mort(P)
    RzD = lemma_detritus_from_z_mort(Z)
    denom = (k_rem + k_sink) + 1e-12
    return (RgD + RpD + RzD) / denom

def lemma_remineralization(P, Z):
    # Rem = k_rem * D*
    return k_rem * lemma_D_star(P, Z)

def lemma_sinking(P, Z):
    # Snk = k_sink * D*
    return k_sink * lemma_D_star(P, Z)

# ------------------ Faceted plotting ------------------
P_range = np.linspace(0.0, 2.0, 400)
N_range = np.linspace(0.0, 2.0, 400)
Z_range = np.linspace(0.0, 2.0, 400)

Z_fixed = Z_init
N_fixed = N_init

# facets: list of tuples (key, title, data_fn)
# where data_fn() -> (x, y_truth, y_lemma, xlabel)
# facets: list of tuples (key, title, data_fn)
# where data_fn() -> (x, y_truth, y_lemma, xlabel)
facets = [
    # ---------- In dN/dt ----------
    ("nutrient_equation_uptake", "Nutrient uptake",
     lambda: (P_range,
              uptake_truth(N_fixed, P_range),
              uptake_lemma(N_fixed, P_range),
              "x=P (N fixed)")),

    ("nutrient_equation_recycling", "Recycling to N",
     lambda: (P_range,
              recycling_truth(N_fixed, P_range, Z_fixed),
              recycling_lemma(N_fixed, P_range, Z_fixed),
              "x=P (N,Z fixed)")),

    ("nutrient_equation_mixing", "N mixing",
     lambda: (N_range,
              mixing_truth_N(N_range),
              mixing_lemma_N(N_range),
              "x=N")),

    # ---------- In dP/dt ----------
    ("phytoplankton_equation_growth", "P growth",
     lambda: (P_range,
              growth_truth_P(N_fixed, P_range),
              growth_lemma_P(N_fixed, P_range),
              "x=P (N fixed)")),

    ("phytoplankton_equation_grazing_loss", "P grazing loss",
     lambda: (P_range,
              grazing_loss_truth(P_range, Z_fixed),
              grazing_loss_lemma(P_range, Z_fixed),
              "x=P (Z fixed)")),

    ("phytoplankton_equation_mortality", "P mortality",
     lambda: (P_range,
              mortality_truth_P(P_range),
              mortality_lemma_P(P_range),
              "x=P")),

    ("phytoplankton_equation_mixing", "P mixing",
     lambda: (P_range,
              mixing_truth_P(P_range),
              None,  # Missing in LEMMA
              "x=P")),

    # ---------- In dZ/dt ----------
    ("zooplankton_equation_growth", "Z growth",
     lambda: (P_range,
              growth_truth_Z(P_range, Z_fixed),
              growth_lemma_Z(P_range, Z_fixed),
              "x=P (Z fixed)")),

    ("zooplankton_equation_mortality", "Z mortality",
     lambda: (Z_range,
              mortality_truth_Z(Z_range),
              mortality_lemma_Z(Z_range),
              "x=Z")),

    # ---------- LEMMA-only extras ----------
    ("lemma_only_z_to_n", "Z to N recycling",
     lambda: (Z_range,
              None,  # Missing in TRUTH
              lemma_z_to_n_recycling(Z_range),
              "x=Z")),

    ("lemma_only_rgD", "Detritus from messy grazing",
     lambda: (P_range,
              None,  # Missing in TRUTH
              lemma_detritus_from_unassimilated(P_range, Z_fixed),
              "x=P (Z fixed)")),

    ("lemma_only_rpD", "Detritus from P mortality",
     lambda: (P_range,
              None,  # Missing in TRUTH
              lemma_detritus_from_p_mort(P_range),
              "x=P")),

    ("lemma_only_rzD", "Detritus from Z mortality",
     lambda: (Z_range,
              None,  # Missing in TRUTH
              lemma_detritus_from_z_mort(Z_range),
              "x=Z")),

    ("lemma_only_rem", "Remineralization to N",
     lambda: (P_range,
              None,  # Missing in TRUTH
              lemma_remineralization(P_range, Z_fixed),
              "x=P (Z fixed)")),

    ("lemma_only_snk", "Detritus sinking/export",
     lambda: (P_range,
              None,  # Missing in TRUTH
              lemma_sinking(P_range, Z_fixed),
              "x=P (Z fixed)")),
]

# ------------------ Plotting ------------------
n = len(facets)
cols = 3
rows = int(np.ceil(n / cols))

fig, axes = plt.subplots(
    rows, cols,
    figsize=(min(6, 2 * cols), 1.5 * rows),
    sharex=False, sharey=False,
    constrained_layout=True
)
axes = np.atleast_1d(axes).flatten()

legend_proxies = [
    Line2D([0], [0], color="tab:blue", label="TRUTH"),
    Line2D([0], [0], color="tab:orange", linestyle="--", label="LEMMA"),
]

bg_color = "white"
axis_linewidth = 0.8
tick_width = 0.8
tick_length = 3
fig.patch.set_facecolor(bg_color)

for idx, (key, title, data_fn) in enumerate(facets):
    ax = axes[idx]
    x, y_truth, y_lemma, xlabel = data_fn()
    ax.set_title(title, fontsize=10)
    ax.set_facecolor(bg_color)

    if y_truth is not None:
        ax.plot(x, y_truth, label="TRUTH", color="tab:blue")
    if y_lemma is not None:
        ax.plot(x, y_lemma, label="LEMMA", color="tab:orange", linestyle="--")

    # Missing-series annotation
    missing = []
    if y_truth is None:
        missing.append("TRUTH")
    if y_lemma is None:
        missing.append("LEMMA")

    if missing:
        # Determine color based on which series is missing
        if len(missing) == 1:
            missing_color = "tab:blue" if missing[0] == "TRUTH" else "tab:orange"
        else:
            # Both missing: use a neutral color or orange for consistency
            missing_color = "tab:gray"

        msg = "Missing from " + (" & ".join(missing) if len(missing) > 1 else missing[0])
        ax.text(0.65, 0.2, msg, ha="center", va="center", transform=ax.transAxes,
                fontsize=7, color=missing_color, bbox=dict(facecolor="white", alpha=0.6))
    # Style (classic, no grid)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(axis_linewidth)
    ax.spines["bottom"].set_linewidth(axis_linewidth)

    ax.tick_params(axis="both", which="both",
                   direction="out", length=tick_length, width=tick_width, labelsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    if idx % cols == 0:
        ax.set_ylabel("Flux (g C m$^{-3}$ d$^{-1}$)", fontsize=9)
    if idx // cols == rows - 1:
        ax.set_xlabel(xlabel, fontsize=9)

# Hide any unused axes
for j in range(n, rows * cols):
    if j >= len(axes): break
    if j >= n:
        fig.delaxes(axes[j])

# Legend outside on the right
fig.legend(
    handles=legend_proxies,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False,
    fontsize=9,
    ncol=1,
    borderaxespad=0.0
)

plt.savefig('Figures/NPZ_form_check.png', dpi=300, bbox_inches='tight')
plt.close(fig)