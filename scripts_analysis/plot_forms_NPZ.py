# Compare TRUTH (NPZ_model.py) vs LEMMA (model.cpp) functional components
# Faceted overlays + "Missing from LEMMA | TRUTH" annotations
import numpy as np
import matplotlib.pyplot as plt

# ---------- TRUTH params (from NPZ_model.py) ----------
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

# ---------- LEMMA params (mapped to TRUTH to enable comparison) ----------
mu_max = a / b          # baseline rate
K_N = e                 # nutrient half-sat
f_T = 1.0               # hold temp effect neutral
f_I = 1.0               # hold light effect neutral

g_max = lambda_
K_G = mu_hsat
h = 2.0

e_Z = alpha
m_P = r + s
r_P = (r / (r + s)) if (r + s) > 0 else 0.0
m_Z = q
gamma_Z = 0.0           # disable quadratic Z mortality for shape-only compare
r_Z = 0.0               # TRUTH has no Z->N mortality term
ex_Z = gamma * q

k_mix = k
N_star = N0
k_rem = 0.0             # TRUTH has no D pool
k_sink = 0.0
# ---------- TRUTH components ----------
def uptake_truth(N, P):
    return (N / (e + N)) * (a / (b + c * P)) * P

def recycling_truth(N, P, Z):
    graz = (lambda_ * P**2 / (mu_hsat**2 + P**2)) * Z
    return r * P + beta * graz + gamma * q * Z

def mixing_truth_N(N):
    return k * (N0 - N)

def growth_truth_P(N, P):
    return uptake_truth(N, P)

def grazing_loss_truth(P, Z):
    return (lambda_ * P**2 / (mu_hsat**2 + P**2)) * Z

def mortality_truth_P(P):
    return (r + s) * P

def mixing_truth_P(P):
    return k * P

def growth_truth_Z(P, Z):
    return alpha * (lambda_ * P**2 / (mu_hsat**2 + P**2)) * Z

def mortality_truth_Z(Z):
    return q * Z

# ---------- LEMMA components ----------
def uptake_lemma(N, P):
    f_N = N / (K_N + N)
    mu = mu_max * f_T * f_I * f_N
    return mu * P

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

def grazing_loss_lemma(P, Z):
    holl = (P**h) / (K_G**h + P**h)
    g_rate = g_max * f_T * holl
    return g_rate * Z

def mortality_lemma_P(P):
    return m_P * P

def growth_lemma_Z(P, Z):
    return e_Z * grazing_loss_lemma(P, Z)

def mortality_lemma_Z(Z):
    return m_Z * Z + gamma_Z * Z**2

# ---------- Faceted plotting ----------
P_range = np.linspace(0.0, 2.0, 400)
N_range = np.linspace(0.0, 2.0, 400)
Z_range = np.linspace(0.0, 2.0, 400)

Z_fixed = Z_init
N_fixed = N_init

facets = [
    ("nutrient_equation_uptake", "Uptake U(N,P)",
     lambda: (P_range,
              uptake_truth(N_fixed, P_range),
              uptake_lemma(N_fixed, P_range),
              "x=P (N fixed)")
    ),
    ("nutrient_equation_recycling", "Recycling to N",
     lambda: (P_range,
              recycling_truth(N_fixed, P_range, Z_fixed),
              recycling_lemma(N_fixed, P_range, Z_fixed),
              "x=P (N,Z fixed)")
    ),
    ("nutrient_equation_mixing", "N mixing",
     lambda: (N_range,
              mixing_truth_N(N_range),
              mixing_lemma_N(N_range),
              "x=N")
    ),
    ("phytoplankton_equation_growth", "P growth",
     lambda: (P_range,
              growth_truth_P(N_fixed, P_range),
              growth_lemma_P(N_fixed, P_range),
              "x=P (N fixed)")
    ),
    ("phytoplankton_equation_grazing_loss", "P grazing loss",
     lambda: (P_range,
              grazing_loss_truth(P_range, Z_fixed),
              grazing_loss_lemma(P_range, Z_fixed),
              "x=P (Z fixed)")
    ),
    ("phytoplankton_equation_mortality", "P mortality",
     lambda: (P_range,
              mortality_truth_P(P_range),
              mortality_lemma_P(P_range),
              "x=P")
    ),
    ("phytoplankton_equation_mixing", "P mixing",
     lambda: (P_range,
              mixing_truth_P(P_range),
              None,  # Missing in LEMMA
              "x=P")
    ),
    ("zooplankton_equation_growth", "Z growth",
     lambda: (P_range,
              growth_truth_Z(P_range, Z_fixed),
              growth_lemma_Z(P_range, Z_fixed),
              "x=P (Z fixed)")
    ),
    ("zooplankton_equation_mortality", "Z mortality",
     lambda: (Z_range,
              mortality_truth_Z(Z_range),
              mortality_lemma_Z(Z_range),
              "x=Z")
    ),
    # ("zooplankton_equation_mixing", "Z mixing",
    #  lambda: (Z_range,
    #           None,   # Missing in TRUTH
    #           None,   # Missing in LEMMA
    #           "x=Z")
    # ),
]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# facets: list of tuples (key, title, data_fn)
# where data_fn() -> (x, y_truth, y_lemma, xlabel)

n = len(facets)
cols = 3
rows = int(np.ceil(n / cols))

# Compact figure; keep scales free
fig, axes = plt.subplots(
    rows, cols,
    figsize=(min(9.5, 3.1 * cols), 2.6 * rows),  # tune to taste
    sharex=False, sharey=False,
    constrained_layout=True
)
axes = np.atleast_1d(axes).flatten()

# Single figure-level legend proxies (so legend shows even if a series is missing)
legend_proxies = [
    Line2D([0], [0], color="tab:blue", label="TRUTH"),
    Line2D([0], [0], color="tab:orange", linestyle="--", label="LEMMA"),
]

# ---- theme_classic()-like styling parameters ----
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

    # Plot series (per-panel free scales)
    if y_truth is not None:
        ax.plot(x, y_truth, label="TRUTH", color="tab:blue")
    if y_lemma is not None:
        ax.plot(x, y_lemma, label="LEMMA", color="tab:orange", linestyle="--")

    # Missing-series message
    missing = []
    if y_truth is None: missing.append("TRUTH")
    if y_lemma is None: missing.append("LEMMA")
    if missing:
        msg = "Missing from " + (" & ".join(missing) if len(missing) > 1 else missing[0])
        ax.text(0.5, 0.9, msg, ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="tab:orange", bbox=dict(facecolor="white", alpha=0.6))

    # --- ggplot theme_classic() look ---
    # No grid
    ax.grid(False)

    # Only left and bottom spines visible; clean, classic look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(axis_linewidth)
    ax.spines["bottom"].set_linewidth(axis_linewidth)

    # Outward ticks, compact labels
    ax.tick_params(axis="both", which="both",
                   direction="out", length=tick_length, width=tick_width, labelsize=8)

    # Modest number of ticks (keeps it tidy with free scales)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    # Axis labels only on the outer panels to save space
    if idx % cols == 0:
        ax.set_ylabel("Flux (g C m$^{-3}$ d$^{-1}$)", fontsize=9)
    if idx // cols == rows - 1:
        ax.set_xlabel(xlabel, fontsize=9)

# Hide any unused axes
for j in range(n, rows * cols):
    fig.delaxes(axes[j])

# ---- Legend outside on the right ----
# Vertical stack (ncol=1) is usually best for a right-side legend.
# bbox_inches='tight' in savefig ensures it’s included in the output.
fig.legend(
    handles=legend_proxies,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),  # just outside the right edge
    frameon=False,
    fontsize=9,
    ncol=1,
    borderaxespad=0.0
)


# Save tightly to include the external legend
plt.savefig('Figures/NPZ_form_check.png', dpi=300, bbox_inches='tight')
plt.close(fig)
