#include <TMB.hpp>

/*
Modeling episodic outbreaks of Crown-of-Thorns starfish (COTS) with coral feedbacks and environmental forcing.

Equations (discrete yearly steps; t indexes years; all transitions use previous states only):

Let:
  C_t   = cots_pred[t]   (ind m^-2)
  A_t   = fast_pred[t]   (% cover)
  S_t   = slow_pred[t]   (% cover)
  P_t   = w_fast*A_t + (1-w_fast)*S_t  (prey index in % cover)
  K     = 100 (% cover, total space)

Forcing:
  IMM_t = cotsimm_dat[t]  (larval immigration, ind m^-2 yr^-1)
  SST_t = sst_dat[t]      (°C)

Helper functions:
  softplus(x) = log1p(exp(-|x|)) + max(x, 0)     // smooth ReLU to keep C >= 0
  invlogit(x) = 1 / (1 + exp(-x))                // smooth ramp in [0,1]
  cons_t = c_max * C_t * P_t / (Hc + P_t + eps)  // Holling type-II (1)
  pA_t   = w_fast*A_t / (P_t + eps)              // diet share on fast coral
  pS_t   = 1 - pA_t                              // diet share on slow coral

Environmental effects:
  temp_fac_t = exp(-0.5 * pow((SST_t - sst_opt)/sst_sd, 2))                  // Gaussian thermal performance (2)
  bleach_idx_t = invlogit(k_bleach*(SST_t - sst_bleach_thr))                 // smooth threshold for bleaching (3)

State transitions (all use t-1 on RHS):
  1) Coral losses to COTS at t-1:
     cons_{t-1} as (1) with P_{t-1}
     lossA_{t-1} = cons_{t-1} * pA_{t-1}
     lossS_{t-1} = cons_{t-1} * pS_{t-1}

  2) SST bleaching losses:
     bleachA_{t-1} = bA * bleach_idx_{t-1} * A_{t-1}
     bleachS_{t-1} = bS * bleach_idx_{t-1} * S_{t-1}

  3) Coral growth (logistic, shared space K=100):
     A_prop = A_{t-1} + rA*A_{t-1}*(1 - (A_{t-1}+S_{t-1})/K) - lossA_{t-1} - bleachA_{t-1}
     S_prop = S_{t-1} + rS*S_{t-1}*(1 - (A_{t-1}+S_{t-1})/K) - lossS_{t-1} - bleachS_{t-1}
     Then smoothly squash to [0,K] via tanh-squash:
     X_t = 0.5*K * (tanh((X_prop - 0.5*K)/squash_scale) + Type(1)) for X in {A,S}      (4)

  4) COTS dynamics (bottom-up, Allee, density dependence, immigration):
     prey_idx_{t-1} = P_{t-1}/K
     mate_{t-1} = invlogit(k_allee * (C_{t-1} - C_crit))
     growth_{t-1} = rC * C_{t-1} * mate_{t-1} * pow(prey_idx_{t-1} + eps, theta_prey) * temp_fac_{t-1}
     mort_{t-1}   = mC*C_{t-1} + m_food*C_{t-1}*(Type(1) - prey_idx_{t-1})
     dens_{t-1}   = densC * C_{t-1]*C_{t-1}
     C_prop = C_{t-1} + growth_{t-1} - mort_{t-1} - dens_{t-1} + beta_imm * IMM_{t-1}
     C_t = softplus(C_prop)                                                              (5)

Observation models (all time steps used):
  cots_dat[t] ~ LogNormal(log(C_t + eps), sigma_cots)
  fast_dat[t] ~ Normal(A_t, sigma_fast_floor)
  slow_dat[t] ~ Normal(S_t, sigma_slow_floor)
with sigma floors added to ensure numerical stability.

All "_pred" variables are reported via REPORT/ADREPORT.
*/

template<class Type>
Type objective_function<Type>::operator() ()
{
  using CppAD::abs;

  // ----------------------------- Data -----------------------------
  DATA_VECTOR(Year);        // Calendar year (integer-like), used only for indexing and reporting
  DATA_VECTOR(cots_dat);    // Adult COTS density [ind m^-2], strictly positive
  DATA_VECTOR(fast_dat);    // Fast coral cover [%]
  DATA_VECTOR(slow_dat);    // Slow coral cover [%]
  DATA_VECTOR(cotsimm_dat); // Larval immigration [ind m^-2 yr^-1]
  DATA_VECTOR(sst_dat);     // Sea Surface Temperature [°C]

  int N = Year.size();      // Number of years
  Type eps = Type(1e-8);    // Small constant to avoid division by zero
  Type K = Type(100.0);     // Total benthic space in % cover

  // --------------------------- Parameters --------------------------
  // Initial states (free, estimated; no data leakage)
  PARAMETER(C0);           // Initial adult COTS density [ind m^-2], initial estimate or expert opinion
  PARAMETER(A0);           // Initial fast coral cover [%], initial estimate or expert opinion
  PARAMETER(S0);           // Initial slow coral cover [%], initial estimate or expert opinion

  // COTS demographic parameters
  PARAMETER(rC);           // COTS intrinsic per-capita growth/recruitment rate [yr^-1], modulated by prey & SST
  PARAMETER(mC);           // COTS baseline mortality rate [yr^-1]
  PARAMETER(m_food);       // Additional mortality when prey is scarce [yr^-1]
  PARAMETER(densC);        // Density-dependent mortality coefficient [m^2 ind^-1 yr^-1]
  PARAMETER(beta_imm);     // Conversion of larval immigration to recruits [ind m^-2 per IMM unit]
  PARAMETER(k_allee);      // Slope of mate-finding Allee effect [m^2 ind^-1]
  PARAMETER(C_crit);       // COTS density at 50% mate success [ind m^-2]
  PARAMETER(theta_prey);   // Nonlinearity of bottom-up control (>=1), unitless

  // Coral growth and predation
  PARAMETER(rA);           // Fast coral intrinsic growth rate [yr^-1]
  PARAMETER(rS);           // Slow coral intrinsic growth rate [yr^-1]
  PARAMETER(c_max);        // Max coral consumption rate per COTS [%-cover yr^-1 per ind m^-2]
  PARAMETER(Hc);           // Half-saturation constant for coral prey [%-cover]
  PARAMETER(w_fast);       // Preference weight for fast coral (0..1), unitless

  // Bleaching (SST stress) on corals
  PARAMETER(bA);           // Fast coral bleaching proportional loss (fraction per year)
  PARAMETER(bS);           // Slow coral bleaching proportional loss (fraction per year)
  PARAMETER(sst_bleach_thr); // SST threshold (°C) where bleaching accelerates (smoothly)
  PARAMETER(k_bleach);     // Steepness of bleaching logistic ramp [°C^-1]

  // SST thermal performance for COTS recruitment
  PARAMETER(sst_opt);      // Optimal SST for COTS recruitment (°C)
  PARAMETER(sst_sd);       // Width of thermal performance curve (°C), >0

  // Observation error
  PARAMETER(log_sigma_cots); // Log SD for lognormal COTS observation error
  PARAMETER(log_sigma_fast); // Log SD for fast coral (Normal) observation error
  PARAMETER(log_sigma_slow); // Log SD for slow coral (Normal) observation error

  // Numerical squashing scale for coral bounds
  PARAMETER(squash_scale); // Scale (in %-cover) for tanh squashing to [0,K], smoothness control

  // -------------------- Transforms and constants -------------------
  Type sigma_cots = exp(log_sigma_cots); // ensure >0
  Type sigma_fast = exp(log_sigma_fast); // ensure >0
  Type sigma_slow = exp(log_sigma_slow); // ensure >0

  // Minimum SD floors for numerical stability
  Type sigma_min_pos = Type(1e-3);
  Type sigma_cots_use = (sigma_cots < sigma_min_pos ? sigma_min_pos : sigma_cots);
  Type sigma_fast_use = (sigma_fast < Type(0.1) ? Type(0.1) : sigma_fast);
  Type sigma_slow_use = (sigma_slow < Type(0.1) ? Type(0.1) : sigma_slow);

  // Helper lambdas
  auto invlogit = [&](Type x) {
    // stable invlogit
    return Type(1) / (Type(1) + exp(-x));
  };
  auto softplus = [&](Type x) {
    // Numerically stable softplus
    Type ax = CppAD::abs(x);
    return log1p(exp(-ax)) + (x > Type(0) ? x : Type(0));
  };
  auto squash_to_0K = [&](Type x_prop) {
    // Smoothly clamp to [0, K] using tanh; identity near interior when squash_scale is reasonably large
    return Type(0.5) * K * (tanh((x_prop - Type(0.5) * K) / (squash_scale + eps)) + Type(1));
  };

  // ----------------------- State containers ------------------------
  vector<Type> cots_pred(N);  // predicted COTS [ind m^-2]
  vector<Type> fast_pred(N);  // predicted fast coral [%]
  vector<Type> slow_pred(N);  // predicted slow coral [%]

  // --------------------------- Initialization ----------------------
  cots_pred(0) = softplus(C0);        // nonnegative and smooth
  fast_pred(0) = squash_to_0K(A0);    // smoothly bounded [0,100]
  slow_pred(0) = squash_to_0K(S0);    // smoothly bounded [0,100]

  // ------------------------------ Process --------------------------
  for (int t = 1; t < N; ++t) {
    // Previous states
    Type C_prev = cots_pred(t-1);
    Type A_prev = fast_pred(t-1);
    Type S_prev = slow_pred(t-1);

    // Forcing at t-1
    Type IMM_prev = cotsimm_dat(t-1);
    Type SST_prev = sst_dat(t-1);

    // Prey and preference
    Type P_prev = w_fast * A_prev + (Type(1) - w_fast) * S_prev; // % cover, prey index
    Type cons_prev = c_max * C_prev * P_prev / (Hc + P_prev + eps); // Holling II consumption [%-cover yr^-1]
    // Diet allocation (avoid division by zero)
    Type pA_prev = (w_fast * A_prev) / (P_prev + eps);
    Type pS_prev = Type(1) - pA_prev;

    // Coral predation losses [%-cover]
    Type lossA = cons_prev * pA_prev;
    Type lossS = cons_prev * pS_prev;

    // Bleaching (smooth logistic ramp)
    Type bleach_idx = invlogit(k_bleach * (SST_prev - sst_bleach_thr)); // 0..1
    Type bleachA = bA * bleach_idx * A_prev; // % cover lost
    Type bleachS = bS * bleach_idx * S_prev; // % cover lost

    // Coral growth (logistic sharing space K=100)
    Type tot_prev = A_prev + S_prev;
    Type growA = rA * A_prev * (Type(1) - tot_prev / K);
    Type growS = rS * S_prev * (Type(1) - tot_prev / K);

    Type A_prop = A_prev + growA - lossA - bleachA;
    Type S_prop = S_prev + growS - lossS - bleachS;

    // Smoothly bound coral states in [0, K]
    fast_pred(t) = squash_to_0K(A_prop);
    slow_pred(t) = squash_to_0K(S_prop);

    // COTS growth modifiers
    Type temp_fac = exp(-Type(0.5) * pow((SST_prev - sst_opt) / (sst_sd + eps), 2)); // 0..1+
    Type prey_idx = P_prev / K; // 0..1
    if (prey_idx < Type(0)) prey_idx = Type(0); // should not happen due to squash, kept for safety
    if (prey_idx > Type(1)) prey_idx = Type(1);

    // Allee (mate finding)
    Type mate = invlogit(k_allee * (C_prev - C_crit)); // 0..1

    // COTS net change
    Type growthC = rC * C_prev * mate * pow(prey_idx + eps, theta_prey) * temp_fac;
    Type mortC   = mC * C_prev + m_food * C_prev * (Type(1) - prey_idx);
    Type densDep = densC * C_prev * C_prev;
    Type C_prop  = C_prev + growthC - mortC - densDep + beta_imm * IMM_prev;

    cots_pred(t) = softplus(C_prop); // keep >=0 smoothly
  }

  // ----------------------------- Likelihood ------------------------
  Type nll = 0.0;

  // COTS: lognormal likelihood on strictly positive scale
  for (int t = 0; t < N; ++t) {
    Type mu = log(cots_pred(t) + eps);
    Type y  = log((cots_dat(t) > Type(0) ? cots_dat(t) : eps));
    nll -= dnorm(y, mu, sigma_cots_use, true); // includes Jacobian implicitly via log-transform of obs+eps
  }

  // Corals: normal likelihood with floors
  for (int t = 0; t < N; ++t) {
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_fast_use, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_slow_use, true);
  }

  // ---------------------------- Reporting --------------------------
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  ADREPORT(cots_pred);
  ADREPORT(fast_pred);
  ADREPORT(slow_pred);

  return nll;
}
