#include <TMB.hpp>

// Smooth positive part to avoid hard cutoffs and preserve differentiability
template<class Type>
inline Type pospart(const Type& x) {
  return (x + CppAD::sqrt(x * x + Type(1e-8))) / Type(2.0); // smooth ReLU
}

// Smooth quadratic penalty for parameters outside [lo, hi]
template<class Type>
inline Type range_penalty(const Type& x, const Type& lo, const Type& hi, const Type& w) {
  Type below = pospart(lo - x);    // >0 when x < lo
  Type above = pospart(x - hi);    // >0 when x > hi
  return w * (below * below + above * above); // quadratic penalty outside range
}

template<class Type>
Type objective_function<Type>::operator() () {
  // ------------------------
  // DATA
  // ------------------------
  DATA_VECTOR(Year);     // time index (numeric, typically integer years)
  DATA_VECTOR(sst_dat);  // Sea Surface Temperature (°C), used as environmental driver

  // Legacy/compatibility response variables from previous codebase (not used in NPZ likelihood)
  DATA_VECTOR(slow_dat); // Legacy "slow coral" proxy; predictions provided for compatibility only
  DATA_VECTOR(fast_dat); // Legacy "fast coral" proxy; predictions provided for compatibility only
  DATA_VECTOR(cots_dat); // Legacy COTS abundance; predictions provided for compatibility only

  int T = Year.size(); // number of time steps

  // ------------------------
  // PARAMETERS
  // ------------------------
  // Phytoplankton growth and nutrient limitation
  PARAMETER(mu_max);   // Maximum phytoplankton growth rate (yr^-1)
  PARAMETER(K_N);      // Half-saturation constant for nutrient uptake (same units as N)
  // Zooplankton grazing on phytoplankton (Holling II/III blend via exponent eta)
  PARAMETER(g_max);    // Maximum grazing rate (yr^-1)
  PARAMETER(K_G);      // Half-saturation for grazing response (same units as P)
  PARAMETER(eta);      // Shape exponent for grazing response (>=1; 1: Type II, >1: Type III-like)
  // Efficiency and mortalities
  PARAMETER(e_Z);      // Assimilation efficiency of Z (0-1)
  PARAMETER(mP);       // Phytoplankton mortality/exudation rate (yr^-1)
  PARAMETER(mZ);       // Zooplankton mortality rate (yr^-1)
  PARAMETER(r_min);    // Fraction of mortality that remineralizes to dissolved nutrient (0-1)
  PARAMETER(I_N);      // External nutrient supply (e.g., mixing/upwelling; units of N per year)
  // Temperature modifier on phytoplankton growth (environmental improvement)
  PARAMETER(T_opt);    // Optimal SST for phytoplankton growth (°C)
  PARAMETER(beta_T);   // Curvature of Gaussian temperature effect (°C^-2)
  // Observation error parameters (log-space SDs) — retained for future use
  PARAMETER(sigma_N);  // Log-space SD for nutrient
  PARAMETER(sigma_P);  // Log-space SD for phytoplankton
  PARAMETER(sigma_Z);  // Log-space SD for zooplankton

  // NPZ initial conditions (parameterized to avoid requiring observation data)
  PARAMETER(N0);
  PARAMETER(P0);
  PARAMETER(Z0);

  // Legacy initial conditions (to avoid using observations in prediction equations)
  PARAMETER(initial_slow);
  PARAMETER(initial_fast);
  PARAMETER(initial_cots);

  // ------------------------
  // NEGATIVE LOG-LIKELIHOOD ACCUMULATOR AND CONSTANTS
  // ------------------------
  Type nll = 0.0;
  const Type eps = Type(1e-8);      // numerical stabilizer
  const Type sd_floor = Type(0.05); // minimum SD floor for stability
  const Type w_pen = Type(1e-3);    // weak penalties to discourage implausible parameter values

  // Smooth range penalties to keep parameters in biologically plausible ranges
  nll += range_penalty(mu_max,  Type(0.0),   Type(20.0),   w_pen);
  nll += range_penalty(K_N,     Type(1e-3),  Type(10.0),   w_pen);
  nll += range_penalty(g_max,   Type(0.0),   Type(20.0),   w_pen);
  nll += range_penalty(K_G,     Type(1e-3),  Type(10.0),   w_pen);
  nll += range_penalty(eta,     Type(1.0),   Type(3.0),    w_pen);
  nll += range_penalty(e_Z,     Type(0.0),   Type(1.0),    w_pen);
  nll += range_penalty(mP,      Type(0.0),   Type(5.0),    w_pen);
  nll += range_penalty(mZ,      Type(0.0),   Type(5.0),    w_pen);
  nll += range_penalty(r_min,   Type(0.0),   Type(1.0),    w_pen);
  nll += range_penalty(I_N,     Type(0.0),   Type(10.0),   w_pen);
  nll += range_penalty(T_opt,   Type(0.0),   Type(40.0),   w_pen);
  nll += range_penalty(beta_T,  Type(0.0),   Type(2.0),    w_pen);
  nll += range_penalty(sigma_N, Type(0.01),  Type(2.0),    w_pen);
  nll += range_penalty(sigma_P, Type(0.01),  Type(2.0),    w_pen);
  nll += range_penalty(sigma_Z, Type(0.01),  Type(2.0),    w_pen);
  // NPZ initial conditions penalties (broad and weak)
  nll += range_penalty(N0,       Type(0.0),  Type(100.0),  w_pen);
  nll += range_penalty(P0,       Type(0.0),  Type(100.0),  w_pen);
  nll += range_penalty(Z0,       Type(0.0),  Type(100.0),  w_pen);
  // Legacy initial condition penalties (broad and weak)
  nll += range_penalty(initial_slow, Type(0.0),  Type(100.0),  w_pen);  // e.g., percent cover
  nll += range_penalty(initial_fast, Type(0.0),  Type(100.0),  w_pen);  // e.g., percent cover
  nll += range_penalty(initial_cots, Type(0.0),  Type(10000.0), w_pen); // e.g., abundance

  // Effective observation SDs (floor-added smoothly) — retained for possible future likelihood use
  Type s_N = CppAD::sqrt(sigma_N * sigma_N + sd_floor * sd_floor);
  Type s_P = CppAD::sqrt(sigma_P * sigma_P + sd_floor * sd_floor);
  Type s_Z = CppAD::sqrt(sigma_Z * sigma_Z + sd_floor * sd_floor);

  // ------------------------
  // STATE PREDICTIONS
  // ------------------------
  vector<Type> N_pred(T);
  vector<Type> P_pred(T);
  vector<Type> Z_pred(T);

  // Initialize with parameterized initial states (no use of observations)
  N_pred(0) = pospart(N0);
  P_pred(0) = pospart(P0);
  Z_pred(0) = pospart(Z0);

  // Legacy prediction vectors sized to the main timeline (persistence dynamics; no use of *_dat)
  vector<Type> slow_pred(T);
  vector<Type> fast_pred(T);
  vector<Type> cots_pred(T);
  slow_pred(0) = pospart(initial_slow);
  fast_pred(0) = pospart(initial_fast);
  cots_pred(0) = pospart(initial_cots);

  // Optional diagnostics
  vector<Type> uptake_vec(T);   // nutrient uptake by phytoplankton
  vector<Type> grazing_vec(T);  // zooplankton grazing on phytoplankton
  vector<Type> remin_vec(T);    // remineralization (to N) including egestion

  uptake_vec.setZero();
  grazing_vec.setZero();
  remin_vec.setZero();

  // Time stepping using only previous-step states (no use of current observations)
  for (int t = 1; t < T; t++) {
    // Previous states (ensure non-negativity)
    Type N_prev = pospart(N_pred(t - 1));
    Type P_prev = pospart(P_pred(t - 1));
    Type Z_prev = pospart(Z_pred(t - 1));

    // Nutrient limitation and temperature modifier
    Type f_N = N_prev / (K_N + N_prev + eps);
    Type dT = sst_dat(t - 1) - T_opt;
    Type g_T = exp(-beta_T * dT * dT); // 0-1 modifier on growth

    // Phytoplankton uptake and zooplankton grazing (Holling II/III blend)
    Type Uptake = mu_max * g_T * f_N * P_prev;

    Type P_eta = pow(P_prev + eps, eta);
    Type Kg_eta = pow(K_G + eps, eta);
    Type ing_per_Z = g_max * P_eta / (Kg_eta + P_eta); // ingestion rate per Z
    Type Grazing = ing_per_Z * Z_prev;

    // Mortality and recycling/egestion to dissolved nutrient
    Type P_mort = mP * P_prev;
    Type Z_mort = mZ * Z_prev;
    Type Remin = r_min * (P_mort + Z_mort) + (Type(1.0) - e_Z) * Grazing;

    // State updates (non-negative via pospart)
    Type P_next = pospart(P_prev + Uptake - Grazing - P_mort);
    Type Z_next = pospart(Z_prev + e_Z * Grazing - Z_mort);
    Type N_next = pospart(N_prev - Uptake + Remin + I_N);

    // Store predictions and diagnostics
    P_pred(t) = P_next;
    Z_pred(t) = Z_next;
    N_pred(t) = N_next;

    uptake_vec(t) = Uptake;
    grazing_vec(t) = Grazing;
    remin_vec(t) = Remin;

    // Legacy persistence dynamics (predictions defined at every step)
    slow_pred(t) = slow_pred(t - 1);
    fast_pred(t) = fast_pred(t - 1);
    cots_pred(t) = cots_pred(t - 1);
  }

  // ------------------------
  // LIKELIHOOD
  // ------------------------
  // No NPZ likelihood terms are included here because NPZ observation data (N_dat, P_dat, Z_dat)
  // are not provided in the current dataset used by the pipeline. This avoids data-reading errors
  // and preserves a no-data-leakage setup.

  // ------------------------
  // REPORTING
  // ------------------------
  REPORT(Year);
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);
  REPORT(uptake_vec);
  REPORT(grazing_vec);
  REPORT(remin_vec);

  // Legacy outputs for compatibility with residuals tooling
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(cots_pred);

  // Provide objective function value in report for downstream tooling
  Type objective = nll;
  REPORT(objective);
  ADREPORT(objective);

  return nll;
}
