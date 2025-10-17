#include <TMB.hpp>

// Helper: square
template<class Type>
inline Type sq(const Type& x){ return x*x; }

// Helper: logistic
template<class Type>
inline Type inv_logit(const Type& x){ return Type(1) / (Type(1) + exp(-x)); }

// Helper: positive part in AD-safe way
template<class Type>
inline Type pospart(const Type& x){ return CppAD::CondExpGt(x, Type(0), x, Type(0)); }

// Helper: clamp to [lo, hi] in AD-safe way
template<class Type>
inline Type clamp(const Type& x, const Type& lo, const Type& hi){
  return CppAD::CondExpLt(x, lo, lo, CppAD::CondExpGt(x, hi, hi, x));
}

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Time vector: Use the exact name from the data (first column header)
  DATA_VECTOR(Time);              // Time in days; must match the CSV "Time" column provided by the loader

  // Observations (names match the CSV column names)
  DATA_VECTOR(N_dat);             // Nutrient concentration (g C m^-3), strictly positive
  DATA_VECTOR(P_dat);             // Phytoplankton concentration (g C m^-3), strictly positive
  DATA_VECTOR(Z_dat);             // Zooplankton concentration (g C m^-3), strictly positive

  // Scalars formerly provided as DATA_SCALAR are treated as PARAMETERS for robustness to missing inputs
  PARAMETER(season_period_days);  // days | Period of seasonal forcing; soft-bounded positive to ensure stability
  PARAMETER(obs_min_sd);          // dimensionless | Minimum SD stabilizer on lognormal likelihood; soft-bounded to [0.01, 0.5]
  PARAMETER(penalty_weight_neg);  // penalty weight | Weight for quadratic penalties discouraging negative states; positive

  // -----------------------------
  // Parameters (with inline comments explaining units and selection)
  // -----------------------------
  PARAMETER(mu_max);              // d^-1 | Max phytoplankton specific growth rate; start from literature ranges 0.3–2 d^-1
  PARAMETER(K_N);                 // g C m^-3 | Half-saturation for nutrient uptake; start 0.01–0.2 g C m^-3 from literature/initial estimate
  PARAMETER(g_max);               // d^-1 | Max zooplankton grazing rate; typical 0.2–2 d^-1 from literature
  PARAMETER(K_G);                 // g C m^-3 | Half-saturation scale for grazing response; initial 0.01–0.3 g C m^-3
  PARAMETER(hill_exponent);       // dimensionless | Holling III shape (>=1); initial 1.2–2 based on observed refuges
  PARAMETER(beta);                // dimensionless | Zooplankton assimilation efficiency (0–1); baseline
  PARAMETER(beta_slope_fN);       // dimensionless | Slope for nutrient-dependent assimilation efficiency on logit scale
  PARAMETER(mP);                  // d^-1 | Phytoplankton linear mortality/exudation; initial 0.01–0.2 d^-1
  PARAMETER(mZ_quadratic);        // (g C m^-3)^-1 d^-1 | Quadratic Z mortality; initial 0.1–10 range
  PARAMETER(remin_frac);          // dimensionless | Fraction of losses immediately remineralized to N; remainder to detritus D
  PARAMETER(remin_rate_D);        // d^-1 | First-order remineralization rate of detritus D to N
  PARAMETER(sink_rate_D);         // d^-1 | First-order sinking/export rate of detritus D from mixed layer
  PARAMETER(N_deep);              // g C m^-3 | Deep/mixing source concentration; initial near observed N range
  PARAMETER(k_mix);               // d^-1 | First-order mixing/entrainment rate; initial 0.001–0.2 d^-1
  PARAMETER(env_logit_intercept); // dimensionless | Controls baseline of seasonal modifier on logit scale; tune by seasonality
  PARAMETER(env_logit_amp);       // dimensionless | Amplitude of seasonal modifier on logit scale; ~0–3
  PARAMETER(env_phase);           // radians | Phase shift of seasonal cycle; 0–2π

  // Observation model standard deviations (log-scale parameters)
  PARAMETER(log_sigma_N);         // log(SD) | Observation log-SD for N (lognormal); initialize ~log(0.1–0.3)
  PARAMETER(log_sigma_P);         // log(SD) | Observation log-SD for P (lognormal); initialize ~log(0.1–0.3)
  PARAMETER(log_sigma_Z);         // log(SD) | Observation log-SD for Z (lognormal); initialize ~log(0.1–0.3)

  // Predator interference (Beddington–DeAngelis) parameter
  PARAMETER(gamma_interference);  // (g C m^-3)^-1 | Strength of Z-Z interference in grazing denominator; >= 0

  // -----------------------------
  // Numerical constants and setup
  // -----------------------------
  Type nll = 0.0;                 // Negative log-likelihood accumulator
  const int n = N_dat.size();     // Length of time series (must match P_dat, Z_dat, Time)
  Type eps = Type(1e-8);          // Small constant for numerical stability in divisions/logs
  Type pi = Type(3.141592653589793238462643383279502884L);

  // Predicted state vectors (must align with _dat names)
  vector<Type> N_pred(n);
  vector<Type> P_pred(n);
  vector<Type> Z_pred(n);
  vector<Type> D_pred(n);         // Detritus pool (unobserved state)
  // Derived time-varying efficiency for diagnostics
  vector<Type> beta_eff_pred(n);

  // Initialize from data (avoid data leakage; we only use previous-step states thereafter)
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);
  D_pred(0) = Type(0.0); // start with no detritus in mixed layer (can build dynamically)

  // -----------------------------
  // Smooth soft bounds (penalties) for biologically plausible parameter ranges
  // AD-safe implementation using positive parts (no branching on AD types)
  // -----------------------------
  auto soft_bound_pen = [&](Type x, Type lo, Type hi){
    Type below = pospart(lo - x); // amount below lower bound
    Type above = pospart(x - hi); // amount above upper bound
    return sq(below) + sq(above); // quadratic penalty outside [lo, hi]
  };

  Type pen_par = Type(0);
  pen_par += soft_bound_pen(mu_max,         Type(0.0),    Type(2.0));
  pen_par += soft_bound_pen(K_N,            Type(0.0),    Type(1.0));
  pen_par += soft_bound_pen(g_max,          Type(0.0),    Type(3.0));
  pen_par += soft_bound_pen(K_G,            Type(0.0),    Type(1.0));
  pen_par += soft_bound_pen(hill_exponent,  Type(1.0),    Type(4.0));
  pen_par += soft_bound_pen(beta,           Type(0.0),    Type(1.0));
  pen_par += soft_bound_pen(beta_slope_fN,  Type(-6.0),   Type(6.0));
  pen_par += soft_bound_pen(mP,             Type(0.0),    Type(1.0));
  pen_par += soft_bound_pen(mZ_quadratic,   Type(0.0),    Type(50.0));
  pen_par += soft_bound_pen(remin_frac,     Type(0.0),    Type(1.0));
  pen_par += soft_bound_pen(remin_rate_D,   Type(0.0),    Type(1.0));
  pen_par += soft_bound_pen(sink_rate_D,    Type(0.0),    Type(1.0));
  pen_par += soft_bound_pen(N_deep,         Type(0.0),    Type(10.0));
  pen_par += soft_bound_pen(k_mix,          Type(0.0),    Type(1.0));
  pen_par += soft_bound_pen(env_logit_amp,  Type(0.0),    Type(5.0));
  pen_par += soft_bound_pen(env_phase,      Type(0.0),    Type(2.0)*pi);
  pen_par += soft_bound_pen(env_logit_intercept, Type(-6.0), Type(6.0));
  // Soft bounds for former data scalars now parameters
  pen_par += soft_bound_pen(season_period_days, Type(1.0),   Type(1000.0));   // ensure positive, reasonable period
  pen_par += soft_bound_pen(obs_min_sd,         Type(0.01),  Type(0.5));      // stabilize likelihood, avoid zero
  pen_par += soft_bound_pen(penalty_weight_neg, Type(10.0),  Type(100000.0)); // keep penalty weight positive and large enough
  // Soft bounds for new predator interference parameter
  pen_par += soft_bound_pen(gamma_interference, Type(0.0),   Type(10.0));

  // Add small weight to parameter penalties to softly confine optimization
  nll += pen_par * Type(1.0); // Tune weight if needed

  // -----------------------------
  // Equations (documentation)
  // (1) f_N(N) = N / (K_N + N)  [Nutrient limitation; saturating Monod]
  // (2) f_env(t) = inv_logit( env_logit_intercept + env_logit_amp * sin( 2π * t / T + env_phase ) )
  //               [Seasonal modifier in (0,1) capturing light/temperature]
  // (3) G(P,Z) = g_max * (P^h / (K_G^h + P^h)) * Z / (1 + gamma_interference * Z)  [Beddington–DeAngelis with Holling III prey term]
  // (4) dP/dt = μ_max * f_N(N) * f_env(t) * P  −  G(P,Z)  −  mP * P
  // (5) dZ/dt = β_eff(f_N) * G(P,Z)  −  mZ_quadratic * Z^2
  // (6) dD/dt = (1 − remin_frac) * [ (1 − β_eff) * G + mP * P + mZ_quadratic * Z^2 ] − remin_rate_D * D − sink_rate_D * D
  // (7) dN/dt = − (μ_max * f_N(N) * f_env(t) * P)
  //             + remin_frac * [ (1 − β_eff) * G + mP * P + mZ_quadratic * Z^2 ]
  //             + remin_rate_D * D
  //             + k_mix * (N_deep − N)
  // Integration: forward Euler with variable dt from Time, using previous-step states only.
  // -----------------------------

  // Precompute baseline logit(beta) safely
  Type beta_clamped = clamp(beta, Type(1e-6), Type(1.0 - 1e-6));
  Type beta_logit_base = log(beta_clamped) - log(Type(1.0) - beta_clamped);

  // Compute initial f_N and beta_eff for i=0 (diagnostic only)
  {
    Type fN0 = N_pred(0) / (K_N + N_pred(0) + eps);
    Type beta_logit0 = beta_logit_base + beta_slope_fN * (fN0 - Type(0.5));
    beta_eff_pred(0) = inv_logit(beta_logit0);
  }

  // Time integration loop
  for (int i = 1; i < n; i++) {
    // Previous-step states (do not use current observations to avoid leakage)
    Type N_prev = N_pred(i-1);
    Type P_prev = P_pred(i-1);
    Type Z_prev = Z_pred(i-1);
    Type D_prev = D_pred(i-1);

    // Time step
    Type dt = Time(i) - Time(i-1);
    // Guard against non-positive or extremely small dt
    dt = CppAD::CondExpLt(dt, eps, eps, dt);

    // Environmental seasonal modifier in [0,1]
    Type denom = CppAD::CondExpLt(season_period_days, eps, Type(1.0), season_period_days);
    Type angle = Type(2.0) * pi * (Time(i-1) / denom) + env_phase;
    Type f_env = inv_logit(env_logit_intercept + env_logit_amp * sin(angle)); // smooth seasonal control

    // Resource limitation and grazing functional response (add eps to denominators)
    Type f_N = N_prev / (K_N + N_prev + eps); // Monod limitation
    Type P_h = pow(P_prev + eps, hill_exponent); // smooth threshold/saturation (Holling III)

    // Predator interference term ensures reduced per-capita grazing at high Z (AD-safe)
    Type denom_graz = pow(K_G + eps, hill_exponent) + P_h + eps;
    Type interference = Type(1.0) + gamma_interference * pospart(Z_prev);
    Type G = g_max * (P_h / denom_graz) * Z_prev / (interference + eps); // Beddington–DeAngelis

    // Nutrient-dependent assimilation efficiency on logit scale
    Type beta_logit = beta_logit_base + beta_slope_fN * (f_N - Type(0.5));
    Type beta_eff = inv_logit(beta_logit);
    beta_eff_pred(i) = beta_eff;

    // Process rates
    Type growth_P = mu_max * f_N * f_env * P_prev;            // Primary production (C-specific)
    Type mort_P   = mP * P_prev;                               // Linear phytoplankton loss
    Type mort_Z   = mZ_quadratic * Z_prev * Z_prev;            // Quadratic zooplankton loss
    Type uptake_N = growth_P;                                  // Assume 1:1 C transfer N->P (carbon-equivalent units)
    Type unassimilated = (Type(1.0) - beta_eff) * G;           // Unassimilated ingestion (nutrient-dependent)

    // Partition losses between immediate remineralization and detritus pool
    Type total_losses = unassimilated + mort_P + mort_Z;
    Type to_N_immediate = remin_frac * total_losses;
    Type to_D = (Type(1.0) - remin_frac) * total_losses;

    // Detritus transformation and loss
    Type D_remin = remin_rate_D * D_prev;
    Type D_sink  = sink_rate_D * D_prev;

    // State updates (Euler forward, from previous-step states)
    Type dP = (growth_P - G - mort_P) * dt;
    Type dZ = (beta_eff * G - mort_Z) * dt;
    Type dD = (to_D - D_remin - D_sink) * dt;
    Type mixing_flux = k_mix * (N_deep - N_prev);
    Type dN = (-uptake_N + to_N_immediate + D_remin + mixing_flux) * dt;

    Type N_next = N_prev + dN;
    Type P_next = P_prev + dP;
    Type Z_next = Z_prev + dZ;
    Type D_next = D_prev + dD;

    // Smooth penalty discouraging negative states (keeps optimizer in feasible region)
    // Use quadratic penalty of negative parts scaled by penalty_weight_neg
    Type negN = CppAD::CondExpLt(N_next, Type(0), -N_next, Type(0));
    Type negP = CppAD::CondExpLt(P_next, Type(0), -P_next, Type(0));
    Type negZ = CppAD::CondExpLt(Z_next, Type(0), -Z_next, Type(0));
    Type negD = CppAD::CondExpLt(D_next, Type(0), -D_next, Type(0));
    nll += penalty_weight_neg * (sq(negN) + sq(negP) + sq(negZ) + sq(negD));

    // Assign (allowing small negative excursions by penalty rather than hard truncation)
    N_pred(i) = N_next;
    P_pred(i) = P_next;
    Z_pred(i) = Z_next;
    D_pred(i) = D_next;
  }

  // -----------------------------
  // Observation model: Lognormal errors with minimum SD
  // Use all observations (including initial), stabilized by obs_min_sd
  // -----------------------------
  Type sigma_N = sqrt( exp(Type(2.0)*log_sigma_N) + sq(obs_min_sd) );
  Type sigma_P = sqrt( exp(Type(2.0)*log_sigma_P) + sq(obs_min_sd) );
  Type sigma_Z = sqrt( exp( Type(2.0)*log_sigma_Z) + sq(obs_min_sd) );

  // Collect residuals (log scale) for diagnostics
  vector<Type> N_resid_log(n);
  vector<Type> P_resid_log(n);
  vector<Type> Z_resid_log(n);

  for (int i = 0; i < n; i++) {
    // Add eps inside logs to avoid log(0)
    Type rN = log(N_dat(i) + eps) - log(N_pred(i) + eps);
    Type rP = log(P_dat(i) + eps) - log(P_pred(i) + eps);
    Type rZ = log(Z_dat(i) + eps) - log(Z_pred(i) + eps);

    N_resid_log(i) = rN;
    P_resid_log(i) = rP;
    Z_resid_log(i) = rZ;

    nll -= dnorm(log(N_dat(i) + eps), log(N_pred(i) + eps), sigma_N, true);
    nll -= dnorm(log(P_dat(i) + eps), log(P_pred(i) + eps), sigma_P, true);
    nll -= dnorm(log(Z_dat(i) + eps), log(Z_pred(i) + eps), sigma_Z, true);
  }

  // -----------------------------
  // Report predictions (must REPORT all *_pred variables)
  // -----------------------------
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);
  REPORT(D_pred);
  REPORT(beta_eff_pred);

  // Residual diagnostics
  REPORT(N_resid_log);
  REPORT(P_resid_log);
  REPORT(Z_resid_log);

  // Optional: report some derived rate parameters for diagnostics
  REPORT(mu_max);
  REPORT(K_N);
  REPORT(g_max);
  REPORT(K_G);
  REPORT(hill_exponent);
  REPORT(beta);
  REPORT(beta_slope_fN);
  REPORT(mP);
  REPORT(mZ_quadratic);
  REPORT(remin_frac);
  REPORT(remin_rate_D);
  REPORT(sink_rate_D);
  REPORT(N_deep);
  REPORT(k_mix);
  REPORT(env_logit_intercept);
  REPORT(env_logit_amp);
  REPORT(env_phase);
  REPORT(log_sigma_N);
  REPORT(log_sigma_P);
  REPORT(log_sigma_Z);
  REPORT(season_period_days);
  REPORT(obs_min_sd);
  REPORT(penalty_weight_neg);
  REPORT(gamma_interference);

  return nll;
}
