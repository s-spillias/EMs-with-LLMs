#include <TMB.hpp>

// Helper: inverse logit
template<class Type>
Type inv_logit(Type x) {
  return Type(1) / (Type(1) + exp(-x));
}

// Helper: softplus for smooth non-negativity transformations
template<class Type>
Type softplus(Type x) {
  // Numerically stable softplus without using std::log1p (AD-safe)
  if (x > Type(20)) return x;                 // avoids overflow
  if (x < Type(-20)) return exp(x);           // underflow-safe approximation
  return log(Type(1) + exp(x));
}

// Helper: smooth maximum to avoid hard clipping: max(a, b) ~ b + (1/k)*log(1 + exp(k*(a-b)))
template<class Type>
Type smooth_max(Type a, Type b, Type k) {
  return b + (Type(1)/k) * log(Type(1) + exp(k * (a - b)));
}

// Helper: smooth penalty for suggested bounds (near-zero inside bounds; increases smoothly outside)
template<class Type>
Type bounds_penalty(Type x, Type lb, Type ub, Type weight, Type alpha) {
  // AD-safe; avoid std::log1p
  Type pl = log(Type(1) + exp(alpha * (lb - x)));
  Type pu = log(Type(1) + exp(alpha * (x - ub)));
  return weight * (pl * pl + pu * pu);
}

template<class Type>
Type objective_function<Type>::operator() () {
  // Small constants for numerical stability
  const Type eps = Type(1e-8);                  // prevents division by zero and log(0)
  const Type min_pred = Type(1e-10);            // lower floor for state predictions
  const Type k_smax = Type(20.0);               // smoothness for smooth_max
  const Type two_pi = Type(6.28318530717958647692);
  const Type sigma_min = Type(0.05);            // minimum observation SD in log space
  const Type pen_wt = Type(1e-3);               // global weight for bound penalties
  const Type pen_alpha = Type(5.0);             // steepness for soft penalties

  // Data vectors:
  // Use the exact time variable name provided by the data source: "Time"
  DATA_VECTOR(Time);         // time in days (from CSV first column)
  DATA_VECTOR(N_dat);        // observed nutrient (g C m^-3)
  DATA_VECTOR(P_dat);        // observed phytoplankton (g C m^-3)
  DATA_VECTOR(Z_dat);        // observed zooplankton (g C m^-3)

  int n = N_dat.size();

  // Parameters (declared with clear units and intended ranges)
  // Growth and uptake
  PARAMETER(log_mu_max);       // log of max phytoplankton specific uptake rate (day^-1). Start from literature/plankton culture rates.
  PARAMETER(log_K_N);          // log of half-saturation constant for nutrient uptake (g C m^-3). Start from literature on nutrient half-sat.
  PARAMETER(log_alpha_shade);  // log of self-shading coefficient (m^3 gC^-1, dimensionless in this box model). Estimated; initial estimate.

  // Grazing
  PARAMETER(log_g_max);        // log of max zooplankton ingestion rate (day^-1). Literature/prior estimates.
  PARAMETER(log_K_P);          // log of P half-saturation for grazing (g C m^-3). Literature/initial estimate.
  PARAMETER(log_h_minus1);     // log of (h-1) for generalized Holling exponent h≥1 (dimensionless). Estimated.
  PARAMETER(log_P_star);       // log of smooth grazing threshold concentration (g C m^-3). Estimated.
  PARAMETER(log_k_thr);        // log of steepness for grazing threshold ( (g C m^-3)^-1 ). Estimated.

  // Mortality
  PARAMETER(log_m_P);          // log of phytoplankton mortality rate (day^-1). Literature/initial estimate.
  PARAMETER(log_m_Z);          // log of zooplankton mortality rate (day^-1). Literature/initial estimate.

  // Efficiencies
  PARAMETER(logit_epsilon_P);  // logit of phytoplankton carbon-use efficiency in [0,1] (dimensionless). Estimated.
  PARAMETER(logit_e_Z);        // logit of zooplankton assimilation efficiency in [0,1] (dimensionless). Estimated.

  // Environmental seasonality (growth and grazing can differ)
  PARAMETER(a_season_grow);    // amplitude of seasonal modulation for growth (dimensionless, used in exp(a*sin(...))). Estimated.
  PARAMETER(phi_grow);         // phase of seasonal modulation for growth (radians). Estimated.
  PARAMETER(a_season_graze);   // amplitude of seasonal modulation for grazing (dimensionless). Estimated.
  PARAMETER(phi_graze);        // phase of seasonal modulation for grazing (radians). Estimated.
  PARAMETER(T_season);         // period of seasonal modulation (days). Estimated/anchored near 365.

  // Observation model (lognormal on each state)
  PARAMETER(log_sigma_N);      // log SD of observation error for N (log-space, dimensionless). Estimated.
  PARAMETER(log_sigma_P);      // log SD of observation error for P (log-space, dimensionless). Estimated.
  PARAMETER(log_sigma_Z);      // log SD of observation error for Z (log-space, dimensionless). Estimated.

  // Transform parameters to working scale
  Type mu_max   = exp(log_mu_max);                 // day^-1
  Type K_N      = exp(log_K_N);                    // g C m^-3
  Type alpha_sh = exp(log_alpha_shade);            // m^3 gC^-1 (proxy scale)

  Type g_max    = exp(log_g_max);                  // day^-1
  Type K_P      = exp(log_K_P);                    // g C m^-3
  Type h        = Type(1.0) + exp(log_h_minus1);   // dimensionless exponent >= 1
  Type P_star   = exp(log_P_star);                 // g C m^-3
  Type k_thr    = exp(log_k_thr);                  // (g C m^-3)^-1

  Type m_P      = exp(log_m_P);                    // day^-1
  Type m_Z      = exp(log_m_Z);                    // day^-1

  Type epsilon_P = inv_logit(logit_epsilon_P);     // in (0,1)
  Type e_Z       = inv_logit(logit_e_Z);           // in (0,1)

  // Positive period with gentle penalty if unreasonable
  Type T_seas    = T_season;

  // Observation SDs with fixed minimum floor
  Type sdN = sqrt( exp(Type(2.0) * log_sigma_N) + sigma_min * sigma_min );
  Type sdP = sqrt( exp(Type(2.0) * log_sigma_P) + sigma_min * sigma_min );
  Type sdZ = sqrt( exp(Type(2.0) * log_sigma_Z) + sigma_min * sigma_min );

  // Parameter bound penalties (biologically plausible ranges)
  Type nll = Type(0.0);
  nll += bounds_penalty(mu_max,   Type(0.05),  Type(3.0),  pen_wt, pen_alpha);
  nll += bounds_penalty(K_N,      Type(0.005), Type(3.0),  pen_wt, pen_alpha);
  nll += bounds_penalty(alpha_sh, Type(0.0),   Type(10.0), pen_wt, pen_alpha);

  nll += bounds_penalty(g_max,    Type(0.01),  Type(5.0),  pen_wt, pen_alpha);
  nll += bounds_penalty(K_P,      Type(0.005), Type(2.0),  pen_wt, pen_alpha);
  nll += bounds_penalty(h,        Type(1.0),   Type(3.0),  pen_wt, pen_alpha);
  nll += bounds_penalty(P_star,   Type(0.0),   Type(0.2),  pen_wt, pen_alpha);
  nll += bounds_penalty(k_thr,    Type(1.0),   Type(200.0),pen_wt, pen_alpha);

  nll += bounds_penalty(m_P,      Type(0.001), Type(0.5),  pen_wt, pen_alpha);
  nll += bounds_penalty(m_Z,      Type(0.001), Type(0.5),  pen_wt, pen_alpha);

  nll += bounds_penalty(epsilon_P,Type(0.2),   Type(0.9),  pen_wt, pen_alpha);
  nll += bounds_penalty(e_Z,      Type(0.2),   Type(0.8),  pen_wt, pen_alpha);

  nll += bounds_penalty(a_season_grow,  Type(-1.0), Type(1.0), pen_wt, pen_alpha);
  nll += bounds_penalty(a_season_graze, Type(-1.0), Type(1.0), pen_wt, pen_alpha);
  nll += bounds_penalty(phi_grow,       Type(0.0),  two_pi,    pen_wt, pen_alpha);
  nll += bounds_penalty(phi_graze,      Type(0.0),  two_pi,    pen_wt, pen_alpha);
  nll += bounds_penalty(T_seas,         Type(50.0), Type(500.0),pen_wt, pen_alpha);

  // Prediction vectors; initialize with observed initial conditions (no leakage beyond i=0)
  vector<Type> N_pred(n); // predicted nutrient (g C m^-3)
  vector<Type> P_pred(n); // predicted phytoplankton (g C m^-3)
  vector<Type> Z_pred(n); // predicted zooplankton (g C m^-3)

  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);

  // Process model integration (forward Euler with variable dt, using previous-step predictions only)
  for (int i = 1; i < n; i++) {
    Type dt = Time(i) - Time(i - 1);
    // Ensure positive dt with a small floor and smooth max to avoid zero/negative steps
    dt = smooth_max(dt, eps, k_smax);

    // Previous state (from predictions only)
    Type Np = N_pred(i - 1);
    Type Pp = P_pred(i - 1);
    Type Zp = Z_pred(i - 1);

    // Seasonality modifiers (always positive via exp(a*sin(...)))
    Type season_grow  = exp(a_season_grow  * sin(two_pi * (Time(i-1) / T_seas) + phi_grow));
    Type season_graze = exp(a_season_graze * sin(two_pi * (Time(i-1) / T_seas) + phi_graze));

    // (1) Resource limitation for phytoplankton
    //     - Nutrient limitation: f_N = N / (K_N + N)
    Type f_N = Np / (K_N + Np + eps);
    //     - Self-shading (light proxy): f_L = 1 / (1 + alpha_sh * P)
    Type f_L = Type(1.0) / (Type(1.0) + alpha_sh * Pp);

    // Uptake (carbon-equivalent) and production
    Type U = mu_max * season_grow * f_N * f_L * Pp;           // gross uptake (g C m^-3 d^-1)
    Type P_prod = epsilon_P * U;                               // net biomass production retained in P

    // (2) Grazing with generalized Holling and smooth threshold
    Type s_thr = Type(1.0) / (Type(1.0) + exp(-k_thr * (Pp - P_star))); // smooth onset from 0 to 1
    Type P_h = pow(Pp + eps, h);
    Type KP_h = pow(K_P + eps, h);
    Type f_g = P_h / (KP_h + P_h + eps);                      // saturation with exponent h
    Type graze_rate = g_max * season_graze * f_g * s_thr;      // ingestion per Z (d^-1)
    Type G = graze_rate * Zp;                                  // total grazing (g C m^-3 d^-1)

    // (3) Mortality
    Type M_P = m_P * Pp;                                       // P mortality (g C m^-3 d^-1)
    Type M_Z = m_Z * Zp;                                       // Z mortality (g C m^-3 d^-1)

    // (4) Mass-balanced flows (recycling to N)
    //     - Phytoplankton: uptake draws from N; respiration (1 - epsilon_P)*U returns to N
    //     - Grazing: e_Z fraction to Z, (1 - e_Z) returns to N (sloppy feeding + egestion)
    //     - Mortality: linear mortality fully remineralized to N
    Type dN = -U + (Type(1.0) - epsilon_P) * U + (Type(1.0) - e_Z) * G + M_P + M_Z;
    Type dP =  P_prod - G - M_P;
    Type dZ =  e_Z * G - M_Z;

    // Euler update with smooth non-negativity enforcement
    Type N_next_raw = Np + dt * dN;
    Type P_next_raw = Pp + dt * dP;
    Type Z_next_raw = Zp + dt * dZ;

    N_pred(i) = smooth_max(N_next_raw, min_pred, k_smax);
    P_pred(i) = smooth_max(P_next_raw, min_pred, k_smax);
    Z_pred(i) = smooth_max(Z_next_raw, min_pred, k_smax);
  }

  // Likelihood: lognormal errors on N, P, Z with SD floors; include all observations
  for (int i = 0; i < n; i++) {
    // Stabilize logs with tiny floor
    Type N_obs = N_dat(i);
    Type P_obs = P_dat(i);
    Type Z_obs = Z_dat(i);
    Type N_prd = N_pred(i);
    Type P_prd = P_pred(i);
    Type Z_prd = Z_pred(i);

    // Lognormal negative log-likelihood contributions
    nll -= dnorm(log(N_obs + eps), log(N_prd + eps), sdN, true);
    nll -= dnorm(log(P_obs + eps), log(P_prd + eps), sdP, true);
    nll -= dnorm(log(Z_obs + eps), log(Z_prd + eps), sdZ, true);
  }

  // Reporting: predictions for all observed series
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);

  // Optional reporting of transformed parameters (useful for diagnostics)
  REPORT(mu_max);
  REPORT(K_N);
  REPORT(alpha_sh);
  REPORT(g_max);
  REPORT(K_P);
  REPORT(h);
  REPORT(P_star);
  REPORT(k_thr);
  REPORT(m_P);
  REPORT(m_Z);
  REPORT(epsilon_P);
  REPORT(e_Z);
  REPORT(a_season_grow);
  REPORT(phi_grow);
  REPORT(a_season_graze);
  REPORT(phi_graze);
  REPORT(T_seas);
  REPORT(sdN);
  REPORT(sdP);
  REPORT(sdZ);

  return nll;
}

/*
Equation summary (all rates in g C m^-3 d^-1 unless noted)
1) f_N = N / (K_N + N), nutrient limitation (saturating Monod).
2) f_L = 1 / (1 + alpha_sh * P), self-shading (light) limitation (saturating with P).
3) season_grow = exp(a_season_grow * sin(2π t / T_season + phi_grow)), growth modifier.
4) U = mu_max * season_grow * f_N * f_L * P, gross uptake of N by phytoplankton.
5) P_prod = epsilon_P * U, biomass retained by phytoplankton after respiration.
6) s_thr = 1 / (1 + exp(-k_thr * (P - P_star))), smooth grazing threshold.
7) f_g = P^h / (K_P^h + P^h), generalized Holling saturation with exponent h ≥ 1.
8) season_graze = exp(a_season_graze * sin(2π t / T_season + phi_graze)), grazing modifier.
9) graze_rate = g_max * season_graze * f_g * s_thr (d^-1 per unit Z biomass).
10) G = graze_rate * Z, total grazing flux from P to Z+N.
11) M_P = m_P * P, M_Z = m_Z * Z, linear mortalities.
12) dN = -U + (1 - epsilon_P)*U + (1 - e_Z)*G + M_P + M_Z, recycling to nutrients.
13) dP = P_prod - G - M_P, phytoplankton biomass balance.
14) dZ = e_Z * G - M_Z, zooplankton biomass balance.
15) Discrete update: X(t+dt) = smooth_max( X(t) + dt * dX, min_pred ), applied to N, P, Z.
16) Observation model: log Y ~ Normal( log X_pred, sd ), with sd = sqrt(exp(2*log_sigma) + sigma_min^2).
*/
