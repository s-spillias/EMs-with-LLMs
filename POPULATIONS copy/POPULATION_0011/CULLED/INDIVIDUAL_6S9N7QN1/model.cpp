#include <TMB.hpp>

// Utility functions for stability and smoothness
template<class Type>
Type inv_logit(Type x) { // Smooth map R -> (0,1)
  return Type(1) / (Type(1) + exp(-x));
}

template<class Type>
Type softplus(Type x) { // Smooth nonlinearity to keep variables positive (AD-safe)
  // Use AD-safe log(1 + exp(x)) without std::log1p to avoid double-only overloads
  return log(Type(1) + exp(x));
}

template<class Type>
Type safe_div(Type num, Type den, Type eps) { // Prevent division by zero
  return num / (den + eps);
}

// Smooth hinge penalties: 0 when inside bound; increases smoothly outside
template<class Type>
Type hinge_upper(Type x, Type upper) {
  return softplus(x - upper); // ~0 if x<=upper, smooth increase if x>upper
}

template<class Type>
Type hinge_lower(Type x, Type lower) {
  return softplus(lower - x); // ~0 if x>=lower, smooth increase if x<lower
}

template<class Type>
Type enforce_min_sd(Type sd_raw, Type min_sd) {
  // Smoothly ensures sd >= min_sd
  return min_sd + softplus(sd_raw - min_sd);
}

// Model
template<class Type>
Type objective_function<Type>::operator() () {
  using CppAD::pow;

  // -----------------------------
  // DATA: names mirror the CSV headers for N,P,Z; time vector corresponds to "Time"
  // -----------------------------
  DATA_VECTOR(Time);                    // Time in days; corresponds to "Time" in the input
  DATA_VECTOR(N_dat);                   // Nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);                   // Phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);                   // Zooplankton concentration (g C m^-3)

  int n = N_dat.size();                // Number of time steps
  Type eps = Type(1e-8);               // Small constant for numerical stability
  Type pi = Type(3.1415926535897932384626433832795);
  Type omega = Type(2.0) * pi / Type(365.0); // Seasonal frequency (per day)

  // -----------------------------
  // PARAMETERS (with comments on units and rationale)
  // -----------------------------
  PARAMETER(log_mu_max);   // ln(d^-1): Maximum specific uptake rate of P; literature: 0.2–2 d^-1, modulated by environment
  PARAMETER(log_K_N);      // ln(g C m^-3): Half-saturation constant for nutrient uptake (Monod)
  PARAMETER(log_a_shade);  // ln((g C m^-3)^-1): Self-shading/crowding coefficient reducing growth as P increases
  PARAMETER(logit_y_P);    // logit(unitless): Growth efficiency converting uptake into P biomass (0–1)
  PARAMETER(log_g_max);    // ln(d^-1): Max zooplankton ingestion rate
  PARAMETER(log_h_Z);      // ln(g C m^-3): Grazing half-saturation/scale for Holling-III
  PARAMETER(logit_nu01);   // logit(unitless): maps -> (0,1), then nu = 1 + 2*nu01 gives Holling-III exponent in [1,3]
  PARAMETER(log_P_thresh); // ln(g C m^-3): Prey refuge/threshold for grazing onset (smooth)
  PARAMETER(log_kappa);    // ln((g C m^-3)^-1): Steepness of smooth prey threshold (higher = sharper)
  PARAMETER(logit_e_Z);    // logit(unitless): Zooplankton assimilation efficiency (0–1)
  PARAMETER(log_m_P);      // ln(d^-1): Phytoplankton linear mortality rate
  PARAMETER(log_m_Z);      // ln(d^-1): Zooplankton linear mortality rate
  PARAMETER(log_m_Zq);     // ln((g C m^-3)^-1 d^-1): Zooplankton quadratic mortality rate
  PARAMETER(log_k_mix);    // ln(d^-1): Vertical mixing/relaxation rate toward external nutrient
  PARAMETER(log_N_star);   // ln(g C m^-3): External/deep nutrient concentration target
  PARAMETER(logit_r_e);    // logit(unitless): Fraction of unassimilated ingestion routed to N (0–1)
  PARAMETER(logit_r_mp);   // logit(unitless): Fraction of P mortality remineralized to N (0–1)
  PARAMETER(logit_r_mz);   // logit(unitless): Fraction of Z mortality remineralized to N (0–1)
  PARAMETER(logit_r_rp);   // logit(unitless): Fraction of P growth inefficiency routed to N (0–1)
  PARAMETER(b0_env);       // unitless: Intercept for seasonal environmental modifier (mapped through logistic to 0–1)
  PARAMETER(b1_env);       // unitless: Cosine coefficient for seasonality of environment
  PARAMETER(b2_env);       // unitless: Sine coefficient for seasonality of environment
  PARAMETER(log_sigma_N);  // ln: Observation SD (log-space) for N
  PARAMETER(log_sigma_P);  // ln: Observation SD (log-space) for P
  PARAMETER(log_sigma_Z);  // ln: Observation SD (log-space) for Z

  // Transformations to natural scales
  Type mu_max   = exp(log_mu_max);                  // d^-1
  Type K_N      = exp(log_K_N);                     // g C m^-3
  Type a_shade  = exp(log_a_shade);                 // (g C m^-3)^-1
  Type y_P      = inv_logit(logit_y_P);             // 0–1
  Type g_max    = exp(log_g_max);                   // d^-1
  Type h_Z      = exp(log_h_Z);                     // g C m^-3
  Type nu       = Type(1.0) + Type(2.0) * inv_logit(logit_nu01); // 1–3
  Type P_thresh = exp(log_P_thresh);                // g C m^-3
  Type kappa    = exp(log_kappa);                   // (g C m^-3)^-1
  Type e_Z      = inv_logit(logit_e_Z);             // 0–1
  Type m_P      = exp(log_m_P);                     // d^-1
  Type m_Z      = exp(log_m_Z);                     // d^-1
  Type m_Zq     = exp(log_m_Zq);                    // (g C m^-3)^-1 d^-1
  Type k_mix    = exp(log_k_mix);                   // d^-1
  Type N_star   = exp(log_N_star);                  // g C m^-3
  Type r_e      = inv_logit(logit_r_e);             // 0–1
  Type r_mp     = inv_logit(logit_r_mp);            // 0–1
  Type r_mz     = inv_logit(logit_r_mz);            // 0–1
  Type r_rp     = inv_logit(logit_r_rp);            // 0–1

  // Observation SDs with minimum SD
  Type min_sd = Type(0.05);                         // minimum SD in log-space to avoid overconfidence
  Type sdN = enforce_min_sd(exp(log_sigma_N), min_sd);
  Type sdP = enforce_min_sd(exp(log_sigma_P), min_sd);
  Type sdZ = enforce_min_sd(exp(log_sigma_Z), min_sd);

  // -----------------------------
  // STATE PREDICTIONS (initialize from first observation to avoid data leakage)
  // -----------------------------
  vector<Type> N_pred(n); // Nutrient predictions (g C m^-3)
  vector<Type> P_pred(n); // Phytoplankton predictions (g C m^-3)
  vector<Type> Z_pred(n); // Zooplankton predictions (g C m^-3)

  N_pred(0) = N_dat(0); // initialize with observed initial condition
  P_pred(0) = P_dat(0); // initialize with observed initial condition
  Z_pred(0) = Z_dat(0); // initialize with observed initial condition

  // -----------------------------
  // NEGATIVE LOG-LIKELIHOOD
  // -----------------------------
  Type nll = 0.0;

  // -----------------------------
  // DYNAMICS
  // Numbered equation summary:
  // (1) M_env(t) = inv_logit(b0 + b1 cos(ωt) + b2 sin(ωt))      [environmental modifier 0–1]
  // (2) f_N = N / (K_N + N)                                      [Monod nutrient limitation]
  // (3) f_shade = 1 / (1 + a_shade P)                            [self-shading/crowding]
  // (4) U = μ_max * M_env * f_N * f_shade * P                    [nutrient uptake rate]
  // (5) P_eff = P * inv_logit(κ (P - P_thresh))                  [smooth prey threshold]
  // (6) f_graz = P_eff^ν / (h_Z^ν + P_eff^ν)                     [Holling-III preference]
  // (7) I = g_max * M_env * f_graz * Z                           [ingestion rate]
  // (8) Mort_P = m_P P;   Mort_Z = m_Z Z + m_Zq Z^2              [mortality terms]
  // (9) dN = -U + r_rp(1 - y_P)U + r_e(1 - e_Z)I + r_mp Mort_P + r_mz Mort_Z + k_mix(N* - N)
  // (10) dP = y_P U - I - Mort_P
  // (11) dZ = e_Z I - Mort_Z
  // (12) Forward Euler: X_{t+1} = X_t + dt * dX, then smoothed positivity via softplus
  // -----------------------------
  for (int i = 1; i < n; i++) {
    // Previous step states (predictions only; no data leakage)
    Type N_prev = N_pred(i - 1);
    Type P_prev = P_pred(i - 1);
    Type Z_prev = Z_pred(i - 1);

    // Time step (smoothly enforce dt >= eps)
    Type raw_dt = Time(i) - Time(i - 1);            // days
    Type dt = eps + softplus(raw_dt - eps);         // ensure positive in a smooth way

    // (1) Environmental modifier (0–1)
    Type env = inv_logit(b0_env + b1_env * cos(omega * Time(i - 1))
                                   + b2_env * sin(omega * Time(i - 1)));

    // (2) Nutrient limitation (Monod)
    Type fN = safe_div(N_prev, (K_N + N_prev), eps);

    // (3) Self-shading limitation
    Type f_shade = safe_div(Type(1.0), (Type(1.0) + a_shade * P_prev), eps);

    // (4) Nutrient uptake (per volume)
    Type U = mu_max * env * fN * f_shade * P_prev; // g C m^-3 d^-1

    // (5) Smooth prey threshold/refuge for grazing
    Type thresh = inv_logit(kappa * (P_prev - P_thresh)); // 0–1
    Type P_eff = P_prev * thresh;                         // effective prey

    // (6) Holling type-III functional response
    Type Pnu = pow(P_eff + eps, nu);
    Type hnu = pow(h_Z + eps, nu);
    Type f_graz = safe_div(Pnu, (hnu + Pnu), eps);

    // (7) Zooplankton ingestion
    Type I = g_max * env * f_graz * Z_prev; // g C m^-3 d^-1

    // (8) Mortality terms
    Type Mort_P = m_P * P_prev;                          // g C m^-3 d^-1
    Type Mort_Z = m_Z * Z_prev + m_Zq * Z_prev * Z_prev; // g C m^-3 d^-1

    // Recycling and flows
    Type remin_growth_ineff = r_rp * (Type(1.0) - y_P) * U;     // P growth inefficiency to N
    Type remin_egestion     = r_e  * (Type(1.0) - e_Z) * I;     // unassimilated ingestion to N
    Type remin_P_mort       = r_mp * Mort_P;                    // P mortality to N
    Type remin_Z_mort       = r_mz * Mort_Z;                    // Z mortality to N
    Type mixing_flux        = k_mix * (N_star - N_prev);        // physical supply/sink

    // (9-11) Rates of change
    Type dN = -U + remin_growth_ineff + remin_egestion + remin_P_mort + remin_Z_mort + mixing_flux;
    Type dP =  y_P * U - I - Mort_P;
    Type dZ =  e_Z * I - Mort_Z;

    // (12) Forward Euler updates with smooth positivity
    Type N_new = N_prev + dt * dN;
    Type P_new = P_prev + dt * dP;
    Type Z_new = Z_prev + dt * dZ;

    // Smooth floor at ~eps using softplus: x_pos = eps + softplus(x - eps)
    N_pred(i) = eps + softplus(N_new - eps);
    P_pred(i) = eps + softplus(P_new - eps);
    Z_pred(i) = eps + softplus(Z_new - eps);
  }

  // -----------------------------
  // OBSERVATION LIKELIHOOD (lognormal; include all observations)
  // -----------------------------
  for (int i = 0; i < n; i++) {
    // Use log of strictly positive variables with small epsilon to avoid log(0)
    nll -= dnorm(log(N_dat(i) + eps), log(N_pred(i) + eps), sdN, true);
    nll -= dnorm(log(P_dat(i) + eps), log(P_pred(i) + eps), sdP, true);
    nll -= dnorm(log(Z_dat(i) + eps), log(Z_pred(i) + eps), sdZ, true);
  }

  // -----------------------------
  // SMOOTH BIOLOGICAL BOUND PENALTIES (discourage extreme/unrealistic values)
  // -----------------------------
  Type lambda_bound = Type(1.0); // weight of penalties (tunable)

  // Suggested biological ranges (see parameters.json for documentation)
  nll += lambda_bound * hinge_lower(mu_max,  Type(0.05));  nll += lambda_bound * hinge_upper(mu_max,  Type(2.0));
  nll += lambda_bound * hinge_lower(K_N,     Type(1e-4));  nll += lambda_bound * hinge_upper(K_N,     Type(2.0));
  nll += lambda_bound * hinge_lower(a_shade, Type(1e-4));  nll += lambda_bound * hinge_upper(a_shade, Type(10.0));
  nll += lambda_bound * hinge_lower(y_P,     Type(0.1));   nll += lambda_bound * hinge_upper(y_P,     Type(0.95));
  nll += lambda_bound * hinge_lower(g_max,   Type(0.05));  nll += lambda_bound * hinge_upper(g_max,   Type(3.0));
  nll += lambda_bound * hinge_lower(h_Z,     Type(1e-4));  nll += lambda_bound * hinge_upper(h_Z,     Type(2.0));
  nll += lambda_bound * hinge_lower(nu,      Type(1.0));   nll += lambda_bound * hinge_upper(nu,      Type(3.0));
  nll += lambda_bound * hinge_lower(P_thresh,Type(0.0));   nll += lambda_bound * hinge_upper(P_thresh,Type(0.3));
  nll += lambda_bound * hinge_lower(kappa,   Type(1.0));   nll += lambda_bound * hinge_upper(kappa,   Type(300.0));
  nll += lambda_bound * hinge_lower(e_Z,     Type(0.3));   nll += lambda_bound * hinge_upper(e_Z,     Type(0.95));
  nll += lambda_bound * hinge_lower(m_P,     Type(0.0));   nll += lambda_bound * hinge_upper(m_P,     Type(0.6));
  nll += lambda_bound * hinge_lower(m_Z,     Type(0.0));   nll += lambda_bound * hinge_upper(m_Z,     Type(0.6));
  nll += lambda_bound * hinge_lower(m_Zq,    Type(0.0));   nll += lambda_bound * hinge_upper(m_Zq,    Type(5.0));
  nll += lambda_bound * hinge_lower(k_mix,   Type(0.0));   nll += lambda_bound * hinge_upper(k_mix,   Type(1.0));
  nll += lambda_bound * hinge_lower(N_star,  Type(0.0));   nll += lambda_bound * hinge_upper(N_star,  Type(5.0));
  nll += lambda_bound * hinge_lower(r_e,     Type(0.0));   nll += lambda_bound * hinge_upper(r_e,     Type(1.0));
  nll += lambda_bound * hinge_lower(r_mp,    Type(0.0));   nll += lambda_bound * hinge_upper(r_mp,    Type(1.0));
  nll += lambda_bound * hinge_lower(r_mz,    Type(0.0));   nll += lambda_bound * hinge_upper(r_mz,    Type(1.0));
  nll += lambda_bound * hinge_lower(r_rp,    Type(0.0));   nll += lambda_bound * hinge_upper(r_rp,    Type(1.0));
  nll += lambda_bound * hinge_lower(b0_env,  Type(-5.0));  nll += lambda_bound * hinge_upper(b0_env,  Type(5.0));
  nll += lambda_bound * hinge_lower(b1_env,  Type(-3.0));  nll += lambda_bound * hinge_upper(b1_env,  Type(3.0));
  nll += lambda_bound * hinge_lower(b2_env,  Type(-3.0));  nll += lambda_bound * hinge_upper(b2_env,  Type(3.0));
  // Observation sd log-scale loose bounds
  nll += lambda_bound * hinge_lower(log_sigma_N, Type(-10.0)); nll += lambda_bound * hinge_upper(log_sigma_N, Type(2.0));
  nll += lambda_bound * hinge_lower(log_sigma_P, Type(-10.0)); nll += lambda_bound * hinge_upper(log_sigma_P, Type(2.0));
  nll += lambda_bound * hinge_lower(log_sigma_Z, Type(-10.0)); nll += lambda_bound * hinge_upper(log_sigma_Z, Type(2.0));

  // -----------------------------
  // REPORTING
  // -----------------------------
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);

  return nll;
}
