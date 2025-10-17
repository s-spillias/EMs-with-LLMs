#include <TMB.hpp>

// Robust softplus to ensure smooth positivity and stable penalties
template<class Type>
Type softplus(Type x) {
  // Numerically stable softplus: log(1 + exp(-|x|)) + max(x, 0)
  // Use only operations compatible with CppAD/TMB AD types.
  Type zero = Type(0);
  Type one  = Type(1);
  Type ax = CppAD::CondExpLt(x, zero, -x, x);                 // |x|
  return log(one + exp(-ax)) + CppAD::CondExpLt(x, zero, zero, x); // log(1+e^-|x|) + max(x,0)
}

// Smooth "box" penalty: zero-ish inside [lo, hi], increases smoothly outside
template<class Type>
Type smooth_box_penalty(Type x, Type lo, Type hi, Type sharpness) {
  // Returns small values when lo <= x <= hi; grows approximately linearly outside.
  // sharpness controls the transition steepness.
  Type below = softplus((lo - x) * sharpness) / sharpness;
  Type above = softplus((x - hi) * sharpness) / sharpness;
  return below + above;
}

// Smooth non-negative mapping for state updates to avoid hard truncation
template<class Type>
Type smooth_pos(Type x, Type eps) {
  // Maps real x to (0, inf) approximately equal to x when x >> eps, but smoothly > 0
  return eps + softplus(x - eps);
}

template<class Type>
Type objective_function<Type>::operator() () {
  // -------------------------------------------------------------------------
  // DATA (observations and time)
  // -------------------------------------------------------------------------
  // Note: The CSV column is "Time (days)". Identifiers in C++ cannot include spaces/parentheses;
  // the data interface provides it as 'Time'. We document this mapping here.
  DATA_VECTOR(Time);          // Observation time points in days; corresponds to "Time (days)" in the CSV
  DATA_VECTOR(N_dat);         // Observed nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);         // Observed phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);         // Observed zooplankton concentration (g C m^-3)

  int T = N_dat.size();       // Number of time points (must match P_dat and Z_dat)

  // -------------------------------------------------------------------------
  // PARAMETERS (unconstrained; smooth penalties impose biological ranges)
  // Each line documents: units and typical literature/data-informed ranges.
  // -------------------------------------------------------------------------
  PARAMETER(mu_max);           // day^-1 | Max phytoplankton growth rate; initial from literature/estimate
  PARAMETER(k_N);              // g C m^-3 | Half-saturation for nutrient uptake (Michaelis–Menten)
  PARAMETER(phi_colim);        // dimensionless | Smooth co-limitation curvature (>= ~0.5; higher approaches Liebig minimum)
  PARAMETER(g_max);            // day^-1 | Max zooplankton ingestion rate per biomass
  PARAMETER(k_P);              // g C m^-3 | Half-saturation for grazing (Holling type II/III base)
  PARAMETER(h_fr);             // dimensionless | Functional response shape (h=1: Type II; h>1: Type III-like)
  PARAMETER(beta_assim);       // dimensionless (0-1) | Zooplankton assimilation efficiency
  PARAMETER(m_p);              // day^-1 | Phytoplankton non-grazing loss (mortality/exudation)
  PARAMETER(m1);               // day^-1 | Zooplankton linear mortality (e.g., background predation)
  PARAMETER(m2);               // (g C m^-3)^-1 day^-1 | Zooplankton density-dependent mortality (quadratic)
  PARAMETER(r_rem);            // dimensionless (0-1) | Remineralization efficiency to dissolved nutrient
  PARAMETER(k_mix);            // day^-1 | Vertical mixing/relaxation rate to external nutrient
  PARAMETER(N_ext);            // g C m^-3 | External nutrient concentration (deep source)
  PARAMETER(env_amp);          // dimensionless (0-<1) | Amplitude of seasonal environmental modulation
  PARAMETER(env_phase);        // radians | Phase shift of seasonal modulation
  PARAMETER(env_period);       // days | Period of seasonal modulation
  PARAMETER(theta_E);          // dimensionless | Steepness of environmental logistic modifier
  PARAMETER(E50);              // dimensionless (0-1) | Midpoint (semi-saturation) of environmental logistic modifier
  PARAMETER(sigma_N);          // sd of log(errors) | Observation error (N), lognormal
  PARAMETER(sigma_P);          // sd of log(errors) | Observation error (P), lognormal
  PARAMETER(sigma_Z);          // sd of log(errors) | Observation error (Z), lognormal

  // -------------------------------------------------------------------------
  // Constants and numerical safeguards
  // -------------------------------------------------------------------------
  Type eps = Type(1e-8);                // Small constant to avoid division by zero
  Type pi  = Type(3.141592653589793238462643383279502884);
  Type obs_sd_floor = Type(0.05);       // Minimum log-space observation SD to prevent degeneracy
  Type pen_sharp = Type(10.0);          // Penalty sharpness: bigger -> steeper rise outside bounds
  Type pen_wt    = Type(1.0);           // Penalty weight added to nll (acts like weak prior)

  // -------------------------------------------------------------------------
  // Suggested biological bounds encoded as smooth penalties (not hard constraints)
  // Keep these consistent with parameters.json for transparency.
  // -------------------------------------------------------------------------
  Type nll = Type(0.0); // negative log-likelihood accumulator

  nll += pen_wt * smooth_box_penalty(mu_max,   Type(0.05),  Type(3.0),  pen_sharp);
  nll += pen_wt * smooth_box_penalty(k_N,      Type(0.005), Type(1.0),  pen_sharp);
  nll += pen_wt * smooth_box_penalty(phi_colim,Type(0.5),   Type(10.0), pen_sharp);
  nll += pen_wt * smooth_box_penalty(g_max,    Type(0.05),  Type(3.0),  pen_sharp);
  nll += pen_wt * smooth_box_penalty(k_P,      Type(0.005), Type(1.0),  pen_sharp);
  nll += pen_wt * smooth_box_penalty(h_fr,     Type(1.0),   Type(2.5),  pen_sharp);
  nll += pen_wt * smooth_box_penalty(beta_assim,Type(0.1),  Type(0.9),  pen_sharp);
  nll += pen_wt * smooth_box_penalty(m_p,      Type(0.001), Type(0.2),  pen_sharp);
  nll += pen_wt * smooth_box_penalty(m1,       Type(0.001), Type(0.5),  pen_sharp);
  nll += pen_wt * smooth_box_penalty(m2,       Type(0.0),   Type(1.0),  pen_sharp);
  nll += pen_wt * smooth_box_penalty(r_rem,    Type(0.1),   Type(1.0),  pen_sharp);
  nll += pen_wt * smooth_box_penalty(k_mix,    Type(0.0),   Type(0.5),  pen_sharp);
  nll += pen_wt * smooth_box_penalty(N_ext,    Type(0.0),   Type(3.0),  pen_sharp);
  nll += pen_wt * smooth_box_penalty(env_amp,  Type(0.0),   Type(0.95), pen_sharp);
  nll += pen_wt * smooth_box_penalty(env_phase,Type(-pi),   Type(pi),   pen_sharp);
  nll += pen_wt * smooth_box_penalty(env_period,Type(20.0), Type(400.0),pen_sharp);
  nll += pen_wt * smooth_box_penalty(theta_E,  Type(0.5),   Type(10.0), pen_sharp);
  nll += pen_wt * smooth_box_penalty(E50,      Type(0.0),   Type(1.0),  pen_sharp);
  nll += pen_wt * smooth_box_penalty(sigma_N,  Type(0.01),  Type(2.0),  pen_sharp);
  nll += pen_wt * smooth_box_penalty(sigma_P,  Type(0.01),  Type(2.0),  pen_sharp);
  nll += pen_wt * smooth_box_penalty(sigma_Z,  Type(0.01),  Type(2.0),  pen_sharp);

  // -------------------------------------------------------------------------
  // Derived observation SDs with floors for stability
  // -------------------------------------------------------------------------
  Type sdN = sqrt(sigma_N * sigma_N + obs_sd_floor * obs_sd_floor);
  Type sdP = sqrt(sigma_P * sigma_P + obs_sd_floor * obs_sd_floor);
  Type sdZ = sqrt(sigma_Z * sigma_Z + obs_sd_floor * obs_sd_floor);

  // -------------------------------------------------------------------------
  // STATE PREDICTIONS (initialize from data) to avoid optimizing initial states
  // -------------------------------------------------------------------------
  vector<Type> N_pred(T);  // Nutrient predictions
  vector<Type> P_pred(T);  // Phytoplankton predictions
  vector<Type> Z_pred(T);  // Zooplankton predictions

  N_pred(0) = N_dat(0);    // Initial condition from observed data
  P_pred(0) = P_dat(0);    // Initial condition from observed data
  Z_pred(0) = Z_dat(0);    // Initial condition from observed data

  // For diagnostics (optional)
  vector<Type> mu_t(T);    // Realized phyto growth rate (day^-1)
  vector<Type> graze_t(T); // Realized ingestion rate per Z (day^-1)
  vector<Type> env_t(T);   // Environmental driver [0,1]
  mu_t.setZero();
  graze_t.setZero();
  env_t.setZero();

  // -------------------------------------------------------------------------
  // PROCESS MODEL (Euler forward; only uses previous time-step states)
  // Equations (per time step t-1 -> t, dt = Time(t)-Time(t-1)):
  // 1) Environmental driver: E = 0.5 + 0.5 * amp * sin(2π t/period + phase)
  // 2) Nutrient limitation: f_N = N / (k_N + N)
  // 3) Environmental modifier (logistic): f_E = 1 / (1 + exp(-theta_E * (E - E50)))
  // 4) Smooth co-limitation: f_lim = [ (f_N^-phi) + (f_E^-phi) ]^(-1/phi)  (smoothly approximates min(f_N, f_E))
  // 5) Phyto growth: mu = mu_max * f_lim; Growth = mu * P
  // 6) Grazing response: g = g_max * P^h / (k_P^h + P^h)
  // 7) Ingestion: I = g * Z; Z growth = beta * I; Unassimilated = (1 - beta) * I -> N via remineralization
  // 8) Losses: P loss = m_p * P; Z loss = m1*Z + m2*Z^2; Remineralized fraction r_rem returns to N
  // 9) Mixing: N_mix = k_mix * (N_ext - N)
  // 10) Euler updates: X_next = smooth_pos(X_prev + dt * dXdt, eps)
  // -------------------------------------------------------------------------
  for (int t = 1; t < T; t++) {
    Type dt = Time(t) - Time(t-1);               // Variable time step (days)
    dt = CppAD::CondExpLe(dt, Type(0), Type(1e-6), dt); // Guard against non-positive dt

    // Previous states (no data leakage)
    Type Np = N_pred(t-1);
    Type Pp = P_pred(t-1);
    Type Zp = Z_pred(t-1);

    // (1) Environmental driver in [0,1]
    Type Eraw = sin(Type(2.0) * pi * (Time(t) / (env_period + eps)) + env_phase);
    Type E = Type(0.5) + Type(0.5) * env_amp * Eraw;   // scaled seasonal driver, centered ~0.5
    env_t(t) = E;

    // (2) Nutrient limitation (Michaelis–Menten)
    Type fN = Np / (k_N + Np + eps);

    // (3) Environmental logistic modifier
    Type fE = Type(1.0) / (Type(1.0) + exp(-theta_E * (E - E50)));

    // (4) Smooth co-limitation
    Type inv_phi = Type(1.0) / (phi_colim + eps);
    Type fN_negphi = pow(fN + eps, -phi_colim);
    Type fE_negphi = pow(fE + eps, -phi_colim);
    Type f_lim = pow(fN_negphi + fE_negphi, -inv_phi);

    // (5) Phytoplankton specific growth rate and gross primary production
    Type mu = mu_max * f_lim;       // day^-1
    mu_t(t) = mu;
    Type Gp = mu * Pp;              // g C m^-3 day^-1

    // (6) Holling functional response with shape h_fr
    Type Ph = pow(Pp + eps, h_fr);
    Type g = g_max * Ph / (pow(k_P + eps, h_fr) + Ph); // day^-1
    graze_t(t) = g;

    // (7) Ingestion and allocation
    Type I = g * Zp;                            // g C m^-3 day^-1 (ingestion proportional to Z)
    Type Z_growth = beta_assim * I;             // growth of Z
    Type Unass = (Type(1.0) - beta_assim) * I;  // unassimilated ingestion

    // (8) Losses and remineralization
    Type P_loss = m_p * Pp;                     // non-grazing P loss
    Type Z_loss = m1 * Zp + m2 * Zp * Zp;       // Z mortalities
    Type Remin = r_rem * (P_loss + Unass + Z_loss); // return to dissolved N

    // (9) Mixing source/sink on N
    Type N_mix = k_mix * (N_ext - Np);

    // (10) State derivatives
    Type dNdt = -Gp + Remin + N_mix;            // nutrient change
    Type dPdt =  Gp - I - P_loss;               // phyto change
    Type dZdt =  Z_growth - Z_loss;             // zoop change

    // Euler update with smooth positivity
    N_pred(t) = smooth_pos(Np + dt * dNdt, eps);
    P_pred(t) = smooth_pos(Pp + dt * dPdt, eps);
    Z_pred(t) = smooth_pos(Zp + dt * dZdt, eps);
  }

  // -------------------------------------------------------------------------
  // LIKELIHOOD: Lognormal errors on N, P, Z (include all observations, t=0..T-1)
  // -------------------------------------------------------------------------
  for (int t = 0; t < T; t++) {
    nll -= dnorm(log(N_dat(t) + eps), log(N_pred(t) + eps), sdN, true);
    nll -= dnorm(log(P_dat(t) + eps), log(P_pred(t) + eps), sdP, true);
    nll -= dnorm(log(Z_dat(t) + eps), log(Z_pred(t) + eps), sdZ, true);
  }

  // -------------------------------------------------------------------------
  // REPORTING
  // -------------------------------------------------------------------------
  REPORT(N_pred);  // Predicted nutrient (g C m^-3)
  REPORT(P_pred);  // Predicted phytoplankton (g C m^-3)
  REPORT(Z_pred);  // Predicted zooplankton (g C m^-3)
  REPORT(mu_t);    // Realized phyto growth rate (day^-1)
  REPORT(graze_t); // Realized grazing rate per Z (day^-1)
  REPORT(env_t);   // Environmental driver [0,1]

  return nll;
}
