#include <TMB.hpp>

// Softplus for smooth nonlinearity and penalties (AD-safe with overflow guard)
template<class Type>
Type softplus(Type x) {
  Type threshold = Type(20);                       // threshold to avoid overflow
  return CppAD::CondExpGt(x, threshold,            // if x > threshold:
                          x,                       //   softplus ~ x
                          log(Type(1) + exp(x)));  // else: log(1 + exp(x))
}

// Smooth penalty for staying within [lower, upper] without hard constraints
template<class Type>
Type bound_penalty(Type x, Type lower, Type upper, Type scale) {
  Type pen = Type(0);                                         // accumulate penalty
  pen += pow(softplus(lower - x) / scale, 2);                 // below lower bound
  pen += pow(softplus(x - upper) / scale, 2);                 // above upper bound
  return pen;                                                 // larger when far outside bounds, ~0 inside
}

// Smooth positivity enforcement to replace posfun (keeps value > eps and penalizes violations)
template<class Type>
Type keep_positive(Type x, Type eps, Type &pen) {
  // Smoothly lower-bound x at eps using softplus; always > eps
  Type y = eps + softplus(x - eps);
  // Add a soft penalty when x < eps; ~0 penalty when x >= eps
  Type shortfall = softplus(eps - x);
  pen += pow(shortfall, 2);
  return y;
}

/*
Equation summary (all rates per day unless noted):

State variables: N (nutrient; g C m^-3), P (phytoplankton; g C m^-3), Z (zooplankton; g C m^-3)

Let:
1) f_N = N / (K_N + N)  [Monod nutrient limitation; saturating and smooth]
2) f_L(t) = invlogit(beta_L0 + beta_env_sin*sin(2*pi*t/T_season) + beta_env_cos*cos(2*pi*t/T_season))
   [Smooth seasonal light proxy; 0..1; modifies growth and grazing]
3) mu_eff = mu_max * f_N * f_L  [Effective phytoplankton specific growth rate]
4) q = 1 + nu  [Switching exponent; nu>=0; q in [1, inf); Type II at q≈1; Type III at q≈2]
5) g_fun = g_max * (0.5 + 0.5*f_L) * P^q / (K_G^q + P^q)
   [Grazing functional response; smoothly modified by season via f_L]
6) Uptake (U) = mu_eff * P  [N consumed; P-specific uptake]
7) Grazing loss of P (G) = g_fun * Z  [Per-capita grazing times zooplankton biomass]
8) P growth = e_P * U  [Photosynthate allocation efficiency to P biomass]
9) Z growth = e_Z * G  [Assimilation efficiency of grazed P to Z]
10) Remineralization/recycling to N:
    R = phi_rec * ((1 - e_P)*U + m_P*P + (1 - e_Z)*G + m_Z*Z)
    [Fraction of all non-growth losses returned to N; remainder exported]
11) Physical nutrient supply (mixing): M = mix_rate * (N_ext - N)
12) ODEs (Euler step with dt):
    dN/dt = -U + R + M
    dP/dt =  e_P*U - G - m_P*P
    dZ/dt =  e_Z*G - m_Z*Z

Observation model (for each i):
13) y_var(i) ~ LogNormal(log(var_pred(i)), sigma_var), with min sigma floor
    Implemented as: -loglik = -dnorm(log(y), log(var_pred), sigma, true) + log(y)
*/

template<class Type>
Type objective_function<Type>::operator() () {
  using CppAD::pow;

  // Small constants for numerical stability
  const Type tiny = Type(1e-8);          // prevents division by zero
  const Type min_sigma = Type(0.05);     // minimum observation SD on log-scale

  // ---------------------------
  // Data inputs (TMB conventions)
  // ---------------------------
  DATA_VECTOR(Time);   // Time (days); from CSV column "Time (days)" mapped to a code-safe name
  DATA_VECTOR(N_dat);  // Observed nutrient (g C m^-3)
  DATA_VECTOR(P_dat);  // Observed phytoplankton (g C m^-3)
  DATA_VECTOR(Z_dat);  // Observed zooplankton (g C m^-3)

  // ---------------------------
  // Parameters (estimation targets)
  // ---------------------------
  PARAMETER(log_mu_max);    // ln of max phyto specific growth rate, mu_max (d^-1); literature 0.05–2.5 d^-1
  PARAMETER(log_K_N);       // ln of half-saturation for nutrient limitation, K_N (g C m^-3); literature 0.001–1
  PARAMETER(log_g_max);     // ln of max zooplankton grazing rate, g_max (d^-1); literature 0.05–3 d^-1
  PARAMETER(log_K_G);       // ln of half-saturation for grazing response, K_G (g C m^-3); literature 0.001–1.5
  PARAMETER(log_nu);        // ln of switching parameter nu (dimensionless, >=0); nu = exp(log_nu)
  PARAMETER(logit_e_Z);     // logit of zooplankton assimilation efficiency e_Z (0–1); literature 0.3–0.9
  PARAMETER(logit_e_P);     // logit of phytoplankton growth allocation efficiency e_P (0–1); literature 0.5–0.99
  PARAMETER(log_m_P);       // ln of phytoplankton mortality rate m_P (d^-1); literature 0.001–0.5
  PARAMETER(log_m_Z);       // ln of zooplankton mortality rate m_Z (d^-1); literature 0.001–0.5
  PARAMETER(logit_phi_rec); // logit of recycling fraction phi_rec (0–1); literature 0.2–0.95
  PARAMETER(log_mix_rate);  // ln of nutrient mixing exchange rate (d^-1); suggested 1e-5–0.2
  PARAMETER(log_N_ext);     // ln of external nutrient concentration N_ext (g C m^-3); suggested 0.01–1.0
  PARAMETER(beta_env_sin);  // seasonal sine coefficient for light proxy (dimensionless); initial estimate
  PARAMETER(beta_env_cos);  // seasonal cosine coefficient for light proxy (dimensionless); initial estimate
  PARAMETER(beta_L0);       // seasonal light proxy intercept (dimensionless); sets baseline f_L (~invlogit(beta_L0))
  PARAMETER(log_sigma_N);   // ln observation SD for N on log scale (dimensionless); min 0.05
  PARAMETER(log_sigma_P);   // ln observation SD for P on log scale (dimensionless); min 0.05
  PARAMETER(log_sigma_Z);   // ln observation SD for Z on log scale (dimensionless); min 0.05

  // ---------------------------
  // Transform parameters to natural scales
  // ---------------------------
  Type mu_max   = exp(log_mu_max);        // d^-1; max phyto growth rate
  Type K_N      = exp(log_K_N);           // g C m^-3; nutrient half-sat
  Type g_max    = exp(log_g_max);         // d^-1; max grazing rate
  Type K_G      = exp(log_K_G);           // g C m^-3; grazing half-sat
  Type nu       = exp(log_nu);            // dimensionless >= 0; switching
  Type q        = Type(1.0) + nu;         // functional response exponent (>=1)
  Type e_Z      = invlogit(logit_e_Z);    // 0..1; assimilation efficiency (TMB's invlogit)
  Type e_P      = invlogit(logit_e_P);    // 0..1; P growth allocation efficiency
  Type m_P      = exp(log_m_P);           // d^-1; P mortality
  Type m_Z      = exp(log_m_Z);           // d^-1; Z mortality
  Type phi_rec  = invlogit(logit_phi_rec);// 0..1; recycling fraction
  Type mix_rate = exp(log_mix_rate);      // d^-1; mixing exchange
  Type N_ext    = exp(log_N_ext);         // g C m^-3; external nutrient pool
  // Observation SDs (log-scale) with floor for numerical stability
  Type sigma_N  = exp(log_sigma_N) + min_sigma; // >= min_sigma
  Type sigma_P  = exp(log_sigma_P) + min_sigma; // >= min_sigma
  Type sigma_Z  = exp(log_sigma_Z) + min_sigma; // >= min_sigma

  // ---------------------------
  // Smooth parameter bounds penalties (no hard constraints)
  // ---------------------------
  Type nll = Type(0.0);                   // negative log-likelihood accumulator
  const Type w_bounds = Type(10.0);       // weight for boundary penalties
  nll += w_bounds * bound_penalty(mu_max,  Type(0.05),  Type(2.5),  Type(0.1));   // mu_max bounds
  nll += w_bounds * bound_penalty(K_N,     Type(0.001), Type(1.0),  Type(0.05));  // K_N bounds
  nll += w_bounds * bound_penalty(g_max,   Type(0.05),  Type(3.0),  Type(0.1));   // g_max bounds
  nll += w_bounds * bound_penalty(K_G,     Type(0.001), Type(1.5),  Type(0.05));  // K_G bounds
  nll += w_bounds * bound_penalty(nu,      Type(0.0),   Type(4.0),  Type(0.25));  // nu bounds
  nll += w_bounds * bound_penalty(e_Z,     Type(0.3),   Type(0.9),  Type(0.05));  // e_Z bounds
  nll += w_bounds * bound_penalty(e_P,     Type(0.5),   Type(0.99), Type(0.05));  // e_P bounds
  nll += w_bounds * bound_penalty(m_P,     Type(0.001), Type(0.5),  Type(0.02));  // m_P bounds
  nll += w_bounds * bound_penalty(m_Z,     Type(0.001), Type(0.5),  Type(0.02));  // m_Z bounds
  nll += w_bounds * bound_penalty(phi_rec, Type(0.2),   Type(0.95), Type(0.05));  // phi_rec bounds
  nll += w_bounds * bound_penalty(mix_rate,Type(1e-5),  Type(0.2),  Type(0.01));  // mixing bounds
  nll += w_bounds * bound_penalty(N_ext,   Type(0.01),  Type(1.0),  Type(0.05));  // external N bounds
  nll += w_bounds * bound_penalty(beta_env_sin, Type(-2.0), Type(2.0), Type(0.25)); // seasonal coeff bounds
  nll += w_bounds * bound_penalty(beta_env_cos, Type(-2.0), Type(2.0), Type(0.25)); // seasonal coeff bounds
  nll += w_bounds * bound_penalty(beta_L0,     Type(-4.0), Type(4.0), Type(0.5));   // light intercept bounds
  nll += w_bounds * bound_penalty(exp(log_sigma_N), Type(0.05), Type(1.0), Type(0.05)); // obs SD bounds (natural)
  nll += w_bounds * bound_penalty(exp(log_sigma_P), Type(0.05), Type(1.0), Type(0.05));
  nll += w_bounds * bound_penalty(exp(log_sigma_Z), Type(0.05), Type(1.0), Type(0.05));

  // ---------------------------
  // Predictions (state trajectories)
  // ---------------------------
  int nT = Time.size();                   // number of time points
  vector<Type> N_pred(nT);                // predicted nutrient trajectory (g C m^-3)
  vector<Type> P_pred(nT);                // predicted phyto trajectory (g C m^-3)
  vector<Type> Z_pred(nT);                // predicted zoo trajectory (g C m^-3)

  // Initialize predictions with observed initial conditions (no optimization on IC)
  N_pred(0) = N_dat(0);                   // initial N from data
  P_pred(0) = P_dat(0);                   // initial P from data
  Z_pred(0) = Z_dat(0);                   // initial Z from data

  // Keep track of diagnostic flows (optional reporting)
  vector<Type> U_flow(nT);                // nutrient uptake by P (g C m^-3 d^-1)
  vector<Type> G_flow(nT);                // grazing flow P->Z (g C m^-3 d^-1)
  vector<Type> R_flow(nT);                // recycling flow to N (g C m^-3 d^-1)
  vector<Type> M_flow(nT);                // mixing-driven N flux (g C m^-3 d^-1)
  vector<Type> fL_series(nT);             // seasonal light proxy (0..1)

  // Precompute constants for seasonal forcing
  const Type two_pi = Type(6.28318530717958647692); // 2*pi
  const Type T_season = Type(365.0);                // days; seasonal period
  fL_series(0) = invlogit(beta_L0 + beta_env_sin * sin(two_pi * Time(0) / T_season)
                                   + beta_env_cos * cos(two_pi * Time(0) / T_season));
  U_flow(0) = Type(0); G_flow(0) = Type(0); R_flow(0) = Type(0); M_flow(0) = Type(0); // no flows at t0

  // Time stepping with Euler integration; use only previous step predictions (no data leakage)
  Type pen_pos = Type(0.0);               // accumulate positivity penalties
  for (int i = 1; i < nT; i++) {
    // Time step
    Type dt = Time(i) - Time(i - 1);                          // step size (days)
    dt = CppAD::CondExpGt(dt, tiny, dt, tiny);                 // enforce minimum positive dt

    // Previous states
    Type N = N_pred(i - 1);                                    // nutrient at t-1
    Type P = P_pred(i - 1);                                    // phyto at t-1
    Type Z = Z_pred(i - 1);                                    // zoo at t-1

    // Seasonal light proxy at mid-step (smooth environmental modifier)
    Type tmid = (Time(i) + Time(i - 1)) / Type(2.0);           // midpoint time
    Type f_L = invlogit(beta_L0 + beta_env_sin * sin(two_pi * tmid / T_season)
                                  + beta_env_cos * cos(two_pi * tmid / T_season)); // 0..1
    fL_series(i) = f_L;                                        // store modifier

    // Resource limitation and process rates (smooth, stabilized)
    Type f_N = N / (K_N + N + tiny);                           // Monod limitation 0..1
    Type mu_eff = mu_max * f_N * f_L;                          // effective phyto specific growth
    Type qpowP = pow(P + tiny, q);                             // P^q with small offset
    Type g_eff = g_max * (Type(0.5) + Type(0.5) * f_L);        // seasonal modifier on grazing (>=0.5*g_max)
    Type denomG = pow(K_G, q) + qpowP + tiny;                  // denominator for grazing function
    Type g_fun = g_eff * qpowP / denomG;                       // per-capita grazing rate on P

    // Flows (all in g C m^-3 d^-1)
    Type U = mu_eff * P;                                       // uptake of N by P
    Type G = g_fun * Z;                                        // grazing loss of P (ingestion by Z)
    Type R = phi_rec * ((Type(1.0) - e_P) * U                  // P exudation/leakage recycled
                        + m_P * P                               // P mortality recycled
                        + (Type(1.0) - e_Z) * G                 // unassimilated ingestion recycled
                        + m_Z * Z);                             // Z mortality recycled
    Type M = mix_rate * (N_ext - N);                           // physical nutrient supply/removal

    // ODEs (Euler step)
    Type dN = -U + R + M;                                      // nutrient change
    Type dP =  e_P * U - G - m_P * P;                          // phyto change
    Type dZ =  e_Z * G - m_Z * Z;                              // zoo change

    // Update states
    Type N_next = N + dt * dN;                                 // Euler step for N
    Type P_next = P + dt * dP;                                 // Euler step for P
    Type Z_next = Z + dt * dZ;                                 // Euler step for Z

    // Smooth positivity enforcement (no hard truncation)
    N_next = keep_positive(N_next, tiny, pen_pos);             // keep > tiny, accumulate penalty if adjusted
    P_next = keep_positive(P_next, tiny, pen_pos);             // keep > tiny
    Z_next = keep_positive(Z_next, tiny, pen_pos);             // keep > tiny

    // Store predictions and flows
    N_pred(i) = N_next;                                        // predicted N at time i
    P_pred(i) = P_next;                                        // predicted P at time i
    Z_pred(i) = Z_next;                                        // predicted Z at time i
    U_flow(i) = U; G_flow(i) = G; R_flow(i) = R; M_flow(i) = M;// diagnostics
  }
  // Add accumulated positivity penalties to objective
  nll += pen_pos;

  // ---------------------------
  // Observation likelihood (lognormal for strictly positive data)
  // ---------------------------
  for (int i = 0; i < nT; i++) {
    // Stabilize observations with tiny offset to avoid log(0)
    Type yN = N_dat(i) + tiny;                                 // observed N
    Type yP = P_dat(i) + tiny;                                 // observed P
    Type yZ = Z_dat(i) + tiny;                                 // observed Z

    // Predicted means on log-scale
    Type muN = log(N_pred(i) + tiny);                          // log-mean for N
    Type muP = log(P_pred(i) + tiny);                          // log-mean for P
    Type muZ = log(Z_pred(i) + tiny);                          // log-mean for Z

    // Lognormal likelihood: log f(y) = dnorm(log y | mu, sigma, true) - log y
    nll -= dnorm(log(yN), muN, sigma_N, true); nll += log(yN); // include Jacobian -log(y)
    nll -= dnorm(log(yP), muP, sigma_P, true); nll += log(yP);
    nll -= dnorm(log(yZ), muZ, sigma_Z, true); nll += log(yZ);
  }

  // ---------------------------
  // Report predictions and diagnostics
  // ---------------------------
  REPORT(N_pred); // predicted nutrient trajectory (g C m^-3)
  REPORT(P_pred); // predicted phytoplankton trajectory (g C m^-3)
  REPORT(Z_pred); // predicted zooplankton trajectory (g C m^-3)
  REPORT(U_flow); // nutrient uptake by phytoplankton (g C m^-3 d^-1)
  REPORT(G_flow); // grazing flow P->Z (g C m^-3 d^-1)
  REPORT(R_flow); // recycling to N (g C m^-3 d^-1)
  REPORT(M_flow); // mixing-driven N flux (g C m^-3 d^-1)
  REPORT(fL_series); // seasonal light proxy (0..1)
  REPORT(mu_max); REPORT(K_N); REPORT(g_max); REPORT(K_G); REPORT(nu); // key natural-scale parameters
  REPORT(e_Z); REPORT(e_P); REPORT(m_P); REPORT(m_Z); REPORT(phi_rec); // efficiencies and mortalities
  REPORT(mix_rate); REPORT(N_ext);                                     // physical supply parameters
  REPORT(sigma_N); REPORT(sigma_P); REPORT(sigma_Z);                   // observation SDs (log-scale effective)

  return nll; // total negative log-likelihood
}
