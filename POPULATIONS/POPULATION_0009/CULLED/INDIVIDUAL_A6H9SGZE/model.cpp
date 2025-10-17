#include <TMB.hpp>

// Utility smooth functions to ensure numerical stability and positivity
template<class Type> inline Type softplus(Type x) {
  // Smooth, everywhere-differentiable approximation to max(0,x) to keep values positive
  // Avoid std::log1p (not AD-overloaded); use AD-safe log and exp
  return log(Type(1) + exp(x));
}
template<class Type> inline Type inv_logit(Type x) {
  // Logistic transform for smooth mapping from R -> (0,1)
  return Type(1) / (Type(1) + exp(-x));
}
template<class Type> inline Type sqr(Type x) { return x * x; }

// Smooth penalty for bounds (adds zero when x in [lo, hi], quadratic growth outside)
template<class Type> inline Type smooth_bound_penalty(Type x, bool has_lo, Type lo, bool has_hi, Type hi, Type scale){
  Type pen = Type(0);
  if (has_lo) {
    pen += sqr(softplus(lo - x) / scale);
  }
  if (has_hi) {
    pen += sqr(softplus(x - hi) / scale);
  }
  return pen;
}

/*
Model overview and numbered equations:

State variables (g C m^-3):
  N = dissolved nutrient (as carbon equivalent)
  P = phytoplankton biomass
  Z = zooplankton biomass

Environmental driver:
  Seasonal light limitation represented by a sinusoid, mapped to (0,1) via logit link.

Process formulations:
  (1) Nutrient limitation:    f_N = N / (K_N + N + eps)                [Monod saturation]
  (2) Light limitation:       f_I = inv_logit(light_logit0 + light_amp * sin(2π t / T + phase))
  (3) Smooth co-limitation:   f_lim = 1 / ((f_N^-s + f_I^-s)^(1/s))    [smooth Liebig-like min]
  (4) Phyto growth:           G_P = μ_max * f_lim * P
  (5) Grazing per Z (HIII):   g0 = g_max * P^h / (K_P_g^h + P^h + eps)
  (6) Predator interference:  g = g0 / (1 + i_Z * Z)                   [Beddington-DeAngelis modifier]
  (7) Ingestion flux:         C = g * Z
  (8) Zoop growth:            G_Z = β_Z * C
  (9) Phyto non-graze losses: L_P = m_P * P + q_P * P^2
  (10) Zoop losses:           L_Z = m_Z * Z + q_Z * Z^2
  (11) Remineralization:      R = φ_remin * (L_P + L_Z + (1 - β_Z) * C)
  (12) Nutrient uptake:       U = G_P / e_P_uptake_eff                 [production/uptake efficiency]
  (13) Mixing supply:         S = k_mix * (N_ext - N)
  (14) Euler updates:         X(t+dt) = X(t) + dt * dX/dt, all flows computed from previous step

Observation model (applied to all time points):
  Lognormal errors for N_dat, P_dat, Z_dat with SD floors for numerical stability.

Notes:
  - All flows use previous-step predictions (no data leakage).
  - Initial conditions are set from the first observed values.
  - Small constants (eps) are used to protect denominators and logs.
  - Parameters are softly bounded by penalties, and rates are mapped through softplus/logit where needed.
*/

template<class Type>
Type objective_function<Type>::operator() ()
{
  // -----------------------
  // Data inputs
  // -----------------------
  DATA_VECTOR(Time);        // Time in days (pipeline provides 'Time')
  DATA_VECTOR(N_dat);       // Observed nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);       // Observed phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);       // Observed zooplankton concentration (g C m^-3)
  // season_period_days is not provided by the pipeline; use a constant inside the model
  Type season_period_days = Type(365.0); // days | Fixed seasonal period used in light limitation

  // -----------------------
  // Parameters (unconstrained; mapped/penalized for biology and stability)
  // -----------------------
  PARAMETER(mu_max);        // day^-1 | Maximum phytoplankton specific growth rate; literature/initial bounds [0,5]
  PARAMETER(K_N);           // g C m^-3 | Half-sat for nutrient limitation; literature bounds [1e-6,1]
  PARAMETER(light_logit0);  // dimensionless | Baseline logit of light limitation f_I; initial bounds [-5,5]
  PARAMETER(light_amp);     // dimensionless | Amplitude on logit scale for seasonal light limitation; bounds [0,5]
  PARAMETER(season_phase);  // radians | Phase of the seasonal sinusoid; bounds [-pi, pi]
  PARAMETER(s_lim);         // dimensionless | Smoothness exponent for co-limitation; bounds [0.5,10]
  PARAMETER(g_max);         // day^-1 | Max per-capita grazing rate; literature bounds [0,5]
  PARAMETER(K_P_g);         // g C m^-3 | Half-sat for grazing; literature bounds [1e-6,1]
  PARAMETER(h_g);           // dimensionless | Hill exponent for grazing (>=1); bounds [1,3]
  PARAMETER(i_Z);           // (g C m^-3)^-1 | Predator interference coefficient (BD); bounds [0, 10]
  PARAMETER(beta_Z);        // dimensionless (0-1) | Zooplankton assimilation efficiency; bounds [0,1]
  PARAMETER(phi_remin);     // dimensionless (0-1) | Fraction of losses instantly remineralized; bounds [0,1]
  PARAMETER(m_P);           // day^-1 | Linear phytoplankton loss rate; bounds [0,1]
  PARAMETER(q_P);           // (g C m^-3)^-1 day^-1 | Quadratic P losses; bounds [0,1]
  PARAMETER(m_Z);           // day^-1 | Linear zooplankton mortality; bounds [0,1]
  PARAMETER(q_Z);           // (g C m^-3)^-1 day^-1 | Quadratic Z mortality; bounds [0,1]
  PARAMETER(k_mix);         // day^-1 | Mixing rate toward N_ext; bounds [0,1]
  PARAMETER(N_ext);         // g C m^-3 | External nutrient concentration; bounds [0,10]
  PARAMETER(e_P_uptake_eff);// dimensionless (0-1) | Efficiency converting nutrient uptake to P growth; bounds [0,1]
  PARAMETER(sigma_N);       // log SD | Observation error for N (lognormal); bounds [0.01,1]
  PARAMETER(sigma_P);       // log SD | Observation error for P (lognormal); bounds [0.01,1]
  PARAMETER(sigma_Z);       // log SD | Observation error for Z (lognormal); bounds [0.01,1]

  // -----------------------
  // Constants and helpers
  // -----------------------
  int n = N_dat.size();                 // Number of time steps
  Type eps = Type(1e-8);                // Small constant for stability in divisions and logs
  Type pi = Type(3.14159265358979323846); // Pi constant

  // Realized, stabilized parameters (smoothly mapped to positive domains where needed)
  Type mu = softplus(mu_max);                 // day^-1, positive
  Type Kn = softplus(K_N);                    // g C m^-3, positive
  Type gmax = softplus(g_max);                // day^-1, positive
  Type Kpg = softplus(K_P_g);                 // g C m^-3, positive
  Type mP = softplus(m_P);                    // day^-1, positive
  Type qP = softplus(q_P);                    // (g C m^-3)^-1 day^-1, positive
  Type mZ = softplus(m_Z);                    // day^-1, positive
  Type qZ = softplus(q_Z);                    // (g C m^-3)^-1 day^-1, positive
  Type kmix = softplus(k_mix);                // day^-1, positive
  Type Next = softplus(N_ext);                // g C m^-3, positive
  Type eP = inv_logit(e_P_uptake_eff);        // (0,1)
  Type betaZ = inv_logit(beta_Z);             // (0,1)
  Type phiRem = inv_logit(phi_remin);         // (0,1)
  Type sLim = softplus(s_lim);                // >=0, smoothness exponent
  // h_g should be >= 1; we map to >=1 smoothly by 1 + softplus(h_g - 1)
  Type h = Type(1) + softplus(h_g - Type(1)); // Hill exponent >= 1
  // light_amp should be >= 0; keep positive via softplus
  Type lamp = softplus(light_amp);            // >=0
  // predator interference coefficient >= 0
  Type iZ = softplus(i_Z);                    // (g C m^-3)^-1, >= 0
  // sigma floors via quadrature (no hard max)
  Type sig_floor = Type(0.05);                // Minimum SD on log scale
  Type sN = sqrt(sqr(sigma_N) + sqr(sig_floor)); // >= sig_floor
  Type sP = sqrt(sqr(sigma_P) + sqr(sig_floor)); // >= sig_floor
  Type sZ = sqrt(sqr(sigma_Z) + sqr(sig_floor)); // >= sig_floor

  // -----------------------
  // Smooth penalties to softly enforce biological bounds
  // -----------------------
  Type penalty = Type(0);
  Type pen_scale = Type(1.0); // penalty scale (tunable)
  penalty += smooth_bound_penalty(mu, true, Type(0.0), true, Type(5.0), pen_scale);
  penalty += smooth_bound_penalty(Kn, true, Type(1e-6), true, Type(1.0), pen_scale);
  penalty += smooth_bound_penalty(light_logit0, true, Type(-5.0), true, Type(5.0), pen_scale);
  penalty += smooth_bound_penalty(lamp, true, Type(0.0), true, Type(5.0), pen_scale);
  penalty += smooth_bound_penalty(season_phase, true, Type(-pi), true, Type(pi), pen_scale);
  penalty += smooth_bound_penalty(sLim, true, Type(0.5), true, Type(10.0), pen_scale);
  penalty += smooth_bound_penalty(gmax, true, Type(0.0), true, Type(5.0), pen_scale);
  penalty += smooth_bound_penalty(Kpg, true, Type(1e-6), true, Type(1.0), pen_scale);
  penalty += smooth_bound_penalty(h, true, Type(1.0), true, Type(3.0), pen_scale);
  penalty += smooth_bound_penalty(iZ, true, Type(0.0), true, Type(10.0), pen_scale);
  penalty += smooth_bound_penalty(betaZ, true, Type(0.0), true, Type(1.0), pen_scale);
  penalty += smooth_bound_penalty(phiRem, true, Type(0.0), true, Type(1.0), pen_scale);
  penalty += smooth_bound_penalty(mP, true, Type(0.0), true, Type(1.0), pen_scale);
  penalty += smooth_bound_penalty(qP, true, Type(0.0), true, Type(1.0), pen_scale);
  penalty += smooth_bound_penalty(mZ, true, Type(0.0), true, Type(1.0), pen_scale);
  penalty += smooth_bound_penalty(qZ, true, Type(0.0), true, Type(1.0), pen_scale);
  penalty += smooth_bound_penalty(kmix, true, Type(0.0), true, Type(1.0), pen_scale);
  penalty += smooth_bound_penalty(Next, true, Type(0.0), true, Type(10.0), pen_scale);
  penalty += smooth_bound_penalty(eP, true, Type(0.0), true, Type(1.0), pen_scale);

  // -----------------------
  // Prediction vectors and initialization from data (no estimation of ICs)
  // -----------------------
  vector<Type> N_pred(n);  // Predicted nutrients
  vector<Type> P_pred(n);  // Predicted phytoplankton
  vector<Type> Z_pred(n);  // Predicted zooplankton
  vector<Type> fN(n);      // Nutrient limitation factor (for reporting)
  vector<Type> fI(n);      // Light limitation factor (for reporting)
  vector<Type> env_mult(n); // Seasonal environment signal (sinusoid argument)

  // Initialize with observed initial conditions (no data leakage in transitions)
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);

  // Compute limitation at the first time (for completeness in reporting)
  env_mult(0) = sin(Type(2.0) * pi * (Time(0) / season_period_days) + season_phase); // dimensionless sinusoid
  fI(0) = inv_logit(light_logit0 + lamp * env_mult(0));                              // in (0,1)
  fN(0) = N_pred(0) / (Kn + N_pred(0) + eps);                                        // in (0,1)

  // -----------------------
  // Time stepping using forward Euler; all flows use previous-step states
  // -----------------------
  for (int i = 1; i < n; i++) {
    // Time step size with small floor
    Type dt = Time(i) - Time(i - 1);
    dt = dt + eps; // ensure strictly positive to avoid division problems

    // Previous-step states, mapped to positive domain smoothly for stable fluxes
    Type Nprev = softplus(N_pred(i - 1));
    Type Pprev = softplus(P_pred(i - 1));
    Type Zprev = softplus(Z_pred(i - 1));

    // Limitation factors
    env_mult(i) = sin(Type(2.0) * pi * (Time(i - 1) / season_period_days) + season_phase); // use previous time
    fI(i) = inv_logit(light_logit0 + lamp * env_mult(i));                                  // (0,1)
    fN(i) = Nprev / (Kn + Nprev + eps);                                                    // (0,1)

    // Smooth Liebig-like co-limitation
    Type inv_s = Type(1.0) / (sLim + eps);
    Type f_lim = Type(1.0) / pow(pow(fN(i) + eps, -sLim) + pow(fI(i) + eps, -sLim) + eps, inv_s);

    // Phytoplankton specific growth and biomass production
    Type Gp_spec = mu * f_lim;                   // day^-1
    Type Gp = Gp_spec * Pprev;                   // g C m^-3 day^-1

    // Grazing functional response (Holling III with predator interference)
    Type Ph = pow(Pprev + eps, h);               // P^h
    Type denom_P = pow(Kpg + eps, h) + Ph + eps;
    // Beddington-DeAngelis predator interference modifier: divides per-capita rate by (1 + iZ * Zprev)
    Type g_rate = gmax * Ph / denom_P / (Type(1.0) + iZ * Zprev); // day^-1 per Z
    Type C = g_rate * Zprev;                     // ingestion flux, g C m^-3 day^-1

    // Losses and remineralization
    Type Lp = mP * Pprev + qP * Pprev * Pprev;   // P non-grazing losses, g C m^-3 day^-1
    Type Lz = mZ * Zprev + qZ * Zprev * Zprev;   // Z losses, g C m^-3 day^-1
    Type Remin = phiRem * (Lp + Lz + (Type(1.0) - betaZ) * C); // remin flux to N

    // Nutrient uptake required to support Gp given efficiency
    Type U = Gp / (eP + eps);                    // g C m^-3 day^-1

    // Nutrient mixing supply
    Type S = kmix * (Next - Nprev);              // g C m^-3 day^-1

    // State derivatives
    Type dP = Gp - C - Lp;                       // g C m^-3 day^-1
    Type dZ = betaZ * C - Lz;                    // g C m^-3 day^-1
    Type dN = -U + Remin + S;                    // g C m^-3 day^-1

    // Euler updates with smooth positivity via softplus on incremented states
    Type Nnext = softplus(N_pred(i - 1) + dN * dt - eps); // keep >0 smoothly
    Type Pnext = softplus(P_pred(i - 1) + dP * dt - eps);
    Type Znext = softplus(Z_pred(i - 1) + dZ * dt - eps);

    N_pred(i) = Nnext + eps;
    P_pred(i) = Pnext + eps;
    Z_pred(i) = Znext + eps;
  }

  // -----------------------
  // Likelihood: lognormal for strictly positive data with SD floor
  // -----------------------
  Type nll = Type(0);
  for (int i = 0; i < n; i++) {
    // Apply lognormal likelihood to all observations
    nll -= dnorm(log(N_dat(i) + eps), log(N_pred(i) + eps), sN, true);
    nll -= dnorm(log(P_dat(i) + eps), log(P_pred(i) + eps), sP, true);
    nll -= dnorm(log(Z_dat(i) + eps), log(Z_pred(i) + eps), sZ, true);
  }

  // -----------------------
  // Objective: negative log-likelihood plus smooth penalties
  // -----------------------
  Type obj = nll + penalty;

  // -----------------------
  // Reporting
  // -----------------------
  REPORT(N_pred);     // Predicted nutrient trajectory
  REPORT(P_pred);     // Predicted phytoplankton trajectory
  REPORT(Z_pred);     // Predicted zooplankton trajectory
  REPORT(fN);         // Nutrient limitation factor over time
  REPORT(fI);         // Light limitation factor over time
  REPORT(env_mult);   // Seasonal sinusoid driver (for diagnostics)
  REPORT(nll);        // Likelihood component
  REPORT(penalty);    // Penalty component
  ADREPORT(N_pred);
  ADREPORT(P_pred);
  ADREPORT(Z_pred);

  return obj;
}
