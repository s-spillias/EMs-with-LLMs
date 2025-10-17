#include <TMB.hpp>

// Small numerical constant to avoid division by zero and log(0)
template<class Type>
inline Type tiny_const() { return Type(1e-12); }

// Smooth, differentiable nonlinearity to keep values near-positive while
// behaving approximately like identity for large positive inputs.
// k controls the sharpness; larger k -> closer to identity for x>0.
// Note: Use AD-safe formulation; avoid log1p which is not overloaded for AD.
template<class Type>
inline Type softplus(Type x, Type k = Type(20.0)) {
  Type z = k * x;
  Type absz = CppAD::abs(z);
  // softplus(z) = log(1 + exp(z)) = log(1 + exp(-|z|)) + max(z, 0)
  Type max_z_0 = CppAD::CondExpGt(z, Type(0.0), z, Type(0.0));
  return (log(Type(1.0) + exp(-absz)) + max_z_0) / k;
}

// Map any real x to a near-nonnegative value. Approaches identity for x >> 0.
template<class Type>
inline Type positive_part(Type x, Type k = Type(20.0)) {
  // Shift to reduce bias near zero
  return softplus(x - tiny_const<Type>(), k) + tiny_const<Type>();
}

// Smooth penalty to softly keep a parameter within [lower, upper].
// Uses softplus so the penalty is differentiable everywhere.
template<class Type>
inline Type bounds_penalty(Type x, Type lower, Type upper, Type scale = Type(10.0)) {
  Type pen_low  = softplus(lower - x); // >0 only when x < lower (smoothly)
  Type pen_high = softplus(x - upper); // >0 only when x > upper (smoothly)
  Type pen = pen_low * pen_low + pen_high * pen_high;
  return scale * pen;
}

/*
Numbered ecological equations used in simulation (all rates per day; state units g C m^-3):

Let:
- LN = N / (K_N + N)                                  [1] Nutrient limitation (Monod, saturating)
- I(t) = max(0, I0 * (1 + I_amp * sin(2*pi*(t/365) + phi_I)))        [2] Seasonal light proxy
- LI = I / (K_I + I)                                   [3] Light limitation (saturating)
- Theta_T_P(t) = Q10_P^((T(t) - T_ref)/10)             [4a] Temperature modifier for phytoplankton processes
- Theta_T_Z(t) = Q10_Z^((T(t) - T_ref)/10)             [4b] Temperature modifier for zooplankton processes
- T(t) = T_ref + T_amp * sin(2*pi*(t/365) + phi_T)     [5] Seasonal temperature

Phytoplankton growth (carbon):
- rP = mu_max * Theta_T_P * LN * LI                    [6] Specific growth rate (d^-1)
- G_P = rP * P                                         [7] Gross P growth flux (g C m^-3 d^-1)
- U_N = G_P / y_P                                      [8] Nutrient uptake to support P growth (g C m^-3 d^-1)

Zooplankton grazing (Holling-type with order h):
- g_fun = g_max * Theta_T_Z * P^h / (K_P^h + P^h)      [9] Per-Z ingestion rate (d^-1)
- G = g_fun * Z                                        [10] Total grazing flux (g C m^-3 d^-1)

Partitioning and losses:
- Z_gain = beta * G                                    [11] Assimilated grazing to Z (growth)
- Z_excr = (ex_z * Theta_T_Z) * Z                      [12] Metabolic/excretory losses (to N), temperature-modified
- P_mort = mPl * P + mPq * P^2                         [13] Phytoplankton mortality (linear + quadratic)
- Z_mort = (mzl * Theta_T_Z) * Z + (mZq * Theta_T_Z) * Z^2   [14] Zooplankton mortality (linear + quadratic), temperature-modified

Remineralization:
- N_remin = (1 - beta) * G + rho_P * P_mort + rho_Z * Z_mort   [15] Flows routed back to N

Physical mixing:
- N_mix = k_mix * (N_ext - N)                          [16] Relaxation toward external nutrient

State dynamics (Euler stepping with dt):
- dP = G_P - G - P_mort                                [17]
- dZ = Z_gain - Z_excr - Z_mort                        [18]
- dN = -U_N + N_remin + N_mix                          [19]

All fluxes use the previous time-step states only to avoid data leakage.
Observations (strictly positive) are modeled with lognormal errors.
*/

template<class Type>
Type objective_function<Type>::operator() () {
  Type nll = 0.0;
  const Type eps = tiny_const<Type>();
  const Type two_pi = Type(6.28318530717958647692);

  // DATA ----
  // The data interface provides a column named "Time"
  DATA_VECTOR(Time);       // Time in days
  DATA_VECTOR(N_dat);      // Observed nutrient (g C m^-3)
  DATA_VECTOR(P_dat);      // Observed phytoplankton (g C m^-3)
  DATA_VECTOR(Z_dat);      // Observed zooplankton (g C m^-3)
  int n = N_dat.size();

  // PARAMETERS ----
  // Phytoplankton growth and resource limitation
  PARAMETER(mu_max);        // Maximum specific P growth rate (d^-1); literature ranges 0.1-2 d^-1; fit to bloom rise rate
  PARAMETER(K_N);           // Half-saturation for nutrient (g C m^-3); sets curvature of Monod limitation
  PARAMETER(y_P);           // Yield: g P biomass produced per g nutrient consumed (dimensionless, ~1 for C-based)
  // Light-seasonality limitation
  PARAMETER(I0);            // Mean light proxy (relative units); scales LI; can be set near 1
  PARAMETER(I_amp);         // Seasonal light amplitude (0-1); 0=no seasonality
  PARAMETER(K_I);           // Half-saturation for light limitation (relative units)
  PARAMETER(phi_I);         // Light phase (radians), shifts seasonal cycle timing
  // Temperature modifiers (Q10-style), separated by trophic level
  PARAMETER(Q10_P);         // Q10 for phytoplankton processes (dimensionless)
  PARAMETER(Q10_Z);         // Q10 for zooplankton processes (dimensionless)
  PARAMETER(T_ref);         // Reference temperature (°C) where temperature multipliers = 1
  PARAMETER(T_amp);         // Seasonal temperature amplitude (°C)
  PARAMETER(phi_T);         // Temperature phase (radians)
  // Grazing
  PARAMETER(g_max);         // Maximum per-capita ingestion rate (d^-1)
  PARAMETER(K_P);           // Half-saturation for grazing saturation (g C m^-3)
  PARAMETER(h);             // Holling order (>=1), controls switching strength Type II (h~1) to Type III (h>1)
  PARAMETER(beta);          // Assimilation efficiency of grazing to Z (dimensionless, 0-1)
  // Losses and remineralization
  PARAMETER(mPl);           // Linear P mortality rate (d^-1)
  PARAMETER(mPq);           // Quadratic P mortality coefficient ((g C m^-3)^-1 d^-1)
  PARAMETER(mzl);           // Linear Z mortality rate (d^-1)
  PARAMETER(mZq);           // Quadratic Z mortality coefficient ((g C m^-3)^-1 d^-1)
  PARAMETER(ex_z);          // Zooplankton excretion rate (d^-1)
  PARAMETER(rho_P);         // Fraction of P mortality routed to N (dimensionless, 0-1)
  PARAMETER(rho_Z);         // Fraction of Z mortality routed to N (dimensionless, 0-1)
  // Physical mixing
  PARAMETER(k_mix);         // Nutrient mixing rate (d^-1)
  PARAMETER(N_ext);         // External (deep) nutrient concentration (g C m^-3)
  // Observation model (lognormal)
  PARAMETER(log_sigma_N);   // Log SD for N observations
  PARAMETER(log_sigma_P);   // Log SD for P observations
  PARAMETER(log_sigma_Z);   // Log SD for Z observations

  // Smooth penalties to softly enforce biologically meaningful ranges
  // Suggested bounds (see parameters.json); penalties are differentiable.
  {
    nll += bounds_penalty(mu_max, Type(0.01), Type(3.0));
    nll += bounds_penalty(K_N,    Type(1e-4), Type(2.0));
    nll += bounds_penalty(y_P,    Type(0.3),  Type(2.0));
    nll += bounds_penalty(I0,     Type(1e-3), Type(10.0));
    nll += bounds_penalty(I_amp,  Type(0.0),  Type(1.0));
    nll += bounds_penalty(K_I,    Type(1e-4), Type(10.0));
    nll += bounds_penalty(phi_I,  Type(0.0),  Type(two_pi));
    nll += bounds_penalty(Q10_P,  Type(1.0),  Type(3.5));
    nll += bounds_penalty(Q10_Z,  Type(1.0),  Type(3.5));
    // T_ref can vary broadly; softly bound to [-2, 35] °C
    nll += bounds_penalty(T_ref,  Type(-2.0), Type(35.0));
    nll += bounds_penalty(T_amp,  Type(0.0),  Type(12.0));
    nll += bounds_penalty(phi_T,  Type(0.0),  Type(two_pi));
    nll += bounds_penalty(g_max,  Type(0.01), Type(5.0));
    nll += bounds_penalty(K_P,    Type(1e-4), Type(2.0));
    nll += bounds_penalty(h,      Type(1.0),  Type(3.0));
    nll += bounds_penalty(beta,   Type(0.1),  Type(0.9));
    nll += bounds_penalty(mPl,    Type(0.0),  Type(0.5));
    nll += bounds_penalty(mPq,    Type(0.0),  Type(5.0));
    nll += bounds_penalty(mzl,    Type(0.0),  Type(0.5));
    nll += bounds_penalty(mZq,    Type(0.0),  Type(5.0));
    nll += bounds_penalty(ex_z,   Type(0.0),  Type(0.7));
    nll += bounds_penalty(rho_P,  Type(0.0),  Type(1.0));
    nll += bounds_penalty(rho_Z,  Type(0.0),  Type(1.0));
    nll += bounds_penalty(k_mix,  Type(0.0),  Type(0.7));
    nll += bounds_penalty(N_ext,  Type(0.0),  Type(2.0));
  }

  // Prepare prediction vectors (initialized with observations at t0)
  vector<Type> N_pred(n); // model prediction for N (g C m^-3)
  vector<Type> P_pred(n); // model prediction for P (g C m^-3)
  vector<Type> Z_pred(n); // model prediction for Z (g C m^-3)

  // INITIAL CONDITIONS: set to first observed data point (no optimization here)
  N_pred(0) = (N_dat.size() > 0 ? N_dat(0) : Type(0));
  P_pred(0) = (P_dat.size() > 0 ? P_dat(0) : Type(0));
  Z_pred(0) = (Z_dat.size() > 0 ? Z_dat(0) : Type(0));

  // Simulate dynamics using previous step states only (no data leakage)
  for (int i = 1; i < n; ++i) {
    Type dt = Time(i) - Time(i - 1);                 // time step (days)
    dt = (dt > eps ? dt : eps);                      // ensure positive small dt

    // Previous states
    Type Nprev = N_pred(i - 1);
    Type Pprev = P_pred(i - 1);
    Type Zprev = Z_pred(i - 1);

    // Environmental modifiers (seasonal)
    Type tt = Time(i - 1);                           // use previous time for rates
    Type It = I0 * (Type(1.0) + I_amp * sin(two_pi * (tt / Type(365.0)) + phi_I));
    It = positive_part(It);                          // smooth non-negativity
    Type LI = It / (K_I + It + eps);                 // [3] Light limitation (saturating)

    Type Tt = T_ref + T_amp * sin(two_pi * (tt / Type(365.0)) + phi_T);
    Type Theta_T_P = pow(Q10_P, (Tt - T_ref) / Type(10.0)); // [4a] Phytoplankton temperature multiplier
    Type Theta_T_Z = pow(Q10_Z, (Tt - T_ref) / Type(10.0)); // [4b] Zooplankton temperature multiplier

    // Resource limitation and growth
    Type LN = Nprev / (K_N + Nprev + eps);          // [1] Nutrient limitation
    Type rP = mu_max * Theta_T_P * LN * LI;         // [6] Specific growth rate
    Type G_P = rP * Pprev;                          // [7] Gross P growth
    Type U_N = G_P / (y_P + eps);                   // [8] Nutrient uptake for growth

    // Grazing (Holling with order h)
    Type P_pow_h = pow(Pprev + eps, h);             // avoid 0^h
    Type denom = pow(K_P + eps, h) + P_pow_h;       // saturation denominator
    Type g_fun = g_max * Theta_T_Z * P_pow_h / (denom + eps); // [9]
    Type G = g_fun * Zprev;                         // [10] Total grazing

    // Partitioning and losses
    Type Z_gain = beta * G;                         // [11]
    Type Z_excr = (ex_z * Theta_T_Z) * Zprev;       // [12]
    Type P_mort = mPl * Pprev + mPq * Pprev * Pprev;                     // [13]
    Type Z_mort = (mzl * Theta_T_Z) * Zprev + (mZq * Theta_T_Z) * Zprev * Zprev; // [14]

    // Remineralization and mixing
    Type N_remin = (Type(1.0) - beta) * G + rho_P * P_mort + rho_Z * Z_mort; // [15]
    Type N_mix = k_mix * (N_ext - Nprev);            // [16]

    // State derivatives
    Type dP = G_P - G - P_mort;                      // [17]
    Type dZ = Z_gain - Z_excr - Z_mort;              // [18]
    Type dN = -U_N + N_remin + N_mix;                // [19]

    // Euler update with smooth positivity enforcement
    Type N_cand = Nprev + dt * dN;
    Type P_cand = Pprev + dt * dP;
    Type Z_cand = Zprev + dt * dZ;

    N_pred(i) = positive_part(N_cand);
    P_pred(i) = positive_part(P_cand);
    Z_pred(i) = positive_part(Z_cand);
  }

  // Observation model: lognormal with a minimum SD
  Type sigma_min = Type(1e-3);                       // fixed minimum SD on log scale for stability
  Type sigmaN = exp(log_sigma_N) + sigma_min;
  Type sigmaP = exp(log_sigma_P) + sigma_min;
  Type sigmaZ = exp(log_sigma_Z) + sigma_min;

  // All observations contribute; add small positive offset before log
  for (int i = 0; i < n; ++i) {
    Type yN = log(N_dat(i) + eps);
    Type mN = log(N_pred(i) + eps) - Type(0.5) * sigmaN * sigmaN; // mean correction for lognormal
    nll -= dnorm(yN, mN, sigmaN, true);

    Type yP = log(P_dat(i) + eps);
    Type mP = log(P_pred(i) + eps) - Type(0.5) * sigmaP * sigmaP;
    nll -= dnorm(yP, mP, sigmaP, true);

    Type yZ = log(Z_dat(i) + eps);
    Type mZ = log(Z_pred(i) + eps) - Type(0.5) * sigmaZ * sigmaZ;
    nll -= dnorm(yZ, mZ, sigmaZ, true);
  }

  // REPORT predictions and useful diagnostics
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);

  // Also report some derived parameters for diagnostics
  ADREPORT(mu_max);
  ADREPORT(g_max);
  ADREPORT(Q10_P);
  ADREPORT(Q10_Z);
  ADREPORT(k_mix);

  return nll;
}
