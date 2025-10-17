#include <TMB.hpp>

// Small helper: numerically stable softplus to ensure smooth non-negativity
template<class Type>
Type softplus(const Type& x) {
  // Stable formulation compatible with CppAD/TMB (avoids log1p):
  // softplus(x) = log(1 + exp(-|x|)) + max(x, 0)
  // max(x,0) written as 0.5*(x + |x|) to avoid branching
  return log(Type(1.0) + exp(-CppAD::abs(x))) + Type(0.5) * (x + CppAD::abs(x));
}

// Smooth lower bound: returns a value >= a with smooth transition
template<class Type>
Type clamp_min_smooth(const Type& x, const Type& a) {
  // a + softplus(x - a) behaves like max(x, a) but is smooth
  return a + softplus(x - a);
}

// Smooth penalty for staying within [low, high] without hard constraints
template<class Type>
Type bound_penalty(const Type& x, const Type& low, const Type& high, const Type& strength) {
  // Penalize excursions outside range with a smooth quadratic on softplus exceedance
  Type below = softplus(low - x);   // >0 if x < low
  Type above = softplus(x - high);  // >0 if x > high
  return strength * (below * below + above * above);
}

template<class Type>
Type objective_function<Type>::operator() ()
{
  using CppAD::pow;

  // Constants for numerical stability
  const Type eps = Type(1e-8);             // small positive constant to avoid division by zero
  const Type min_sigma = Type(0.05);       // minimum observational SD on log scale
  const Type pi = Type(3.141592653589793238462643383279502884);

  // -----------------------------
  // DATA (read-only)
  // -----------------------------
  // NOTE: "Time" corresponds to the CSV time column (originally labeled "Time (days)").
  DATA_VECTOR(Time);                      // Time in days from data file; uneven time step in days
  DATA_VECTOR(N_dat);                     // Observed nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);                     // Observed phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);                     // Observed zooplankton concentration (g C m^-3)

  // Provide a robust default seasonal period as data is not supplied by the driver
  const Type period_days = Type(365.0);   // days | Seasonal period for light/temperature forcing

  // -----------------------------
  // PARAMETERS (to estimate)
  // -----------------------------
  // Primary production parameters
  PARAMETER(mu_max);                      // d^-1 | Max specific growth rate of phytoplankton; literature or initial estimate from bloom rise rate
  PARAMETER(k_N);                         // g C m^-3 | Half-saturation for nutrient uptake; literature or fit from low-N curvature
  PARAMETER(light_amp);                   // dimensionless (0-1) | Amplitude of seasonal light limitation; 0=no seasonality, ~0.5 typical
  PARAMETER(light_phase);                 // days | Phase shift for seasonal light (days), aligns seasonal peak
  // Temperature modulation (Q10)
  PARAMETER(Q10);                         // dimensionless | Temperature sensitivity factor per 10°C; commonly ~2
  PARAMETER(T_mean);                      // °C | Mean seasonal temperature
  PARAMETER(T_amp);                       // °C | Amplitude of seasonal temperature cycle (>=0)
  PARAMETER(T_phase);                     // days | Phase shift for seasonal temperature cycle
  PARAMETER(T_ref);                       // °C | Reference temperature for Q10 scaling (often equal to T_mean)

  // Grazing and zooplankton parameters
  PARAMETER(g_max);                       // d^-1 | Max specific grazing rate
  PARAMETER(k_G);                         // g C m^-3 | Half-saturation (scale) for grazing functional response
  PARAMETER(hill);                        // dimensionless | Hill exponent (>=1) for Holling II/III behavior
  PARAMETER(gamma_Z);                     // dimensionless (0-1) | Fraction of grazed P converted into Z growth (assimilation * growth efficiency)

  // Loss and exchange parameters
  PARAMETER(mP);                          // d^-1 | Non-grazing phytoplankton loss rate; remineralized to N
  PARAMETER(mZ);                          // d^-1 | Zooplankton linear mortality; remineralized to N
  PARAMETER(mZ2);                         // (g C m^-3)^-1 d^-1 | Zooplankton quadratic mortality; remineralized to N
  PARAMETER(k_exch);                      // d^-1 | Exchange/mixing rate with external nutrient reservoir
  PARAMETER(N_ext);                       // g C m^-3 | External (deep/source) nutrient concentration

  // Observation error (lognormal) parameters; ensure sigma >= min_sigma using softplus
  PARAMETER(log_sigma_N);                 // log-scale free parameter for nutrient observation error
  PARAMETER(log_sigma_P);                 // log-scale free parameter for phyto observation error
  PARAMETER(log_sigma_Z);                 // log-scale free parameter for zoo observation error

  // Effective observation sigmas with smooth minimum
  Type sigma_N = min_sigma + softplus(log_sigma_N);  // >= min_sigma
  Type sigma_P = min_sigma + softplus(log_sigma_P);  // >= min_sigma
  Type sigma_Z = min_sigma + softplus(log_sigma_Z);  // >= min_sigma

  int Tn = N_dat.size();                  // Number of time points

  // -----------------------------
  // STATE PREDICTIONS
  // -----------------------------
  vector<Type> N_pred(Tn);                // Predicted nutrient (g C m^-3)
  vector<Type> P_pred(Tn);                // Predicted phytoplankton (g C m^-3)
  vector<Type> Z_pred(Tn);                // Predicted zooplankton (g C m^-3)

  // Initialize states from observed initial conditions to avoid optimizing initial states
  N_pred(0) = N_dat(0);                   // Initial nutrient = first observation
  P_pred(0) = P_dat(0);                   // Initial phytoplankton = first observation
  Z_pred(0) = Z_dat(0);                   // Initial zooplankton = first observation

  // For diagnostics: store some forcings/limiters
  vector<Type> temp_t(Tn);                // Seasonal temperature (°C)
  vector<Type> theta_T(Tn);               // Q10 temperature multiplier (dimensionless)
  vector<Type> L_season(Tn);              // Seasonal light limitation (0-1)
  vector<Type> fN(Tn);                    // Nutrient limitation (0-1)
  vector<Type> G_t(Tn);                   // Grazing flux (g C m^-3 d^-1)
  vector<Type> U_t(Tn);                   // Primary production flux (g C m^-3 d^-1)

  // -----------------------------
  // TIME-STEPPING DYNAMICS
  // -----------------------------
  for (int t = 1; t < Tn; t++) {
    // Time step (days), ensure strictly positive with smooth clamp
    Type dt_raw = Time(t) - Time(t - 1);   // from time column
    Type dt = clamp_min_smooth(dt_raw, eps);         // enforce dt > 0 smoothly

    // Previous step states (use predictions only; no data leakage)
    Type N_prev = clamp_min_smooth(N_pred(t - 1), eps);
    Type P_prev = clamp_min_smooth(P_pred(t - 1), eps);
    Type Z_prev = clamp_min_smooth(Z_pred(t - 1), eps);

    // Seasonal temperature and Q10 modulation (environmental effect)
    Type angle_T = Type(2.0) * pi * (Time(t - 1) - T_phase) / period_days;
    temp_t(t) = T_mean + T_amp * cos(angle_T);                         // seasonal temperature
    theta_T(t) = pow(Q10, (temp_t(t) - T_ref) / Type(10.0));           // Q10 temperature multiplier

    // Seasonal light limitation proxy (0..1), using cosine; min value = 1 - light_amp, max = 1
    Type angle_L = Type(2.0) * pi * (Time(t - 1) - light_phase) / period_days;
    L_season(t) = (Type(1.0) - light_amp) + light_amp * (Type(0.5) + Type(0.5) * cos(angle_L));
    // Smooth clamp to [eps, 1] to ensure strictly positive
    L_season(t) = clamp_min_smooth(L_season(t), eps);
    if (L_season(t) > Type(1.0)) L_season(t) = Type(1.0); // Cosine construction keeps <=1; keep this guard

    // Nutrient limitation (Michaelis-Menten)
    fN(t) = N_prev / (k_N + N_prev + eps);                               // 0..1

    // Primary production flux U (g C m^-3 d^-1)
    U_t(t) = mu_max * theta_T(t) * L_season(t) * fN(t) * P_prev;         // co-limited by nutrient and light, temp-scaled

    // Grazing flux G using Hill-type (Holling II/III) functional response: g_max * (P^h / (K^h + P^h)) * Z
    Type P_pow = pow(P_prev + eps, hill);
    Type K_pow = pow(k_G + eps,  hill);
    Type phi = P_pow / (K_pow + P_pow + eps);                             // 0..1 saturation function
    G_t(t) = g_max * phi * Z_prev;

    // Losses and remineralization
    Type lossP = mP * P_prev;                                             // phytoplankton non-grazing losses
    Type mortZ_lin = mZ * Z_prev;                                         // zooplankton linear mortality
    Type mortZ_quad = mZ2 * Z_prev * Z_prev;                              // zooplankton quadratic mortality

    // External mixing of nutrient
    Type mixN = k_exch * (N_ext - N_prev);

    // Differential changes (per day)
    // 1) dP/dt = U - G - mP*P
    Type dPdt = U_t(t) - G_t(t) - lossP;

    // 2) dZ/dt = gamma_Z * G - mZ*Z - mZ2*Z^2
    Type dZdt = gamma_Z * G_t(t) - mortZ_lin - mortZ_quad;

    // 3) dN/dt = -U + (1 - gamma_Z)*G + mP*P + mZ*Z + mZ2*Z^2 + mixing
    Type dNdt = -U_t(t) + (Type(1.0) - gamma_Z) * G_t(t) + lossP + mortZ_lin + mortZ_quad + mixN;

    // Euler update with smooth positivity
    Type N_next_raw = N_prev + dt * dNdt;
    Type P_next_raw = P_prev + dt * dPdt;
    Type Z_next_raw = Z_prev + dt * dZdt;

    N_pred(t) = clamp_min_smooth(N_next_raw, eps);
    P_pred(t) = clamp_min_smooth(P_next_raw, eps);
    Z_pred(t) = clamp_min_smooth(Z_next_raw, eps);
  }

  // Initialize diagnostics for t=0 (for completeness)
  temp_t(0)  = T_mean + T_amp * cos(Type(2.0) * pi * (Time(0) - T_phase) / period_days);
  theta_T(0) = pow(Q10, (temp_t(0) - T_ref) / Type(10.0));
  L_season(0)= (Type(1.0) - light_amp) + light_amp * (Type(0.5) + Type(0.5) * cos(Type(2.0) * pi * (Time(0) - light_phase) / period_days));
  L_season(0)= clamp_min_smooth(L_season(0), eps);
  fN(0)      = N_pred(0) / (k_N + N_pred(0) + eps);
  U_t(0)     = mu_max * theta_T(0) * L_season(0) * fN(0) * P_pred(0);
  {
    Type P_pow0 = pow(P_pred(0) + eps, hill);
    Type K_pow0 = pow(k_G + eps,  hill);
    Type phi0 = P_pow0 / (K_pow0 + P_pow0 + eps);
    G_t(0) = g_max * phi0 * clamp_min_smooth(Z_pred(0), eps);
  }

  // -----------------------------
  // LIKELIHOOD (lognormal errors)
  // -----------------------------
  Type nll = 0.0;

  for (int t = 0; t < Tn; t++) {
    // lognormal density: lnY ~ Normal(ln(pred), sigma); include Jacobian -log(Y)
    nll -= dnorm(log(N_dat(t) + eps), log(N_pred(t) + eps), sigma_N, true) - log(N_dat(t) + eps);
    nll -= dnorm(log(P_dat(t) + eps), log(P_pred(t) + eps), sigma_P, true) - log(P_dat(t) + eps);
    nll -= dnorm(log(Z_dat(t) + eps), log(Z_pred(t) + eps), sigma_Z, true) - log(Z_dat(t) + eps);
  }

  // -----------------------------
  // Smooth parameter bound penalties (no hard constraints)
  // -----------------------------
  // Suggested biological bounds (documented in parameters.json as well)
  const Type pen_w = Type(100.0); // penalty strength (tunable)

  nll += bound_penalty(mu_max,   Type(0.0),  Type(3.0),  pen_w);
  nll += bound_penalty(k_N,      Type(1e-6), Type(5.0),  pen_w);
  nll += bound_penalty(light_amp,Type(0.0),  Type(0.99), pen_w);
  nll += bound_penalty(light_phase, Type(0.0), period_days, pen_w);

  nll += bound_penalty(Q10,      Type(1.0),  Type(3.0),  pen_w);
  nll += bound_penalty(T_mean,   Type(-2.0), Type(30.0), pen_w);
  nll += bound_penalty(T_amp,    Type(0.0),  Type(15.0), pen_w);
  nll += bound_penalty(T_phase,  Type(0.0),  period_days, pen_w);
  nll += bound_penalty(T_ref,    Type(-2.0), Type(30.0), pen_w);

  nll += bound_penalty(g_max,    Type(0.0),  Type(5.0),  pen_w);
  nll += bound_penalty(k_G,      Type(1e-6), Type(5.0),  pen_w);
  nll += bound_penalty(hill,     Type(1.0),  Type(3.0),  pen_w);
  nll += bound_penalty(gamma_Z,  Type(0.0),  Type(1.0),  pen_w);

  nll += bound_penalty(mP,       Type(0.0),  Type(1.0),  pen_w);
  nll += bound_penalty(mZ,       Type(0.0),  Type(1.0),  pen_w);
  nll += bound_penalty(mZ2,      Type(0.0),  Type(1.0),  pen_w);
  nll += bound_penalty(k_exch,   Type(0.0),  Type(1.0),  pen_w);
  nll += bound_penalty(N_ext,    Type(0.0),  Type(5.0),  pen_w);

  // -----------------------------
  // REPORTING
  // -----------------------------
  // Predictions corresponding to observations
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);

  // Report diagnostic/derived quantities
  REPORT(temp_t);
  REPORT(theta_T);
  REPORT(L_season);
  REPORT(fN);
  REPORT(G_t);
  REPORT(U_t);

  // Report sigmas
  REPORT(sigma_N);
  REPORT(sigma_P);
  REPORT(sigma_Z);

  return nll;
}

/*
Model equations (per-step, using forward Euler with irregular dt):

Let the state at time t-1 be (N, P, Z). Define:
1) Nutrient limitation:      fN = N / (k_N + N)
2) Light limitation:         L(t) = (1 - light_amp) + light_amp * (0.5) * (1 + cos(2π*(Time - light_phase)/period_days))
3) Temperature multiplier:   θ(t) = Q10^((T(t) - T_ref)/10), T(t) = T_mean + T_amp * cos(2π*(Time - T_phase)/period_days)
4) Primary production:       U = μ_max * θ(t) * L(t) * fN * P
5) Grazing saturation:       φ = P^h / (k_G^h + P^h)
6) Grazing flux:             G = g_max * φ * Z
7) Phyto loss (non-grazing): L_P = mP * P
8) Zoo mortality:            M_Z = mZ * Z + mZ2 * Z^2
9) External mixing (N):      Mix = k_exch * (N_ext - N)

State changes (per day):
10) dP/dt = U - G - L_P
11) dZ/dt = γ_Z * G - M_Z
12) dN/dt = -U + (1 - γ_Z)*G + L_P + M_Z + Mix

Discretization:
X_next = clamp_min_smooth(X_prev + dt * dX/dt, eps) where clamp_min_smooth(x,a) = a + softplus(x - a).

All observations are modeled as lognormal:
ln(Y_obs) ~ Normal(ln(Y_pred), σ), with σ ≥ min_sigma via σ = min_sigma + softplus(log_sigma_param).
*/
