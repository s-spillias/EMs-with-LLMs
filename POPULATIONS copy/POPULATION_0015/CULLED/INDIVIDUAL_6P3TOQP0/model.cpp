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
  PARAMETER(log_mu_max);       // log of max phytoplankton specific uptake rate (day^-1)
  PARAMETER(log_K_N);          // log of half-saturation constant for nutrient uptake (g C m^-3)
  PARAMETER(log_alpha_shade);  // log of self-shading coefficient (m^3 gC^-1)

  // Grazing
  PARAMETER(log_g_max);        // log of max zooplankton ingestion rate per Z biomass (day^-1)
  PARAMETER(log_K_P);          // log half-saturation for grazing (g C m^-3)
  PARAMETER(log_h_minus1);     // log(h - 1) where h >= 1
  PARAMETER(log_P_star);       // log of smooth grazing threshold P_star (g C m^-3)
  PARAMETER(log_k_thr);        // log of steepness k_thr for threshold ((g C m^-3)^-1)

  // Mortality and efficiencies
  PARAMETER(log_m_P);          // log P linear mortality (day^-1)
  PARAMETER(log_m_Z);          // log Z linear mortality (day^-1)
  PARAMETER(log_m2_Z);         // log quadratic Z mortality (m^3 gC^-1 day^-1)
  PARAMETER(logit_epsilon_P);  // logit of phytoplankton carbon-use efficiency (fraction 0–1)
  PARAMETER(logit_e_Z);        // logit of zooplankton assimilation efficiency (fraction 0–1)

  // Seasonality
  PARAMETER(a_season_grow);    // amplitude for growth modulation (dimensionless)
  PARAMETER(phi_grow);         // phase for growth modulation (radians)
  PARAMETER(a_season_graze);   // amplitude for grazing modulation (dimensionless)
  PARAMETER(phi_graze);        // phase for grazing modulation (radians)
  PARAMETER(T_season);         // seasonal period (days)

  // Observation error (log-space SDs)
  PARAMETER(log_sigma_N);
  PARAMETER(log_sigma_P);
  PARAMETER(log_sigma_Z);

  // Transform parameters to real scales
  Type mu_max       = exp(log_mu_max);
  Type K_N          = exp(log_K_N);
  Type alpha_shade  = exp(log_alpha_shade);

  Type g_max        = exp(log_g_max);
  Type K_P          = exp(log_K_P);
  Type h            = Type(1.0) + exp(log_h_minus1);
  Type P_star       = exp(log_P_star);
  Type k_thr        = exp(log_k_thr);

  Type m_P          = exp(log_m_P);
  Type m_Z          = exp(log_m_Z);
  Type m2_Z         = exp(log_m2_Z);
  Type epsilon_P    = inv_logit(logit_epsilon_P);
  Type e_Z          = inv_logit(logit_e_Z);

  Type sigmaN       = (exp(log_sigma_N) < sigma_min ? sigma_min : exp(log_sigma_N));
  Type sigmaP       = (exp(log_sigma_P) < sigma_min ? sigma_min : exp(log_sigma_P));
  Type sigmaZ       = (exp(log_sigma_Z) < sigma_min ? sigma_min : exp(log_sigma_Z));

  // State predictions
  vector<Type> N_pred(n);
  vector<Type> P_pred(n);
  vector<Type> Z_pred(n);

  // Initialize states using first observations as initial conditions (not reused thereafter)
  N_pred(0) = smooth_max(N_dat(0), min_pred, k_smax);
  P_pred(0) = smooth_max(P_dat(0), min_pred, k_smax);
  Z_pred(0) = smooth_max(Z_dat(0), min_pred, k_smax);

  // Process model: forward Euler integration using previous-step predictions only
  for (int i = 1; i < n; i++) {
    Type dt = Time(i) - Time(i - 1);
    if (!(dt > Type(0))) dt = Type(1); // fallback to 1 day if non-increasing or invalid

    // Previous-step predicted states
    Type N_prev = N_pred(i - 1);
    Type P_prev = P_pred(i - 1);
    Type Z_prev = Z_pred(i - 1);

    // Seasonal modifiers evaluated at previous time
    Type t_prev = Time(i - 1);
    Type grow_mod  = exp(a_season_grow  * sin(two_pi * t_prev / T_season + phi_grow));
    Type graze_mod = exp(a_season_graze * sin(two_pi * t_prev / T_season + phi_graze));

    // Limitations and rates
    Type fN   = N_prev / (K_N + N_prev + eps);                // Monod nutrient limitation
    Type fL   = Type(1) / (Type(1) + alpha_shade * P_prev);   // self-shading light proxy
    Type U    = mu_max * grow_mod * fN * fL * P_prev;         // phytoplankton carbon uptake (flux)

    // Grazing functional response with smooth threshold
    Type fP_num = pow(P_prev, h);
    Type fP_den = pow(K_P, h) + fP_num + eps;
    Type fP     = fP_num / fP_den;                            // Hill response
    Type thr    = inv_logit(k_thr * (P_prev - P_star));       // smooth on/off near P_star
    Type G      = g_max * graze_mod * thr * fP * Z_prev;      // ingestion flux (P -> Z)

    // Mortalities
    Type M_P   = m_P  * P_prev;
    Type M_Z   = m_Z  * Z_prev;
    Type M2_Z  = m2_Z * Z_prev * Z_prev;                      // quadratic closure

    // State derivatives (mass balanced)
    Type dN = -U + (Type(1) - epsilon_P) * U + (Type(1) - e_Z) * G + M_P + M_Z + M2_Z;
    Type dP =  epsilon_P * U - G - M_P;
    Type dZ =  e_Z * G - M_Z - M2_Z;

    // Euler update with smooth non-negativity enforcement
    N_pred(i) = smooth_max(N_prev + dt * dN, min_pred, k_smax);
    P_pred(i) = smooth_max(P_prev + dt * dP, min_pred, k_smax);
    Z_pred(i) = smooth_max(Z_prev + dt * dZ, min_pred, k_smax);
  }

  // Observation likelihood: lognormal errors on N, P, Z
  using namespace density;
  Type nll = 0.0;
  for (int i = 0; i < n; i++) {
    // floor to ensure positivity in log
    Type N_obs = smooth_max(N_dat(i), min_pred, k_smax);
    Type P_obs = smooth_max(P_dat(i), min_pred, k_smax);
    Type Z_obs = smooth_max(Z_dat(i), min_pred, k_smax);

    nll -= dnorm(log(N_obs), log(N_pred(i)), sigmaN, true);
    nll -= dnorm(log(P_obs), log(P_pred(i)), sigmaP, true);
    nll -= dnorm(log(Z_obs), log(Z_pred(i)), sigmaZ, true);
  }

  // Soft penalties for plausible ranges (example for m2_Z as documented)
  nll += bounds_penalty(m2_Z, Type(0.0), Type(5.0), pen_wt, pen_alpha);

  // Reports
  ADREPORT(mu_max);
  ADREPORT(K_N);
  ADREPORT(g_max);
  ADREPORT(K_P);
  ADREPORT(h);
  ADREPORT(P_star);
  ADREPORT(k_thr);
  ADREPORT(m_P);
  ADREPORT(m_Z);
  ADREPORT(m2_Z);
  REPORT(N_pred);
  REPORT(P_pred);
  REPORT(Z_pred);

  return nll;
}
