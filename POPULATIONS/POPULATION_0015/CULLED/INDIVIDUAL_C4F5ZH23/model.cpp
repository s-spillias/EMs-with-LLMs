#include <TMB.hpp>

// Template Model Builder (TMB) NPZ model
// All parameters and equations are documented inline for clarity and reproducibility.

// Helper: smooth floor using softplus; returns approximately max(x, floor) but smooth and differentiable
template<class Type>
Type smooth_floor(Type x, Type floor_val, Type k = Type(20)) {
  // softplus_k(z) = log(1 + exp(k*z)) / k; shift by floor_val; use CppAD::log for AD types
  Type z = x - floor_val;                                // difference from floor (same units as x)
  return floor_val + CppAD::log(Type(1) + exp(k * z)) / k; // smooth floor; avoids log1p(double)
}

// Helper: soft penalty outside [lo, hi] using softplus; near-zero inside, grows smoothly outside
template<class Type>
Type soft_bound_penalty(Type x, Type lo, Type hi, Type softness = Type(0.1)) {
  // Unconditional smooth penalties to avoid AD/double issues
  Type pen = Type(0);                                                        // initialize penalty (dimensionless)
  pen += CppAD::log(Type(1) + exp((lo - x) / softness));                     // lower-bound softplus penalty
  pen += CppAD::log(Type(1) + exp((x - hi) / softness));                     // upper-bound softplus penalty
  return pen;
}

template<class Type>
Type objective_function<Type>::operator() ()
{
  using CppAD::log;
  using CppAD::sin;

  // Small constants for numerical stability
  const Type eps = Type(1e-8);    // to avoid division by zero and log(0)
  const Type sd_min = Type(0.05); // minimum log-space SD for observation model to prevent collapse
  const Type pen_w = Type(10.0);  // global weight for soft bound penalties
  const Type two_pi = Type(6.2831853071795864769); // 2π constant for seasonal cycle

  // DATA ----------------------------------------------------------------------
  // Use 'Time' to match the sanitized CSV time column name.
  DATA_VECTOR(Time);  // time in days; corresponds to CSV column name 'Time'
  DATA_VECTOR(N_dat); // observed nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat); // observed phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat); // observed zooplankton concentration (g C m^-3)

  int n = N_dat.size();   // number of observations

  // PARAMETERS (transformed for positivity/bounds) ----------------------------
  PARAMETER(log_g_max);        // log of max specific P growth rate (day^-1); typical 0.1-2 day^-1
  PARAMETER(log_K_N);          // log half-saturation for N uptake (g C m^-3); typical 0.005-1
  PARAMETER(log_m_P);          // log linear P mortality (day^-1)
  PARAMETER(log_g_Z);          // log max zooplankton grazing rate (day^-1)
  PARAMETER(log_K_P);          // log scale (half-sat) for Holling III grazing (g C m^-3)
  PARAMETER(log_m_Z_lin);      // log linear Z mortality (day^-1)
  PARAMETER(log_gamma_Z);      // log quadratic Z mortality coefficient (m^3 gC^-1 day^-1)
  PARAMETER(logit_e_Z);        // logit zooplankton assimilation efficiency (0-1)
  PARAMETER(logit_y_P);        // logit phytoplankton growth yield (0-1)
  PARAMETER(logit_r_excr);     // logit remin fraction of unassimilated grazing to N (0-1)
  PARAMETER(logit_r_mort_P);   // logit fraction of P mortality remineralized (0-1)
  PARAMETER(logit_r_mort_Z);   // logit fraction of Z mortality remineralized (0-1)
  PARAMETER(log_s_N);          // log exchange rate with deep nutrient (day^-1)
  PARAMETER(log_N_deep);       // log deep nutrient concentration (g C m^-3)
  PARAMETER(logit_light_amp);  // logit amplitude of light modulation (0-1)
  PARAMETER(light_phase);      // phase shift (days) of seasonal light
  PARAMETER(log_photoperiod_days); // log period (days) of seasonal cycle
  PARAMETER(log_light_k);      // log light logistic steepness (dimensionless)
  PARAMETER(log_sigma_N);      // log observation SD for log(N) (lognormal)
  PARAMETER(log_sigma_P);      // log observation SD for log(P) (lognormal)
  PARAMETER(log_sigma_Z);      // log observation SD for log(Z) (lognormal)

  // Transform to natural scales ------------------------------------------------
  Type g_max = exp(log_g_max);              // day^-1
  Type K_N = exp(log_K_N);                  // g C m^-3
  Type m_P = exp(log_m_P);                  // day^-1
  Type g_Z = exp(log_g_Z);                  // day^-1
  Type K_P = exp(log_K_P);                  // g C m^-3
  Type m_Z_lin = exp(log_m_Z_lin);          // day^-1
  Type gamma_Z = exp(log_gamma_Z);          // m^3 gC^-1 day^-1
  Type e_Z = invlogit(logit_e_Z);           // 0-1 (use TMB-provided invlogit)
  Type y_P = invlogit(logit_y_P);           // 0-1
  Type r_excr = invlogit(logit_r_excr);     // 0-1
  Type r_mort_P = invlogit(logit_r_mort_P); // 0-1
  Type r_mort_Z = invlogit(logit_r_mort_Z); // 0-1
  Type s_N = exp(log_s_N);                  // day^-1
  Type N_deep = exp(log_N_deep);            // g C m^-3
  Type light_amp = invlogit(logit_light_amp); // 0-1
  Type photoperiod_days = exp(log_photoperiod_days); // days
  Type light_k = exp(log_light_k);          // dimensionless
  Type sigma_N = exp(log_sigma_N);          // log-space SD
  Type sigma_P = exp(log_sigma_P);          // log-space SD
  Type sigma_Z = exp(log_sigma_Z);          // log-space SD

  // Soft penalties for biologically plausible ranges --------------------------
  // Ranges reflect common NPZ literature; penalties are smooth and do not hard-constrain optimization.
  Type nll = Type(0);
  nll += pen_w * soft_bound_penalty(g_max, Type(0.05), Type(3.0));          // P growth rate
  nll += pen_w * soft_bound_penalty(K_N, Type(0.001), Type(1.0));           // N half-sat
  nll += pen_w * soft_bound_penalty(m_P, Type(0.001), Type(0.5));           // P mortality
  nll += pen_w * soft_bound_penalty(g_Z, Type(0.05), Type(3.0));            // max grazing
  nll += pen_w * soft_bound_penalty(K_P, Type(0.001), Type(1.0));           // grazing half-sat
  nll += pen_w * soft_bound_penalty(m_Z_lin, Type(0.001), Type(0.5));       // Z linear mort
  nll += pen_w * soft_bound_penalty(gamma_Z, Type(0.0), Type(2.0));         // Z quadratic mort
  nll += pen_w * soft_bound_penalty(e_Z, Type(0.3), Type(0.95));            // assimilation
  nll += pen_w * soft_bound_penalty(y_P, Type(0.3), Type(1.0));             // yield
  nll += pen_w * soft_bound_penalty(r_excr, Type(0.3), Type(1.0));          // excretion remin
  nll += pen_w * soft_bound_penalty(r_mort_P, Type(0.3), Type(1.0));        // P mort remin
  nll += pen_w * soft_bound_penalty(r_mort_Z, Type(0.3), Type(1.0));        // Z mort remin
  nll += pen_w * soft_bound_penalty(s_N, Type(0.0), Type(0.5));             // mixing rate
  nll += pen_w * soft_bound_penalty(N_deep, Type(0.05), Type(3.0));         // deep nutrient
  nll += pen_w * soft_bound_penalty(light_amp, Type(0.0), Type(0.95));      // light amplitude
  nll += pen_w * soft_bound_penalty(photoperiod_days, Type(20.0), Type(400.0)); // period
  nll += pen_w * soft_bound_penalty(light_k, Type(0.2), Type(10.0));        // light steepness
  nll += pen_w * soft_bound_penalty(sigma_N, sd_min, Type(1.0));            // obs SDs
  nll += pen_w * soft_bound_penalty(sigma_P, sd_min, Type(1.0));
  nll += pen_w * soft_bound_penalty(sigma_Z, sd_min, Type(1.0));

  // STATE PREDICTIONS ----------------------------------------------------------
  vector<Type> N_pred(n); // predicted nutrient concentration (g C m^-3)
  vector<Type> P_pred(n); // predicted phytoplankton concentration (g C m^-3)
  vector<Type> Z_pred(n); // predicted zooplankton concentration (g C m^-3)

  // Initialize with observed initial conditions (avoid data leakage during propagation)
  N_pred(0) = N_dat(0); // IC from data
  P_pred(0) = P_dat(0); // IC from data
  Z_pred(0) = Z_dat(0); // IC from data

  // Diagnostics (optional): realized growth modifier and grazing
  vector<Type> fN_t(n);      // nutrient limitation factor
  vector<Type> fL_t(n);      // light limitation factor
  vector<Type> Graz_t(n);    // grazing flux (g C m^-3 day^-1)
  vector<Type> mu_t(n);      // realized specific growth rate (day^-1)

  // TIME INTEGRATION (forward Euler with variable dt)
  for (int i = 1; i < n; i++) {
    // time step (days), ensure strictly positive
    Type dt = Time(i) - Time(i - 1);
    dt = CppAD::CondExpGt(dt, eps, dt, eps);

    // State at previous time step (predictions only; no data leakage)
    Type N_prev = N_pred(i - 1);
    Type P_prev = P_pred(i - 1);
    Type Z_prev = Z_pred(i - 1);

    // 1) Resource limitations and environmental modifiers ---------------------
    // Seasonal light modifier in [1 - light_amp, 1]; smoothly varies with sin() via logistic
    Type s = sin(two_pi * ((Time(i - 1) - light_phase) / photoperiod_days));
    Type f_light = (Type(1.0) - light_amp) + light_amp * invlogit(light_k * s); // 0 < f_light <= 1
    fL_t(i - 1) = f_light;

    // Nutrient limitation (Michaelis-Menten)
    Type f_nut = N_prev / (K_N + N_prev + eps); // in [0,1)
    fN_t(i - 1) = f_nut;

    // Realized specific P growth rate
    Type mu = g_max * f_nut * f_light; // day^-1
    mu_t(i - 1) = mu;

    // 2) Trophic interactions (Holling type III grazing) ----------------------
    // Functional response f(P) = P^2 / (K_P^2 + P^2)
    Type P2 = P_prev * P_prev;
    Type KP2 = K_P * K_P;
    Type f_graz = P2 / (KP2 + P2 + eps);
    Type Graz = g_Z * f_graz * Z_prev; // g C m^-3 day^-1
    Graz_t(i - 1) = Graz;

    // 3) Flux definitions ------------------------------------------------------
    // Uptake from nutrient pool (accounts for yield y_P)
    Type Uptake = (mu * P_prev) / (y_P + eps); // g C m^-3 day^-1 removed from N
    // Phytoplankton mortality (linear)
    Type MortP = m_P * P_prev; // g C m^-3 day^-1
    // Zooplankton mortality (linear + quadratic)
    Type MortZ = m_Z_lin * Z_prev + gamma_Z * Z_prev * Z_prev; // g C m^-3 day^-1
    // Remineralization sources to N
    Type Remin_excr = r_excr * ((Type(1.0) - e_Z) * Graz); // unassimilated grazing to N
    Type Remin_P = r_mort_P * MortP;                       // fraction of P mortality to N
    Type Remin_Z = r_mort_Z * MortZ;                       // fraction of Z mortality to N
    // Vertical exchange with deep nutrient
    Type MixN = s_N * (N_deep - N_prev); // g C m^-3 day^-1

    // 4) State derivatives -----------------------------------------------------
    Type dPdt = (mu * P_prev) - Graz - MortP;                 // P dynamics
    Type dZdt = (e_Z * Graz) - MortZ;                         // Z dynamics
    Type dNdt = -Uptake + Remin_excr + Remin_P + Remin_Z + MixN; // N dynamics

    // 5) Euler update with smooth floor to maintain non-negativity ------------
    Type N_next = smooth_floor(N_prev + dt * dNdt, eps);
    Type P_next = smooth_floor(P_prev + dt * dPdt, eps);
    Type Z_next = smooth_floor(Z_prev + dt * dZdt, eps);

    // Store predictions
    N_pred(i) = N_next;
    P_pred(i) = P_next;
    Z_pred(i) = Z_next;
  }

  // Fill diagnostics last entries safely
  if (n >= 2) {
    fN_t(n - 1) = fN_t(n - 2);
    fL_t(n - 1) = fL_t(n - 2);
    Graz_t(n - 1) = Graz_t(n - 2);
    mu_t(n - 1) = mu_t(n - 2);
  } else if (n == 1) {
    // Compute diagnostics at t0 from initial conditions
    Type t0 = Time(0);
    Type s0 = sin(two_pi * ((t0 - light_phase) / photoperiod_days));
    Type f_light0 = (Type(1.0) - light_amp) + light_amp * invlogit(light_k * s0);
    fL_t(0) = f_light0;
    Type f_nut0 = N_pred(0) / (K_N + N_pred(0) + eps);
    fN_t(0) = f_nut0;
    mu_t(0) = g_max * f_nut0 * f_light0;
    Type P20 = P_pred(0) * P_pred(0);
    Type KP20 = K_P * K_P;
    Type f_graz0 = P20 / (KP20 + P20 + eps);
    Graz_t(0) = g_Z * f_graz0 * Z_pred(0);
  }

  // LIKELIHOOD (lognormal for strictly positive data) -------------------------
  // Apply minimum SD to prevent numerical issues for small values
  Type sN = (sigma_N > sd_min ? sigma_N : sd_min);
  Type sP = (sigma_P > sd_min ? sigma_P : sd_min);
  Type sZ = (sigma_Z > sd_min ? sigma_Z : sd_min);

  for (int i = 0; i < n; i++) {
    // Always include all observations (including t0)
    nll -= dnorm(log(N_dat(i) + eps), log(N_pred(i) + eps), sN, true);
    nll -= dnorm(log(P_dat(i) + eps), log(P_pred(i) + eps), sP, true);
    nll -= dnorm(log(Z_dat(i) + eps), log(Z_pred(i) + eps), sZ, true);
  }

  // REPORTING -----------------------------------------------------------------
  REPORT(N_pred); // predicted nutrient vector
  REPORT(P_pred); // predicted phytoplankton vector
  REPORT(Z_pred); // predicted zooplankton vector
  REPORT(fN_t);   // nutrient limitation over time
  REPORT(fL_t);   // light modifier over time
  REPORT(Graz_t); // grazing flux over time
  REPORT(mu_t);   // realized P growth rate over time

  // Also report key parameters on natural scales for interpretability
  REPORT(g_max);
  REPORT(K_N);
  REPORT(m_P);
  REPORT(g_Z);
  REPORT(K_P);
  REPORT(m_Z_lin);
  REPORT(gamma_Z);
  REPORT(e_Z);
  REPORT(y_P);
  REPORT(r_excr);
  REPORT(r_mort_P);
  REPORT(r_mort_Z);
  REPORT(s_N);
  REPORT(N_deep);
  REPORT(light_amp);
  REPORT(photoperiod_days);
  REPORT(light_k);
  REPORT(sigma_N);
  REPORT(sigma_P);
  REPORT(sigma_Z);

  // MODEL EQUATIONS (documentation)
  // 1) f_light(t) = (1 - A) + A * invlogit(k * sin(2π (t - φ) / T)), A ∈ (0,1), k > 0
  // 2) f_nut(N)   = N / (K_N + N)
  // 3) μ(t)       = g_max * f_light(t) * f_nut(N)
  // 4) f_graz(P)  = P^2 / (K_P^2 + P^2)  (Holling III)
  // 5) Graz       = g_Z * f_graz(P) * Z
  // 6) Uptake     = (μ * P) / y_P
  // 7) MortP      = m_P * P
  // 8) MortZ      = m_Z_lin * Z + γ_Z * Z^2
  // 9) Remin      = r_excr * (1 - e_Z) * Graz + r_mort_P * MortP + r_mort_Z * MortZ
  // 10) MixN      = s_N * (N_deep - N)
  // 11) dP/dt     = μ P - Graz - MortP
  // 12) dZ/dt     = e_Z Graz - MortZ
  // 13) dN/dt     = -Uptake + Remin + MixN
  // 14) Euler     : X(t+dt) = smooth_floor( X(t) + dt * dX/dt, ε )

  return nll;
}
