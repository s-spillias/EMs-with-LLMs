#include <TMB.hpp>

// Helper: softplus for smooth positivity (AD-compatible)
template<class Type>
Type softplus(Type x) {
  // Numerically stable softplus without log1p (works for AD types):
  // softplus(x) = log(1 + exp(-|x|)) + max(x, 0)
  Type zero = Type(0.0);
  Type one  = Type(1.0);
  Type ax   = CppAD::abs(x);
  return log(one + exp(-ax)) + CppAD::CondExpGt(x, zero, x, zero);
}

// Helper: smooth non-negative transform approximating max(x, 0) without kinks
template<class Type>
Type soft_relu(Type x, Type eps) {
  // Returns ~max(x,0) but smooth near 0; eps sets smoothness scale
  return (x + sqrt(x * x + eps)) / Type(2.0);
}

// Helper: safe division
template<class Type>
Type safediv(Type num, Type den, Type tiny) {
  return num / (den + tiny);
}

// Helper: smooth penalty if parameter outside [lo, hi]
template<class Type>
Type smooth_bound_penalty(Type x, Type lo, Type hi, Type scale) {
  // Zero-ish inside bounds; increases smoothly outside via softplus
  // Note: softplus of negative values is near zero, positive side penalizes out-of-bounds.
  return softplus((lo - x) / scale) + softplus((x - hi) / scale);
}

/*
Equations (per time step, Euler-forward with dt):

Let f_T = q10^((T_C - T_ref)/10)           [temperature modifier, dimensionless]
    f_I = I / (K_I + I)                    [light limitation, dimensionless]
    f_N = N / (K_N + N)                    [nutrient limitation, dimensionless]
    mu  = mu_max * f_T * f_I * f_N         [d^-1, realized phyto growth rate]
    g   = g_max * f_T * (P^h / (K_G^h + P^h))   [d^-1, grazing rate per Z]

Flows (g C m^-3 d^-1):
  1) Primary production:          U  = mu * P
  2) Grazing flux (ingestion):    G  = g * Z
  3) Z production (assim.):       Zg = e_Z * G
  4) Unassimilated to N:          Rg = (1 - e_Z) * G
  5) P mortality remineralized:   Rp = r_P * m_P * P
  6) Z mortality remineralized:   Rz = r_Z * m_Z * Z
  7) Z excretion to N:            Ex = ex_Z * Z
  8) Mixing supply to N:          Mx = k_mix * (N_star - N)

Dynamics:
  dN/dt = -U + Rg + Rp + Rz + Ex + Mx
  dP/dt =  U - G - m_P * P
  dZ/dt =  Zg - m_Z * Z - gamma_Z * Z^2

All states are kept non-negative using a smooth rectifier. Initial conditions:
  N_pred(0) = N_dat(0), P_pred(0) = P_dat(0), Z_pred(0) = Z_dat(0).
Observation model (for i = 0..T-1):
  log(N_dat[i]) ~ Normal(log(N_pred[i]), sigma_N) and similarly for P, Z.
*/

template<class Type>
Type objective_function<Type>::operator() () {
  // -----------------------------
  // Data
  // -----------------------------
  // Use the exact time variable name provided by the data layer: "Time"
  DATA_VECTOR(Time);   // Time in days, strictly increasing
  DATA_VECTOR(N_dat);  // Observed nutrient concentration (g C m^-3)
  DATA_VECTOR(P_dat);  // Observed phytoplankton concentration (g C m^-3)
  DATA_VECTOR(Z_dat);  // Observed zooplankton concentration (g C m^-3)

  int Tn = N_dat.size(); // Number of time points
  // Safety: all vectors should be same length
  if (P_dat.size() != Tn || Z_dat.size() != Tn || Time.size() != Tn) {
    error("All data vectors must have the same length.");
  }

  // -----------------------------
  // Parameters (all transformed to their natural scales where needed)
  // -----------------------------
  // Growth and limitation parameters
  PARAMETER(log_mu_max);    // log of maximum phyto growth rate (d^-1); expected ~ log(0.1-2 d^-1)
  PARAMETER(log_K_N);       // log of half-saturation for nutrient (g C m^-3); expected ~ log(0.01-0.5)
  PARAMETER(I);             // Irradiance proxy (W m^-2), treated as constant over period
  PARAMETER(log_K_I);       // log of light half-saturation (W m^-2)

  // Grazing parameters
  PARAMETER(log_g_max);     // log of max grazing rate per Z biomass (d^-1)
  PARAMETER(log_K_G);       // log of P half-saturation for grazing (g C m^-3)
  PARAMETER(h_grazing);     // Holling type III shape exponent h (dimensionless, >=1)

  // Efficiencies and losses
  PARAMETER(logit_e_Z);     // logit of Z assimilation efficiency (0..1), dimensionless
  PARAMETER(log_m_P);       // log of P linear mortality rate (d^-1)
  PARAMETER(log_m_Z);       // log of Z linear mortality rate (d^-1)
  PARAMETER(log_gamma_Z);   // log of Z quadratic self-limitation coefficient ((g C m^-3)^-1 d^-1)
  PARAMETER(logit_r_P);     // logit of fraction of P mortality remineralized to N (0..1)
  PARAMETER(logit_r_Z);     // logit of fraction of Z mortality remineralized to N (0..1)
  PARAMETER(log_ex_Z);      // log of Z excretion rate to N (d^-1)

  // Physical supply
  PARAMETER(log_k_mix);     // log of mixing rate (d^-1)
  PARAMETER(N_star);        // Deep/source nutrient concentration (g C m^-3)

  // Temperature modifier
  PARAMETER(log_q10);       // log of Q10 (dimensionless), e.g., log(2)
  PARAMETER(T_C);           // Ambient temperature (deg C)
  PARAMETER(T_ref);         // Reference temperature for Q10 (deg C)

  // Observation error (lognormal SDs)
  PARAMETER(log_sigma_N);   // log of observation SD on log-scale for N
  PARAMETER(log_sigma_P);   // log of observation SD on log-scale for P
  PARAMETER(log_sigma_Z);   // log of observation SD on log-scale for Z

  // -----------------------------
  // Transforms and constants
  // -----------------------------
  Type tiny = Type(1e-8);            // Small constant to avoid division by zero
  Type pos_eps = Type(1e-12);        // For smooth non-negativity
  Type pen_scale = Type(0.05);       // Scale for smooth bound penalties (larger = gentler)
  Type pen_weight = Type(10.0);      // Weight for penalties in NLL

  Type mu_max = exp(log_mu_max);     // d^-1
  Type K_N    = exp(log_K_N);        // g C m^-3
  Type K_I    = exp(log_K_I);        // W m^-2
  Type g_max  = exp(log_g_max);      // d^-1
  Type K_G    = exp(log_K_G);        // g C m^-3
  Type e_Z    = Type(1.0) / (Type(1.0) + exp(-logit_e_Z)); // (0,1)
  Type m_P    = exp(log_m_P);        // d^-1
  Type m_Z    = exp(log_m_Z);        // d^-1
  Type gamma_Z= exp(log_gamma_Z);    // (g C m^-3)^-1 d^-1
  Type r_P    = Type(1.0) / (Type(1.0) + exp(-logit_r_P)); // (0,1)
  Type r_Z    = Type(1.0) / (Type(1.0) + exp(-logit_r_Z)); // (0,1)
  Type ex_Z   = exp(log_ex_Z);       // d^-1
  Type k_mix  = exp(log_k_mix);      // d^-1
  Type q10    = exp(log_q10);        // dimensionless

  // Temperature and light modifiers
  // f_T applies to biological rates; f_I saturates with I
  Type f_T = pow(q10, (T_C - T_ref) / Type(10.0));           // dimensionless
  Type f_I = safediv(I, (K_I + I), tiny);                     // dimensionless, in (0,1)

  // Observation SDs with minimum floors for stability
  Type min_sd = Type(0.05); // Minimum SD on log-scale
  Type sigma_N = exp(log_sigma_N) + min_sd;
  Type sigma_P = exp(log_sigma_P) + min_sd;
  Type sigma_Z = exp(log_sigma_Z) + min_sd;

  // -----------------------------
  // State predictions
  // -----------------------------
  vector<Type> N_pred(Tn);
  vector<Type> P_pred(Tn);
  vector<Type> Z_pred(Tn);

  // Initial conditions from data (no leakage beyond t=0)
  N_pred(0) = N_dat(0);
  P_pred(0) = P_dat(0);
  Z_pred(0) = Z_dat(0);

  // -----------------------------
  // Likelihood
  // -----------------------------
  Type nll = Type(0);

  // Penalize parameter bounds smoothly (suggested biological ranges)
  // mu_max: [0.05, 2] d^-1
  nll += pen_weight * smooth_bound_penalty(mu_max, Type(0.05), Type(2.0), pen_scale);
  // K_N: [0.001, 1] g C m^-3
  nll += pen_weight * smooth_bound_penalty(K_N, Type(0.001), Type(1.0), pen_scale);
  // I: [0, 500] W m^-2
  nll += pen_weight * smooth_bound_penalty(I, Type(0.0), Type(500.0), pen_scale);
  // K_I: [1, 300] W m^-2
  nll += pen_weight * smooth_bound_penalty(K_I, Type(1.0), Type(300.0), pen_scale);
  // g_max: [0.05, 2] d^-1
  nll += pen_weight * smooth_bound_penalty(g_max, Type(0.05), Type(2.0), pen_scale);
  // K_G: [0.001, 1] g C m^-3
  nll += pen_weight * smooth_bound_penalty(K_G, Type(0.001), Type(1.0), pen_scale);
  // h_grazing: [1, 3]
  nll += pen_weight * smooth_bound_penalty(h_grazing, Type(1.0), Type(3.0), pen_scale);
  // e_Z: [0.3, 0.9]
  nll += pen_weight * smooth_bound_penalty(e_Z, Type(0.3), Type(0.9), pen_scale);
  // m_P: [0.001, 0.3] d^-1
  nll += pen_weight * smooth_bound_penalty(m_P, Type(0.001), Type(0.3), pen_scale);
  // m_Z: [0.001, 0.3] d^-1
  nll += pen_weight * smooth_bound_penalty(m_Z, Type(0.001), Type(0.3), pen_scale);
  // gamma_Z: [1e-4, 0.2] (g C m^-3)^-1 d^-1
  nll += pen_weight * smooth_bound_penalty(gamma_Z, Type(1e-4), Type(0.2), pen_scale);
  // r_P: [0.3, 1]
  nll += pen_weight * smooth_bound_penalty(r_P, Type(0.3), Type(1.0), pen_scale);
  // r_Z: [0.3, 1]
  nll += pen_weight * smooth_bound_penalty(r_Z, Type(0.3), Type(1.0), pen_scale);
  // ex_Z: [0.0, 0.2] d^-1
  nll += pen_weight * smooth_bound_penalty(ex_Z, Type(0.0), Type(0.2), pen_scale);
  // k_mix: [0.0, 0.5] d^-1
  nll += pen_weight * smooth_bound_penalty(k_mix, Type(0.0), Type(0.5), pen_scale);
  // N_star: [0.0, 2.0] g C m^-3
  nll += pen_weight * smooth_bound_penalty(N_star, Type(0.0), Type(2.0), pen_scale);
  // q10: [1.3, 3.0]
  nll += pen_weight * smooth_bound_penalty(q10, Type(1.3), Type(3.0), pen_scale);
  // T_C, T_ref: [0, 35] deg C
  nll += pen_weight * smooth_bound_penalty(T_C, Type(0.0), Type(35.0), pen_scale);
  nll += pen_weight * smooth_bound_penalty(T_ref, Type(0.0), Type(35.0), pen_scale);

  // -----------------------------
  // Time stepping
  // -----------------------------
  for (int i = 1; i < Tn; i++) {
    Type dt = Time(i) - Time(i - 1);
    // Enforce positive dt smoothly
    if (dt <= Type(0)) dt = tiny;

    // State at previous step (predictions only—no data leakage)
    Type Np = N_pred(i - 1);
    Type Pp = P_pred(i - 1);
    Type Zp = Z_pred(i - 1);

    // Limitation functions (use small constants for stability)
    Type f_N = safediv(Np, (K_N + Np), tiny);                                // [0,1]
    Type mu  = mu_max * f_T * f_I * f_N;                                     // d^-1
    Type holl_num = pow(Pp + tiny, h_grazing);                               // P^h
    Type holl_den = pow(K_G + tiny, h_grazing) + holl_num;                   // K^h + P^h
    Type g_rate   = g_max * f_T * safediv(holl_num, holl_den, tiny);         // d^-1

    // Fluxes
    Type U  = mu * Pp;                   // Primary production (g C m^-3 d^-1)
    Type G  = g_rate * Zp;               // Grazing ingestion (g C m^-3 d^-1)
    Type Zg = e_Z * G;                   // Z growth (g C m^-3 d^-1)
    Type Rg = (Type(1.0) - e_Z) * G;     // Unassimilated to N
    Type Rp = r_P * m_P * Pp;            // P mortality remineralized to N
    Type Rz = r_Z * m_Z * Zp;            // Z mortality remineralized to N
    Type Ex = ex_Z * Zp;                 // Z excretion to N
    Type Mx = k_mix * (N_star - Np);     // Mixing supply to N

    // Euler updates
    Type dN = -U + Rg + Rp + Rz + Ex + Mx;
    Type dP =  U - G - m_P * Pp;
    Type dZ =  Zg - m_Z * Zp - gamma_Z * Zp * Zp;

    Type N_next_raw = Np + dt * dN;
    Type P_next_raw = Pp + dt * dP;
    Type Z_next_raw = Zp + dt * dZ;

    // Smooth non-negativity
    N_pred(i) = soft_relu(N_next_raw, pos_eps);
    P_pred(i) = soft_relu(P_next_raw, pos_eps);
    Z_pred(i) = soft_relu(Z_next_raw, pos_eps);
  }

  // -----------------------------
  // Observation likelihood (lognormal)
  // -----------------------------
  for (int i = 0; i < Tn; i++) {
    // Add tiny offsets to ensure positivity inside logs
    Type lnN_obs = log(N_dat(i) + tiny);
    Type lnP_obs = log(P_dat(i) + tiny);
    Type lnZ_obs = log(Z_dat(i) + tiny);

    Type lnN_pred = log(N_pred(i) + tiny);
    Type lnP_pred = log(P_pred(i) + tiny);
    Type lnZ_pred = log(Z_pred(i) + tiny);

    nll -= dnorm(lnN_obs, lnN_pred, sigma_N, true);
    nll -= dnorm(lnP_obs, lnP_pred, sigma_P, true);
    nll -= dnorm(lnZ_obs, lnZ_pred, sigma_Z, true);
  }

  // -----------------------------
  // Reporting
  // -----------------------------
  REPORT(N_pred); // Model predictions for Nutrient (g C m^-3)
  REPORT(P_pred); // Model predictions for Phytoplankton (g C m^-3)
  REPORT(Z_pred); // Model predictions for Zooplankton (g C m^-3)

  return nll;
}
