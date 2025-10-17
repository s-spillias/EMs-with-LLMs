#include <TMB.hpp>

// Helper: softplus for smooth non-negativity (AD-safe, numerically stable)
template <class Type>
Type softplus(Type x) {
  Type zero = Type(0.0);
  // max_part = max(x, 0) using AD-safe conditional
  Type max_part = CppAD::CondExpGt(x, zero, x, zero);
  // abs_x = |x| using AD-safe conditional
  Type abs_x = CppAD::CondExpGt(x, zero, x, -x);
  // softplus(x) = max(x,0) + log(1 + exp(-|x|)) to avoid overflow
  return max_part + log(Type(1.0) + exp(-abs_x));
}

// Helper: logistic and logit with small eps for stability
template <class Type>
Type inv_logit_eps(Type x, Type eps) {
  return Type(1.0) / (Type(1.0) + exp(-x)) * (Type(1.0) - Type(2.0) * eps) + eps; // maps R -> (eps, 1-eps)
}

template <class Type>
Type logit_clip01(Type p, Type eps) {
  // Smoothly keep p inside (eps, 1-eps) before logit
  Type pe = (Type(1.0) - Type(2.0) * eps) * p + eps;
  pe = CppAD::CondExpLt(pe, eps, eps, pe);
  pe = CppAD::CondExpGt(pe, Type(1.0) - eps, Type(1.0) - eps, pe);
  return log(pe + eps) - log(Type(1.0) - pe + eps);
}

// Helper: smooth ReLU via softplus scaled by k
template <class Type>
Type smooth_relu(Type x, Type k) {
  return softplus(x / k) * k; // smooth approximation to max(0, x)
}

// Helper: add smooth penalty for bounds [lo, hi]
template <class Type>
Type bound_penalty(Type x, Type lo, Type hi, Type scale, Type k) {
  Type pen_low = smooth_relu(lo - x, k);
  Type pen_high = smooth_relu(x - hi, k);
  return scale * (pen_low * pen_low + pen_high * pen_high);
}

/*
Numbered model equations (discrete annual steps; t = 0..T-1):

State variables (predicted):
A_pred(t): Adult COTS density (individuals m^-2)
fast_pred(t): Fast coral cover (%), Acropora
slow_pred(t): Slow coral cover (%), Faviidae/Porites

Forcings:
sst_dat(t): Sea-surface temperature (°C)
cotsimm_dat(t): External larval immigration (individuals m^-2 yr^-1)

Auxiliary definitions:
F_t = fast_pred(t)/100, S_t = slow_pred(t)/100  (proportions)
P_t = w_fast*F_t + w_slow*S_t                  (food index; dimensionless)
S_env,t = 1 / (1 + exp(-k_T*(sst_dat(t) - T_thr)))  (environmental larval survival trigger; dimensionless)
A_allee(A) = A / (A + A_Allee)                 (smooth Allee; dimensionless)
food_sat(P) = P / (K_food + P)                 (Monod saturation; dimensionless)
m_T(t) = exp(gamma_mT * (sst_dat(t) - T_m_ref)) (temperature multiplier on mortality; dimensionless)
g_T(t) = exp(-beta_bleach * softplus(sst_dat(t) - T_bleach)) (bleaching penalty on coral growth; dimensionless)
Multi-prey Holling type II denominator:
H_t = h_hand + q_fast*F_t + q_slow*S_t

Process model (using previous time step values; t>=1):
(1) Recruitment precursor: R_t = r_max * S_env,t-1 * food_sat(P_t-1) * A_allee(A_pred(t-1)) * A_pred(t-1)
(2) Beverton–Holt recruitment: R_BH,t = R_t / (1 + A_pred(t-1)/K_A)
(3) Adult survival: S_A,t = exp(-m_base * m_T(t-1))
(4) COTS update: A_pred(t) = softplus( A_pred(t-1)*S_A,t + e_conv*R_BH,t + imm_eff*cotsimm_dat(t-1) )
(5) Per-adult coral consumption rates (proportion yr^-1):
    C_fast,t = c_max * A_pred(t-1) * (q_fast*F_t-1) / (H_t-1 + 1e-8)
    C_slow,t = c_max * A_pred(t-1) * (q_slow*S_t-1) / (H_t-1 + 1e-8)
(6) Coral logistic growth with competition and bleaching:
    G_fast,t = g_T(t-1) * g_fast * F_t-1 * (1 - (F_t-1 + alpha_fs*S_t-1))
    G_slow,t = g_T(t-1) * g_slow * S_t-1 * (1 - (S_t-1 + alpha_sf*F_t-1))
(7) Coral updates in logit space to keep bounds (smooth):
    zF_t = logit(F_t-1) + (G_fast,t - C_fast,t)
    zS_t = logit(S_t-1) + (G_slow,t - C_slow,t)
    F_t   = inv_logit_eps(zF_t, eps_prop)
    S_t   = inv_logit_eps(zS_t, eps_prop)
    fast_pred(t) = 100 * F_t
    slow_pred(t) = 100 * S_t

Observation models (use all observations):
(8) log(cots_dat(t)) ~ Normal( log(cots_pred(t) + 1e-8), sqrt(sd_log_cots^2 + sd_min^2) ), log-normal for strictly positive COTS
(9) logit(fast_dat(t)/100) ~ Normal( logit(fast_pred(t)/100), sqrt(sd_logit_fast^2 + sd_min^2) ), logit-normal for % cover
(10) logit(slow_dat(t)/100) ~ Normal( logit(slow_pred(t)/100), sqrt(sd_logit_slow^2 + sd_min^2) ), logit-normal for % cover
*/

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ---- Data ----
  DATA_VECTOR(Year);           // observation year (yr); used for alignment and reporting
  DATA_VECTOR(cots_dat);       // observed adult COTS (individuals m^-2)
  DATA_VECTOR(fast_dat);       // observed fast coral cover (%)
  DATA_VECTOR(slow_dat);       // observed slow coral cover (%)
  DATA_VECTOR(sst_dat);        // sea-surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);    // larval immigration (individuals m^-2 yr^-1)

  int n = cots_dat.size();     // number of time steps

  // ---- Parameters (with units and guidance) ----
  PARAMETER(r_max);        // year^-1; max per-capita larval production rate; informed by outbreak fecundity; start from literature or initial estimate
  PARAMETER(K_food);       // proportion (0-1); half-saturation constant for food limitation; initial estimate from coral cover scales
  PARAMETER(A_Allee);      // individuals m^-2; Allee scale for reproduction; initial estimate tuned to outbreak onset
  PARAMETER(K_A);          // individuals m^-2; Beverton–Holt density scale for recruitment saturation; initial estimate from peak densities
  PARAMETER(m_base);       // year^-1; baseline adult mortality rate; literature/initial estimate

  PARAMETER(k_T);          // °C^-1; steepness of SST-trigger on larval survival (logistic); initial estimate
  PARAMETER(T_thr);        // °C; SST threshold for boosting larval survival; initial estimate/literature

  PARAMETER(e_conv);       // individuals per unit (dimensionless wrt process units); conversion efficiency from recruits to new adults; initial estimate
  PARAMETER(imm_eff);      // dimensionless; efficiency scaling for external larval immigration assimilation; initial estimate

  PARAMETER(c_max);        // proportion per (ind*yr); max area cleared per adult per year; literature/initial estimate
  PARAMETER(h_hand);       // proportion; handling/time-scale term in multi-prey Type II denominator; initial estimate
  PARAMETER(q_fast);       // dimensionless; vulnerability/selectivity weight for fast coral; literature/initial estimate
  PARAMETER(q_slow);       // dimensionless; vulnerability/selectivity weight for slow coral; literature/initial estimate
  PARAMETER(w_fast);       // dimensionless; food quality weight of fast coral for COTS fecundity; initial estimate
  PARAMETER(w_slow);       // dimensionless; food quality weight of slow coral for COTS fecundity; initial estimate

  PARAMETER(g_fast);       // year^-1; intrinsic growth rate of fast coral; literature (Acropora) or initial estimate
  PARAMETER(g_slow);       // year^-1; intrinsic growth rate of slow coral; literature (Faviidae/Porites) or initial estimate
  PARAMETER(alpha_fs);     // dimensionless; competition impact of slow on fast (space preemption); initial estimate
  PARAMETER(alpha_sf);     // dimensionless; competition impact of fast on slow; initial estimate

  PARAMETER(T_bleach);     // °C; onset temperature for bleaching penalty; literature/initial estimate
  PARAMETER(beta_bleach);  // °C^-1; strength of bleaching penalty on coral growth; initial estimate

  PARAMETER(gamma_mT);     // °C^-1; temperature sensitivity for multiplicative mortality of COTS; initial estimate
  PARAMETER(T_m_ref);      // °C; reference temperature for COTS mortality baseline; initial estimate

  // Observation error standard deviations (minimum enforced in likelihood)
  PARAMETER(sd_log_cots);      // dimensionless; SD on log-scale for COTS abundance (log-normal); initial estimate
  PARAMETER(sd_logit_fast);    // dimensionless; SD on logit-scale for fast coral % (logit-normal); initial estimate
  PARAMETER(sd_logit_slow);    // dimensionless; SD on logit-scale for slow coral % (logit-normal); initial estimate

  // ---- Numerical stability constants ----
  Type tiny = Type(1e-8);          // small number to avoid division by zero
  Type eps_prop = Type(1e-6);      // bounds for proportions (0,1)
  Type sd_min = Type(0.05);        // minimum SD for likelihoods
  Type pen_scale = Type(100.0);    // penalty scale for bounds
  Type pen_k = Type(10.0);         // softness of smooth ReLU

  // ---- Smooth parameter bounds via penalties (see parameters.json for same ranges) ----
  Type nll = 0.0;
  nll += bound_penalty(r_max,       Type(0.0),  Type(20.0), pen_scale, pen_k);
  nll += bound_penalty(K_food,      Type(1e-3), Type(1.0),  pen_scale, pen_k);
  nll += bound_penalty(A_Allee,     Type(0.0),  Type(5.0),  pen_scale, pen_k);
  nll += bound_penalty(K_A,         Type(0.1),  Type(50.0), pen_scale, pen_k);
  nll += bound_penalty(m_base,      Type(0.0),  Type(3.0),  pen_scale, pen_k);

  nll += bound_penalty(k_T,         Type(0.0),  Type(5.0),  pen_scale, pen_k);
  nll += bound_penalty(T_thr,       Type(23.0), Type(32.0), pen_scale, pen_k);

  nll += bound_penalty(e_conv,      Type(0.0),  Type(5.0),  pen_scale, pen_k);
  nll += bound_penalty(imm_eff,     Type(0.0),  Type(2.0),  pen_scale, pen_k);

  nll += bound_penalty(c_max,       Type(0.0),  Type(5.0),  pen_scale, pen_k);
  nll += bound_penalty(h_hand,      Type(1e-3), Type(5.0),  pen_scale, pen_k);
  nll += bound_penalty(q_fast,      Type(0.0),  Type(10.0), pen_scale, pen_k);
  nll += bound_penalty(q_slow,      Type(0.0),  Type(10.0), pen_scale, pen_k);
  nll += bound_penalty(w_fast,      Type(0.0),  Type(5.0),  pen_scale, pen_k);
  nll += bound_penalty(w_slow,      Type(0.0),  Type(5.0),  pen_scale, pen_k);

  nll += bound_penalty(g_fast,      Type(0.0),  Type(2.0),  pen_scale, pen_k);
  nll += bound_penalty(g_slow,      Type(0.0),  Type(1.0),  pen_scale, pen_k);
  nll += bound_penalty(alpha_fs,    Type(0.0),  Type(2.0),  pen_scale, pen_k);
  nll += bound_penalty(alpha_sf,    Type(0.0),  Type(2.0),  pen_scale, pen_k);

  nll += bound_penalty(T_bleach,    Type(26.0), Type(33.0), pen_scale, pen_k);
  nll += bound_penalty(beta_bleach, Type(0.0),  Type(2.0),  pen_scale, pen_k);

  nll += bound_penalty(gamma_mT,    Type(0.0),  Type(1.0),  pen_scale, pen_k);
  nll += bound_penalty(T_m_ref,     Type(20.0), Type(32.0), pen_scale, pen_k);

  nll += bound_penalty(sd_log_cots,   Type(0.05), Type(2.0), pen_scale, pen_k);
  nll += bound_penalty(sd_logit_fast, Type(0.05), Type(1.0), pen_scale, pen_k);
  nll += bound_penalty(sd_logit_slow, Type(0.05), Type(1.0), pen_scale, pen_k);

  // ---- Prediction vectors ----
  vector<Type> cots_pred(n);  // Adult COTS predicted (individuals m^-2)
  vector<Type> fast_pred(n);  // Fast coral predicted (%)
  vector<Type> slow_pred(n);  // Slow coral predicted (%)

  // Initialize predictions with first observed values to anchor initial conditions (no data leakage beyond t=0)
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // ---- Forward simulation ----
  for (int t = 1; t < n; t++) {
    // Previous states
    Type A_prev = cots_pred(t-1);                    // individuals m^-2
    Type F_prev = fast_pred(t-1) / Type(100.0);      // proportion
    Type S_prev = slow_pred(t-1) / Type(100.0);      // proportion

    // Clamp to (eps, 1-eps) smoothly for stability in logit
    F_prev = (Type(1.0) - Type(2.0) * eps_prop) * F_prev + eps_prop;
    S_prev = (Type(1.0) - Type(2.0) * eps_prop) * S_prev + eps_prop;

    // Environmental modifiers (use previous time step forcings to avoid leakage)
    Type sst_prev = sst_dat(t-1);
    Type S_env = Type(1.0) / (Type(1.0) + exp(-k_T * (sst_prev - T_thr)));                    // larval survival trigger
    Type m_T = exp(gamma_mT * (sst_prev - T_m_ref));                                           // mortality multiplier
    Type g_T = exp(-beta_bleach * softplus(sst_prev - T_bleach));                              // bleaching penalty

    // Food limitation and Allee effects
    Type P_prev = w_fast * F_prev + w_slow * S_prev;                                           // food index
    Type food_sat = P_prev / (K_food + P_prev + tiny);                                         // Monod saturation
    Type A_allee_fac = A_prev / (A_prev + A_Allee + tiny);                                     // Allee factor

    // Recruitment with Beverton–Holt saturation
    Type R_t = r_max * S_env * food_sat * A_allee_fac * A_prev;                                // pre-saturation recruits
    Type R_BH = R_t / (Type(1.0) + A_prev / (K_A + tiny));                                     // Beverton–Holt

    // Adult update (survival + recruits + immigration), bounded smoothly non-negative
    Type surv = exp(-m_base * m_T);
    Type A_next_raw = A_prev * surv + e_conv * R_BH + imm_eff * cotsimm_dat(t-1);
    Type A_next = tiny + softplus(A_next_raw - tiny);

    // Multi-prey Holling Type II predation (per adult area cleared; capped by denominator)
    Type denom = h_hand + q_fast * F_prev + q_slow * S_prev + tiny;
    Type C_fast = c_max * A_prev * (q_fast * F_prev) / denom;                                   // fast coral consumption (proportion/yr)
    Type C_slow = c_max * A_prev * (q_slow * S_prev) / denom;                                   // slow coral consumption (proportion/yr)

    // Coral logistic growth with competition and bleaching penalty
    Type G_fast = g_T * g_fast * F_prev * (Type(1.0) - (F_prev + alpha_fs * S_prev));           // fast coral growth (proportion/yr)
    Type G_slow = g_T * g_slow * S_prev * (Type(1.0) - (S_prev + alpha_sf * F_prev));           // slow coral growth (proportion/yr)

    // Update corals in logit space for smooth [0,1] bounds
    Type zF_prev = log(F_prev + eps_prop) - log(Type(1.0) - F_prev + eps_prop);
    Type zS_prev = log(S_prev + eps_prop) - log(Type(1.0) - S_prev + eps_prop);
    Type zF_next = zF_prev + (G_fast - C_fast);
    Type zS_next = zS_prev + (G_slow - C_slow);
    Type F_next = inv_logit_eps(zF_next, eps_prop);                                            // stays within (eps,1-eps)
    Type S_next = inv_logit_eps(zS_next, eps_prop);                                            // stays within (eps,1-eps)

    // Write predictions
    cots_pred(t) = A_next;
    fast_pred(t) = Type(100.0) * F_next;
    slow_pred(t) = Type(100.0) * S_next;
  }

  // ---- Likelihood: include all observations ----
  for (int t = 0; t < n; t++) {
    // COTS log-normal
    Type mu_log = log(cots_pred(t) + tiny);
    Type y_log = log(cots_dat(t) + tiny);
    Type sd_log = sqrt(sd_log_cots * sd_log_cots + sd_min * sd_min);
    nll -= dnorm(y_log, mu_log, sd_log, true);

    // Fast coral logit-normal
    Type yF = (fast_dat(t) / Type(100.0));
    Type muF = (fast_pred(t) / Type(100.0));
    // Stabilize to (eps,1-eps)
    Type yF_c = (Type(1.0) - Type(2.0) * eps_prop) * yF + eps_prop;
    Type muF_c = (Type(1.0) - Type(2.0) * eps_prop) * muF + eps_prop;
    Type yF_logit = log(yF_c + eps_prop) - log(Type(1.0) - yF_c + eps_prop);
    Type muF_logit = log(muF_c + eps_prop) - log(Type(1.0) - muF_c + eps_prop);
    Type sd_logitF = sqrt(sd_logit_fast * sd_logit_fast + sd_min * sd_min);
    nll -= dnorm(yF_logit, muF_logit, sd_logitF, true);

    // Slow coral logit-normal
    Type yS = (slow_dat(t) / Type(100.0));
    Type muS = (slow_pred(t) / Type(100.0));
    Type yS_c = (Type(1.0) - Type(2.0) * eps_prop) * yS + eps_prop;
    Type muS_c = (Type(1.0) - Type(2.0) * eps_prop) * muS + eps_prop;
    Type yS_logit = log(yS_c + eps_prop) - log(Type(1.0) - yS_c + eps_prop);
    Type muS_logit = log(muS_c + eps_prop) - log(Type(1.0) - muS_c + eps_prop);
    Type sd_logitS = sqrt(sd_logit_slow * sd_logit_slow + sd_min * sd_min);
    nll -= dnorm(yS_logit, muS_logit, sd_logitS, true);
  }

  // ---- Reporting ----
  REPORT(Year);
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  // Optional reporting of intermediate ecological signals for diagnostics
  // (commented out to keep output concise; can be enabled if desired)
  // REPORT(r_max); REPORT(g_fast); REPORT(g_slow); REPORT(c_max);

  return nll;
}
