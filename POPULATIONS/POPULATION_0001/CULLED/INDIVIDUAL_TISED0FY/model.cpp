#include <TMB.hpp>

// Helper: inverse-logit with numerical safety
template<class Type>
Type invlogit_safe(Type x) {
  return Type(1.0) / (Type(1.0) + exp(-x));
}

// Helper: softplus for smooth thresholding (k controls sharpness)
// Numerically stable and AD-safe implementation: softplus(z) = max(z,0) + log(1 + exp(-|z|))
template<class Type>
Type softplus(Type x, Type k) {
  Type z = k * x;
  Type absz = CppAD::abs(z);
  Type term = log(Type(1.0) + exp(-absz));
  // If z > 0: z + term; else: term
  Type val = CppAD::CondExpGt(z, Type(0.0), z + term, term);
  return (Type(1.0) / k) * val;
}

/*
Ecological model equations (annual time step). Variables with suffix _pred are predictions.
1) Food availability for reproduction (saturating, preference-weighted):
   P_t = pref_fast * F_{t}/100 + (1 - pref_fast) * S_{t}/100
   g_food_t = P_t / (K_food + P_t)

2) Temperature effect on larval processes (smooth logistic around T_ref):
   g_T_larv_t = 0.5 + invlogit(beta_T_larvae * (sst_{t} - T_ref))   in (0.5, 1.5)

3) Larval supply combining local reproduction and immigration:
   L_t = r_rep * A_t * g_food_t * g_T_larv_t + q_imm * cotsimm_{t}

4) Juvenile dynamics with density dependence:
   J_{t+1} = (1 - mu_J) * J_t + eta_settle * L_t / (1 + kJ_dd * J_t)

5) Adult dynamics with maturation and density dependence:
   A_{t+1} = (1 - mu_A) * A_t + m_mat * J_t / (1 + kA_dd * A_t)

6) Selective predation (Holling Type II) on coral proportions:
   a_F = a0 * pref_fast
   a_S = a0 * (1 - pref_fast)
   denom_t = 1 + h_hand * (a_F * Fp_t + a_S * Sp_t)
   c_F_t = a_F * Fp_t / denom_t
   c_S_t = a_S * Sp_t / denom_t
   pred_F_t = alpha_pred * A_t * c_F_t
   pred_S_t = alpha_pred * A_t * c_S_t

7) Coral temperature sensitivity (smooth bleaching penalty):
   g_T_coral_t = exp(-beta_bleach * softplus(sst_{t} - T_bleach, k_bleach))

8) Coral growth (logistic under shared carrying capacity K_tot, proportions):
   dF_growth_t = rF * Fp_t * (1 - (Fp_t + Sp_t) / K_tot) * g_T_coral_t
   dS_growth_t = rS * Sp_t * (1 - (Fp_t + Sp_t) / K_tot) * g_T_coral_t

9) Coral updates with background mortality and predation (proportions):
   Fp_{t+1} = Fp_t + dF_growth_t - pred_F_t - mu_F_bg * Fp_t
   Sp_{t+1} = Sp_t + dS_growth_t - pred_S_t - mu_S_bg * Sp_t

Observation models:
10) Adult COTS (lognormal): log(cots_dat_t) ~ N(log(cots_pred_t), sigma_cots)
11) Coral cover (logit-normal on proportions in (0,1)): logit(fast_dat_t/100) ~ N(logit(fast_pred_t/100), sigma_fast)
                                                    and  logit(slow_dat_t/100) ~ N(logit(slow_pred_t/100), sigma_slow)
12) Forcings (sst_dat, cotsimm_dat) are passed-through as predictions and contribute only constants to the likelihood with small fixed SDs.

Notes:
- All dynamics at time t+1 use states and forcings from time t (no data leakage).
- Initial conditions for A, F, S are set to the first observed values.
- Small constants and smooth functions are used for stability, and soft penalties nudge parameters toward biological ranges.
*/

template<class Type>
Type objective_function<Type>::operator() ()
{
  // -----------------------------
  // Numerical constants
  // -----------------------------
  Type eps = Type(1e-8);                          // Small positive constant to avoid division by zero and logs
  Type min_sd = Type(1e-3);                       // Minimum observation SD to avoid overly tight fits
  Type k_bleach = Type(10.0);                     // Sharpness of softplus threshold for bleaching (higher = sharper)
  Type penalty_w = Type(1.0);                     // Weight for smooth bound penalties

  // -----------------------------
  // Data (time series)
  // -----------------------------
  DATA_VECTOR(Year);           // Year (calendar year), used as the time index; size n
  DATA_VECTOR(sst_dat);        // Sea-surface temperature (°C), annual mean
  DATA_VECTOR(cotsimm_dat);    // Crown-of-thorns larval immigration (ind/m2/year)
  DATA_VECTOR(cots_dat);       // Adult COTS density (ind/m2)
  DATA_VECTOR(fast_dat);       // Fast coral cover (%; Acropora)
  DATA_VECTOR(slow_dat);       // Slow coral cover (%; Faviidae + Porites)

  int n = Year.size();         // Length of time series

  // -----------------------------
  // Parameters (see parameters.json for bounds and descriptions)
  // -----------------------------
  PARAMETER(r_rep);            // Reproductive output to larval pool per adult per year (ind larva equiv. per ind adult per year); initial estimate from literature/initial guess
  PARAMETER(eta_settle);       // Settlement/early survival efficiency to juveniles (dimensionless 0-1); initial estimate
  PARAMETER(mu_J);             // Juvenile annual mortality rate (year^-1, 0-1); initial estimate
  PARAMETER(mu_A);             // Adult annual mortality rate (year^-1, 0-1); initial estimate
  PARAMETER(m_mat);            // Juvenile-to-adult maturation rate (year^-1, 0-1); initial estimate
  PARAMETER(kJ_dd);            // Density dependence for juvenile settlement (per ind m^-2); initial estimate
  PARAMETER(kA_dd);            // Density dependence for adult recruitment (per ind m^-2); initial estimate
  PARAMETER(q_imm);            // Scaling of immigration forcing to larval pool (ind/m2 per unit cotsimm_dat); initial estimate

  PARAMETER(beta_T_larvae);    // Temperature sensitivity of larval processes (per °C; logistic scaling); initial estimate
  PARAMETER(T_ref);            // Reference SST around which larval performance is centered (°C); initial estimate

  PARAMETER(rF);               // Intrinsic growth rate of fast coral (year^-1); initial estimate
  PARAMETER(rS);               // Intrinsic growth rate of slow coral (year^-1); initial estimate
  PARAMETER(K_tot);            // Carrying capacity for total coral cover (proportion 0-1); initial estimate

  PARAMETER(a0);               // Overall encounter/attack rate scaling for predation (year^-1); initial estimate
  PARAMETER(h_hand);           // Handling time parameter for Type II functional response (year); initial estimate
  PARAMETER(pref_fast);        // Preference for fast coral in [0,1]; higher favors Acropora; initial estimate
  PARAMETER(alpha_pred);       // Scaling from per-capita feeding to proportional cover loss (per ind m^-2); initial estimate

  PARAMETER(beta_bleach);      // Sensitivity of coral to heat stress (per °C); initial estimate
  PARAMETER(T_bleach);         // SST threshold for bleaching onset (°C); initial estimate

  PARAMETER(K_food);           // Half-saturation for fecundity vs. prey availability (proportion cover); initial estimate

  PARAMETER(mu_F_bg);          // Background annual mortality for fast coral (year^-1); initial estimate
  PARAMETER(mu_S_bg);          // Background annual mortality for slow coral (year^-1); initial estimate

  PARAMETER(sigma_cots);       // Observation SD for log(COTS) (dimensionless); to be estimated with floor
  PARAMETER(sigma_fast);       // Observation SD for logit(fast proportion) (dimensionless); to be estimated with floor
  PARAMETER(sigma_slow);       // Observation SD for logit(slow proportion) (dimensionless); to be estimated with floor

  // -----------------------------
  // Soft bound penalties (encourage biological ranges without hard constraints)
  // -----------------------------
  Type nll = 0.0; // Negative log-likelihood accumulator

  auto pen = [&](Type x, Type lo, Type hi) {
    Type below = softplus(lo - x, Type(5.0));   // Smoothly penalize x < lo
    Type above = softplus(x - hi, Type(5.0));   // Smoothly penalize x > hi
    return penalty_w * (below*below + above*above);
  };

  // Suggested biological ranges (documented in parameters.json)
  nll += pen(r_rep,       Type(0.0),  Type(20.0));
  nll += pen(eta_settle,  Type(0.0),  Type(1.0));
  nll += pen(mu_J,        Type(0.0),  Type(1.0));
  nll += pen(mu_A,        Type(0.0),  Type(1.0));
  nll += pen(m_mat,       Type(0.0),  Type(1.0));
  nll += pen(kJ_dd,       Type(0.0),  Type(10.0));
  nll += pen(kA_dd,       Type(0.0),  Type(10.0));
  nll += pen(q_imm,       Type(0.0),  Type(50.0));

  nll += pen(beta_T_larvae, Type(-2.0), Type(2.0));
  nll += pen(T_ref,         Type(20.0), Type(35.0));

  nll += pen(rF,          Type(0.0),  Type(2.0));
  nll += pen(rS,          Type(0.0),  Type(1.0));
  nll += pen(K_tot,       Type(0.1),  Type(1.0));

  nll += pen(a0,          Type(0.0),  Type(100.0));
  nll += pen(h_hand,      Type(0.0),  Type(10.0));
  nll += pen(pref_fast,   Type(0.0),  Type(1.0));
  nll += pen(alpha_pred,  Type(0.0),  Type(10.0));

  nll += pen(beta_bleach, Type(0.0),  Type(5.0));
  nll += pen(T_bleach,    Type(25.0), Type(35.0));

  nll += pen(K_food,      Type(1e-6), Type(1.0));
  nll += pen(mu_F_bg,     Type(0.0),  Type(0.5));
  nll += pen(mu_S_bg,     Type(0.0),  Type(0.5));

  nll += pen(sigma_cots,  Type(1e-6), Type(5.0));
  nll += pen(sigma_fast,  Type(1e-6), Type(5.0));
  nll += pen(sigma_slow,  Type(1e-6), Type(5.0));

  // -----------------------------
  // State vectors (predictions)
  // -----------------------------
  vector<Type> cots_pred(n);      // Adult COTS prediction (ind/m2)
  vector<Type> fast_pred(n);      // Fast coral cover prediction (%)
  vector<Type> slow_pred(n);      // Slow coral cover prediction (%)
  vector<Type> sst_pred(n);       // SST prediction (pass-through)
  vector<Type> cotsimm_pred(n);   // Immigration prediction (pass-through)
  vector<Type> J_pred(n);         // Juvenile COTS (latent; ind/m2)

  // Initialize predictions with observed initial conditions (no optimization on ICs)
  cots_pred(0) = cots_dat(0);     // Adult COTS at t0 equals observed t0
  fast_pred(0) = fast_dat(0);     // Fast coral at t0 equals observed t0
  slow_pred(0) = slow_dat(0);     // Slow coral at t0 equals observed t0
  sst_pred = sst_dat;             // Forcings are passed through as-is
  cotsimm_pred = cotsimm_dat;     // Forcings are passed through as-is
  // Latent juvenile initial condition set to adult abundance at t0 (uninformative neutral start)
  J_pred(0) = cots_pred(0);

  // -----------------------------
  // Time loop: dynamics (use only previous time step states and forcings)
  // -----------------------------
  for (int t = 1; t < n; t++) {
    // Previous states (percent for corals, convert to proportions)
    Type A_prev = cots_pred(t-1) + eps;               // Adults at t-1 (ind/m2), ensure >0
    Type F_prev = fast_pred(t-1);                     // Fast coral at t-1 (%)
    Type S_prev = slow_pred(t-1);                     // Slow coral at t-1 (%)
    Type Fp_prev = (F_prev / Type(100.0));            // Fast as proportion
    Type Sp_prev = (S_prev / Type(100.0));            // Slow as proportion

    // Forcings from previous year (no data leakage)
    Type sst_prev = sst_dat(t-1);                     // SST at t-1 (°C)
    Type imm_prev = cotsimm_dat(t-1);                 // Immigration at t-1 (ind/m2/year)

    // 1) Food availability for reproduction (saturating, preference-weighted)
    Type P_prev = pref_fast * Fp_prev + (Type(1.0) - pref_fast) * Sp_prev;  // Weighted prey availability (proportion)
    Type g_food = P_prev / (K_food + P_prev + eps);                          // Saturating effect on fecundity

    // 2) Temperature effect on larval processes (smooth logistic around T_ref)
    Type g_T_larv = Type(0.5) + invlogit_safe(beta_T_larvae * (sst_prev - T_ref)); // In (0.5,1.5)

    // 3) Larval supply: local reproduction + exogenous immigration
    Type L_prev = r_rep * A_prev * g_food * g_T_larv + q_imm * imm_prev;    // Total larvae contributing to settlement

    // 4) Juvenile update with density dependence
    Type J_prev = J_pred(t-1) + eps;                                        // Juveniles at t-1 (ind/m2)
    Type J_gain = eta_settle * L_prev / (Type(1.0) + kJ_dd * J_prev);        // Settlement limited by juvenile density
    Type J_next = (Type(1.0) - mu_J) * J_prev + J_gain;                      // Juvenile survivors + new settlers

    // 5) Adult update with maturation and density dependence
    Type A_gain = m_mat * J_prev / (Type(1.0) + kA_dd * A_prev);             // Maturation limited by adult density
    Type A_next = (Type(1.0) - mu_A) * A_prev + A_gain;                      // Adult survivors + recruits

    // 6) Selective predation functional response (Holling Type II on coral proportions)
    Type aF = a0 * pref_fast;                                                // Encounter rate on fast coral
    Type aS = a0 * (Type(1.0) - pref_fast);                                  // Encounter rate on slow coral
    Type denom = Type(1.0) + h_hand * (aF * Fp_prev + aS * Sp_prev + eps);   // Denominator with handling time
    Type cF = aF * Fp_prev / denom;                                          // Per-adult consumption of fast (proportion/year)
    Type cS = aS * Sp_prev / denom;                                          // Per-adult consumption of slow (proportion/year)
    Type predF = alpha_pred * A_prev * cF;                                    // Total predation loss for fast (proportion/year)
    Type predS = alpha_pred * A_prev * cS;                                    // Total predation loss for slow (proportion/year)

    // 7) Coral temperature sensitivity (smooth bleaching penalty via softplus above threshold)
    Type heat_excess = sst_prev - T_bleach;                                   // °C above bleaching threshold
    Type g_T_coral = exp(-beta_bleach * softplus(heat_excess, k_bleach));     // Multiplier <= 1

    // 8) Coral growth (logistic with shared K)
    Type total_prev = Fp_prev + Sp_prev + eps;                                // Total coral proportion
    Type crowd = (Type(1.0) - (total_prev / (K_tot + eps)));                  // Space remaining (can be negative if over K)
    Type dF_grow = rF * Fp_prev * crowd * g_T_coral;                          // Logistic growth of fast
    Type dS_grow = rS * Sp_prev * crowd * g_T_coral;                          // Logistic growth of slow

    // 9) Coral updates with predation and background mortality (proportions)
    Type Fp_next = Fp_prev + dF_grow - predF - mu_F_bg * Fp_prev;             // Fast coral next proportion
    Type Sp_next = Sp_prev + dS_grow - predS - mu_S_bg * Sp_prev;             // Slow coral next proportion

    // Map back to % for reporting (ensure non-negative by flooring at eps to avoid logit issues later)
    fast_pred(t) = Type(100.0) * (Fp_next > eps ? Fp_next : eps);             // Fast coral cover (%)
    slow_pred(t) = Type(100.0) * (Sp_next > eps ? Sp_next : eps);             // Slow coral cover (%)

    // Adults and juveniles (ensure strictly positive)
    cots_pred(t) = (A_next > eps ? A_next : eps);                              // Adult COTS (ind/m2)
    J_pred(t) = (J_next > eps ? J_next : eps);                                 // Juvenile COTS (ind/m2)
  }

  // -----------------------------
  // Likelihoods
  // -----------------------------
  // Observation SD floors
  Type sd_cots = sigma_cots + min_sd;   // Ensure positive SD
  Type sd_fast = sigma_fast + min_sd;   // Ensure positive SD
  Type sd_slow = sigma_slow + min_sd;   // Ensure positive SD

  // 10) COTS adults: lognormal
  for (int t = 0; t < n; t++) {
    Type y = log(cots_dat(t) + eps);                 // Observed on log scale
    Type mu = log(cots_pred(t) + eps);               // Predicted on log scale
    nll -= dnorm(y, mu, sd_cots, true);              // Add log-density
  }

  // 11) Corals: logit-normal on proportions
  for (int t = 0; t < n; t++) {
    // Fast coral
    Type p_dat_f = (fast_dat(t) / Type(100.0));                         // Observed proportion
    Type p_pred_f = (fast_pred(t) / Type(100.0));                        // Predicted proportion
    // Clip to (eps, 1-eps) to avoid logit extremes
    p_dat_f = CppAD::CondExpLt(p_dat_f, eps, eps, CppAD::CondExpGt(p_dat_f, Type(1.0) - eps, Type(1.0) - eps, p_dat_f));
    p_pred_f = CppAD::CondExpLt(p_pred_f, eps, eps, CppAD::CondExpGt(p_pred_f, Type(1.0) - eps, Type(1.0) - eps, p_pred_f));
    Type y_f = log(p_dat_f / (Type(1.0) - p_dat_f));                    // Logit observed
    Type mu_f = log(p_pred_f / (Type(1.0) - p_pred_f));                 // Logit predicted
    nll -= dnorm(y_f, mu_f, sd_fast, true);

    // Slow coral
    Type p_dat_s = (slow_dat(t) / Type(100.0));                          // Observed proportion
    Type p_pred_s = (slow_pred(t) / Type(100.0));                         // Predicted proportion
    p_dat_s = CppAD::CondExpLt(p_dat_s, eps, eps, CppAD::CondExpGt(p_dat_s, Type(1.0) - eps, Type(1.0) - eps, p_dat_s));
    p_pred_s = CppAD::CondExpLt(p_pred_s, eps, eps, CppAD::CondExpGt(p_pred_s, Type(1.0) - eps, Type(1.0) - eps, p_pred_s));
    Type y_s = log(p_dat_s / (Type(1.0) - p_dat_s));                    // Logit observed
    Type mu_s = log(p_pred_s / (Type(1.0) - p_pred_s));                 // Logit predicted
    nll -= dnorm(y_s, mu_s, sd_slow, true);
  }

  // 12) Forcings: constant-contribution check (predictions equal data; residuals=0, adds constant)
  Type sd_sst_fixed = Type(1e-2);       // Small fixed SD (°C)
  Type sd_imm_fixed = Type(1e-2);       // Small fixed SD (ind/m2/yr)
  for (int t = 0; t < n; t++) {
    nll -= dnorm(sst_dat(t), sst_pred(t), sd_sst_fixed, true);         // Adds constant when equal
    nll -= dnorm(cotsimm_dat(t), cotsimm_pred(t), sd_imm_fixed, true); // Adds constant when equal
  }

  // -----------------------------
  // Reporting
  // -----------------------------
  REPORT(Year);            // Time index
  REPORT(sst_dat);         // Forcing (data)
  REPORT(cotsimm_dat);     // Forcing (data)
  REPORT(sst_pred);        // Forcing (predictions)
  REPORT(cotsimm_pred);    // Forcing (predictions)

  REPORT(cots_dat);        // Observations: adult COTS
  REPORT(fast_dat);        // Observations: fast coral
  REPORT(slow_dat);        // Observations: slow coral

  REPORT(cots_pred);       // Predictions: adult COTS
  REPORT(fast_pred);       // Predictions: fast coral
  REPORT(slow_pred);       // Predictions: slow coral
  REPORT(J_pred);          // Latent juveniles (for diagnostics)

  // Derived helpful diagnostics (optional)
  Type mean_sst = Type(0.0);
  for (int i = 0; i < n; i++) mean_sst += sst_dat(i);
  mean_sst /= Type(n);
  REPORT(mean_sst);

  // Return negative log-likelihood
  return nll;
}
