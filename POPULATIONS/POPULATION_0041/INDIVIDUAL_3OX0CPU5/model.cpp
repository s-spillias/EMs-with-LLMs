#include <TMB.hpp>

// Small helpers
template<class Type>
Type invlogit(Type x) { return Type(1) / (Type(1) + exp(-x)); }

template<class Type>
Type clamp01(Type x) {
  Type eps = Type(1e-8);
  return CppAD::CondExpLt(x, eps, eps,
         CppAD::CondExpGt(x, Type(1) - eps, Type(1) - eps, x));
}

/*
Numbered model equations (all transitions use t-1 states; no data leakage):

Let:
C_t   = COTS adults (indiv m^-2)
F_t   = Fast coral cover (Acropora, %)
S_t   = Slow coral cover (Faviidae/Porites, %)
A_t   = Weighted coral availability for COTS
T_t   = Sea surface temperature (°C)
I_t   = External larval immigration (indiv m^-2 yr^-1)

Auxiliary smooth indices (sigmoids, values in (0,1)):
(1) Food index:     food_t   = logistic((A_t - food_thr)/food_slope)
(2) Outbreak index: sst_out_t= logistic((T_t - sst_thr)/sst_slope)
(3) Bleach index:   bleach_t = logistic((T_t - bleach_thr)/bleach_slope)

Functional response and predation allocation:
(4) A_t = wF*F_t + (1-wF)*S_t
(5) q   = 1 + exp(log_q)   (Type-II when q≈1, Type-III when q>1)
(6) cons_per_star_t = a * A_t^q / (H^q + A_t^q)        // % cover consumed per star per year
(7) phiF_t = (selF*F_t) / (selF*F_t + (1-selF)*S_t)    // selectivity-weighted share to fast coral
(8) E_F_t  = pred_eff * cons_per_star_t * C_t * phiF_t // % cover yr^-1 consumed from fast coral
(9) E_S_t  = pred_eff * cons_per_star_t * C_t * (1-phiF_t)

Population and coral updates (Ricker-style for COTS, logistic for corals):
(10) outbreak_t = food_t * sst_out_t
(11) C_{t+1} = C_t * s_C * exp( r_max*outbreak_t - d_self*C_t ) + imm_coeff * I_t
(12) F_{t+1} = F_t + r_F*F_t*(1 - (F_t+S_t)/K) - E_F_t - m_bleach_F*bleach_t*F_t
(13) S_{t+1} = S_t + r_S*S_t*(1 - (F_t+S_t)/K) - E_S_t - m_bleach_S*bleach_t*S_t

Observation models (for all t including t=0):
(14) log(C_obs_t) ~ Normal( log(C_t + eps), sigma_cots )
(15) logit(F_obs_t/K) ~ Normal( logit(F_t/K), sigma_coral )
(16) logit(S_obs_t/K) ~ Normal( logit(S_t/K), sigma_coral )
*/

template<class Type>
Type objective_function<Type>::operator() () {
  using namespace density;

  // -------------------- DATA --------------------
  DATA_VECTOR(Year);              // Year (calendar years), used only for reporting/plotting
  DATA_VECTOR(cots_dat);          // Observed COTS adults (indiv m^-2), strictly positive in practice
  DATA_VECTOR(fast_dat);          // Observed fast coral cover (%)
  DATA_VECTOR(slow_dat);          // Observed slow coral cover (%)
  DATA_VECTOR(sst_dat);           // Sea-surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);       // External larval immigration (indiv m^-2 yr^-1)

  int T = cots_dat.size();        // Length of time series
  Type eps = Type(1e-8);          // Small constant for numerical stability
  Type K = Type(100.0);           // Total cover (%) carrying capacity (fixed at 100%)

  // -------------------- PARAMETERS --------------------
  // Coral intrinsic growth (year^-1), positive via log-transform
  PARAMETER(log_rF);              // log r_F (yr^-1), fast coral intrinsic growth
  PARAMETER(log_rS);              // log r_S (yr^-1), slow coral intrinsic growth

  // COTS growth controls
  PARAMETER(log_rmax);            // log r_max (yr^-1), max per-capita growth at full triggers
  PARAMETER(log_d_self);          // log d_self ((indiv m^-2)^-1 yr^-1), self-limitation strength
  PARAMETER(logit_s_C);           // logit s_C (unitless), annual adult survival probability in (0,1)

  // Predation and resource limitation
  PARAMETER(selF_logit);          // logit sel_F (unitless), selectivity to fast coral in (0,1)
  PARAMETER(log_a);               // log a (% per indiv per yr), max per-capita coral consumption scale
  PARAMETER(log_H);               // log H (% cover), half-saturation of functional response
  PARAMETER(log_q);               // log q' (unitless), q = 1 + exp(log_q) => Type-II to Type-III
  PARAMETER(pred_eff_logit);      // logit pred_eff (unitless), efficiency converting consumption to coral loss

  // Food trigger (weighted coral)
  PARAMETER(food_wF_logit);       // logit wF (unitless), weight on fast coral in [0,1]
  PARAMETER(food_thr_logit);      // logit(food_thr/K) (unitless), threshold as fraction of K
  PARAMETER(log_food_slope);      // log food_slope (% cover), smoothness of food trigger

  // SST triggers
  PARAMETER(sst_thr);             // °C, outbreak SST threshold (center of sigmoid)
  PARAMETER(log_sst_slope);       // log °C, smoothness of outbreak SST trigger
  PARAMETER(bleach_thr);          // °C, bleaching threshold (center of sigmoid)
  PARAMETER(log_bleach_slope);    // log °C, smoothness of bleaching sigmoid

  // Bleaching sensitivities (fraction per year)
  PARAMETER(log_m_bleach_F);      // log m_bleach_F (yr^-1), fast coral extra mortality at full bleach
  PARAMETER(log_m_bleach_S);      // log m_bleach_S (yr^-1), slow coral extra mortality at full bleach

  // Immigration scaling
  PARAMETER(log_imm);             // log imm_coeff (yr), scales cotsimm_dat contribution

  // Initial states (t = 0)
  PARAMETER(F0_logit);            // logit(F0/K) (unitless), initial fast coral fraction of K
  PARAMETER(S0_logit);            // logit(S0/K) (unitless), initial slow coral fraction of K
  PARAMETER(log_C0);              // log C0 (indiv m^-2), initial adult COTS density

  // Observation standard deviations (lower-bounded via exponentials and minimums)
  PARAMETER(log_sigma_cots);      // log sigma_cots (log scale)
  PARAMETER(log_sigma_coral);     // log sigma_coral (logit scale)

  // -------------------- TRANSFORMS --------------------
  Type r_F = exp(log_rF);                         // yr^-1
  Type r_S = exp(log_rS);                         // yr^-1
  Type r_max = exp(log_rmax);                     // yr^-1
  Type d_self = exp(log_d_self);                  // (indiv m^-2)^-1 yr^-1
  Type s_C = invlogit(logit_s_C);                 // survival in (0,1)

  Type sel_F = invlogit(selF_logit);              // preference for fast coral in (0,1)
  Type a = exp(log_a);                            // % per indiv per yr
  Type H = exp(log_H);                            // % cover
  Type q = Type(1.0) + exp(log_q);                // >= 1
  Type pred_eff = invlogit(pred_eff_logit);       // in (0,1)

  Type wF = invlogit(food_wF_logit);              // weight in (0,1)
  Type food_thr = K * invlogit(food_thr_logit);   // % cover threshold
  Type food_slope = exp(log_food_slope);          // % cover (>0)

  Type sst_slope = exp(log_sst_slope);            // °C (>0)
  Type bleach_slope = exp(log_bleach_slope);      // °C (>0)

  Type m_bleach_F = exp(log_m_bleach_F);          // yr^-1
  Type m_bleach_S = exp(log_m_bleach_S);          // yr^-1

  Type imm_coeff = exp(log_imm);                  // scales immigration

  Type F0 = K * invlogit(F0_logit);               // % cover initial
  Type S0 = K * invlogit(S0_logit);               // % cover initial
  Type C0 = exp(log_C0);                          // indiv m^-2 initial

  // Observation error SDs with fixed minimums for stability
  Type sigma_cots = exp(log_sigma_cots) + Type(0.05);    // log scale SD, min ~0.05
  Type sigma_coral = exp(log_sigma_coral) + Type(0.05);  // logit scale SD, min ~0.05

  // -------------------- STATE TRAJECTORIES --------------------
  vector<Type> cots_pred(T); cots_pred.setZero();
  vector<Type> fast_pred(T); fast_pred.setZero();
  vector<Type> slow_pred(T); slow_pred.setZero();

  // Indices for diagnostics
  vector<Type> food_index(T); food_index.setZero();
  vector<Type> sst_out_index(T); sst_out_index.setZero();
  vector<Type> bleach_index(T); bleach_index.setZero();
  vector<Type> avail_coral(T); avail_coral.setZero();
  vector<Type> cons_per_star(T); cons_per_star.setZero();

  // Initialize states at t=0 via parameters (no leakage from data)
  cots_pred(0) = C0;                          // indiv m^-2
  fast_pred(0) = F0;                          // %
  slow_pred(0) = S0;                          // %

  // Time loop: transitions use t-1 predicted states and exogenous forcing at t-1
  for (int t = 1; t < T; ++t) {
    // Previous states
    Type Cprev = cots_pred(t-1);
    Type Fprev = fast_pred(t-1);
    Type Sprev = slow_pred(t-1);

    // Environmental forcing at previous year
    Type Tprev = sst_dat(t-1);
    Type Iprev = cotsimm_dat(t-1);

    // 1) Availability and smooth indices
    Type Aprev = wF * Fprev + (Type(1.0) - wF) * Sprev;              // weighted % cover
    Type food_sig = invlogit((Aprev - food_thr) / (food_slope + eps));
    Type sst_sig = invlogit((Tprev - sst_thr) / (sst_slope + eps));
    Type bleach_sig = invlogit((Tprev - bleach_thr) / (bleach_slope + eps));

    // Save diagnostics
    avail_coral(t) = Aprev;
    food_index(t) = food_sig;
    sst_out_index(t) = sst_sig;
    bleach_index(t) = bleach_sig;

    // 2) Functional response and consumption
    Type Aq = pow(Aprev, q);
    Type Hq = pow(H, q);
    Type cons_star = a * Aq / (Hq + Aq + eps);                      // % per star per yr
    cons_per_star(t) = cons_star;

    // Selectivity-weighted allocation
    Type denom_sel = sel_F * Fprev + (Type(1.0) - sel_F) * Sprev + eps;
    Type phiF = (sel_F * Fprev) / denom_sel;                        // to fast coral
    Type total_consumption = pred_eff * cons_star * Cprev;          // % cover yr^-1
    Type E_F = total_consumption * phiF;                            // fast removal
    Type E_S = total_consumption * (Type(1.0) - phiF);              // slow removal

    // 3) COTS population update (Ricker with immigration)
    Type outbreak = food_sig * sst_sig;                             // 0..1
    Type g = r_max * outbreak - d_self * Cprev;                     // per-capita net growth
    Type Cnext = Cprev * s_C * exp(g) + imm_coeff * Iprev;          // indiv m^-2
    Cnext = CppAD::CondExpLt(Cnext, eps, eps, Cnext);               // clamp to positive

    // 4) Coral updates (logistic growth - predation - bleaching)
    Type cover_sum = Fprev + Sprev;
    Type logistic_termF = r_F * Fprev * (Type(1.0) - cover_sum / K);
    Type logistic_termS = r_S * Sprev * (Type(1.0) - cover_sum / K);

    Type Fnext = Fprev + logistic_termF - E_F - m_bleach_F * bleach_sig * Fprev;
    Type Snext = Sprev + logistic_termS - E_S - m_bleach_S * bleach_sig * Sprev;

    // Clamp to [eps, K-eps] smoothly
    Fnext = CppAD::CondExpLt(Fnext, eps, eps,
             CppAD::CondExpGt(Fnext, K - eps, K - eps, Fnext));
    Snext = CppAD::CondExpLt(Snext, eps, eps,
             CppAD::CondExpGt(Snext, K - eps, K - eps, Snext));

    // Save states
    cots_pred(t) = Cnext;
    fast_pred(t) = Fnext;
    slow_pred(t) = Snext;
  }

  // -------------------- LIKELIHOOD --------------------
  Type nll = Type(0.0);

  // Observation model for COTS (lognormal on abundance)
  for (int t = 0; t < T; ++t) {
    Type mu_logC = log(cots_pred(t) + eps);
    Type obs_logC = log(cots_dat(t) + eps);
    nll -= dnorm(obs_logC, mu_logC, sigma_cots, true);
  }

  // Observation model for corals (logit-normal on % cover / K)
  for (int t = 0; t < T; ++t) {
    // Fast coral
    Type pF_pred = clamp01(fast_pred(t) / K);
    Type pF_obs  = clamp01(fast_dat(t) / K);
    Type zF_pred = log(pF_pred + eps) - log(Type(1.0) - pF_pred + eps);
    Type zF_obs  = log(pF_obs  + eps) - log(Type(1.0) - pF_obs  + eps);
    nll -= dnorm(zF_obs, zF_pred, sigma_coral, true);

    // Slow coral
    Type pS_pred = clamp01(slow_pred(t) / K);
    Type pS_obs  = clamp01(slow_dat(t) / K);
    Type zS_pred = log(pS_pred + eps) - log(Type(1.0) - pS_pred + eps);
    Type zS_obs  = log(pS_obs  + eps) - log(Type(1.0) - pS_obs  + eps);
    nll -= dnorm(zS_obs, zS_pred, sigma_coral, true);
  }

  // -------------------- SOFT PRIORS / PENALTIES --------------------
  // Weakly-informative priors to encourage biologically plausible ranges without hard constraints
  // Survival centered near 0.8 on logit scale
  nll -= dnorm(logit_s_C, Type(1.38629436112), Type(1.5), true); // mean logit(0.8), sd ~1.5

  // Keep selectivity biased to fast coral but flexible
  nll -= dnorm(selF_logit, Type(1.0), Type(2.0), true);

  // Growth/bleach rates regularization (log-scale)
  nll -= dnorm(log_rF, Type(log(0.4)), Type(1.0), true);
  nll -= dnorm(log_rS, Type(log(0.2)), Type(1.0), true);
  nll -= dnorm(log_m_bleach_F, Type(log(0.3)), Type(1.0), true);
  nll -= dnorm(log_m_bleach_S, Type(log(0.1)), Type(1.0), true);

  // -------------------- REPORTS --------------------
  REPORT(Year);                 // year axis
  REPORT(cots_pred);            // predicted COTS adults (indiv m^-2)
  REPORT(fast_pred);            // predicted fast coral cover (%)
  REPORT(slow_pred);            // predicted slow coral cover (%)

  // Diagnostics for understanding outbreak mechanisms
  REPORT(avail_coral);          // weighted availability (%)
  REPORT(food_index);           // food trigger (0..1)
  REPORT(sst_out_index);        // SST outbreak trigger (0..1)
  REPORT(bleach_index);         // bleaching index (0..1)
  REPORT(cons_per_star);        // % cover consumed per star per year

  REPORT(r_F);
  REPORT(r_S);
  REPORT(r_max);
  REPORT(d_self);
  REPORT(s_C);
  REPORT(sel_F);
  REPORT(a);
  REPORT(H);
  REPORT(q);
  REPORT(pred_eff);
  REPORT(wF);
  REPORT(food_thr);
  REPORT(food_slope);
  REPORT(sst_thr);
  REPORT(sst_slope);
  REPORT(bleach_thr);
  REPORT(bleach_slope);
  REPORT(m_bleach_F);
  REPORT(m_bleach_S);
  REPORT(imm_coeff);
  REPORT(F0);
  REPORT(S0);
  REPORT(C0);
  REPORT(sigma_cots);
  REPORT(sigma_coral);

  return nll;
}
