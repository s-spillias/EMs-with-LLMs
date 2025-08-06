#include <TMB.hpp>

// 1. Model equations are described at the end of this file.

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- DATA INPUTS ---
  DATA_VECTOR(Year); // Time variable (years)
  DATA_VECTOR(cots_dat); // Adult COTS abundance (ind/m2)
  DATA_VECTOR(fast_dat); // Fast-growing coral cover (%) (Acropora spp.)
  DATA_VECTOR(slow_dat); // Slow-growing coral cover (%) (Faviidae/Porites)
  DATA_VECTOR(sst_dat); // Sea-surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration rate (ind/m2/year)

  int n = Year.size();

  // --- PARAMETERS ---
  PARAMETER(log_r_cots); // log intrinsic growth rate of COTS (year^-1)
  PARAMETER(log_K_cots); // log carrying capacity for COTS (ind/m2)
  PARAMETER(log_alpha_fast); // log COTS predation rate on fast coral (% cover^-1 year^-1)
  PARAMETER(log_alpha_slow); // log COTS predation rate on slow coral (% cover^-1 year^-1)
  PARAMETER(log_e_fast); // log assimilation efficiency from fast coral (unitless)
  PARAMETER(log_e_slow); // log assimilation efficiency from slow coral (unitless)
  PARAMETER(log_r_fast); // log recovery rate of fast coral (% cover/year)
  PARAMETER(log_r_slow); // log recovery rate of slow coral (% cover/year)
  PARAMETER(log_K_fast); // log max cover of fast coral (%)
  PARAMETER(log_K_slow); // log max cover of slow coral (%)
  PARAMETER(log_beta_sst); // log effect of SST on COTS growth (per deg C)
  PARAMETER(log_sigma_cots); // log SD of observation error for COTS
  PARAMETER(log_sigma_fast); // log SD of observation error for fast coral
  PARAMETER(log_sigma_slow); // log SD of observation error for slow coral
  PARAMETER(log_eps_imm); // log immigration efficiency for COTS (unitless)
  PARAMETER(log_thresh_coral); // log coral threshold for outbreak (cover %)
  PARAMETER(log_K_food); // log half-saturation constant for COTS food limitation (coral cover %)
  PARAMETER(logit_phi_fast); // logit selectivity of COTS for fast coral (unitless, 0-1)
  PARAMETER(logit_phi_slow); // logit selectivity of COTS for slow coral (unitless, 0-1)

  // --- TRANSFORM PARAMETERS TO NATURAL SCALE ---
  Type r_cots = exp(log_r_cots); // COTS intrinsic growth rate
  Type K_cots = exp(log_K_cots); // COTS carrying capacity
  Type alpha_fast = exp(log_alpha_fast); // predation rate on fast coral
  Type alpha_slow = exp(log_alpha_slow); // predation rate on slow coral
  Type e_fast = exp(log_e_fast); // assimilation efficiency fast coral
  Type e_slow = exp(log_e_slow); // assimilation efficiency slow coral
  Type r_fast = exp(log_r_fast); // fast coral recovery rate
  Type r_slow = exp(log_r_slow); // slow coral recovery rate
  Type K_fast = exp(log_K_fast); // fast coral max cover
  Type K_slow = exp(log_K_slow); // slow coral max cover
  Type beta_sst = exp(log_beta_sst); // SST effect on COTS
  Type sigma_cots = exp(log_sigma_cots); // obs error COTS
  Type sigma_fast = exp(log_sigma_fast); // obs error fast coral
  Type sigma_slow = exp(log_sigma_slow); // obs error slow coral
  Type eps_imm = exp(log_eps_imm); // immigration efficiency
  //Type thresh_coral = exp(log_thresh_coral); // coral threshold for outbreak (no longer used)
  Type K_food = exp(log_K_food); // half-saturation constant for food limitation
  Type phi_fast = 1/(1+exp(-logit_phi_fast)); // selectivity for fast coral (0-1)
  Type phi_slow = 1/(1+exp(-logit_phi_slow)); // selectivity for slow coral (0-1)

  // --- INITIAL CONDITIONS ---
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Use positive initial values to avoid log(0) or NA/NaN in likelihood
  cots_pred(0) = (CppAD::isnan(cots_dat(0)) || cots_dat(0) <= Type(0.0)) ? Type(1e-8) : cots_dat(0); // initial COTS abundance
  fast_pred(0) = (CppAD::isnan(fast_dat(0)) || fast_dat(0) <= Type(0.0)) ? Type(1e-8) : fast_dat(0); // initial fast coral cover
  slow_pred(0) = (CppAD::isnan(slow_dat(0)) || slow_dat(0) <= Type(0.0)) ? Type(1e-8) : slow_dat(0); // initial slow coral cover

  // --- MODEL DYNAMICS ---
  for(int t=1; t<n; t++) {
    // Resource limitation: total coral cover
    Type coral_total_prev = fast_pred(t-1) + slow_pred(t-1);

    // Food limitation: saturating function of coral cover
    Type food_lim = (coral_total_prev > Type(0.0)) ? (coral_total_prev / (coral_total_prev + K_food)) : Type(1e-8);

    // SST effect (centered at 27C, positive above, negative below)
    Type sst_effect = exp(beta_sst * (sst_dat(t-1) - Type(27.0)));

    // Immigration input (scaled by efficiency)
    Type imm_input = eps_imm * cotsimm_dat(t-1);

    // COTS predation on corals (functional response: saturating, Holling Type II)
    Type pred_fast = (fast_pred(t-1) > Type(0.0)) ? (alpha_fast * phi_fast * cots_pred(t-1) * fast_pred(t-1) / (fast_pred(t-1) + Type(1.0))) : Type(0.0);
    Type pred_slow = (slow_pred(t-1) > Type(0.0)) ? (alpha_slow * phi_slow * cots_pred(t-1) * slow_pred(t-1) / (slow_pred(t-1) + Type(1.0))) : Type(0.0);

    // COTS population update (logistic growth + immigration + food limitation)
    cots_pred(t) = cots_pred(t-1)
      + r_cots * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1)/K_cots) * food_lim * sst_effect
      + imm_input
      - pred_fast * e_fast
      - pred_slow * e_slow;

    // Prevent negative or NaN values
    cots_pred(t) = (CppAD::isnan(cots_pred(t)) || cots_pred(t) <= Type(0.0)) ? Type(1e-8) : cots_pred(t);

    // Fast coral update (recovery - COTS predation)
    fast_pred(t) = fast_pred(t-1)
      + r_fast * fast_pred(t-1) * (Type(1.0) - (fast_pred(t-1) + slow_pred(t-1))/K_fast)
      - pred_fast;

    fast_pred(t) = (CppAD::isnan(fast_pred(t)) || fast_pred(t) <= Type(0.0)) ? Type(1e-8) : fast_pred(t);

    // Slow coral update (recovery - COTS predation)
    slow_pred(t) = slow_pred(t-1)
      + r_slow * slow_pred(t-1) * (Type(1.0) - (fast_pred(t-1) + slow_pred(t-1))/K_slow)
      - pred_slow;

    slow_pred(t) = (CppAD::isnan(slow_pred(t)) || slow_pred(t) <= Type(0.0)) ? Type(1e-8) : slow_pred(t);
  }

  // --- LIKELIHOOD ---
  Type nll = 0.0;
  Type min_sd = Type(1e-3); // minimum SD for numerical stability

  for(int t=0; t<n; t++) {
    // Lognormal likelihood for strictly positive data
    nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred(t) + Type(1e-8)), sigma_cots + min_sd, true);
    nll -= dnorm(log(fast_dat(t) + Type(1e-8)), log(fast_pred(t) + Type(1e-8)), sigma_fast + min_sd, true);
    nll -= dnorm(log(slow_dat(t) + Type(1e-8)), log(slow_pred(t) + Type(1e-8)), sigma_slow + min_sd, true);
  }

  // --- SMOOTH PENALTIES FOR PARAMETER BOUNDS ---
  // Example: penalize negative or extreme parameter values (soft bounds)
  nll += pow(CppAD::CondExpLt(r_cots, Type(1e-4), r_cots-Type(1e-4), Type(0.0)), 2) * 1e2;
  nll += pow(CppAD::CondExpGt(r_cots, Type(5.0), r_cots-Type(5.0), Type(0.0)), 2) * 1e2;
  // (Add similar penalties for other parameters as needed)

  // --- REPORTING ---
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}

/*
Model equations (numbered):

1. COTS predation on fast coral: pred_fast = alpha_fast * phi_fast * COTS * fast / (fast + 1)
2. COTS predation on slow coral: pred_slow = alpha_slow * phi_slow * COTS * slow / (slow + 1)
3. COTS population: 
   COTS_t = COTS_{t-1} + r_cots * COTS_{t-1} * (1 - COTS_{t-1}/K_cots) * food_lim * sst_effect
            + immigration - pred_fast * e_fast - pred_slow * e_slow
   where food_lim = coral_total_prev / (coral_total_prev + K_food)
4. Fast coral: 
   fast_t = fast_{t-1} + r_fast * fast_{t-1} * (1 - (fast_{t-1} + slow_{t-1})/K_fast) - pred_fast
5. Slow coral: 
   slow_t = slow_{t-1} + r_slow * slow_{t-1} * (1 - (fast_{t-1} + slow_{t-1})/K_slow) - pred_slow
6. SST effect: sst_effect = exp(beta_sst * (sst - 27))
7. Immigration: imm_input = eps_imm * cotsimm_dat
8. Selectivity: phi_fast, phi_slow (bounded 0-1 via logit transform)
9. All predictions use only previous time step values (no data leakage)
*/
