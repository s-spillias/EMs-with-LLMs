#include <TMB.hpp>

// Crown-of-thorns starfish outbreak model on the Great Barrier Reef
// Predictions for:
//   cots_pred (adult density, individuals/m2)
//   fast_pred (fast-growing Acropora % cover)
//   slow_pred (slow-growing Porites/Faviidae % cover)

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --------------------------
  // 1. DATA INPUTS
  // --------------------------
  DATA_VECTOR(Year);         // Observation years
  DATA_VECTOR(cots_dat);     // Observed adult COTS density (indiv/m2)
  DATA_VECTOR(fast_dat);     // Observed % cover fast-growing corals
  DATA_VECTOR(slow_dat);     // Observed % cover slow-growing corals
  DATA_VECTOR(sst_dat);      // Forcing: SST (deg C)
  DATA_VECTOR(cotsimm_dat);  // Forcing: Larval immigration rate (indiv/m2/year)

  int n = Year.size();        // number of timesteps
  Type eps = Type(1e-8);      // small constant for stability

  // --------------------------
  // 2. PARAMETERS
  // --------------------------

  PARAMETER(log_cots_r);           // log intrinsic COTS recruitment rate (year^-1)
  PARAMETER(log_cots_m);           // log natural COTS mortality (year^-1)
  PARAMETER(log_cots_k);           // log density-dependent self-regulation strength
  PARAMETER(log_attack_fast);      // log predation rate on Acropora
  PARAMETER(log_attack_slow);      // log predation rate on Porites/Faviidae
  PARAMETER(log_coral_growth_fast);// log intrinsic growth rate fast corals (year^-1)
  PARAMETER(log_coral_growth_slow);// log intrinsic growth rate slow corals (year^-1)
  PARAMETER(log_coral_K);          // log total carrying capacity coral (% cover)
  PARAMETER(beta_sst);             // effect of SST anomaly on COTS recruitment
  PARAMETER(log_proc_sd);          // log process error SD
  PARAMETER(log_obs_sd);           // log observation error SD

  // Transform parameters to natural scale
  Type cots_r = exp(log_cots_r);               // COTS recruitment rate (>0)
  Type cots_m = exp(log_cots_m);               // COTS mortality (>0)
  Type cots_k = exp(log_cots_k);               // density regulation (>0)
  Type attack_fast = exp(log_attack_fast);     // predation strength fast corals
  Type attack_slow = exp(log_attack_slow);     // predation strength slow corals
  Type coral_growth_fast = exp(log_coral_growth_fast);
  Type coral_growth_slow = exp(log_coral_growth_slow);
  Type coral_K = exp(log_coral_K);
  Type proc_sd = exp(log_proc_sd) + Type(1e-3);
  Type obs_sd = exp(log_obs_sd) + Type(1e-3);

  // --------------------------
  // 3. STATE VECTORS
  // --------------------------
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // --------------------------
  // 4. INITIAL CONDITIONS
  // --------------------------
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // --------------------------
  // 5. PROCESS EQUATIONS
  // --------------------------
  // compute SST mean once for anomaly calculation
  Type sst_mean = sst_dat.mean();
  for(int t=1; t<n; t++){
    // Environmental effect on COTS recruitment:
    // scaling factor = exp(beta_sst * (SST_t - long-term mean))
    Type sst_anom = sst_dat(t-1) - sst_mean;
    Type env_factor = exp(beta_sst * sst_anom);

    // (1) COTS dynamics: boom-bust with density dependence + immigration
    Type recruit = cots_r * cots_pred(t-1) * exp(-cots_k * cots_pred(t-1)) * env_factor;
    Type imm = cotsimm_dat(t-1);
    Type mortality = cots_m * cots_pred(t-1);
    cots_pred(t) = cots_pred(t-1) + recruit + imm - mortality;
    if(cots_pred(t) < eps) cots_pred(t) = eps;

    // (2) Coral dynamics
    // Logistic growth toward carrying capacity, with differential recovery rates
    Type total_coral_prev = fast_pred(t-1) + slow_pred(t-1);
    Type avail_K = coral_K - total_coral_prev;
    if(avail_K < 0) avail_K = 0;

    // predation mortality following a Type II functional response
    Type pred_fast = attack_fast * cots_pred(t-1) * fast_pred(t-1) /
                     (Type(1.0) + attack_fast * fast_pred(t-1) + attack_slow * slow_pred(t-1) + eps);
    Type pred_slow = attack_slow * cots_pred(t-1) * slow_pred(t-1) /
                     (Type(1.0) + attack_fast * fast_pred(t-1) + attack_slow * slow_pred(t-1) + eps);

    fast_pred(t) = fast_pred(t-1) + coral_growth_fast * fast_pred(t-1) * (avail_K/coral_K) - pred_fast;
    slow_pred(t) = slow_pred(t-1) + coral_growth_slow * slow_pred(t-1) * (avail_K/coral_K) - pred_slow;

    if(fast_pred(t) < eps) fast_pred(t) = eps;
    if(slow_pred(t) < eps) slow_pred(t) = eps;
  }

  // --------------------------
  // 6. LIKELIHOOD
  // --------------------------
  Type nll = 0.0;
  for(int t=0; t<n; t++){
    // lognormal likelihood for positive data (manual form since dlnorm not available in TMB)
    Type mu = log(cots_pred(t) + eps);
    Type y = log(cots_dat(t) + eps);
    nll -= dnorm(y, mu, obs_sd, true);

    // normal likelihoods for coral cover
    nll -= dnorm(fast_dat(t), fast_pred(t), obs_sd, true);
    nll -= dnorm(slow_dat(t), slow_pred(t), obs_sd, true);
  }

  // --------------------------
  // 7. REPORTING
  // --------------------------
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}

/*
Equation overview:
1. COTS dynamics follow a density-dependent recruitment function with SST-modulated success and added larval immigration.
2. Coral communities grow logistically toward carrying capacity, with different rates for fast vs slow growers.
3. COTS reduce coral cover through a saturating functional response for each coral group.
4. Outbreaks emerge as positive feedbacks between coral food supply, immigration, and intrinsic COTS growth outweigh density dependence and mortality.
*/
