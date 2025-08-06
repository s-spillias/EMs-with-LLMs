#include <TMB.hpp>

// 1. Adult COTS population dynamics: boom-bust cycles driven by resource availability, environmental forcing, and density dependence
// 2. Fast-growing coral (Acropora) and slow-growing coral (Faviidae/Porites) cover: reduced by COTS predation, recover via growth
// 3. Resource limitation: Michaelis-Menten (saturating) and threshold effects on COTS recruitment/survival
// 4. Environmental drivers: SST and larval immigration modulate COTS recruitment
// 5. Feedbacks: Coral depletion reduces COTS carrying capacity; coral recovery after COTS decline
// 6. All predictions (_pred) use only previous time step values of state variables

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- DATA ---
  DATA_VECTOR(Year); // Observation years
  DATA_VECTOR(cots_dat); // Adult COTS abundance (indiv/m2)
  DATA_VECTOR(fast_dat); // Fast-growing coral cover (%)
  DATA_VECTOR(slow_dat); // Slow-growing coral cover (%)
  DATA_VECTOR(sst_dat); // Sea-surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration (indiv/m2/yr)

  int n = Year.size();

  // --- PARAMETERS ---
  PARAMETER(log_r_cots); // log intrinsic COTS recruitment rate (log(year^-1))
  PARAMETER(log_K_cots); // log COTS carrying capacity (log(indiv/m2))
  PARAMETER(log_alpha_cots); // log COTS density-dependence strength (log(indiv/m2)^-1)
  PARAMETER(log_beta_fast); // log COTS predation rate on fast coral (log(% cover)^-1 yr^-1)
  PARAMETER(log_beta_slow); // log COTS predation rate on slow coral (log(% cover)^-1 yr^-1)
  PARAMETER(log_gamma_fast); // log recovery rate of fast coral (log(% cover)^-1 yr^-1)
  PARAMETER(log_gamma_slow); // log recovery rate of slow coral (log(% cover)^-1 yr^-1)
  PARAMETER(log_sigma_cots); // log obs SD for COTS (lognormal)
  PARAMETER(log_sigma_fast); // log obs SD for fast coral (lognormal)
  PARAMETER(log_sigma_slow); // log obs SD for slow coral (lognormal)
  PARAMETER(log_sst_effect); // log effect of SST on COTS recruitment (unitless)
  PARAMETER(log_immig_effect); // log effect of larval immigration on COTS recruitment (unitless)
  PARAMETER(log_thresh_coral); // log coral cover threshold for COTS recruitment (log(% cover))
  PARAMETER(log_min_sd); // log minimum SD for numerical stability

  // --- TRANSFORM PARAMETERS ---
  Type r_cots = exp(log_r_cots); // COTS recruitment rate (year^-1)
  Type K_cots = exp(log_K_cots); // COTS carrying capacity (indiv/m2)
  Type alpha_cots = exp(log_alpha_cots); // COTS density-dependence (indiv/m2)^-1
  Type beta_fast = exp(log_beta_fast); // COTS predation rate on fast coral (% cover)^-1 yr^-1
  Type beta_slow = exp(log_beta_slow); // COTS predation rate on slow coral (% cover)^-1 yr^-1
  Type gamma_fast = exp(log_gamma_fast); // Fast coral recovery rate (% cover)^-1 yr^-1
  Type gamma_slow = exp(log_gamma_slow); // Slow coral recovery rate (% cover)^-1 yr^-1
  Type sigma_cots = exp(log_sigma_cots); // Obs SD for COTS
  Type sigma_fast = exp(log_sigma_fast); // Obs SD for fast coral
  Type sigma_slow = exp(log_sigma_slow); // Obs SD for slow coral
  Type sst_effect = exp(log_sst_effect); // SST effect (unitless, >=0)
  Type immig_effect = exp(log_immig_effect); // Immigration effect (unitless, >=0)
  Type thresh_coral = exp(log_thresh_coral); // Coral cover threshold (% cover)
  Type min_sd = exp(log_min_sd); // Minimum SD for numerical stability

  // --- INITIAL STATES ---
  // Defensive: check input vectors are non-empty
  if(n == 0) error("Input time series must have at least one time step.");
  if(cots_dat.size() != n || fast_dat.size() != n || slow_dat.size() != n || sst_dat.size() != n || cotsimm_dat.size() != n)
    error("All input data vectors must have the same length as Year.");

  Type cots_prev = cots_dat(0); // Initial COTS abundance (indiv/m2)
  Type fast_prev = fast_dat(0); // Initial fast coral cover (%)
  Type slow_prev = slow_dat(0); // Initial slow coral cover (%)

  // --- STORAGE FOR PREDICTIONS ---
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  cots_pred(0) = cots_prev;
  fast_pred(0) = fast_prev;
  slow_pred(0) = slow_prev;

  // --- MODEL DYNAMICS ---
  for(int t=1; t<n; t++) {
    // 1. Total coral cover (resource for COTS)
    Type coral_total_prev = fast_prev + slow_prev + Type(1e-8); // % cover, avoid zero

    // 2. Resource limitation: Michaelis-Menten + threshold
    Type resource_lim = coral_total_prev / (coral_total_prev + thresh_coral); // [0,1], saturating
    // Smooth threshold: resource_lim ~0 if coral_total_prev << thresh_coral

    // 3. Environmental effects on COTS recruitment
    Type env_effect = pow(sst_dat(t-1)/Type(27.0), sst_effect) * (Type(1.0) + immig_effect * cotsimm_dat(t-1)); // SST and immigration

    // 4. COTS recruitment (boom): resource limitation, environmental forcing, density dependence
    // Prevent denominator from being zero or negative
    Type denom = Type(1.0) + alpha_cots * cots_prev;
    denom = CppAD::CondExpGt(denom, Type(1e-8), denom, Type(1e-8));
    Type cots_recruit = r_cots * cots_prev * resource_lim * env_effect / denom;

    // 5. COTS mortality (bust): density dependence, resource depletion
    Type cots_mortality = cots_prev * (Type(1.0) - resource_lim); // More mortality if coral is low

    // 6. Update COTS
    cots_pred(t) = cots_prev + cots_recruit - cots_mortality;
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), Type(1e-8), cots_pred(t), Type(1e-8)); // Bound away from zero

    // 7. Coral predation by COTS (selective)
    Type coral_sum = fast_prev + slow_prev + Type(1e-8); // Avoid zero denominator
    Type pred_fast = beta_fast * cots_prev * fast_prev / coral_sum; // Prefer fast coral
    Type pred_slow = beta_slow * cots_prev * slow_prev / coral_sum;

    // 8. Coral recovery (logistic)
    Type fast_recovery = gamma_fast * fast_prev * (Type(100.0) - fast_prev - slow_prev) / Type(100.0); // % cover, max 100%
    Type slow_recovery = gamma_slow * slow_prev * (Type(100.0) - fast_prev - slow_prev) / Type(100.0);

    // 9. Update corals
    fast_pred(t) = fast_prev + fast_recovery - pred_fast;
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), Type(1e-8), fast_pred(t), Type(1e-8)); // Bound away from zero
    fast_pred(t) = CppAD::CondExpLt(fast_pred(t), Type(100.0), fast_pred(t), Type(100.0)); // Bound above by 100%

    slow_pred(t) = slow_prev + slow_recovery - pred_slow;
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), Type(1e-8), slow_pred(t), Type(1e-8)); // Bound away from zero
    slow_pred(t) = CppAD::CondExpLt(slow_pred(t), Type(100.0), slow_pred(t), Type(100.0)); // Bound above by 100%

    // 10. Advance state
    cots_prev = cots_pred(t);
    fast_prev = fast_pred(t);
    slow_prev = slow_pred(t);
  }

  // --- LIKELIHOOD ---
  Type nll = 0.0;
  for(int t=0; t<n; t++) {
    // Use lognormal likelihoods, with minimum SD for stability
    Type sd_cots = sqrt(sigma_cots*sigma_cots + min_sd*min_sd);
    Type sd_fast = sqrt(sigma_fast*sigma_fast + min_sd*min_sd);
    Type sd_slow = sqrt(sigma_slow*sigma_slow + min_sd*min_sd);

    // Defensive: ensure predictions are positive for log
    nll -= dnorm(log((cots_dat(t) > 0 ? cots_dat(t) : Type(1e-8))), log((cots_pred(t) > 0 ? cots_pred(t) : Type(1e-8))), sd_cots, true);
    nll -= dnorm(log((fast_dat(t) > 0 ? fast_dat(t) : Type(1e-8))), log((fast_pred(t) > 0 ? fast_pred(t) : Type(1e-8))), sd_fast, true);
    nll -= dnorm(log((slow_dat(t) > 0 ? slow_dat(t) : Type(1e-8))), log((slow_pred(t) > 0 ? slow_pred(t) : Type(1e-8))), sd_slow, true);
  }

  // --- REPORTING ---
  REPORT(cots_pred); // Predicted COTS abundance (indiv/m2)
  REPORT(fast_pred); // Predicted fast coral cover (%)
  REPORT(slow_pred); // Predicted slow coral cover (%)

  return nll;
}
