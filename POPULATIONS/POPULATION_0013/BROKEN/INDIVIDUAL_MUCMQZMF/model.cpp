#include <TMB.hpp>

// 1. Model equations are described at the end of this file.

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- DATA INPUTS ---
  DATA_VECTOR(Year); // Observation year (integer)
  DATA_VECTOR(cots_dat); // Observed adult COTS density (indiv/m2)
  DATA_VECTOR(fast_dat); // Observed fast-growing coral cover (%) (Acropora)
  DATA_VECTOR(slow_dat); // Observed slow-growing coral cover (%) (Faviidae, Porites)
  DATA_VECTOR(sst_dat); // Sea-surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration (indiv/m2/year)

  int n = Year.size();

  // --- PARAMETERS ---
  PARAMETER(log_r_cots); // log intrinsic COTS growth rate (year^-1)
  PARAMETER(log_K_cots); // log COTS carrying capacity (indiv/m2)
  PARAMETER(log_alpha_fast); // log COTS attack rate on fast coral (m2/indiv/year)
  PARAMETER(log_alpha_slow); // log COTS attack rate on slow coral (m2/indiv/year)
  PARAMETER(log_e_fast); // log assimilation efficiency from fast coral (unitless)
  PARAMETER(log_e_slow); // log assimilation efficiency from slow coral (unitless)
  PARAMETER(log_m_cots); // log natural mortality rate of COTS (year^-1)
  PARAMETER(logit_theta_sst); // logit SST sensitivity (unitless, 0-1)
  PARAMETER(log_sigma_cots); // log SD for COTS obs (lognormal)
  PARAMETER(log_sigma_fast); // log SD for fast coral obs (lognormal)
  PARAMETER(log_sigma_slow); // log SD for slow coral obs (lognormal)
  PARAMETER(log_r_fast); // log growth rate of fast coral (year^-1)
  PARAMETER(log_r_slow); // log growth rate of slow coral (year^-1)
  PARAMETER(log_K_fast); // log carrying capacity fast coral (%)
  PARAMETER(log_K_slow); // log carrying capacity slow coral (%)
  PARAMETER(log_m_fast); // log background mortality fast coral (year^-1)
  PARAMETER(log_m_slow); // log background mortality slow coral (year^-1)
  PARAMETER(logit_phi_outbreak); // logit outbreak threshold (unitless, 0-1)
  PARAMETER(log_immig_scale); // log scale for larval immigration effect
  PARAMETER(log_K_coral_effect); // log half-saturation constant for coral effect on COTS growth

  // --- TRANSFORM PARAMETERS ---
  Type r_cots = exp(log_r_cots); // COTS intrinsic growth rate
  Type K_cots = exp(log_K_cots); // COTS carrying capacity
  Type alpha_fast = exp(log_alpha_fast); // COTS attack rate on fast coral
  Type alpha_slow = exp(log_alpha_slow); // COTS attack rate on slow coral
  Type e_fast = exp(log_e_fast); // Assimilation efficiency fast coral
  Type e_slow = exp(log_e_slow); // Assimilation efficiency slow coral
  Type m_cots = exp(log_m_cots); // COTS natural mortality
  Type theta_sst = Type(1)/(Type(1)+exp(-logit_theta_sst)); // SST sensitivity [0,1]
  Type sigma_cots = exp(log_sigma_cots); // SD for COTS obs
  Type sigma_fast = exp(log_sigma_fast); // SD for fast coral obs
  Type sigma_slow = exp(log_sigma_slow); // SD for slow coral obs
  Type r_fast = exp(log_r_fast); // Fast coral growth rate
  Type r_slow = exp(log_r_slow); // Slow coral growth rate
  Type K_fast = exp(log_K_fast); // Fast coral carrying capacity
  Type K_slow = exp(log_K_slow); // Slow coral carrying capacity
  Type m_fast = exp(log_m_fast); // Fast coral background mortality
  Type m_slow = exp(log_m_slow); // Slow coral background mortality
  Type phi_outbreak = Type(1)/(Type(1)+exp(-logit_phi_outbreak)); // Outbreak threshold [0,1]
  Type immig_scale = exp(log_immig_scale); // Immigration effect scale
  Type K_coral_effect = exp(log_K_coral_effect); // Half-saturation constant for coral effect

  // --- INITIAL STATES ---
  // Use small positive values if initial data is NA/negative to avoid NA/NaN in gradients
  Type cots_prev = (CppAD::Value(cots_dat(0)) != CppAD::Value(cots_dat(0)) || !(cots_dat(0) > 0) ? Type(1e-3) : cots_dat(0)); // Initial COTS density (indiv/m2)
  Type fast_prev = (CppAD::Value(fast_dat(0)) != CppAD::Value(fast_dat(0)) || !(fast_dat(0) > 0) ? Type(1e-3) : fast_dat(0)); // Initial fast coral cover (%)
  Type slow_prev = (CppAD::Value(slow_dat(0)) != CppAD::Value(slow_dat(0)) || !(slow_dat(0) > 0) ? Type(1e-3) : slow_dat(0)); // Initial slow coral cover (%)

  // --- SMALL CONSTANTS FOR NUMERICAL STABILITY ---
  Type eps = Type(1e-8);

  // --- OUTPUT VECTORS ---
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // --- PROCESS MODEL ---
  for(int t=1; t<n; t++){
    // 1. Resource limitation: total available coral (sum of fast and slow)
    Type total_coral_prev = fast_prev + slow_prev + eps;

    // 2. Functional response: COTS predation on coral (Holling Type II)
    Type pred_fast = alpha_fast * cots_prev * fast_prev / (fast_prev + slow_prev + eps); // predation on fast coral
    Type pred_slow = alpha_slow * cots_prev * slow_prev / (fast_prev + slow_prev + eps); // predation on slow coral

    // 3. COTS population growth (logistic, modified by coral availability and SST)
    // Michaelis-Menten (saturating) resource limitation for coral effect
    Type coral_resource = fast_prev * e_fast + slow_prev * e_slow;
    // Avoid division by zero or negative values in denominator
    Type coral_effect = coral_resource / (CppAD::CondExpGt(K_coral_effect + coral_resource, eps, K_coral_effect + coral_resource, eps)); // saturating effect (0-1)
    Type sst_effect = 1.0 + theta_sst * (sst_dat(t) - 27.0); // SST modifies growth (centered at 27C)
    Type immig_effect = immig_scale * cotsimm_dat(t); // immigration pulse

    // Outbreak trigger: smooth threshold on coral_effect
    Type outbreak_boost = 1.0 + phi_outbreak * (coral_effect - 0.5);

    Type cots_growth = r_cots * cots_prev * (1.0 - cots_prev / (K_cots + eps)) * coral_effect * sst_effect * outbreak_boost;
    Type cots_mortality = m_cots * cots_prev;

    Type cots_next = cots_prev + cots_growth - cots_mortality + immig_effect;
    cots_next = CppAD::CondExpGt(cots_next, eps, cots_next, eps); // Bound to >= eps

    // 4. Coral dynamics (logistic growth minus COTS predation and background mortality)
    Type fast_growth = r_fast * fast_prev * (1.0 - fast_prev / (K_fast + eps));
    Type fast_mortality = m_fast * fast_prev;
    Type fast_next = fast_prev + fast_growth - pred_fast - fast_mortality;
    fast_next = CppAD::CondExpGt(fast_next, eps, fast_next, eps);

    Type slow_growth = r_slow * slow_prev * (1.0 - slow_prev / (K_slow + eps));
    Type slow_mortality = m_slow * slow_prev;
    Type slow_next = slow_prev + slow_growth - pred_slow - slow_mortality;
    slow_next = CppAD::CondExpGt(slow_next, eps, slow_next, eps);

    // Store predictions
    cots_pred(t) = cots_next;
    fast_pred(t) = fast_next;
    slow_pred(t) = slow_next;

    // Update for next step
    cots_prev = cots_next;
    fast_prev = fast_next;
    slow_prev = slow_next;
  }

  // Set initial predictions to observed initial values
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // --- LIKELIHOOD ---
  Type nll = 0.0;
  for(int t=0; t<n; t++){
    // Lognormal likelihood for strictly positive data
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots + eps, true);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_fast + eps, true);
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_slow + eps, true);
  }

  // --- REPORTING ---
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  // --- EQUATION DESCRIPTIONS ---
  /*
  1. COTS predation on coral: Holling Type II functional response, partitioned by coral type.
  2. COTS population growth: Logistic, modified by coral availability, SST, and outbreak threshold.
  3. Coral growth: Logistic, minus COTS predation and background mortality.
  4. Outbreaks triggered by high coral cover and/or larval immigration.
  5. All rates and effects are bounded and smoothed for numerical stability.
  */

  return nll;
}
