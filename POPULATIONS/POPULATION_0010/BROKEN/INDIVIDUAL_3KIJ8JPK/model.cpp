#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
// 1. DATA SECTION
// Data vectors (time series)
  DATA_VECTOR(Year); // Year (time variable, units: year)
  DATA_VECTOR(sst_dat); // Sea-Surface Temperature (deg C)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration rate (indiv/m2/year)
  DATA_VECTOR(cots_dat); // Adult COTS abundance (indiv/m2)
  DATA_VECTOR(fast_dat); // Fast-growing coral cover (Acropora spp., %)
  DATA_VECTOR(slow_dat); // Slow-growing coral cover (Faviidae/Porites spp., %)

  // 2. PARAMETER SECTION
  // COTS population parameters
  PARAMETER(log_r_cots); // log intrinsic growth rate of COTS (year^-1)
  PARAMETER(log_K_cots); // log carrying capacity for COTS (indiv/m2)
  PARAMETER(log_m_cots); // log baseline mortality rate of COTS (year^-1)
  PARAMETER(log_alpha_imm); // log immigration efficiency (unitless)
  PARAMETER(log_beta_sst); // log SST effect on COTS recruitment (unitless)
  PARAMETER(log_gamma_pred); // log predation efficiency on coral (unitless)
  PARAMETER(log_delta_reslim); // log resource limitation threshold (coral % cover)
  PARAMETER(log_sigma_cots); // log SD of COTS observation error (lognormal)
  PARAMETER(logit_phi_outbreak); // logit probability of outbreak trigger (unitless)

  // Coral parameters
  PARAMETER(log_r_fast); // log regrowth rate of fast coral (year^-1)
  PARAMETER(log_r_slow); // log regrowth rate of slow coral (year^-1)
  PARAMETER(log_K_fast); // log carrying capacity of fast coral (% cover)
  PARAMETER(log_K_slow); // log carrying capacity of slow coral (% cover)
  PARAMETER(log_m_fast); // log baseline mortality of fast coral (year^-1)
  PARAMETER(log_m_slow); // log baseline mortality of slow coral (year^-1)
  PARAMETER(log_gamma_fast); // log COTS predation rate on fast coral (unitless)
  PARAMETER(log_gamma_slow); // log COTS predation rate on slow coral (unitless)
  PARAMETER(log_sigma_fast); // log SD of fast coral obs error (lognormal)
  PARAMETER(log_sigma_slow); // log SD of slow coral obs error (lognormal)

  // 3. TRANSFORM PARAMETERS TO NATURAL SCALE
  Type r_cots = exp(log_r_cots); // COTS intrinsic growth rate (year^-1)
  Type K_cots = exp(log_K_cots); // COTS carrying capacity (indiv/m2)
  Type m_cots = exp(log_m_cots); // COTS baseline mortality (year^-1)
  Type alpha_imm = exp(log_alpha_imm); // Immigration efficiency
  Type beta_sst = exp(log_beta_sst); // SST effect on recruitment
  Type gamma_pred = exp(log_gamma_pred); // Predation efficiency
  Type delta_reslim = exp(log_delta_reslim); // Resource limitation threshold (%)
  Type phi_outbreak = 1/(1+exp(-logit_phi_outbreak)); // Outbreak trigger probability (0-1)

  Type r_fast = exp(log_r_fast); // Fast coral regrowth rate (year^-1)
  Type r_slow = exp(log_r_slow); // Slow coral regrowth rate (year^-1)
  Type K_fast = exp(log_K_fast); // Fast coral carrying capacity (%)
  Type K_slow = exp(log_K_slow); // Slow coral carrying capacity (%)
  Type m_fast = exp(log_m_fast); // Fast coral baseline mortality (year^-1)
  Type m_slow = exp(log_m_slow); // Slow coral baseline mortality (year^-1)
  Type gamma_fast = exp(log_gamma_fast); // COTS predation rate on fast coral
  Type gamma_slow = exp(log_gamma_slow); // COTS predation rate on slow coral

  Type sigma_cots = exp(log_sigma_cots); // SD for COTS obs error
  Type sigma_fast = exp(log_sigma_fast); // SD for fast coral obs error
  Type sigma_slow = exp(log_sigma_slow); // SD for slow coral obs error

  // 4. INITIAL CONDITIONS
  int n = Year.size();
  vector<Type> cots_pred(n); // Predicted COTS abundance
  vector<Type> fast_pred(n); // Predicted fast coral cover
  vector<Type> slow_pred(n); // Predicted slow coral cover

  // Set initial conditions from data
  if(n > 0) {
    cots_pred(0) = CppAD::CondExpGt(cots_dat(0), Type(0), cots_dat(0), Type(1e-8));
    fast_pred(0) = CppAD::CondExpGt(fast_dat(0), Type(0), fast_dat(0), Type(1e-8));
    slow_pred(0) = CppAD::CondExpGt(slow_dat(0), Type(0), slow_dat(0), Type(1e-8));
  }
  // No need to fill with sentinel values; let the process model fill all values.

  // 5. PROCESS MODEL
  for(int t=1; t<n; t++) {
    // 1. Resource limitation: total coral cover
    Type coral_total_prev = fast_pred(t-1) + slow_pred(t-1) + Type(1e-8); // % cover, add small constant

    // 2. Immigration: episodic, modulated by SST and threshold
    Type imm_effect = alpha_imm * cotsimm_dat(t) * (1 + beta_sst * (sst_dat(t) - Type(27.0))); // SST modifies immigration
    // Outbreak trigger: smooth threshold on immigration
    Type outbreak_prob = 1/(1+exp(-10*(imm_effect - delta_reslim))); // Smooth transition

    // 3. COTS population growth: logistic with resource limitation and outbreak trigger
    Type resource_lim = coral_total_prev/(coral_total_prev + delta_reslim); // saturating function
    Type cots_growth = r_cots * cots_pred(t-1) * (1 - cots_pred(t-1)/K_cots) * resource_lim;
    Type cots_imm = imm_effect * outbreak_prob;
    Type cots_mort = m_cots * cots_pred(t-1);

    Type cots_next = cots_pred(t-1) + cots_growth + cots_imm - cots_mort;
    cots_pred(t) = CppAD::CondExpGt(cots_next, Type(1e-8), cots_next, Type(1e-8)); // Bound to positive

    // 4. Coral predation: COTS selectively feed on corals
    Type denom_coral = fast_pred(t-1) + slow_pred(t-1) + Type(1e-8);
    Type pred_fast = gamma_fast * cots_pred(t-1) * fast_pred(t-1)/denom_coral;
    Type pred_slow = gamma_slow * cots_pred(t-1) * slow_pred(t-1)/denom_coral;

    // 5. Coral regrowth: logistic, resource-limited, with mortality and predation
    Type fast_next = fast_pred(t-1) + r_fast * fast_pred(t-1) * (1 - fast_pred(t-1)/K_fast) - pred_fast - m_fast * fast_pred(t-1);
    fast_pred(t) = CppAD::CondExpGt(fast_next, Type(1e-8), fast_next, Type(1e-8)); // Bound to positive

    Type slow_next = slow_pred(t-1) + r_slow * slow_pred(t-1) * (1 - slow_pred(t-1)/K_slow) - pred_slow - m_slow * slow_pred(t-1);
    slow_pred(t) = CppAD::CondExpGt(slow_next, Type(1e-8), slow_next, Type(1e-8)); // Bound to positive
  }

  // 6. LIKELIHOOD
  Type nll = 0.0;
  Type min_sd = Type(1e-3); // Minimum SD for numerical stability

  // 1. COTS abundance (lognormal likelihood)
  for(int t=0; t<n; t++) {
    Type obs = CppAD::CondExpGt(cots_dat(t), Type(0), cots_dat(t), Type(1e-8));
    Type pred = CppAD::CondExpGt(cots_pred(t), Type(0), cots_pred(t), Type(1e-8));
    nll -= dnorm(log(obs), log(pred), sigma_cots + min_sd, true);
  }

  // 2. Fast coral cover (lognormal likelihood)
  for(int t=0; t<n; t++) {
    Type obs = CppAD::CondExpGt(fast_dat(t), Type(0), fast_dat(t), Type(1e-8));
    Type pred = CppAD::CondExpGt(fast_pred(t), Type(0), fast_pred(t), Type(1e-8));
    nll -= dnorm(log(obs), log(pred), sigma_fast + min_sd, true);
  }

  // 3. Slow coral cover (lognormal likelihood)
  for(int t=0; t<n; t++) {
    Type obs = CppAD::CondExpGt(slow_dat(t), Type(0), slow_dat(t), Type(1e-8));
    Type pred = CppAD::CondExpGt(slow_pred(t), Type(0), slow_pred(t), Type(1e-8));
    nll -= dnorm(log(obs), log(pred), sigma_slow + min_sd, true);
  }

  // 7. REPORTING
  REPORT(cots_pred); // Predicted COTS abundance (indiv/m2)
  REPORT(fast_pred); // Predicted fast coral cover (%)
  REPORT(slow_pred); // Predicted slow coral cover (%)

  // 8. EQUATION DESCRIPTIONS
  /*
  Equation list:
  1. coral_total_prev = fast_pred(t-1) + slow_pred(t-1) + 1e-8; // total coral cover
  2. imm_effect = alpha_imm * cotsimm_dat(t) * (1 + beta_sst * (sst_dat(t) - 27.0)); // immigration modulated by SST
  3. outbreak_prob = 1/(1+exp(-10*(imm_effect - delta_reslim))); // smooth outbreak threshold
  4. resource_lim = coral_total_prev/(coral_total_prev + delta_reslim); // resource limitation
  5. cots_growth = r_cots * cots_pred(t-1) * (1 - cots_pred(t-1)/K_cots) * resource_lim; // logistic growth
  6. cots_imm = imm_effect * outbreak_prob; // effective immigration
  7. cots_mort = m_cots * cots_pred(t-1); // mortality
  8. pred_fast = gamma_fast * cots_pred(t-1) * fast_pred(t-1)/(fast_pred(t-1) + slow_pred(t-1) + 1e-8); // predation on fast coral
  9. pred_slow = gamma_slow * cots_pred(t-1) * slow_pred(t-1)/(fast_pred(t-1) + slow_pred(t-1) + 1e-8); // predation on slow coral
  10. fast_pred(t) = fast_pred(t-1) + r_fast * fast_pred(t-1) * (1 - fast_pred(t-1)/K_fast) - pred_fast - m_fast * fast_pred(t-1); // fast coral update
  11. slow_pred(t) = slow_pred(t-1) + r_slow * slow_pred(t-1) * (1 - slow_pred(t-1)/K_slow) - pred_slow - m_slow * slow_pred(t-1); // slow coral update
  */

  return nll;
}
