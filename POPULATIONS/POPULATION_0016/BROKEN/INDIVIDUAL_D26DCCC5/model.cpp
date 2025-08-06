#include <TMB.hpp>

// 1. Model equations describe the coupled dynamics of COTS, fast-growing coral, and slow-growing coral.
// 2. COTS population dynamics include intrinsic growth, resource limitation, larval immigration, and mortality.
// 3. Coral dynamics include growth, recovery, and selective predation by COTS.
// 4. Environmental variables (e.g., SST) modulate rates.
// 5. All predictions use lagged (previous time step) state variables to avoid data leakage.
// 6. Likelihoods are calculated for all observations using lognormal errors with fixed minimum SDs.

template<class Type>
Type objective_function<Type>::operator() ()
{
  // DATA INPUTS
  DATA_VECTOR(Year); // Time variable (years)
  DATA_VECTOR(cots_dat); // Adult COTS abundance (indiv/m2)
  DATA_VECTOR(fast_dat); // Fast-growing coral cover (%)
  DATA_VECTOR(slow_dat); // Slow-growing coral cover (%)
  DATA_VECTOR(sst_dat); // Sea Surface Temperature (deg C)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration (indiv/m2/year)

  int n = Year.size();

  // PARAMETERS
  PARAMETER(log_r_cots); // log intrinsic growth rate of COTS (year^-1)
  PARAMETER(log_K_cots); // log carrying capacity for COTS (indiv/m2)
  PARAMETER(log_m_cots); // log natural mortality rate of COTS (year^-1)
  PARAMETER(log_alpha_fast); // log attack rate on fast coral (m2/indiv/year)
  PARAMETER(log_alpha_slow); // log attack rate on slow coral (m2/indiv/year)
  PARAMETER(log_e_fast); // log efficiency of converting fast coral to COTS biomass
  PARAMETER(log_e_slow); // log efficiency of converting slow coral to COTS biomass
  PARAMETER(log_r_fast); // log recovery rate of fast coral (%/year)
  PARAMETER(log_r_slow); // log recovery rate of slow coral (%/year)
  PARAMETER(log_K_fast); // log carrying capacity for fast coral (% cover)
  PARAMETER(log_K_slow); // log carrying capacity for slow coral (% cover)
  PARAMETER(beta_SST); // effect of SST on COTS recruitment (unitless)
  PARAMETER(gamma_eff); // strength of density dependence in COTS conversion efficiency (unitless)
  PARAMETER(log_sigma_cots); // log SD for COTS obs
  PARAMETER(log_sigma_fast); // log SD for fast coral obs
  PARAMETER(log_sigma_slow); // log SD for slow coral obs

  // TRANSFORM PARAMETERS
  Type r_cots = exp(log_r_cots); // COTS intrinsic growth rate
  Type K_cots = exp(log_K_cots); // COTS carrying capacity
  Type m_cots = exp(log_m_cots); // COTS mortality rate
  Type alpha_fast = exp(log_alpha_fast); // COTS attack rate on fast coral
  Type alpha_slow = exp(log_alpha_slow); // COTS attack rate on slow coral
  Type e_fast = exp(log_e_fast); // Conversion efficiency fast coral
  Type e_slow = exp(log_e_slow); // Conversion efficiency slow coral
  Type r_fast = exp(log_r_fast); // Fast coral recovery rate
  Type r_slow = exp(log_r_slow); // Slow coral recovery rate
  Type K_fast = exp(log_K_fast); // Fast coral carrying capacity
  Type K_slow = exp(log_K_slow); // Slow coral carrying capacity
  Type sigma_cots = exp(log_sigma_cots); // SD for COTS obs
  Type sigma_fast = exp(log_sigma_fast); // SD for fast coral obs
  Type sigma_slow = exp(log_sigma_slow); // SD for slow coral obs
  // gamma_eff is already defined by PARAMETER(gamma_eff);

  // SMALL CONSTANTS FOR NUMERICAL STABILITY
  Type eps = Type(1e-8);
  Type min_sigma = Type(1e-3);

  // INITIAL STATES (set to first observed value)
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Ensure initial states are positive (and fallback to eps if not)
  // Defensive: fallback to eps if input is <= 0 or not finite
  cots_pred(0) = (CppAD::isnan(cots_dat(0)) || !CppAD::isfinite(cots_dat(0)) || cots_dat(0) <= eps) ? eps : cots_dat(0);
  fast_pred(0) = (CppAD::isnan(fast_dat(0)) || !CppAD::isfinite(fast_dat(0)) || fast_dat(0) <= eps) ? eps : fast_dat(0);
  slow_pred(0) = (CppAD::isnan(slow_dat(0)) || !CppAD::isfinite(slow_dat(0)) || slow_dat(0) <= eps) ? eps : slow_dat(0);

  // MODEL DYNAMICS
  for(int t=1; t<n; t++) {
    // Environmental modifier for COTS recruitment (centered on mean SST)
    Type env_mod = exp(beta_SST * (sst_dat(t-1) - sst_dat.mean())); // 1. SST effect on COTS

    // Resource limitation for COTS (saturating function)
    Type resource = (alpha_fast * fast_pred(t-1) + alpha_slow * slow_pred(t-1)) / (alpha_fast + alpha_slow + eps); // 2. Weighted coral availability

    // Density-dependent conversion efficiency (new feedback)
    Type e_fast_eff = e_fast * exp(-gamma_eff * cots_pred(t-1));
    Type e_slow_eff = e_slow * exp(-gamma_eff * cots_pred(t-1));

    // COTS population update (Ricker-like with immigration, resource limitation, and mortality)
    // Use density-dependent efficiency in the growth term
    Type cots_growth = cots_pred(t-1) * exp(r_cots * resource * env_mod * (1 - cots_pred(t-1) / (K_cots + eps))) + cotsimm_dat(t-1);
    Type cots_surv = cots_growth * exp(-m_cots);
    cots_pred(t) = CppAD::CondExpGt(cots_surv, eps, cots_surv, eps); // 3. Prevent negative/zero

    // Coral predation (Type II functional response)
    Type pred_fast = (alpha_fast * cots_pred(t-1) * fast_pred(t-1)) / (1 + alpha_fast * cots_pred(t-1) * fast_pred(t-1) + eps);
    Type pred_slow = (alpha_slow * cots_pred(t-1) * slow_pred(t-1)) / (1 + alpha_slow * cots_pred(t-1) * slow_pred(t-1) + eps);

    // Fast coral update
    Type fast_growth = r_fast * fast_pred(t-1) * (1 - fast_pred(t-1) / (K_fast + eps));
    fast_pred(t) = fast_pred(t-1) + fast_growth - pred_fast;
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), eps, fast_pred(t), eps); // 4. Prevent negative/zero

    // Slow coral update
    Type slow_growth = r_slow * slow_pred(t-1) * (1 - slow_pred(t-1) / (K_slow + eps));
    slow_pred(t) = slow_pred(t-1) + slow_growth - pred_slow;
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), eps, slow_pred(t), eps); // 5. Prevent negative/zero
  }

  // LIKELIHOOD
  Type nll = 0.0;
  for(int t=0; t<n; t++) {
    // Lognormal likelihoods (log-transform for positive data)
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), (sigma_cots + min_sigma), true);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), (sigma_fast + min_sigma), true);
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), (sigma_slow + min_sigma), true);
  }

  // SOFT PENALTIES FOR PARAMETER BOUNDS (example: keep rates positive and within plausible ranges)
  nll += pow(CppAD::CondExpLt(r_cots, Type(0.01), r_cots - Type(0.01), Type(0)), 2); // r_cots > 0.01
  nll += pow(CppAD::CondExpGt(r_cots, Type(2.0), r_cots - Type(2.0), Type(0)), 2);   // r_cots < 2.0
  nll += pow(CppAD::CondExpLt(K_cots, Type(0.01), K_cots - Type(0.01), Type(0)), 2); // K_cots > 0.01
  nll += pow(CppAD::CondExpLt(m_cots, Type(0.001), m_cots - Type(0.001), Type(0)), 2); // m_cots > 0.001

  // REPORTING
  REPORT(cots_pred); // Predicted COTS abundance (indiv/m2)
  REPORT(fast_pred); // Predicted fast coral cover (%)
  REPORT(slow_pred); // Predicted slow coral cover (%)

  return nll;
}
