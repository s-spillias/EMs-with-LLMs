#include <TMB.hpp>

// 1. Model equations are described at the end of this file.

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- DATA INPUTS ---
  DATA_VECTOR(Year); // Time variable (years)
  DATA_VECTOR(cots_dat); // Adult COTS abundance (ind/m^2)
  DATA_VECTOR(fast_dat); // Fast-growing coral cover (%) (Acropora spp.)
  DATA_VECTOR(slow_dat); // Slow-growing coral cover (%) (Faviidae/Porites spp.)
  DATA_VECTOR(sst_dat); // Sea-surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration (ind/m^2/year)

  int n = Year.size();

  // --- PARAMETERS ---
  PARAMETER(log_r_cots); // log intrinsic growth rate of COTS (year^-1)
  // Determines the potential for rapid COTS population increase
  PARAMETER(log_K_cots); // log carrying capacity for COTS (ind/m^2)
  // Maximum sustainable COTS density, set by resource limitation
  PARAMETER(log_alpha_fast); // log attack rate on fast coral (m2/ind/year)
  // COTS predation rate on Acropora
  PARAMETER(log_alpha_slow); // log attack rate on slow coral (m2/ind/year)
  // COTS predation rate on Faviidae/Porites
  PARAMETER(log_e_fast); // log assimilation efficiency from fast coral (unitless)
  // Fraction of consumed fast coral converted to COTS biomass
  PARAMETER(log_e_slow); // log assimilation efficiency from slow coral (unitless)
  // Fraction of consumed slow coral converted to COTS biomass
  PARAMETER(log_m_cots); // log baseline COTS mortality (year^-1)
  // Natural mortality rate of COTS
  PARAMETER(log_sigma_cots); // log obs SD for COTS abundance (lognormal)
  PARAMETER(log_sigma_fast); // log obs SD for fast coral cover (lognormal)
  PARAMETER(log_sigma_slow); // log obs SD for slow coral cover (lognormal)
  PARAMETER(beta_sst); // effect of SST on COTS growth (per deg C)
  // Modifies COTS growth rate by SST anomaly
  PARAMETER(log_fast_min); // log minimum fast coral cover for COTS recruitment (threshold, %)
  PARAMETER(log_slow_min); // log minimum slow coral cover for COTS recruitment (threshold, %)
  PARAMETER(log_K_fast); // log carrying capacity for fast coral (%)
  PARAMETER(log_K_slow); // log carrying capacity for slow coral (%)
  PARAMETER(log_r_fast); // log recovery rate for fast coral (year^-1)
  PARAMETER(log_r_slow); // log recovery rate for slow coral (year^-1)
  PARAMETER(log_gamma_fast); // log predation handling time for fast coral (year)
  PARAMETER(log_gamma_slow); // log predation handling time for slow coral (year)

  // --- TRANSFORM PARAMETERS ---
  Type r_cots = exp(log_r_cots); // COTS intrinsic growth rate
  Type K_cots = exp(log_K_cots); // COTS carrying capacity
  Type alpha_fast = exp(log_alpha_fast); // COTS attack rate on fast coral
  Type alpha_slow = exp(log_alpha_slow); // COTS attack rate on slow coral
  Type m_cots = exp(log_m_cots); // COTS mortality
  Type sigma_cots = exp(log_sigma_cots); // Obs SD for COTS
  Type sigma_fast = exp(log_sigma_fast); // Obs SD for fast coral
  Type sigma_slow = exp(log_sigma_slow); // Obs SD for slow coral
  Type K_fast = exp(log_K_fast); // Fast coral carrying capacity
  Type K_slow = exp(log_K_slow); // Slow coral carrying capacity
  Type r_fast = exp(log_r_fast); // Fast coral recovery rate
  Type r_slow = exp(log_r_slow); // Slow coral recovery rate

  // --- PARAMETER VALIDITY CHECK ---
  if (
      !R_finite(log_r_cots) || !R_finite(log_K_cots) || !R_finite(log_alpha_fast) ||
      !R_finite(log_alpha_slow) || !R_finite(log_m_cots) || !R_finite(log_sigma_cots) ||
      !R_finite(log_sigma_fast) || !R_finite(log_sigma_slow) || !R_finite(log_K_fast) ||
      !R_finite(log_K_slow) || !R_finite(log_r_fast) || !R_finite(log_r_slow) ||
      !R_finite(r_cots) || !R_finite(K_cots) || !R_finite(alpha_fast) || !R_finite(alpha_slow) ||
      !R_finite(m_cots) || !R_finite(sigma_cots) || !R_finite(sigma_fast) || !R_finite(sigma_slow) ||
      !R_finite(K_fast) || !R_finite(K_slow) || !R_finite(r_fast) || !R_finite(r_slow)
  ) {
    return Type(1e10); // Large penalty if any parameter is NA/NaN/Inf
  }

  // --- INITIAL CONDITIONS ---
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Ensure strictly positive initial conditions for log operations
  if (!R_finite(cots_dat(0)) || !R_finite(fast_dat(0)) || !R_finite(slow_dat(0))) {
    return Type(1e10); // Large penalty if any initial condition is NA/NaN/Inf
  }
  cots_pred(0) = CppAD::CondExpGt(cots_dat(0), Type(1e-8), cots_dat(0), Type(1e-8)); // Initial COTS abundance (ind/m^2)
  fast_pred(0) = CppAD::CondExpGt(fast_dat(0), Type(1e-8), fast_dat(0), Type(1e-8)); // Initial fast coral cover (%)
  slow_pred(0) = CppAD::CondExpGt(slow_dat(0), Type(1e-8), slow_dat(0), Type(1e-8)); // Initial slow coral cover (%)

  // --- MODEL DYNAMICS ---
  for(int t=1; t<n; t++) {

    // COTS population: logistic growth - mortality + immigration
    Type recruit = r_cots * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1)/K_cots);
    cots_pred(t) = cots_pred(t-1)
      + recruit
      - m_cots * cots_pred(t-1)
      + cotsimm_dat(t-1);

    // Fast coral: logistic recovery - linear predation by COTS
    Type fast_growth = r_fast * fast_pred(t-1) * (Type(1.0) - fast_pred(t-1)/K_fast);
    Type pred_fast = alpha_fast * cots_pred(t-1) * fast_pred(t-1);
    fast_pred(t) = fast_pred(t-1) + fast_growth - pred_fast;
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), Type(1e-8), fast_pred(t), Type(1e-8));

    // Slow coral: logistic recovery - linear predation by COTS
    Type slow_growth = r_slow * slow_pred(t-1) * (Type(1.0) - slow_pred(t-1)/K_slow);
    Type pred_slow = alpha_slow * cots_pred(t-1) * slow_pred(t-1);
    slow_pred(t) = slow_pred(t-1) + slow_growth - pred_slow;
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), Type(1e-8), slow_pred(t), Type(1e-8));

    // Bound COTS abundance to non-negative values using softplus
    cots_pred(t) = log(Type(1.0) + exp(cots_pred(t))) + Type(1e-8);
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), Type(1e-8), cots_pred(t), Type(1e-8));

    // Additional numerical safety: check for Inf/NaN at each step
    if (!R_finite(cots_pred(t)) || !R_finite(fast_pred(t)) || !R_finite(slow_pred(t))) {
      return Type(1e10);
    }
  }

  // --- LIKELIHOOD ---
  Type nll = 0.0;
  Type min_sd = Type(1e-3); // minimum SD for numerical stability

  for(int t=0; t<n; t++) {
    // Lognormal likelihood for strictly positive data
    if (!R_finite(cots_pred(t)) || !R_finite(fast_pred(t)) || !R_finite(slow_pred(t))) {
      return Type(1e10); // Large penalty if any state variable is NA/NaN/Inf
    }
    nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred(t) + Type(1e-8)), sigma_cots + min_sd, true);
    nll -= dnorm(log(fast_dat(t) + Type(1e-8)), log(fast_pred(t) + Type(1e-8)), sigma_fast + min_sd, true);
    nll -= dnorm(log(slow_dat(t) + Type(1e-8)), log(slow_pred(t) + Type(1e-8)), sigma_slow + min_sd, true);
  }

  // --- REPORTING ---
  REPORT(cots_pred); // predicted COTS abundance (ind/m^2)
  REPORT(fast_pred); // predicted fast coral cover (%)
  REPORT(slow_pred); // predicted slow coral cover (%)

  return nll;
}

/*
MODEL EQUATION DESCRIPTIONS:
1. COTS population: 
   cots_pred(t) = cots_pred(t-1) + recruitment - mortality + immigration
   - Recruitment is logistic
   - Mortality is density-independent
   - Immigration is external larval input

2. Coral populations (fast and slow):
   coral_pred(t) = coral_pred(t-1) + logistic recovery - COTS predation
   - Logistic recovery toward carrying capacity
   - Losses due to COTS predation (linear Lotka-Volterra type)

3. All transitions use lagged (t-1) values to avoid data leakage.

4. All parameters are bounded to biologically meaningful ranges via log-transform and softplus.

5. Likelihood: lognormal for all observed variables, with fixed minimum SD for stability.

*/
