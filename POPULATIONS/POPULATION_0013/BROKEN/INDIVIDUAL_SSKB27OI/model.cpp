#include <TMB.hpp>

// 1. Adult COTS abundance (cots_pred) is driven by intrinsic growth, larval immigration, resource limitation (coral cover), and density-dependent mortality.
// 2. Fast-growing coral (fast_pred) and slow-growing coral (slow_pred) are reduced by COTS predation (with different susceptibilities) and recover via their own growth, subject to resource limitation and environmental effects.
// 3. Environmental variables (e.g., SST) and larval immigration are included as data vectors and can modify process rates.
// 4. All transitions use smooth saturating functions to avoid hard cutoffs and ensure numerical stability.
// 5. All predictions (_pred) use only previous time step values of state variables (no data leakage).

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- DATA INPUTS ---
  DATA_VECTOR(Year); // Time variable (years)
  DATA_VECTOR(cots_dat); // Observed adult COTS abundance (indiv/m2)
  DATA_VECTOR(fast_dat); // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat); // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat); // Sea-surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration (indiv/m2/year)

  int n = Year.size();

  // --- PARAMETERS ---
  PARAMETER(log_r_cots); // log intrinsic growth rate of COTS (year^-1)
  PARAMETER(log_K_cots); // log carrying capacity of COTS (indiv/m2)
  PARAMETER(log_alpha_fast); // log attack rate of COTS on fast coral (m2/indiv/year)
  PARAMETER(log_alpha_slow); // log attack rate of COTS on slow coral (m2/indiv/year)
  PARAMETER(log_e_fast); // log efficiency of converting fast coral to COTS biomass
  PARAMETER(log_e_slow); // log efficiency of converting slow coral to COTS biomass
  PARAMETER(log_m_cots); // log baseline mortality rate of COTS (year^-1)
  PARAMETER(log_r_fast); // log growth rate of fast coral (year^-1)
  PARAMETER(log_r_slow); // log growth rate of slow coral (year^-1)
  PARAMETER(log_K_fast); // log carrying capacity of fast coral (% cover)
  PARAMETER(log_K_slow); // log carrying capacity of slow coral (% cover)
  PARAMETER(beta_sst_cots); // effect of SST on COTS growth (unitless)
  PARAMETER(beta_sst_coral); // effect of SST on coral growth (unitless)
  PARAMETER(log_sigma_cots); // log SD for COTS obs (lognormal)
  PARAMETER(log_sigma_fast); // log SD for fast coral obs (lognormal)
  PARAMETER(log_sigma_slow); // log SD for slow coral obs (lognormal)

  // --- TRANSFORM PARAMETERS ---
  Type r_cots = exp(log_r_cots); // COTS intrinsic growth rate (year^-1)
  Type K_cots = exp(log_K_cots); // COTS carrying capacity (indiv/m2)
  Type alpha_fast = exp(log_alpha_fast); // COTS attack rate on fast coral (m2/indiv/year)
  Type alpha_slow = exp(log_alpha_slow); // COTS attack rate on slow coral (m2/indiv/year)
  Type e_fast = exp(log_e_fast); // COTS conversion efficiency from fast coral
  Type e_slow = exp(log_e_slow); // COTS conversion efficiency from slow coral
  Type m_cots = exp(log_m_cots); // COTS baseline mortality (year^-1)
  Type r_fast = exp(log_r_fast); // Fast coral growth rate (year^-1)
  Type r_slow = exp(log_r_slow); // Slow coral growth rate (year^-1)
  Type K_fast = exp(log_K_fast); // Fast coral carrying capacity (%)
  Type K_slow = exp(log_K_slow); // Slow coral carrying capacity (%)
  Type sigma_cots = exp(log_sigma_cots); // SD for COTS obs
  Type sigma_fast = exp(log_sigma_fast); // SD for fast coral obs
  Type sigma_slow = exp(log_sigma_slow); // SD for slow coral obs

  // --- INITIAL STATES ---
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Set initial states to observed values at t=0
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // --- MODEL DYNAMICS ---
  for(int t=1; t<n; t++) {
    // Resource limitation for COTS: saturating function of total coral cover
    Type coral_avail = fast_pred(t-1) + slow_pred(t-1) + Type(1e-8); // % cover, avoid zero
    Type resource_lim = coral_avail / (coral_avail + Type(10.0)); // Half-saturation at 10% cover

    // SST effect on COTS growth (centered at 27C)
    Type sst_effect_cots = exp(beta_sst_cots * (sst_dat(t-1) - Type(27.0)));

    // COTS predation on corals (Holling Type II functional response)
    Type pred_fast = alpha_fast * cots_pred(t-1) * fast_pred(t-1) / (fast_pred(t-1) + Type(5.0)); // 5% half-sat
    Type pred_slow = alpha_slow * cots_pred(t-1) * slow_pred(t-1) / (slow_pred(t-1) + Type(5.0));

    // COTS population update
    Type cots_growth = r_cots * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1)/K_cots) * resource_lim * sst_effect_cots;
    Type cots_gain = cots_growth + e_fast * pred_fast + e_slow * pred_slow + cotsimm_dat(t-1);
    Type cots_loss = m_cots * cots_pred(t-1) + Type(0.01) * pow(cots_pred(t-1), 2); // density-dependent loss
    cots_pred(t) = cots_pred(t-1) + cots_gain - cots_loss;
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), Type(1e-8), cots_pred(t), Type(1e-8)); // prevent negative

    // SST effect on coral growth (centered at 27C)
    Type sst_effect_coral = exp(beta_sst_coral * (sst_dat(t-1) - Type(27.0)));

    // Fast coral update
    Type fast_growth = r_fast * fast_pred(t-1) * (Type(1.0) - (fast_pred(t-1) + slow_pred(t-1))/K_fast) * sst_effect_coral;
    Type fast_loss = pred_fast;
    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_loss;
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), Type(1e-8), fast_pred(t), Type(1e-8)); // prevent negative

    // Slow coral update
    Type slow_growth = r_slow * slow_pred(t-1) * (Type(1.0) - (fast_pred(t-1) + slow_pred(t-1))/K_slow) * sst_effect_coral;
    Type slow_loss = pred_slow;
    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_loss;
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), Type(1e-8), slow_pred(t), Type(1e-8)); // prevent negative

    // Extra: check for NaN using TMB/CppAD-safe logic (replace with small value if not finite)
    // Remove Inf checks: CppAD::isinf is not available and ==self catches NaN only
    // Commented out: these checks can cause segfaults in TMB/CppAD context and are not recommended.
    // if (!(slow_pred(t) == slow_pred(t))) slow_pred(t) = Type(1e-8); // NaN check
    // if (!(fast_pred(t) == fast_pred(t))) fast_pred(t) = Type(1e-8);
    // if (!(cots_pred(t) == cots_pred(t))) cots_pred(t) = Type(1e-8);
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

  // --- REPORTING ---
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);

  return nll;
}
