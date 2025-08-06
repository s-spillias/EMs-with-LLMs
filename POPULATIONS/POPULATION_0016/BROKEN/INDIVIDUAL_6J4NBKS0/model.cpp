#include <TMB.hpp>

// 1. Model equations are described at the end of this file.

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- DATA INPUTS ---
  DATA_VECTOR(Year); // Time variable (years)
  DATA_VECTOR(cots_dat); // Observed COTS abundance (indiv/m2)
  DATA_VECTOR(fast_dat); // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat); // Observed slow-growing coral cover (%)
  DATA_VECTOR(sst_dat); // Sea surface temperature (deg C)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration (indiv/m2/year)

  int n = Year.size();

  // --- PARAMETERS ---
  PARAMETER(log_r_cots); // log intrinsic growth rate of COTS (year^-1)
  PARAMETER(log_K_cots); // log carrying capacity of COTS (indiv/m2)
  PARAMETER(log_alpha_fast); // log attack rate on fast coral (m2/indiv/year)
  PARAMETER(log_alpha_slow); // log attack rate on slow coral (m2/indiv/year)
  PARAMETER(log_h_fast); // log half-saturation constant for fast coral (%)
  PARAMETER(log_h_slow); // log half-saturation constant for slow coral (%)
  PARAMETER(log_e_fast); // log conversion efficiency (fast coral to COTS)
  PARAMETER(log_e_slow); // log conversion efficiency (slow coral to COTS)
  PARAMETER(log_m_cots); // log background mortality rate of COTS (year^-1)
  PARAMETER(beta_sst); // effect of SST on COTS growth (unitless)
  PARAMETER(log_r_fast); // log regrowth rate of fast coral (year^-1)
  PARAMETER(log_r_slow); // log regrowth rate of slow coral (year^-1)
  PARAMETER(log_K_fast); // log carrying capacity of fast coral (%)
  PARAMETER(log_K_slow); // log carrying capacity of slow coral (%)
  PARAMETER(log_sigma_cots); // log obs SD for COTS (lognormal)
  PARAMETER(log_sigma_fast); // log obs SD for fast coral (lognormal)
  PARAMETER(log_sigma_slow); // log obs SD for slow coral (lognormal)

  // --- TRANSFORM PARAMETERS ---
  Type r_cots = exp(log_r_cots); // COTS intrinsic growth rate (year^-1)
  Type K_cots = exp(log_K_cots); // COTS carrying capacity (indiv/m2)
  Type alpha_fast = exp(log_alpha_fast); // COTS attack rate on fast coral (m2/indiv/year)
  Type alpha_slow = exp(log_alpha_slow); // COTS attack rate on slow coral (m2/indiv/year)
  Type h_fast = exp(log_h_fast); // Half-saturation for fast coral (%)
  Type h_slow = exp(log_h_slow); // Half-saturation for slow coral (%)
  Type e_fast = exp(log_e_fast); // Conversion efficiency (fast coral to COTS)
  Type e_slow = exp(log_e_slow); // Conversion efficiency (slow coral to COTS)
  Type m_cots = exp(log_m_cots); // COTS background mortality (year^-1)
  Type r_fast = exp(log_r_fast); // Fast coral regrowth rate (year^-1)
  Type r_slow = exp(log_r_slow); // Slow coral regrowth rate (year^-1)
  Type K_fast = exp(log_K_fast); // Fast coral carrying capacity (%)
  Type K_slow = exp(log_K_slow); // Slow coral carrying capacity (%)
  Type sigma_cots = exp(log_sigma_cots) + Type(1e-8); // SD for COTS obs
  Type sigma_fast = exp(log_sigma_fast) + Type(1e-8); // SD for fast coral obs
  Type sigma_slow = exp(log_sigma_slow) + Type(1e-8); // SD for slow coral obs

  // --- INITIAL STATES ---
  Type cots_pred_t = (cots_dat.size() > 0) ? cots_dat(0) : Type(1e-8); // Initial COTS abundance (indiv/m2)
  Type fast_pred_t = (fast_dat.size() > 0) ? fast_dat(0) : Type(1e-8); // Initial fast coral cover (%)
  Type slow_pred_t = (slow_dat.size() > 0) ? slow_dat(0) : Type(1e-8); // Initial slow coral cover (%)

  // --- OUTPUT VECTORS ---
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // --- LIKELIHOOD ---
  Type nll = 0.0;

  // --- MODEL DYNAMICS ---
  for(int t=0; t<n; t++) {
    // Save predictions
    cots_pred(t) = cots_pred_t;
    fast_pred(t) = fast_pred_t;
    slow_pred(t) = slow_pred_t;

    // Likelihoods (lognormal, always include all obs)
    nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred_t + Type(1e-8)), sigma_cots, true);
    nll -= dnorm(log(fast_dat(t) + Type(1e-8)), log(fast_pred_t + Type(1e-8)), sigma_fast, true);
    nll -= dnorm(log(slow_dat(t) + Type(1e-8)), log(slow_pred_t + Type(1e-8)), sigma_slow, true);

    // Skip update at last time step
    if(t == n-1) break;

    // --- COTS functional response (Holling Type II, resource limitation) ---
    Type predation_fast = alpha_fast * cots_pred_t * fast_pred_t / (h_fast + fast_pred_t + Type(1e-8)); // COTS predation on fast coral
    Type predation_slow = alpha_slow * cots_pred_t * slow_pred_t / (h_slow + slow_pred_t + Type(1e-8)); // COTS predation on slow coral

    // --- COTS population update ---
    // Outbreak triggers: immigration + SST effect
    Type env_effect = exp(beta_sst * (sst_dat(t) - Type(27.0))); // SST modifies growth (centered at 27C)
    Type cots_growth = r_cots * cots_pred_t * (1 - cots_pred_t / (K_cots + Type(1e-8))) * env_effect;
    Type cots_gain = cots_growth + e_fast * predation_fast + e_slow * predation_slow + cotsimm_dat(t);
    Type cots_loss = m_cots * cots_pred_t;
    Type cots_next = cots_pred_t + cots_gain - cots_loss;

    // Smooth penalty to keep COTS abundance positive
    cots_next = CppAD::CondExpGt(cots_next, Type(1e-8), cots_next, Type(1e-8));

    // --- Coral population updates (logistic regrowth minus COTS predation) ---
    Type fast_regrow = r_fast * fast_pred_t * (1 - fast_pred_t / (K_fast + Type(1e-8)));
    Type fast_next = fast_pred_t + fast_regrow - predation_fast;
    fast_next = CppAD::CondExpGt(fast_next, Type(1e-8), fast_next, Type(1e-8));

    Type slow_regrow = r_slow * slow_pred_t * (1 - slow_pred_t / (K_slow + Type(1e-8)));
    Type slow_next = slow_pred_t + slow_regrow - predation_slow;
    slow_next = CppAD::CondExpGt(slow_next, Type(1e-8), slow_next, Type(1e-8));

    // Advance state
    cots_pred_t = cots_next;
    fast_pred_t = fast_next;
    slow_pred_t = slow_next;
  }

  // --- REPORTING ---
  REPORT(cots_pred); // Model-predicted COTS abundance (indiv/m2)
  REPORT(fast_pred); // Model-predicted fast coral cover (%)
  REPORT(slow_pred); // Model-predicted slow coral cover (%)

  return nll;
}

/*
MODEL EQUATIONS (numbered):

1. predation_fast = alpha_fast * COTS * Fast / (h_fast + Fast)
   (Saturating functional response for COTS predation on fast coral)

2. predation_slow = alpha_slow * COTS * Slow / (h_slow + Slow)
   (Saturating functional response for COTS predation on slow coral)

3. env_effect = exp(beta_sst * (SST - 27))
   (Environmental effect of SST on COTS growth, centered at 27C)

4. COTS_next = COTS + r_cots * COTS * (1 - COTS/K_cots) * env_effect
                     + e_fast * predation_fast + e_slow * predation_slow
                     + cotsimm_dat
                     - m_cots * COTS
   (COTS population update with resource limitation, environmental forcing, and immigration)

5. Fast_next = Fast + r_fast * Fast * (1 - Fast/K_fast) - predation_fast
   (Fast coral logistic regrowth minus COTS predation)

6. Slow_next = Slow + r_slow * Slow * (1 - Slow/K_slow) - predation_slow
   (Slow coral logistic regrowth minus COTS predation)

All transitions use smooth minimums to avoid negative states.
All likelihoods are lognormal with fixed minimum SDs for stability.
*/
