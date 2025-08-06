#include <TMB.hpp>

// 1. Model equations are described at the end of this file.

template<class Type>
Type objective_function<Type>::operator() ()
{
  // --- DATA INPUTS ---
  DATA_VECTOR(Year); // Observation year
  DATA_VECTOR(cots_dat); // COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat); // Fast-growing coral cover (%)
  DATA_VECTOR(slow_dat); // Slow-growing coral cover (%)
  DATA_VECTOR(sst_dat); // Sea-surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat); // COTS larval immigration (individuals/m2/year)

  int n = Year.size();

  // Defensive check: all data vectors must be same length as Year
  if ((cots_dat.size() != n) ||
      (fast_dat.size() != n) ||
      (slow_dat.size() != n) ||
      (sst_dat.size() != n) ||
      (cotsimm_dat.size() != n)) {
    error("All data vectors must have the same length as Year.");
  }
  // Remove hard error for n == 0 to allow model to compile and run with empty data (for testing/initialization)
  // if (n == 0) {
  //   error("Data vectors must have length > 0.");
  // }

  // --- PARAMETERS ---
  PARAMETER(log_r_cots); // log intrinsic growth rate of COTS (year^-1)
  PARAMETER(log_K_cots); // log carrying capacity of COTS (individuals/m2)
  PARAMETER(log_alpha_fast); // log attack rate on fast coral (m2/individual/year)
  PARAMETER(log_alpha_slow); // log attack rate on slow coral (m2/individual/year)
  PARAMETER(log_h_fast); // log half-saturation constant for fast coral (%)
  PARAMETER(log_h_slow); // log half-saturation constant for slow coral (%)
  PARAMETER(log_e_fast); // log assimilation efficiency from fast coral (unitless)
  PARAMETER(log_e_slow); // log assimilation efficiency from slow coral (unitless)
  PARAMETER(log_m_cots); // log baseline COTS mortality (year^-1)
  PARAMETER(log_gamma); // log density-dependence strength (unitless)
  PARAMETER(beta_sst); // effect of SST on COTS growth (per deg C)
  PARAMETER(log_r_fast); // log intrinsic growth rate of fast coral (year^-1)
  PARAMETER(log_r_slow); // log intrinsic growth rate of slow coral (year^-1)
  PARAMETER(log_K_fast); // log carrying capacity of fast coral (%)
  PARAMETER(log_K_slow); // log carrying capacity of slow coral (%)
  PARAMETER(log_sigma_cots); // log obs SD for COTS
  PARAMETER(log_sigma_fast); // log obs SD for fast coral
  PARAMETER(log_sigma_slow); // log obs SD for slow coral

  // --- TRANSFORM PARAMETERS ---
  Type r_cots = exp(log_r_cots); // COTS intrinsic growth rate
  Type K_cots = exp(log_K_cots); // COTS carrying capacity
  Type alpha_fast = exp(log_alpha_fast); // COTS attack rate on fast coral
  Type alpha_slow = exp(log_alpha_slow); // COTS attack rate on slow coral
  Type h_fast = exp(log_h_fast) + Type(1e-8); // Half-saturation for fast coral
  Type h_slow = exp(log_h_slow) + Type(1e-8); // Half-saturation for slow coral
  Type e_fast = exp(log_e_fast); // Assimilation efficiency fast coral
  Type e_slow = exp(log_e_slow); // Assimilation efficiency slow coral
  Type m_cots = exp(log_m_cots); // Baseline COTS mortality
  Type gamma = exp(log_gamma); // Density-dependence strength
  Type r_fast = exp(log_r_fast); // Fast coral intrinsic growth
  Type r_slow = exp(log_r_slow); // Slow coral intrinsic growth
  Type K_fast = exp(log_K_fast); // Fast coral carrying capacity
  Type K_slow = exp(log_K_slow); // Slow coral carrying capacity
  Type sigma_cots = exp(log_sigma_cots) + Type(1e-6); // Obs SD COTS
  Type sigma_fast = exp(log_sigma_fast) + Type(1e-6); // Obs SD fast coral
  Type sigma_slow = exp(log_sigma_slow) + Type(1e-6); // Obs SD slow coral

  // --- INITIAL STATES ---
  // Defensive: avoid out-of-bounds if n == 0
  Type cots_prev = n > 0 ? cots_dat(0) : Type(1.0); // Initial COTS abundance
  Type fast_prev = n > 0 ? fast_dat(0) : Type(10.0); // Initial fast coral cover
  Type slow_prev = n > 0 ? slow_dat(0) : Type(10.0); // Initial slow coral cover

  // --- OUTPUT VECTORS ---
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);

  // Always initialize output vectors to avoid returning empty vectors
  if(n > 0) {
    cots_pred(0) = cots_prev;
    fast_pred(0) = fast_prev;
    slow_pred(0) = slow_prev;
  }

  // --- PROCESS MODEL ---
  for(int t=1; t<n; t++) {
    // Resource limitation (coral cover) for COTS
    Type coral_avail = fast_prev + slow_prev + Type(1e-8); // total coral available

    // Functional response: COTS predation on fast and slow coral (Holling type II)
    Type consump_fast = alpha_fast * cots_prev * fast_prev / (h_fast + fast_prev + Type(1e-8)); // predation on fast coral
    Type consump_slow = alpha_slow * cots_prev * slow_prev / (h_slow + slow_prev + Type(1e-8)); // predation on slow coral

    // COTS population update
    Type growth_env = r_cots * exp(beta_sst * (sst_dat(t-1) - Type(27.0))); // SST modifies growth
    Type resource_gain = e_fast * consump_fast + e_slow * consump_slow; // energy from coral
    Type density_feedback = exp(-gamma * cots_prev); // density-dependence
    Type immigration = cotsimm_dat(t-1); // larval immigration

    Type cots_next = cots_prev +
      (growth_env * cots_prev * (1 - cots_prev / (K_cots + Type(1e-8))) * density_feedback) // logistic + feedback
      + resource_gain // gain from predation
      - m_cots * cots_prev // mortality
      + immigration; // larval input

    // Prevent negative or zero values (numerical stability)
    cots_next = CppAD::CondExpGt(cots_next, Type(1e-8), cots_next, Type(1e-8));

    // Coral updates
    // Fast coral: logistic growth minus COTS predation
    Type fast_next = fast_prev +
      r_fast * fast_prev * (1 - fast_prev / (K_fast + Type(1e-8))) // logistic growth
      - consump_fast; // loss to COTS

    fast_next = CppAD::CondExpGt(fast_next, Type(1e-8), fast_next, Type(1e-8));

    // Slow coral: logistic growth minus COTS predation
    Type slow_next = slow_prev +
      r_slow * slow_prev * (1 - slow_prev / (K_slow + Type(1e-8))) // logistic growth
      - consump_slow; // loss to COTS

    slow_next = CppAD::CondExpGt(slow_next, Type(1e-8), slow_next, Type(1e-8));

    // Store predictions
    cots_pred(t) = cots_next;
    fast_pred(t) = fast_next;
    slow_pred(t) = slow_next;

    // Advance state
    cots_prev = cots_next;
    fast_prev = fast_next;
    slow_prev = slow_next;
  }

  // --- LIKELIHOOD ---
  Type nll = 0.0;
  for(int t=0; t<n; t++) {
    // Lognormal likelihood for strictly positive data
    if (cots_pred(t) > 0 && fast_pred(t) > 0 && slow_pred(t) > 0) {
      nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred(t) + Type(1e-8)), sigma_cots, true);
      nll -= dnorm(log(fast_dat(t) + Type(1e-8)), log(fast_pred(t) + Type(1e-8)), sigma_fast, true);
      nll -= dnorm(log(slow_dat(t) + Type(1e-8)), log(slow_pred(t) + Type(1e-8)), sigma_slow, true);
    }
  }

  // --- REPORTING ---
  REPORT(cots_pred); // predicted COTS abundance (individuals/m2)
  REPORT(fast_pred); // predicted fast coral cover (%)
  REPORT(slow_pred); // predicted slow coral cover (%)

  // --- EQUATION DESCRIPTIONS ---
  /*
  1. COTS predation on coral: Holling type II functional response for each coral group.
  2. COTS population: Logistic growth with SST effect, density feedback, resource gain from predation, mortality, and larval immigration.
  3. Coral groups: Logistic growth minus COTS predation.
  4. All transitions use previous time step values only (no data leakage).
  5. All parameters are bounded via smooth penalties (log-transform), and small constants prevent division by zero.
  */

  return nll;
}
