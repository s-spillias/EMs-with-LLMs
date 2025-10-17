#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // ------------------------------------------------------------------------
  // DATA INPUTS
  // ------------------------------------------------------------------------
  
  // Forcing data vectors
  DATA_VECTOR(Year);          // Year of observation
  DATA_VECTOR(sst_dat);       // Sea-Surface Temperature in Celsius
  DATA_VECTOR(cotsimm_dat);   // COTS larval immigration rate (individuals/m2/year)

  // Response data vectors for likelihood calculation
  DATA_VECTOR(cots_dat);      // Adult COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);      // Fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);      // Slow-growing coral cover (%)

  int n_obs = Year.size();    // Number of observations in the time-series

  // ------------------------------------------------------------------------
  // PARAMETERS
  // ------------------------------------------------------------------------

  // COTS parameters
  PARAMETER(m_cots);          // Natural mortality rate of COTS (year^-1). literature
  PARAMETER(m_cots_dd);       // COTS density-dependent mortality coefficient due to starvation. initial estimate
  PARAMETER(cots_repro_a);    // Maximum COTS recruitment rate (Beverton-Holt 'a' parameter, year^-1). initial estimate
  PARAMETER(cots_repro_b);    // COTS recruitment density-dependence (Beverton-Holt 'b' parameter, m2/individual). initial estimate
  PARAMETER(cots_allee);      // Allee effect threshold for COTS reproduction (individuals/m2). initial estimate
  
  // Predation parameters
  PARAMETER(P_max);           // Maximum predation rate by COTS per individual (% coral cover / year). literature
  PARAMETER(H_total);         // Half-saturation constant for COTS predation on total coral cover (%). literature
  PARAMETER(pref_fast);       // COTS preference for fast-growing corals (dimensionless, 0.5-1.0). literature

  // Coral parameters
  PARAMETER(r_fast);          // Intrinsic growth rate of fast-growing corals (year^-1). literature
  PARAMETER(r_slow);          // Intrinsic growth rate of slow-growing corals (year^-1). literature
  PARAMETER(K_coral_total);   // Total carrying capacity for all corals on the reef (%). initial estimate
  PARAMETER(sst_opt_coral);   // Optimal SST for coral growth (Celsius). literature
  PARAMETER(sst_width_coral); // SD of the thermal tolerance curve for coral growth (Celsius). initial estimate

  // Standard deviation parameters for the lognormal likelihood
  PARAMETER(sd_cots);         // SD for the lognormal error of COTS abundance. initial estimate
  PARAMETER(sd_fast);         // SD for the lognormal error of fast-growing coral cover. initial estimate
  PARAMETER(sd_slow);         // SD for the lognormal error of slow-growing coral cover. initial estimate

  // ------------------------------------------------------------------------
  // MODEL PREDICTIONS
  // ------------------------------------------------------------------------

  // Prediction vectors for state variables
  vector<Type> cots_pred(n_obs);
  vector<Type> fast_pred(n_obs);
  vector<Type> slow_pred(n_obs);

  // Initialize predictions with the first data point
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // Negative log-likelihood accumulator
  Type nll = 0.0;

  // Time-stepping loop for dynamic model
  for (int t = 1; t < n_obs; ++t) {
    
    // --- Intermediate variables for clarity (calculated at t-1) ---
    
    // Total available coral food for COTS
    Type total_coral_prev = fast_pred(t-1) + slow_pred(t-1);
    
    // Total coral eaten by COTS, using a Holling Type II functional response on total coral
    Type total_coral_eaten = (P_max * cots_pred(t-1) * total_coral_prev) / (H_total + total_coral_prev + Type(1e-8));
    
    // Partitioning of eaten coral based on preference for fast-growing corals
    Type fast_coral_preference_term = pref_fast * fast_pred(t-1);
    Type slow_coral_preference_term = (Type(1.0) - pref_fast) * slow_pred(t-1);
    Type total_preference_term = fast_coral_preference_term + slow_coral_preference_term + Type(1e-8);
    
    Type predation_on_fast = total_coral_eaten * (fast_coral_preference_term / total_preference_term);
    Type predation_on_slow = total_coral_eaten * (slow_coral_preference_term / total_preference_term);

    // Effect of SST on coral growth (Gaussian thermal performance curve)
    Type sst_effect_coral = exp(Type(-0.5) * pow((sst_dat(t-1) - sst_opt_coral) / sst_width_coral, 2));

    // --- Main model equations ---
    // Equation descriptions:
    // 1. COTS abundance: Previous abundance declines due to natural and starvation-based mortality, and increases with local reproduction and larval immigration.
    //    - Mortality is modeled with an exponential decay function. Starvation mortality increases as COTS density per unit of coral food increases.
    //    - Recruitment is a sum of local reproduction (Beverton-Holt model with an Allee effect) and external larval immigration.
    // 2. Fast-growing coral cover: Previous cover changes based on logistic growth, competition for space with slow corals, and predation by COTS.
    //    - Growth is influenced by the SST effect.
    // 3. Slow-growing coral cover: Similar to fast corals but with a lower intrinsic growth rate and lower predation pressure.
    
    // 1. COTS dynamics
    Type cots_starvation_rate = m_cots_dd * cots_pred(t-1) / (total_coral_prev + Type(1e-8));
    Type cots_bh_recruitment = (cots_repro_a * cots_pred(t-1)) / (Type(1.0) + cots_repro_b * cots_pred(t-1));
    Type allee_effect = cots_pred(t-1) / (cots_allee + cots_pred(t-1) + Type(1e-8));
    Type cots_recruits_local = cots_bh_recruitment * allee_effect;
    cots_pred(t) = cots_pred(t-1) * exp(-m_cots - cots_starvation_rate) + cots_recruits_local + cotsimm_dat(t-1);

    // 2. Fast-growing coral dynamics
    Type fast_growth = sst_effect_coral * r_fast * fast_pred(t-1) * (Type(1.0) - total_coral_prev / K_coral_total);
    fast_pred(t) = fast_pred(t-1) + fast_growth - predation_on_fast;
    
    // 3. Slow-growing coral dynamics
    Type slow_growth = sst_effect_coral * r_slow * slow_pred(t-1) * (Type(1.0) - total_coral_prev / K_coral_total);
    slow_pred(t) = slow_pred(t-1) + slow_growth - predation_on_slow;

    // --- Numerical stability constraints ---
    // Ensure state variables do not fall below zero
    if (cots_pred(t) < 0) cots_pred(t) = 0;
    if (fast_pred(t) < 0) fast_pred(t) = 0;
    if (slow_pred(t) < 0) slow_pred(t) = 0;
  }

  // ------------------------------------------------------------------------
  // LIKELIHOOD CALCULATION
  // ------------------------------------------------------------------------
  
  // Use a lognormal distribution for strictly positive data (abundance, cover)
  // This is equivalent to a normal distribution on the log-transformed data and predictions
  // A small constant is added to prevent log(0)
  for (int t = 0; t < n_obs; ++t) {
    nll -= dnorm(log(cots_dat(t) + Type(1e-8)), log(cots_pred(t) + Type(1e-8)), sd_cots, true);
    nll -= dnorm(log(fast_dat(t) + Type(1e-8)), log(fast_pred(t) + Type(1e-8)), sd_fast, true);
    nll -= dnorm(log(slow_dat(t) + Type(1e-8)), log(slow_pred(t) + Type(1e-8)), sd_slow, true);
  }

  // ------------------------------------------------------------------------
  // REPORTING SECTION
  // ------------------------------------------------------------------------
  
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
