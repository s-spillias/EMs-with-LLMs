#include <TMB.hpp>  // TMB library for statistical modeling

template<class Type>
Type objective_function<Type>::operator() () {
  
  // Data: Time and observations (using _dat suffix as in the provided data file)
  DATA_VECTOR(time);            // Time (Year)
  DATA_VECTOR(cots_dat);        // Observed COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);        // Observed fast-growing coral cover (%) for Acropora spp.
  DATA_VECTOR(slow_dat);        // Observed slow-growing coral cover (%) for Faviidae spp. and Porities spp.
  
  int n = time.size();  // Number of time steps
  
  // Parameters defining ecological dynamics:
  PARAMETER(growth_rate);          // (year^-1) Intrinsic growth rate of COTS
  PARAMETER(carrying_capacity);    // (individuals/m2) Carrying capacity for COTS population
  PARAMETER(predation_rate);       // Scaled impact of coral predation on COTS dynamics
  PARAMETER(consumption_rate_fast); // (per year) Consumption rate of fast-growing coral by COTS
  PARAMETER(consumption_rate_slow); // (per year) Consumption rate of slow-growing coral by COTS
  PARAMETER(recovery_rate_fast);    // (year^-1) Recovery rate of fast-growing coral
  PARAMETER(recovery_rate_slow);    // (year^-1) Recovery rate of slow-growing coral
  PARAMETER(log_sd_cots);          // Log-scale SD for COTS process error
  PARAMETER(log_sd_fast);          // Log-scale SD for fast coral process error
  PARAMETER(log_sd_slow);          // Log-scale SD for slow coral process error
  PARAMETER(outbreak_trigger_threshold);          // Coral cover threshold (%) for triggering COTS outbreak
  PARAMETER(growth_boost);                         // Multiplier for growth rate during outbreak conditions
  
  // Convert log standard deviations to SDs with a small constant for numerical stability.
  Type sd_cots = exp(log_sd_cots) + Type(1e-8);
  Type sd_fast = exp(log_sd_fast) + Type(1e-8);
  Type sd_slow = exp(log_sd_slow) + Type(1e-8);
  
  // Initialize prediction vectors (with _pred suffix matching _dat observations)
  vector<Type> cots_pred(n), fast_pred(n), slow_pred(n);
  
  // Set initial conditions from the first observation (ensuring no data leakage)
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Model Equations:
  // 1. COTS Dynamics: Logistic growth modulated by predation.
  // 2. Fast Coral Dynamics: Recovery offset by consumption with a saturating functional response.
  // 3. Slow Coral Dynamics: Similar recovery and consumption dynamics.
  // Note: Predictions at time t are computed using values from time t-1.
  for (int t = 1; t < n; t++) {
    Type effective_growth = growth_rate;
    if ((fast_pred(t-1) + slow_pred(t-1)) < outbreak_trigger_threshold)
      effective_growth = growth_rate * growth_boost;
    // (1) COTS dynamics: Growth minus reduction due to predation
    cots_pred(t) = cots_pred(t-1)
      + effective_growth * cots_pred(t-1) * (1 - cots_pred(t-1) / (carrying_capacity + Type(1e-8)))  // Logistic growth
      - predation_rate * (fast_pred(t-1) + slow_pred(t-1)) * cots_pred(t-1);                // Predation impact
    
    // (2) Fast coral dynamics: Recovery minus consumption by COTS (saturating response)
    fast_pred(t) = fast_pred(t-1)
      + recovery_rate_fast * (1 - fast_pred(t-1))  // Recovery process
      - consumption_rate_fast * cots_pred(t-1) * fast_pred(t-1) / (fast_pred(t-1) + Type(1e-8)); // Consumption pressure
    
    // (3) Slow coral dynamics: Recovery minus consumption by COTS (saturating response)
    slow_pred(t) = slow_pred(t-1)
      + recovery_rate_slow * (1 - slow_pred(t-1))  // Recovery process
      - consumption_rate_slow * cots_pred(t-1) * slow_pred(t-1) / (slow_pred(t-1) + Type(1e-8)); // Consumption pressure
  }
  
  // Likelihood Calculation:
  // Compare the model predictions (from the previous time step) with current observations.
  Type nll = 0.0;
  for (int t = 1; t < n; t++) {
    nll -= dnorm(cots_dat(t), cots_pred(t-1), sd_cots, true);  // COTS likelihood
    nll -= dnorm(fast_dat(t), fast_pred(t-1), sd_fast, true);     // Fast coral likelihood
    nll -= dnorm(slow_dat(t), slow_pred(t-1), sd_slow, true);     // Slow coral likelihood
  }
  
  // Report model predictions for further diagnostics.
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
