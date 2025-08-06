#include <TMB.hpp>

// 1. Data inputs (units and descriptions provided by the data file):
//    - Year: Time (year)
//    - cots_dat: Adult Crown-of-Thorns starfish abundance (individuals/m^2)
//    - slow_dat: Slow-growing coral cover (percentage)
//    - fast_dat: Fast-growing coral cover (percentage)
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data declarations
  DATA_VECTOR(Year);       // Time series (years)
  DATA_VECTOR(cots_dat);   // Observed COTS density (individuals/m^2)
  DATA_VECTOR(slow_dat);   // Observed slow coral cover (%)
  DATA_VECTOR(fast_dat);   // Observed fast coral cover (%)

  // Parameter declarations (log-transformed when necessary)
  PARAMETER(log_growth_rate);    // log(Intrinsic growth rate in year^-1)
  PARAMETER(log_predation_effect); // log(Predation efficiency coefficient; unitless)
  PARAMETER(sigma_cots);         // Standard deviation of COTS observations (log-space)
  PARAMETER(sigma_coral);        // Standard deviation for coral observations (log-space)
  PARAMETER(cots_init);          // Initial COTS density (individuals/m^2)
  PARAMETER(slow_init);          // Initial slow coral cover (%)
  PARAMETER(fast_init);          // Initial fast coral cover (%)
  PARAMETER(log_r_slow);         // log intrinsic recovery rate for slow coral
  PARAMETER(log_r_fast);         // log intrinsic recovery rate for fast coral
  PARAMETER(coral_capacity);     // Maximum coral cover (%) acting as carrying capacity

  // Transformed parameters
  Type growth_rate = exp(log_growth_rate);           // (year^-1)
  Type predation_effect = exp(log_predation_effect);   // (unitless)
  Type r_slow = exp(log_r_slow);
  Type r_fast = exp(log_r_fast);

  // Number of time steps
  int n = cots_dat.size();
  
  // Predicted state vectors
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  
  // Initial conditions setup
  cots_pred(0) = cots_init;  // [1] Initial COTS density (individuals/m^2)
  slow_pred(0) = slow_init;  // [2] Initial slow coral cover (%)
  fast_pred(0) = fast_init;  // [3] Initial fast coral cover (%)
  
  // Small constant for numerical stability
  Type small = Type(1e-8);

  // Loop over time steps; use only values from previous steps to avoid data leakage
  for(int t = 1; t < n; t++){
    // Equation (1): COTS population dynamics
    //   Growth is saturating (limited by coral cover) and reduced by predation on slow coral.
    cots_pred(t) = cots_pred(t-1) 
                  + growth_rate * cots_pred(t-1) * (1 - cots_pred(t-1) / (1 + slow_pred(t-1)*small + fast_pred(t-1)*small))
                  - predation_effect * cots_pred(t-1) * (slow_pred(t-1) / (slow_pred(t-1) + small));
    
    // Equation (2): Slow coral dynamics
    //   Recovery towards maximum cover reduced by competition for space and COTS predation.
    slow_pred(t) = slow_pred(t-1)
                   + r_slow * slow_pred(t-1) * (1 - (slow_pred(t-1) + fast_pred(t-1))/coral_capacity)
                   - predation_effect * cots_pred(t-1) * (slow_pred(t-1) / (slow_pred(t-1) + small));
    
    // Equation (3): Fast coral dynamics
    //   Faster recovery towards maximum cover reduced by competition for space and COTS predation.
    fast_pred(t) = fast_pred(t-1)
                   + r_fast * fast_pred(t-1) * (1 - (slow_pred(t-1) + fast_pred(t-1))/coral_capacity)
                   - predation_effect * cots_pred(t-1) * (fast_pred(t-1) / (fast_pred(t-1) + small));
    
    // Ensure non-negative states using smooth transitions
    cots_pred(t) = cots_pred(t) > small ? cots_pred(t) : small;
    slow_pred(t) = slow_pred(t) > small ? slow_pred(t) : small;
    fast_pred(t) = fast_pred(t) > small ? fast_pred(t) : small;
  }
  
  // Likelihood calculation using lognormal distributions
  Type nll = 0.0;
  // Looping over all time steps
  for(int t = 0; t < n; t++){
    nll -= (dnorm(log(cots_dat(t) + small), log(cots_pred(t) + small), sigma_cots + small, true) - log(cots_dat(t) + small));
    nll -= (dnorm(log(slow_dat(t) + small), log(slow_pred(t) + small), sigma_coral + small, true) - log(slow_dat(t) + small));
    nll -= (dnorm(log(fast_dat(t) + small), log(fast_pred(t) + small), sigma_coral + small, true) - log(fast_dat(t) + small));
  }
  
  // Reporting predictions for inspection
  REPORT(cots_pred);  // Predicted COTS density (individuals/m^2)
  REPORT(slow_pred);  // Predicted slow coral cover (%)
  REPORT(fast_pred);  // Predicted fast coral cover (%)
  
  return nll;
}
