#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(cots_dat);    // COTS abundance observations (individuals/m^2)
  DATA_VECTOR(slow_dat);    // Slow-growing coral cover observations (%)
  DATA_VECTOR(fast_dat);    // Fast-growing coral cover observations (%)
  DATA_VECTOR(sst_dat);     // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat); // COTS immigration rate (individuals/m^2/year)
  
  // Parameters
  PARAMETER(r_cots);      // COTS intrinsic growth rate
  PARAMETER(K_cots);      // COTS carrying capacity
  PARAMETER(r_slow);      // Slow coral growth rate
  PARAMETER(r_fast);      // Fast coral growth rate
  PARAMETER(K_slow);      // Slow coral carrying capacity
  PARAMETER(K_fast);      // Fast coral carrying capacity
  PARAMETER(alpha_slow);  // COTS predation rate on slow corals
  PARAMETER(alpha_fast);  // COTS predation rate on fast corals
  PARAMETER(temp_opt);    // Optimal temperature for COTS
  PARAMETER(temp_width);  // Temperature tolerance width
  PARAMETER(beta_cots);   // Density-dependent reduction in predation
  
  // Standard deviations for observations
  PARAMETER(log_sigma_cots);
  PARAMETER(log_sigma_slow);
  PARAMETER(log_sigma_fast);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Convert log parameters with minimum standard deviation
  Type sigma_cots = exp(log_sigma_cots) + Type(0.01);  // Minimum SD of 0.01
  Type sigma_slow = exp(log_sigma_slow) + Type(0.01);
  Type sigma_fast = exp(log_sigma_fast) + Type(0.01);
  
  // Small constant to prevent division by zero
  const Type eps = Type(1e-8);
  
  // Vectors for predictions
  vector<Type> cots_pred(cots_dat.size());
  vector<Type> slow_pred(slow_dat.size());
  vector<Type> fast_pred(fast_dat.size());
  
  // Initial conditions
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Time series simulation
  for(int t = 1; t < cots_dat.size(); t++) {
    // 1. Temperature effect on COTS (smoother response)
    Type temp_diff = (sst_dat(t) - temp_opt) / temp_width;
    Type temp_effect = Type(1.0) / (Type(1.0) + pow(temp_diff, 2));
    
    // 2. Total coral cover (with minimum to prevent division by zero)
    Type total_coral = slow_pred(t-1) + fast_pred(t-1) + eps;
    
    // 3. Feeding preference weighting
    Type pref_slow = slow_pred(t-1) / total_coral;
    Type pref_fast = fast_pred(t-1) / total_coral;
    
    // 4. COTS population dynamics
    Type cots_growth = r_cots * temp_effect * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1)/K_cots);
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t);
    
    // 5. Coral dynamics with smooth transitions
    Type pred_pressure_slow = alpha_slow * cots_pred(t) * pref_slow;
    Type pred_pressure_fast = alpha_fast * cots_pred(t) * pref_fast;
    
    slow_pred(t) = slow_pred(t-1) + r_slow * slow_pred(t-1) * (Type(1.0) - slow_pred(t-1)/K_slow) - pred_pressure_slow;
    fast_pred(t) = fast_pred(t-1) + r_fast * fast_pred(t-1) * (Type(1.0) - fast_pred(t-1)/K_fast) - pred_pressure_fast;
    
    // Ensure predictions stay positive using conditional operator
    slow_pred(t) = slow_pred(t) > Type(0.0) ? slow_pred(t) : Type(0.0);
    fast_pred(t) = fast_pred(t) > Type(0.0) ? fast_pred(t) : Type(0.0);
  }
  
  // Observation model using lognormal distribution
  for(int t = 0; t < cots_dat.size(); t++) {
    // Add small constant to observations and predictions
    Type cots_obs = cots_dat(t) + eps;
    Type slow_obs = slow_dat(t) + eps;
    Type fast_obs = fast_dat(t) + eps;
    
    Type cots_model = cots_pred(t) + eps;
    Type slow_model = slow_pred(t) + eps;
    Type fast_model = fast_pred(t) + eps;
    
    nll -= dnorm(log(cots_obs), log(cots_model), sigma_cots, true);
    nll -= dnorm(log(slow_obs), log(slow_model), sigma_slow, true);
    nll -= dnorm(log(fast_obs), log(fast_model), sigma_fast, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  // Report derived quantities
  ADREPORT(sigma_cots);
  ADREPORT(sigma_slow);
  ADREPORT(sigma_fast);
  
  return nll;
}
