#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs
  DATA_VECTOR(Year);           // Time series years
  DATA_VECTOR(cots_dat);       // Observed COTS density (individuals/m2)
  DATA_VECTOR(slow_dat);       // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);       // Observed fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);        // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);    // COTS immigration rate (individuals/m2/year)

  // Parameters
  PARAMETER(log_r_cots);       // COTS intrinsic growth rate (year^-1)
  PARAMETER(log_K_cots);       // COTS carrying capacity (individuals/m2)
  PARAMETER(log_alpha_slow);   // Maximum COTS predation rate on slow corals (m2/individual/year)
  PARAMETER(log_alpha_fast);   // Maximum COTS predation rate on fast corals (m2/individual/year)
  PARAMETER(log_h_slow);       // Half-saturation constant for predation on slow corals (%)
  PARAMETER(log_h_fast);       // Half-saturation constant for predation on fast corals (%)
  PARAMETER(log_r_slow);       // Slow coral intrinsic growth rate (year^-1)
  PARAMETER(log_r_fast);       // Fast coral intrinsic growth rate (year^-1)
  PARAMETER(log_temp_opt);     // Optimal temperature for COTS (Celsius)
  PARAMETER(log_temp_range);   // Temperature tolerance range (Celsius)
  
  // Convert log parameters
  Type r_cots = exp(log_r_cots);
  Type K_cots = exp(log_K_cots);
  Type alpha_slow = exp(log_alpha_slow); 
  Type alpha_fast = exp(log_alpha_fast);
  Type h_slow = exp(log_h_slow);
  Type h_fast = exp(log_h_fast);
  Type r_slow = exp(log_r_slow);
  Type r_fast = exp(log_r_fast);
  Type temp_opt = exp(log_temp_opt);
  Type temp_range = exp(log_temp_range);

  // Initialize vectors for predictions
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  
  // Set initial conditions
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Negative log-likelihood
  Type nll = 0;
  
  // Process model
  for(int t = 1; t < n; t++) {
    // 1. Temperature effect on COTS (Gaussian response)
    Type temp_diff = sst_dat(t-1) - temp_opt;
    Type temp_effect = exp(-0.5 * (temp_diff * temp_diff) / (temp_range * temp_range));
    
    // 2. Resource availability effect (total coral cover)
    Type total_coral = slow_pred(t-1) + fast_pred(t-1) + eps;
    Type resource_effect = total_coral / (total_coral + eps);
    
    // 3. COTS population dynamics
    Type density_effect = Type(1.0) - cots_pred(t-1)/K_cots;
    Type cots_growth = r_cots * cots_pred(t-1) * density_effect * temp_effect * resource_effect;
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t-1);
    
    // 4. Coral dynamics with differential predation
    Type coral_space = Type(1.0) - total_coral/Type(100.0);
    // Type II functional response with total coral in denominator for stability
    Type pred_pressure_slow = alpha_slow * cots_pred(t-1) * slow_pred(t-1) / (total_coral + eps);
    Type pred_pressure_fast = alpha_fast * cots_pred(t-1) * fast_pred(t-1) / (total_coral + eps);
    
    slow_pred(t) = slow_pred(t-1) + r_slow * slow_pred(t-1) * coral_space - pred_pressure_slow;
    fast_pred(t) = fast_pred(t-1) + r_fast * fast_pred(t-1) * coral_space - pred_pressure_fast;
    
    // Ensure predictions stay within biological bounds
    cots_pred(t) = Type(1.0)/(Type(1.0) + exp(-cots_pred(t)));
    slow_pred(t) = Type(100.0)/(Type(1.0) + exp(-slow_pred(t)/Type(100.0)));
    fast_pred(t) = Type(100.0)/(Type(1.0) + exp(-fast_pred(t)/Type(100.0)));
  }
  
  // Observation model using log-normal distribution
  Type sigma_cots = Type(0.1); // Minimum SD for numerical stability
  Type sigma_coral = Type(1.0);
  
  for(int t = 0; t < n; t++) {
    // Add to negative log-likelihood using log-normal distribution
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots, true);
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_coral, true);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_coral, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
