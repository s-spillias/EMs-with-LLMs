#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(cots_dat);    // COTS abundance (individuals/m^2)
  DATA_VECTOR(slow_dat);    // Slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);    // Fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);     // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat); // COTS immigration rate (individuals/m^2/year)
  
  // Parameters with bounds using logistic transformations
  PARAMETER(log_r_cots);    // Log of maximum COTS reproduction rate
  PARAMETER(log_T_opt);     // Log of optimal temperature
  PARAMETER(log_T_width);   // Log of temperature width
  PARAMETER(log_m_cots);    // Log of mortality rate
  PARAMETER(log_K_slow);    // Log of slow coral carrying capacity
  PARAMETER(log_K_fast);    // Log of fast coral carrying capacity
  PARAMETER(log_r_slow);    // Log of slow coral growth rate
  PARAMETER(log_r_fast);    // Log of fast coral growth rate
  PARAMETER(log_alpha_slow);// Log of feeding rate on slow corals
  PARAMETER(log_alpha_fast);// Log of feeding rate on fast corals
  
  // Transform parameters to natural scale with bounds
  Type r_cots = exp(log_r_cots);
  Type T_opt = exp(log_T_opt);
  Type T_width = exp(log_T_width);
  Type m_cots = exp(log_m_cots);
  Type K_slow = exp(log_K_slow);
  Type K_fast = exp(log_K_fast);
  Type r_slow = exp(log_r_slow);
  Type r_fast = exp(log_r_fast);
  Type alpha_slow = exp(log_alpha_slow);
  Type alpha_fast = exp(log_alpha_fast);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Constants for numerical stability
  const Type eps = Type(1e-8);
  
  // Vectors for predictions
  int n = cots_dat.size();
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  
  // Initial conditions
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Process model
  for(int t = 1; t < n; t++) {
    // 1. Temperature-dependent reproduction (bounded between 0 and 1)
    Type temp_diff = (sst_dat(t-1) - T_opt) / T_width;
    Type temp_effect = exp(-temp_diff * temp_diff);
    
    // 2. COTS population dynamics
    Type net_growth = r_cots * temp_effect - m_cots * cots_pred(t-1);
    cots_pred(t) = cots_pred(t-1) * (Type(1.0) + net_growth) + cotsimm_dat(t-1);
    cots_pred(t) = posfun(cots_pred(t), eps, nll);
    
    // 3. Coral dynamics
    Type total_coral = slow_pred(t-1) + fast_pred(t-1) + eps;
    
    // Calculate predation pressure
    Type pred_pressure = alpha_slow * cots_pred(t-1) * slow_pred(t-1) / total_coral;
    Type pred_pressure_fast = alpha_fast * cots_pred(t-1) * fast_pred(t-1) / total_coral;
    
    // Update coral cover with logistic growth and predation
    slow_pred(t) = slow_pred(t-1) + r_slow * slow_pred(t-1) * (Type(1.0) - slow_pred(t-1)/K_slow) - pred_pressure;
    fast_pred(t) = fast_pred(t-1) + r_fast * fast_pred(t-1) * (Type(1.0) - fast_pred(t-1)/K_fast) - pred_pressure_fast;
    
    // Ensure coral predictions stay positive
    slow_pred(t) = posfun(slow_pred(t), eps, nll);
    fast_pred(t) = posfun(fast_pred(t), eps, nll);
  }
  
  // Observation model using log-normal distribution
  Type sigma_obs = Type(0.2);
  
  for(int t = 0; t < n; t++) {
    // Add small constant to prevent taking log of zero
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_obs, true);
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_obs, true);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_obs, true);
    
    // Add penalties to keep predictions within reasonable bounds
    if(slow_pred(t) > K_slow) nll += pow(slow_pred(t) - K_slow, 2);
    if(fast_pred(t) > K_fast) nll += pow(fast_pred(t) - K_fast, 2);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
