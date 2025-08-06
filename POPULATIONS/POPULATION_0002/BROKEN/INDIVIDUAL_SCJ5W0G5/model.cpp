#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Year);                  // Time vector
  DATA_VECTOR(cots_dat);             // COTS abundance (individuals/m2)
  DATA_VECTOR(fast_dat);             // Fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);             // Slow-growing coral cover (%)
  DATA_VECTOR(sst_dat);              // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);          // COTS larval immigration rate

  // Parameters
  PARAMETER(r_cots);                 // COTS intrinsic growth rate
  PARAMETER(K_cots);                 // COTS carrying capacity
  PARAMETER(r_fast);                 // Fast coral growth rate
  PARAMETER(r_slow);                 // Slow coral growth rate
  PARAMETER(K_fast);                 // Fast coral carrying capacity
  PARAMETER(K_slow);                 // Slow coral carrying capacity
  PARAMETER(alpha_fast);             // COTS attack rate on fast coral
  PARAMETER(alpha_slow);             // COTS attack rate on slow coral
  PARAMETER(temp_opt);               // Optimal temperature
  PARAMETER(temp_width);             // Temperature tolerance width
  PARAMETER(log_sigma_cots);         // Log SD for COTS observations
  PARAMETER(log_sigma_fast);         // Log SD for fast coral observations
  PARAMETER(log_sigma_slow);         // Log SD for slow coral observations
  PARAMETER(allee_threshold);        // Critical density threshold for Allee effect

  // Derived values
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_fast = exp(log_sigma_fast);
  Type sigma_slow = exp(log_sigma_slow);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Small constant to prevent division by zero
  const Type eps = Type(1e-8);
  
  // Vectors for predictions
  vector<Type> cots_pred(Year.size());
  vector<Type> fast_pred(Year.size());
  vector<Type> slow_pred(Year.size());
  
  // Initialize first predictions with data
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);
  
  // Time series predictions
  for(int t = 1; t < Year.size(); t++) {
    // 1. Temperature effect on COTS reproduction (Gaussian response)
    Type temp_effect = exp(-pow(sst_dat(t-1) - temp_opt, 2.0) / (2.0 * pow(temp_width, 2.0)));
    
    // 2. Resource availability (total coral)
    Type total_coral = fast_pred(t-1) + slow_pred(t-1) + eps;
    
    // 3. COTS functional responses (Holling Type II)
    Type fast_consumed = (alpha_fast * fast_pred(t-1) * cots_pred(t-1)) / (1.0 + alpha_fast * fast_pred(t-1) + alpha_slow * slow_pred(t-1));
    Type slow_consumed = (alpha_slow * slow_pred(t-1) * cots_pred(t-1)) / (1.0 + alpha_fast * fast_pred(t-1) + alpha_slow * slow_pred(t-1));
    
    // 4. Population dynamics
    // COTS: logistic growth with Allee effect + temperature effect + immigration
    Type allee_effect = pow(cots_pred(t-1), 2.0) / (pow(allee_threshold, 2.0) + pow(cots_pred(t-1), 2.0));
    cots_pred(t) = cots_pred(t-1) + 
                   r_cots * cots_pred(t-1) * (1.0 - cots_pred(t-1)/K_cots) * 
                   allee_effect * temp_effect * total_coral/(total_coral + eps) +
                   cotsimm_dat(t-1);
    
    // Fast coral: logistic growth - COTS predation
    fast_pred(t) = fast_pred(t-1) + 
                   r_fast * fast_pred(t-1) * (1.0 - fast_pred(t-1)/K_fast) -
                   fast_consumed;
    
    // Slow coral: logistic growth - COTS predation
    slow_pred(t) = slow_pred(t-1) + 
                   r_slow * slow_pred(t-1) * (1.0 - slow_pred(t-1)/K_slow) -
                   slow_consumed;
    
    // 5. Ensure predictions stay positive
    cots_pred(t) = posfun(cots_pred(t), eps, 0);
    fast_pred(t) = posfun(fast_pred(t), eps, 0);
    slow_pred(t) = posfun(slow_pred(t), eps, 0);
  }
  
  // Observation model using log-normal distribution
  for(int t = 0; t < Year.size(); t++) {
    // Add small constant to both observed and predicted values
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots, true);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_fast, true);
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_slow, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
