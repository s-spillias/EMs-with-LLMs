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
  PARAMETER(alpha_fast);             // Attack rate on fast coral
  PARAMETER(alpha_slow);             // Attack rate on slow coral
  PARAMETER(r_fast);                 // Fast coral growth rate
  PARAMETER(r_slow);                 // Slow coral growth rate
  PARAMETER(temp_opt);               // Optimal temperature
  PARAMETER(temp_width);             // Temperature tolerance width
  PARAMETER(sigma_cots);             // SD for COTS observations
  PARAMETER(sigma_fast);             // SD for fast coral observations
  PARAMETER(sigma_slow);             // SD for slow coral observations
  PARAMETER(allee_threshold);        // COTS density threshold for Allee effects
  PARAMETER(predation_scaling);      // Scaling factor for density-dependent predation

  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Small constant to prevent division by zero
  const Type eps = Type(1e-8);
  
  // Vectors to store predictions
  vector<Type> cots_pred(Year.size());
  vector<Type> fast_pred(Year.size());
  vector<Type> slow_pred(Year.size());

  // Initialize first predictions with first observations
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  // Time series predictions
  for(int t = 1; t < Year.size(); t++) {
    // 1. Temperature effect on COTS reproduction (Gaussian response)
    Type temp_effect = exp(-pow(sst_dat(t-1) - temp_opt, 2) / (2 * pow(temp_width, 2)));
    
    // 2. Resource availability effect (Type II functional response)
    Type resource_effect = (fast_pred(t-1) + slow_pred(t-1)) / 
                         (fast_pred(t-1) + slow_pred(t-1) + Type(10.0));
    
    // 3. Allee effect (gentle sigmoid)
    Type allee_effect = Type(1.0) + Type(0.5) / (Type(1.0) + 
                       exp(-Type(1.0) * (cots_pred(t-1) - allee_threshold)));
    
    // 4. COTS population dynamics with Allee effect
    Type growth_rate = r_cots * temp_effect * resource_effect * allee_effect * 
                      (Type(1.0) - cots_pred(t-1)/K_cots);
    // Bound growth rate with gentle sigmoid
    growth_rate = Type(0.5) * tanh(growth_rate);
    cots_pred(t) = cots_pred(t-1) * (Type(1.0) + growth_rate) + cotsimm_dat(t-1);
    
    // 5. Density-dependent predation (gentle sigmoid)
    Type pred_efficiency = Type(1.0) + predation_scaling / (Type(1.0) + 
                         exp(-Type(1.0) * cots_pred(t-1)/K_cots));
    
    // 6. Coral predation rates
    Type denominator = Type(1.0) + alpha_fast * fast_pred(t-1) + 
                      alpha_slow * slow_pred(t-1);
    Type fast_consumption = pred_efficiency * alpha_fast * cots_pred(t-1) * 
                          fast_pred(t-1) / denominator;
    Type slow_consumption = pred_efficiency * alpha_slow * cots_pred(t-1) * 
                          slow_pred(t-1) / denominator;
    
    // 5. Coral dynamics
    fast_pred(t) = fast_pred(t-1) + r_fast * fast_pred(t-1) * 
                   (1 - (fast_pred(t-1) + slow_pred(t-1))/100) - fast_consumption;
    slow_pred(t) = slow_pred(t-1) + r_slow * slow_pred(t-1) * 
                   (1 - (fast_pred(t-1) + slow_pred(t-1))/100) - slow_consumption;
    
    // 6. Ensure predictions stay positive
    cots_pred(t) = exp(log(cots_pred(t) + eps));
    fast_pred(t) = exp(log(fast_pred(t) + eps));
    slow_pred(t) = exp(log(slow_pred(t) + eps));
  }

  // Observation model using log-normal distribution
  for(int t = 0; t < Year.size(); t++) {
    // Add small constant to prevent taking log of zero
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
