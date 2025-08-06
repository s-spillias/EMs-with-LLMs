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
  PARAMETER(A_cots);                 // COTS Allee effect threshold
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
    // COTS: modified logistic growth with Allee effect + temperature effect + immigration
    Type N = cots_pred(t-1);
    Type allee_term = N * N / (N * N + A_cots * A_cots);  // Quadratic Allee effect
    Type logistic_term = (1.0 - N/K_cots);
    Type cots_growth = r_cots * N * allee_term * logistic_term * temp_effect;
    cots_pred(t) = N + cots_growth + cotsimm_dat(t-1);
    if(cots_pred(t) < eps) cots_pred(t) = eps;
    
    // Fast coral: logistic growth - COTS predation
    Type fast_growth = r_fast * fast_pred(t-1) * (1.0 - fast_pred(t-1)/K_fast) - fast_consumed;
    fast_pred(t) = fast_pred(t-1) + fast_growth;
    if(fast_pred(t) < eps) fast_pred(t) = eps;
    
    // Slow coral: logistic growth - COTS predation
    Type slow_growth = r_slow * slow_pred(t-1) * (1.0 - slow_pred(t-1)/K_slow) - slow_consumed;
    slow_pred(t) = slow_pred(t-1) + slow_growth;
    if(slow_pred(t) < eps) slow_pred(t) = eps;
  }
  
  // Observation model using log-normal distribution
  for(int t = 0; t < Year.size(); t++) {
    if(!R_IsNA(asDouble(cots_dat(t)))) {
      nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots, true);
    }
    if(!R_IsNA(asDouble(fast_dat(t)))) {
      nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_fast, true);
    }
    if(!R_IsNA(asDouble(slow_dat(t)))) {
      nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_slow, true);
    }
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
