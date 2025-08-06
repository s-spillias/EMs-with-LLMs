#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Year);                  // Time vector
  DATA_VECTOR(cots_dat);             // COTS abundance (individuals/m2)
  DATA_VECTOR(slow_dat);             // Slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);             // Fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);              // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);          // COTS immigration rate (individuals/m2/year)

  // Parameters
  PARAMETER(r_cots);                 // COTS intrinsic growth rate
  PARAMETER(K_cots);                 // COTS carrying capacity
  PARAMETER(temp_opt);               // Optimal temperature for COTS
  PARAMETER(temp_tol);               // Temperature tolerance
  PARAMETER(coral_stress);           // Coral stress sensitivity to temperature
  PARAMETER(a_fast);                 // Attack rate on fast coral
  PARAMETER(a_slow);                 // Attack rate on slow coral
  PARAMETER(h_fast);                 // Handling time for fast coral
  PARAMETER(h_slow);                 // Handling time for slow coral
  PARAMETER(r_fast);                 // Fast coral growth rate
  PARAMETER(r_slow);                 // Slow coral growth rate
  PARAMETER(K_coral);                // Total coral carrying capacity
  
  // Transform parameters to valid ranges
  r_cots = exp(r_cots);
  K_cots = exp(K_cots);
  a_fast = exp(a_fast);
  a_slow = exp(a_slow);
  r_fast = exp(r_fast);
  r_slow = exp(r_slow);
  K_coral = exp(K_coral);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Initialize vectors for predictions
  vector<Type> cots_pred(Year.size());
  vector<Type> slow_pred(Year.size());
  vector<Type> fast_pred(Year.size());
  
  // Set initial conditions with small constant to prevent zeros
  Type eps = Type(1e-8);
  cots_pred(0) = cots_dat(0) + eps;
  slow_pred(0) = slow_dat(0) + eps;
  fast_pred(0) = fast_dat(0) + eps;
  
  // Model equations
  for(int t = 1; t < Year.size(); t++) {
    // 1. Temperature effects
    Type temp_effect = exp(-pow(sst_dat(t-1) - temp_opt, 2.0) / (2.0 * pow(temp_tol, 2.0)));
    
    // Simple linear temperature stress effect
    Type stress_effect = Type(1.0) + coral_stress * pow(sst_dat(t-1) - temp_opt, 2.0) / (Type(2.0) * pow(temp_tol, 2.0));
    
    // 2. Type II functional responses for COTS predation
    Type total_coral = fast_pred(t-1) + slow_pred(t-1) + eps;
    Type pred_fast = (a_fast * fast_pred(t-1)) / (Type(1.0) + h_fast * total_coral);
    Type pred_slow = (a_slow * slow_pred(t-1)) / (Type(1.0) + h_slow * total_coral);
    
    // Apply stress effect to predation rates
    pred_fast *= stress_effect;
    pred_slow *= stress_effect;
    
    // 3. COTS population dynamics with temperature effect and immigration
    cots_pred(t) = cots_pred(t-1) + 
                   temp_effect * r_cots * cots_pred(t-1) * (1.0 - cots_pred(t-1)/K_cots) +
                   cotsimm_dat(t-1);
    
    // 4. Coral dynamics with competition and COTS predation
    Type total_cover = fast_pred(t-1) + slow_pred(t-1);
    
    fast_pred(t) = fast_pred(t-1) + 
                   r_fast * fast_pred(t-1) * (1.0 - total_cover/K_coral) -
                   pred_fast * cots_pred(t-1);
    
    slow_pred(t) = slow_pred(t-1) + 
                   r_slow * slow_pred(t-1) * (1.0 - total_cover/K_coral) -
                   pred_slow * cots_pred(t-1);
    
    // Ensure predictions stay positive using softplus
    cots_pred(t) = log(Type(1.0) + exp(cots_pred(t)));
    fast_pred(t) = log(Type(1.0) + exp(fast_pred(t)));
    slow_pred(t) = log(Type(1.0) + exp(slow_pred(t)));
  }
  
  // Calculate negative log-likelihood using lognormal distribution
  Type sigma_cots = Type(0.3);    // Observation error for COTS
  Type sigma_coral = Type(0.3);   // Observation error for coral
  
  for(int t = 0; t < Year.size(); t++) {
    // COTS likelihood
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t)), sigma_cots, true);
    
    // Coral likelihoods
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t)), sigma_coral, true);
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t)), sigma_coral, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
