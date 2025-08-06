#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Year);                  // Time series years
  DATA_VECTOR(sst_dat);              // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);          // COTS immigration rate (individuals/m2/year)
  DATA_VECTOR(cots_dat);             // Observed COTS density (individuals/m2)
  DATA_VECTOR(slow_dat);             // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);             // Observed fast-growing coral cover (%)
  
  // Parameters
  PARAMETER(r_cots);                 // COTS intrinsic growth rate
  PARAMETER(K_cots);                 // COTS carrying capacity
  PARAMETER(temp_opt);               // Optimal temperature for COTS
  PARAMETER(temp_tol);               // Temperature tolerance
  PARAMETER(stress_effect);          // Temperature stress effect on predation
  PARAMETER(attack_rate_slow);       // Attack rate on slow corals
  PARAMETER(attack_rate_fast);       // Attack rate on fast corals
  PARAMETER(handling_time);          // Handling time for predation
  PARAMETER(r_slow);                 // Slow coral growth rate
  PARAMETER(r_fast);                 // Fast coral growth rate
  PARAMETER(K_total);                // Total coral carrying capacity
  
  // Initialize vectors for predictions
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Set initial conditions
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Process model
  for(int t = 1; t < n; t++) {
    // 1. Temperature effect on COTS growth (Gaussian response)
    Type temp_diff = sst_dat(t-1) - temp_opt;
    Type temp_effect = exp(-(temp_diff * temp_diff) / (2.0 * temp_tol * temp_tol));
    
    // 2. Coral availability for predation (previous time step)
    Type total_coral = slow_pred(t-1) + fast_pred(t-1);
    
    // 3. COTS population dynamics with temperature effect and immigration
    Type density_effect = Type(1.0) - cots_pred(t-1) / K_cots;
    cots_pred(t) = cots_pred(t-1) + 
                   (r_cots * temp_effect * cots_pred(t-1) * density_effect) +
                   cotsimm_dat(t-1);
    cots_pred(t) = exp(log(cots_pred(t) + eps)); // Ensure positive values
    
    // 4. Temperature stress effect on predation efficiency (smooth bounded response)
    Type temp_stress = Type(1.0) / (Type(1.0) + exp(-temp_diff / temp_tol));
    Type stress_modifier = Type(1.0) + stress_effect * temp_stress;
    
    // Modified Holling Type II functional responses with stress effect
    Type total_coral_available = slow_pred(t-1) + fast_pred(t-1) + eps;
    Type pred_base_slow = (attack_rate_slow * slow_pred(t-1) * cots_pred(t-1)) /
                         (Type(1.0) + handling_time * total_coral_available);
    Type pred_base_fast = (attack_rate_fast * fast_pred(t-1) * cots_pred(t-1)) /
                         (Type(1.0) + handling_time * total_coral_available);
    
    // Apply stress modifier to predation
    Type pred_slow = pred_base_slow * stress_modifier;
    Type pred_fast = pred_base_fast * stress_modifier;
    
    // 5. Coral dynamics with logistic growth and predation
    Type available_space = (K_total - total_coral) / K_total;
    slow_pred(t) = slow_pred(t-1) + 
                   (r_slow * slow_pred(t-1) * available_space) - 
                   pred_slow;
    fast_pred(t) = fast_pred(t-1) + 
                   (r_fast * fast_pred(t-1) * available_space) - 
                   pred_fast;
    
    // Ensure predictions stay positive using exp(log())
    slow_pred(t) = exp(log(slow_pred(t) + eps));
    fast_pred(t) = exp(log(fast_pred(t) + eps));
  }
  
  // Observation model using log-normal distribution
  Type obs_sd_cots = Type(0.2);    // Minimum observation SD for COTS
  Type obs_sd_coral = Type(0.1);   // Minimum observation SD for coral cover
  
  for(int t = 0; t < n; t++) {
    // Log-normal likelihood for COTS abundance
    nll -= dnorm(log(cots_dat(t) + eps), 
                 log(cots_pred(t) + eps),
                 obs_sd_cots, true);
    
    // Log-normal likelihood for coral cover
    nll -= dnorm(log(slow_dat(t) + eps),
                 log(slow_pred(t) + eps),
                 obs_sd_coral, true);
    
    nll -= dnorm(log(fast_dat(t) + eps),
                 log(fast_pred(t) + eps),
                 obs_sd_coral, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
