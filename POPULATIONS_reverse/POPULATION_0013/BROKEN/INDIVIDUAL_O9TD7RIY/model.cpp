#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data inputs
  DATA_VECTOR(Year);           // Time series years
  DATA_VECTOR(sst_dat);        // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);    // COTS immigration rate (individuals/m2/year)
  DATA_VECTOR(cots_dat);       // Observed COTS density (individuals/m2)
  DATA_VECTOR(slow_dat);       // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);       // Observed fast-growing coral cover (%)
  
  // Parameters
  PARAMETER(r_cots);           // COTS intrinsic growth rate (year^-1)
  PARAMETER(K_cots);           // COTS carrying capacity (individuals/m2)
  PARAMETER(temp_opt);         // Optimal temperature for COTS (°C)
  PARAMETER(temp_tol);         // Temperature tolerance range (°C) 
  PARAMETER(coral_temp_sens);  // Coral vulnerability to temperature stress
  PARAMETER(a_slow);           // Attack rate on slow corals (m2/year/individual)
  PARAMETER(a_fast);           // Attack rate on fast corals (m2/year/individual)
  PARAMETER(h_slow);           // Handling time for slow corals (year)
  PARAMETER(h_fast);           // Handling time for fast corals (year)
  PARAMETER(r_slow);           // Growth rate of slow corals (year^-1)
  PARAMETER(r_fast);           // Growth rate of fast corals (year^-1)
  PARAMETER(K_slow);           // Carrying capacity of slow corals (%)
  PARAMETER(K_fast);           // Carrying capacity of fast corals (%)
  
  // Standard deviations for observation model
  PARAMETER(log_sigma_cots);   // Log SD for COTS observations
  PARAMETER(log_sigma_slow);   // Log SD for slow coral observations
  PARAMETER(log_sigma_fast);   // Log SD for fast coral observations
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Convert log SDs to regular scale
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_slow = exp(log_sigma_slow);
  Type sigma_fast = exp(log_sigma_fast);
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Vectors to store predictions
  vector<Type> cots_pred(Year.size());
  vector<Type> slow_pred(Year.size());
  vector<Type> fast_pred(Year.size());
  
  // Initial conditions (using first observation)
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Time series simulation
  for(int t = 1; t < Year.size(); t++) {
    // 1. Temperature effect on COTS (bounded Gaussian response)
    Type temp_diff = sst_dat(t-1) - temp_opt;
    Type temp_sq = (temp_diff * temp_diff) / (2.0 * temp_tol * temp_tol);
    Type temp_effect = exp(-temp_sq < Type(10.0) ? temp_sq : Type(10.0));
    
    // 2. Temperature stress effect on coral vulnerability (smoothly bounded)
    Type rel_temp_diff = temp_diff/temp_tol;
    Type stress = coral_temp_sens * rel_temp_diff * rel_temp_diff;
    Type coral_temp_effect = Type(1.0) + stress / (Type(1.0) + Type(0.1) * stress);
    
    // 3. COTS population dynamics with temperature effect and immigration 
    Type cots_growth = r_cots * temp_effect * cots_pred(t-1) * (1.0 - cots_pred(t-1)/K_cots);
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t-1);
    cots_pred(t) = cots_pred(t) > eps ? cots_pred(t) : eps;
    
    // 4. Type II functional responses for COTS predation with temperature-modified attack rates
    Type pred_slow = (a_slow * coral_temp_effect * slow_pred(t-1) * cots_pred(t-1)) / 
                    (1.0 + a_slow * h_slow * slow_pred(t-1) + a_fast * h_fast * fast_pred(t-1));
    Type pred_fast = (a_fast * coral_temp_effect * fast_pred(t-1) * cots_pred(t-1)) / 
                    (1.0 + a_slow * h_slow * slow_pred(t-1) + a_fast * h_fast * fast_pred(t-1));
    
    // 4. Coral dynamics with logistic growth and predation
    slow_pred(t) = slow_pred(t-1) + r_slow * slow_pred(t-1) * (1.0 - slow_pred(t-1)/K_slow) - pred_slow;
    fast_pred(t) = fast_pred(t-1) + r_fast * fast_pred(t-1) * (1.0 - fast_pred(t-1)/K_fast) - pred_fast;
    
    // Ensure predictions stay positive
    slow_pred(t) = slow_pred(t) > eps ? slow_pred(t) : eps;
    fast_pred(t) = fast_pred(t) > eps ? fast_pred(t) : eps;
  }
  
  // Observation model using lognormal distribution
  for(int t = 0; t < Year.size(); t++) {
    // Add small constant to prevent taking log of zero
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots, true);
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_slow, true);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_fast, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(sigma_cots);
  REPORT(sigma_slow);
  REPORT(sigma_fast);
  
  return nll;
}
