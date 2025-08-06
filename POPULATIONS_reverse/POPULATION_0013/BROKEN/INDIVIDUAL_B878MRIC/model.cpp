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
  PARAMETER(a_slow);           // Attack rate on slow corals (m2/year/individual)
  PARAMETER(a_fast);           // Attack rate on fast corals (m2/year/individual)
  PARAMETER(h_slow);           // Handling time for slow corals (year)
  PARAMETER(h_fast);           // Handling time for fast corals (year) 
  PARAMETER(q_slow);           // Size-dependence of attack rate on slow corals
  PARAMETER(q_fast);           // Size-dependence of attack rate on fast corals
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
    // 1. Temperature effect on COTS (Gaussian response)
    Type temp_diff = sst_dat(t-1) - temp_opt;
    Type temp_effect = exp(-(temp_diff * temp_diff) / (2.0 * temp_tol * temp_tol));
    
    // 2. COTS population dynamics with temperature effect and immigration
    Type cots_growth = r_cots * temp_effect * cots_pred(t-1) * (1.0 - cots_pred(t-1)/K_cots);
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t-1);
    cots_pred(t) = cots_pred(t) > eps ? cots_pred(t) : eps;
    
    // 3. Type II functional responses with simplified coral-dependent attack rates
    Type slow_ratio = slow_pred(t-1)/(K_slow + slow_pred(t-1));
    Type fast_ratio = fast_pred(t-1)/(K_fast + fast_pred(t-1));
    
    Type pred_slow = (a_slow * (1.0 + q_slow * slow_ratio) * slow_pred(t-1) * cots_pred(t-1)) / 
                    (1.0 + h_slow * slow_pred(t-1) + h_fast * fast_pred(t-1));
    Type pred_fast = (a_fast * (1.0 + q_fast * fast_ratio) * fast_pred(t-1) * cots_pred(t-1)) / 
                    (1.0 + h_slow * slow_pred(t-1) + h_fast * fast_pred(t-1));
    
    // 4. Coral dynamics with logistic growth and predation
    slow_pred(t) = slow_pred(t-1) + r_slow * slow_pred(t-1) * (1.0 - slow_pred(t-1)/K_slow) - pred_slow;
    fast_pred(t) = fast_pred(t-1) + r_fast * fast_pred(t-1) * (1.0 - fast_pred(t-1)/K_fast) - pred_fast;
    
    // Ensure predictions stay positive using addition instead of comparison
    slow_pred(t) = slow_pred(t) + eps;
    fast_pred(t) = fast_pred(t) + eps;
  }
  
  // Observation model using lognormal distribution
  for(int t = 0; t < Year.size(); t++) {
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
