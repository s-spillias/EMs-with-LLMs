#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Year);                  // Time vector (years)
  DATA_VECTOR(cots_dat);             // Observed COTS density (individuals/m2)
  DATA_VECTOR(slow_dat);             // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);             // Observed fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);              // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);          // COTS immigration rate (individuals/m2/year)
  
  // Parameters
  PARAMETER(r_cots);                 // COTS intrinsic growth rate
  PARAMETER(K_cots);                 // COTS carrying capacity
  PARAMETER(temp_opt);               // Optimal temperature for COTS
  PARAMETER(temp_tol);               // Temperature tolerance
  PARAMETER(g_slow);                 // Slow coral growth rate
  PARAMETER(g_fast);                 // Fast coral growth rate
  PARAMETER(K_coral);                // Total coral carrying capacity
  PARAMETER(alpha_slow);             // Predation rate on slow corals
  PARAMETER(alpha_fast);             // Predation rate on fast corals
  PARAMETER(h_coral);                // Half-saturation for predation
  PARAMETER(sigma_cots);             // SD for COTS observations
  PARAMETER(sigma_coral);            // SD for coral observations

  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Initialize vectors for predictions
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  
  // Constants to prevent numerical issues
  Type eps = Type(1e-8);
  
  // Initialize with first observation
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Time series predictions
  for(int t = 1; t < n; t++) {
    // 1. Temperature effect on COTS growth (Gaussian response)
    Type temp_effect = exp(-pow(sst_dat(t-1) - temp_opt, 2) / (2 * pow(temp_tol, 2)));
    
    // 2. Total coral cover for density dependence
    Type total_coral = slow_pred(t-1) + fast_pred(t-1);
    
    // 3. Coral-dependent carrying capacity
    Type effective_K = K_cots * total_coral / (total_coral + h_coral);
    
    // 4. COTS population dynamics with temperature effect and immigration
    Type cots_growth = r_cots * temp_effect * cots_pred(t-1) * 
                      (1 - cots_pred(t-1) / (effective_K + eps));
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t-1);
    cots_pred(t) = exp(log(cots_pred(t) + eps));  // Ensure positivity
    
    // 5. Coral predation rates with functional response
    Type pred_slow = (alpha_slow * cots_pred(t-1) * slow_pred(t-1)) / 
                    (h_coral + slow_pred(t-1) + fast_pred(t-1));
    Type pred_fast = (alpha_fast * cots_pred(t-1) * fast_pred(t-1)) / 
                    (h_coral + slow_pred(t-1) + fast_pred(t-1));
    
    // 6. Coral dynamics with logistic growth and predation
    Type available_space = (K_coral - (slow_pred(t-1) + fast_pred(t-1))) / K_coral;
    slow_pred(t) = slow_pred(t-1) + g_slow * slow_pred(t-1) * available_space - pred_slow;
    fast_pred(t) = fast_pred(t-1) + g_fast * fast_pred(t-1) * available_space - pred_fast;
    
    // Ensure coral cover stays positive
    slow_pred(t) = exp(log(slow_pred(t) + eps));
    fast_pred(t) = exp(log(fast_pred(t) + eps));
  }
  
  // Observation model using log-normal distribution
  for(int t = 0; t < n; t++) {
    // Add small constant to prevent taking log of zero
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
