#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Year);                  // Time vector (years)
  DATA_VECTOR(cots_dat);             // Observed COTS density (individuals/m^2)
  DATA_VECTOR(slow_dat);             // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);             // Observed fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);              // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);          // COTS immigration rate (individuals/m^2/year)
  
  // Parameters
  PARAMETER(r_cots);                 // COTS intrinsic growth rate
  PARAMETER(K_cots);                 // COTS carrying capacity
  PARAMETER(alpha_slow);             // COTS predation rate on slow corals
  PARAMETER(alpha_fast);             // COTS predation rate on fast corals
  PARAMETER(r_slow);                 // Slow coral growth rate
  PARAMETER(r_fast);                 // Fast coral growth rate
  PARAMETER(temp_opt);               // Optimal temperature
  PARAMETER(temp_tol);               // Temperature tolerance
  PARAMETER(h_coral);                // Half-saturation constant for feeding
  PARAMETER(m_cots);                 // COTS natural mortality
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Initialize vectors for predictions
  vector<Type> cots_pred(Year.size());
  vector<Type> slow_pred(Year.size());
  vector<Type> fast_pred(Year.size());
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Initialize first predictions with first observation
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Time series predictions
  for(int t = 1; t < Year.size(); t++) {
    // 1. Temperature effect on growth (Gaussian response)
    Type temp_diff = sst_dat(t-1) - temp_opt;
    Type temp_effect = CppAD::exp(-(temp_diff * temp_diff) / (Type(2.0) * temp_tol * temp_tol));
    
    // 2. Coral space competition factor
    Type space_available = Type(100.0) - (slow_pred(t-1) + fast_pred(t-1));
    
    // 3. COTS functional response (Type II)
    Type feeding_slow = (alpha_slow * slow_pred(t-1)) / (h_coral + slow_pred(t-1) + fast_pred(t-1));
    Type feeding_fast = (alpha_fast * fast_pred(t-1)) / (h_coral + slow_pred(t-1) + fast_pred(t-1));
    
    // 4. COTS population dynamics
    cots_pred(t) = cots_pred(t-1) + 
                   r_cots * cots_pred(t-1) * (1 - cots_pred(t-1)/K_cots) * temp_effect -
                   m_cots * cots_pred(t-1) +
                   cotsimm_dat(t-1);
    
    // 5. Coral dynamics
    slow_pred(t) = slow_pred(t-1) + 
                   r_slow * slow_pred(t-1) * (space_available/(100.0 + eps)) * temp_effect -
                   feeding_slow * cots_pred(t-1);
    
    fast_pred(t) = fast_pred(t-1) + 
                   r_fast * fast_pred(t-1) * (space_available/(100.0 + eps)) * temp_effect -
                   feeding_fast * cots_pred(t-1);
    
    // 6. Ensure predictions stay within biological bounds using smooth penalties
    cots_pred(t) = cots_pred(t) < eps ? eps : cots_pred(t);
    slow_pred(t) = slow_pred(t) < eps ? eps : slow_pred(t);
    fast_pred(t) = fast_pred(t) < eps ? eps : fast_pred(t);
  }
  
  // Calculate negative log-likelihood
  // Using log-normal distribution for all observations
  Type sigma_cots = Type(0.1);    // Minimum SD for COTS
  Type sigma_coral = Type(0.1);   // Minimum SD for coral cover
  
  for(int t = 0; t < Year.size(); t++) {
    // COTS likelihood
    nll -= dnorm(CppAD::log(cots_dat(t) + eps), 
                CppAD::log(cots_pred(t) + eps), 
                sigma_cots, true);
    
    // Coral cover likelihoods
    nll -= dnorm(CppAD::log(slow_dat(t) + eps), 
                CppAD::log(slow_pred(t) + eps), 
                sigma_coral, true);
    nll -= dnorm(CppAD::log(fast_dat(t) + eps), 
                CppAD::log(fast_pred(t) + eps), 
                sigma_coral, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
