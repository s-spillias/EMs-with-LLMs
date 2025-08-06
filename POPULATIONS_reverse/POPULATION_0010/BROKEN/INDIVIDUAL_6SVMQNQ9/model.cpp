#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Year);                  // Time vector
  DATA_VECTOR(sst_dat);              // Sea surface temperature data (°C)
  DATA_VECTOR(cotsimm_dat);          // COTS immigration rate data (ind/m²/year)
  DATA_VECTOR(cots_dat);             // Observed COTS density (ind/m²)
  DATA_VECTOR(slow_dat);             // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);             // Observed fast-growing coral cover (%)
  
  // Parameters
  PARAMETER(r_slow);                 // Growth rate of slow corals
  PARAMETER(r_fast);                 // Growth rate of fast corals
  PARAMETER(K_slow);                 // Carrying capacity of slow corals
  PARAMETER(K_fast);                 // Carrying capacity of fast corals
  PARAMETER(alpha_slow);             // Attack rate on slow corals
  PARAMETER(alpha_fast);             // Attack rate on fast corals
  PARAMETER(h_slow);                 // Handling time for slow corals
  PARAMETER(h_fast);                 // Handling time for fast corals
  PARAMETER(m_cots);                 // Natural mortality rate of COTS
  PARAMETER(q_cots);                 // Density-dependent mortality of COTS
  PARAMETER(temp_opt);               // Optimal temperature for COTS
  PARAMETER(temp_range);             // Temperature range for COTS performance
  PARAMETER(sigma_cots);             // Observation error SD for COTS
  PARAMETER(sigma_slow);             // Observation error SD for slow corals
  PARAMETER(sigma_fast);             // Observation error SD for fast corals

  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Initialize vectors for predictions
  vector<Type> cots_pred(Year.size());
  vector<Type> slow_pred(Year.size());
  vector<Type> fast_pred(Year.size());
  
  // Small constant to prevent division by zero
  const Type eps = Type(1e-8);
  
  // Initial conditions
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Process model
  for(int t = 1; t < Year.size(); t++) {
    // Temperature scaling function (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t) - temp_opt) / temp_range, 2));
    
    // Total coral cover (with small constant for numerical stability)
    Type total_cover = slow_pred(t-1) + fast_pred(t-1) + eps;
    
    // Functional responses with prey switching
    Type f_slow = (alpha_slow * slow_pred(t-1)) / (1 + h_slow * alpha_slow * slow_pred(t-1) + h_fast * alpha_fast * fast_pred(t-1));
    Type f_fast = (alpha_fast * fast_pred(t-1)) / (1 + h_slow * alpha_slow * slow_pred(t-1) + h_fast * alpha_fast * fast_pred(t-1));
    
    // COTS dynamics
    Type mortality = m_cots + q_cots * cots_pred(t-1);
    cots_pred(t) = cots_pred(t-1) + temp_effect * (f_slow + f_fast) * cots_pred(t-1) - mortality * cots_pred(t-1) + cotsimm_dat(t);
    cots_pred(t) = exp(log(cots_pred(t) + eps)); // Ensure positivity
    
    // Coral dynamics with competition and predation
    Type competition_slow = 1 - (slow_pred(t-1) + 0.5 * fast_pred(t-1)) / K_slow;
    Type competition_fast = 1 - (fast_pred(t-1) + 0.5 * slow_pred(t-1)) / K_fast;
    
    slow_pred(t) = slow_pred(t-1) + r_slow * slow_pred(t-1) * competition_slow - f_slow * cots_pred(t-1);
    fast_pred(t) = fast_pred(t-1) + r_fast * fast_pred(t-1) * competition_fast - f_fast * cots_pred(t-1);
    
    // Ensure coral cover stays between 0 and 100%
    slow_pred(t) = exp(log(slow_pred(t) + eps));
    fast_pred(t) = exp(log(fast_pred(t) + eps));
  }
  
  // Observation model (lognormal)
  for(int t = 0; t < Year.size(); t++) {
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots, true);
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_slow, true);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_fast, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
