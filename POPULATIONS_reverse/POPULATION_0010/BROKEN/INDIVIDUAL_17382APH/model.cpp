#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Year);                  // Time points for observations
  DATA_VECTOR(cots_dat);             // COTS abundance data (individuals/m2)
  DATA_VECTOR(slow_dat);             // Slow-growing coral cover data (%)
  DATA_VECTOR(fast_dat);             // Fast-growing coral cover data (%)
  DATA_VECTOR(sst_dat);              // Sea surface temperature data (Celsius)
  DATA_VECTOR(cotsimm_dat);          // COTS immigration rate data (individuals/m2/year)
  
  // Parameters
  PARAMETER(r_slow);                 // Growth rate of slow-growing corals
  PARAMETER(r_fast);                 // Growth rate of fast-growing corals
  PARAMETER(K_total);                // Total carrying capacity for corals
  PARAMETER(a_slow);                 // Attack rate on slow-growing corals
  PARAMETER(a_fast);                 // Attack rate on fast-growing corals
  PARAMETER(h_slow);                 // Handling time for slow-growing corals
  PARAMETER(h_fast);                 // Handling time for fast-growing corals
  PARAMETER(m_cots);                 // Natural mortality rate of COTS
  PARAMETER(q_cots);                 // Density-dependent mortality coefficient
  PARAMETER(T_opt);                  // Optimal temperature for coral growth
  PARAMETER(sigma_temp);             // Temperature sensitivity parameter
  PARAMETER(sigma_obs_cots);         // Observation error SD for COTS
  PARAMETER(sigma_obs_slow);         // Observation error SD for slow corals
  PARAMETER(sigma_obs_fast);         // Observation error SD for fast corals

  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Small constant to prevent division by zero
  const Type eps = Type(1e-8);
  
  // Predicted values vectors
  vector<Type> cots_pred(Year.size());
  vector<Type> slow_pred(Year.size());
  vector<Type> fast_pred(Year.size());
  
  // Initial conditions (first data point)
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Time series predictions
  for(int t = 1; t < Year.size(); t++) {
    // Temperature effect on growth (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t) - T_opt)/sigma_temp, 2));
    
    // Total coral cover
    Type total_cover = slow_pred(t-1) + fast_pred(t-1);
    
    // Space availability factor
    Type space_factor = (K_total - total_cover) / K_total;
    
    // Functional responses for COTS feeding
    Type f_slow = (a_slow * slow_pred(t-1)) / (1 + a_slow * h_slow * slow_pred(t-1) + a_fast * h_fast * fast_pred(t-1));
    Type f_fast = (a_fast * fast_pred(t-1)) / (1 + a_slow * h_slow * slow_pred(t-1) + a_fast * h_fast * fast_pred(t-1));
    
    // COTS dynamics
    Type cots_growth = cots_pred(t-1) * (f_slow + f_fast - m_cots - q_cots * cots_pred(t-1)) + cotsimm_dat(t);
    cots_pred(t) = cots_pred(t-1) + cots_growth;
    cots_pred(t) = exp(log(cots_pred(t) + eps)); // Ensure positivity
    
    // Coral dynamics
    Type slow_growth = r_slow * slow_pred(t-1) * space_factor * temp_effect - f_slow * cots_pred(t-1);
    Type fast_growth = r_fast * fast_pred(t-1) * space_factor * temp_effect - f_fast * cots_pred(t-1);
    
    slow_pred(t) = slow_pred(t-1) + slow_growth;
    fast_pred(t) = fast_pred(t-1) + fast_growth;
    
    // Ensure coral cover stays positive
    slow_pred(t) = exp(log(slow_pred(t) + eps));
    fast_pred(t) = exp(log(fast_pred(t) + eps));
  }
  
  // Observation model (lognormal)
  for(int t = 0; t < Year.size(); t++) {
    // Add small constant to prevent log(0)
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_obs_cots, true);
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_obs_slow, true);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_obs_fast, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  // Report forcing variables (not predictions, just pass-through)
  REPORT(sst_dat);
  REPORT(cotsimm_dat);
  
  return nll;
}
