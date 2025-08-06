#include <TMB.hpp>
#include <iostream>

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;
  
  // Data
  DATA_VECTOR(sst_dat);        // Sea surface temperature time series (°C)
  DATA_VECTOR(cotsimm_dat);    // COTS immigration rate time series (individuals/m^2/year)
  DATA_VECTOR(cots_dat);       // Observed COTS density (individuals/m^2)
  DATA_VECTOR(slow_dat);       // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);       // Observed fast-growing coral cover (%)
  
  // Parameters
  PARAMETER(r_cots);           // COTS intrinsic growth rate (year^-1)
  PARAMETER(K_cots);           // COTS carrying capacity (individuals/m^2)
  PARAMETER(temp_opt);         // Optimal temperature for COTS reproduction (°C)
  PARAMETER(temp_width);       // Temperature tolerance width (°C)
  PARAMETER(g_slow);           // Growth rate of slow-growing corals (year^-1)
  PARAMETER(g_fast);           // Growth rate of fast-growing corals (year^-1)
  PARAMETER(K_coral);          // Combined coral carrying capacity (%)
  PARAMETER(alpha_slow);       // COTS feeding rate on slow corals
  PARAMETER(alpha_fast);       // COTS feeding rate on fast corals

  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Constants for numerical stability
  const Type eps = Type(1e-8);
  const Type max_pred = Type(1e3);  // Maximum prediction value
  
  // Vectors for model predictions
  int n = sst_dat.size();
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  
  // Initialize first values
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Model equations
  for(int t = 0; t < n-1; t++) {
    // 1. Temperature effect on COTS reproduction (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t) - temp_opt) / temp_width, 2));
    
    // 2. COTS population dynamics with temperature-dependent growth and immigration
    Type cots_growth = r_cots * temp_effect * cots_pred(t) * (1 - cots_pred(t) / K_cots);
    cots_pred(t + 1) = cots_pred(t) + cots_growth + cotsimm_dat(t);
    
    // 3. Coral dynamics with competition and COTS predation
    Type total_cover = slow_pred(t) + fast_pred(t);
    Type competition = 1 - total_cover / K_coral;
    
    // Slow-growing coral dynamics
    Type slow_growth = g_slow * slow_pred(t) * competition;
    Type slow_pred_loss = alpha_slow * cots_pred(t) * slow_pred(t);
    slow_pred(t + 1) = slow_pred(t) + slow_growth - slow_pred_loss;
    
    // Fast-growing coral dynamics
    Type fast_growth = g_fast * fast_pred(t) * competition;
    Type fast_pred_loss = alpha_fast * cots_pred(t) * fast_pred(t);
    fast_pred(t + 1) = fast_pred(t) + fast_growth - fast_pred_loss;
    
    // Ensure predictions stay positive using exp(log())
    cots_pred(t + 1) = exp(log(cots_pred(t + 1) + eps));
    slow_pred(t + 1) = exp(log(slow_pred(t + 1) + eps));
    fast_pred(t + 1) = exp(log(fast_pred(t + 1) + eps));
  }
  
  // Observation model using lognormal distribution
  Type obs_sd_cots = Type(0.2);    // Minimum observation SD for COTS
  Type obs_sd_coral = Type(0.1);   // Minimum observation SD for coral cover
  
  for(int t = 0; t < n; t++) {
    // COTS observations
    nll -= dnorm(log(cots_dat(t) + eps), 
                 log(cots_pred(t) + eps), 
                 sqrt(pow(obs_sd_cots, 2)), 
                 true);
    
    // Coral observations
    nll -= dnorm(log(slow_dat(t) + eps), 
                 log(slow_pred(t) + eps), 
                 sqrt(pow(obs_sd_coral, 2)), 
                 true);
    
    nll -= dnorm(log(fast_dat(t) + eps), 
                 log(fast_pred(t) + eps), 
                 sqrt(pow(obs_sd_coral, 2)), 
                 true);
  }
  
  // Report predictions and objective
  ADREPORT(cots_pred);
  ADREPORT(slow_pred);
  ADREPORT(fast_pred);
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(nll);
  
  // Return objective function
  
  return nll;
}
