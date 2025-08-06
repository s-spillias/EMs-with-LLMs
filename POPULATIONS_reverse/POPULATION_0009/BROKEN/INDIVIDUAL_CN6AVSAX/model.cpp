#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(cots_dat);    // Observed COTS density (individuals/m^2)
  DATA_VECTOR(slow_dat);    // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);    // Observed fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);     // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat); // COTS immigration rate (individuals/m^2/year)
  
  // Parameters
  PARAMETER(r_cots);      // COTS intrinsic growth rate (year^-1)
  PARAMETER(K_cots);      // COTS carrying capacity (individuals/m^2)
  PARAMETER(a_fast);      // COTS predation rate on fast coral (m^2/individual/year)
  PARAMETER(a_slow);      // COTS predation rate on slow coral (m^2/individual/year)
  PARAMETER(r_fast);      // Fast coral intrinsic growth rate (year^-1)
  PARAMETER(r_slow);      // Slow coral intrinsic growth rate (year^-1)
  PARAMETER(temp_opt);    // Optimal temperature for COTS (Celsius)
  PARAMETER(temp_width);  // Temperature tolerance width (Celsius)

  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Constants for numerical stability
  const Type eps = Type(1e-8);
  
  // Vectors to store predictions
  vector<Type> cots_pred(cots_dat.size());
  vector<Type> slow_pred(slow_dat.size());
  vector<Type> fast_pred(fast_dat.size());
  
  // Initialize first timestep with reasonable starting values
  cots_pred(0) = Type(0.5); // Initial COTS density
  slow_pred(0) = Type(15.0); // Initial slow coral cover
  fast_pred(0) = Type(12.0); // Initial fast coral cover
  
  // Time series predictions
  for(int t = 1; t < cots_dat.size(); t++) {
    // 1. Temperature effect on COTS growth (Gaussian response)
    Type temp_effect = exp(-pow(sst_dat(t) - temp_opt, 2.0) / (2.0 * pow(temp_width, 2.0)));
    
    // 2. COTS population dynamics - depends on temperature and available coral prey
    Type available_coral = (slow_pred(t-1) + fast_pred(t-1))/100.0; // Proportion of coral available
    Type cots_growth = r_cots * temp_effect * available_coral * cots_pred(t-1) * (1.0 - cots_pred(t-1)/K_cots);
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t);
    
    // 3. Coral dynamics with predation and temperature stress
    Type temp_stress = pow(sst_dat(t) - Type(26.0), 2.0) / (2.0 * pow(Type(2.0), 2.0)); // Temperature stress on corals
    
    // Slow coral dynamics
    Type slow_growth = r_slow * slow_pred(t-1) * (1.0 - slow_pred(t-1)/100.0) * exp(-0.1 * temp_stress);
    slow_pred(t) = slow_pred(t-1) + slow_growth - a_slow * cots_pred(t-1) * slow_pred(t-1);
    
    // Fast coral dynamics - more sensitive to temperature
    Type fast_growth = r_fast * fast_pred(t-1) * (1.0 - fast_pred(t-1)/100.0) * exp(-0.2 * temp_stress);
    fast_pred(t) = fast_pred(t-1) + fast_growth - a_fast * cots_pred(t-1) * fast_pred(t-1);
    
    // Bound predictions to valid ranges
    cots_pred(t) = posfun(cots_pred(t), eps, nll);
    slow_pred(t) = posfun(slow_pred(t), eps, nll);
    fast_pred(t) = posfun(fast_pred(t), eps, nll);
    
    if(slow_pred(t) > 100.0) slow_pred(t) = 100.0;
    if(fast_pred(t) > 100.0) fast_pred(t) = 100.0;
  }
  
  // Observation model using normal distribution on log scale
  Type obs_sd_cots = Type(0.2); // Minimum observation SD for COTS
  Type obs_sd_coral = Type(0.1); // Minimum observation SD for coral cover
  
  for(int t = 0; t < cots_dat.size(); t++) {
    // Add small constant before taking logs
    Type log_cots_obs = log(cots_dat(t) + eps);
    Type log_cots_pred = log(cots_pred(t) + eps);
    Type log_slow_obs = log(slow_dat(t) + eps);
    Type log_slow_pred = log(slow_pred(t) + eps);
    Type log_fast_obs = log(fast_dat(t) + eps);
    Type log_fast_pred = log(fast_pred(t) + eps);
    
    // Normal likelihood on log scale
    nll += 0.5 * pow((log_cots_obs - log_cots_pred) / obs_sd_cots, 2.0);
    nll += 0.5 * pow((log_slow_obs - log_slow_pred) / obs_sd_coral, 2.0);
    nll += 0.5 * pow((log_fast_obs - log_fast_pred) / obs_sd_coral, 2.0);
    
    // Add log of standard deviation to likelihood
    nll += log(obs_sd_cots);
    nll += log(obs_sd_coral);
    nll += log(obs_sd_coral);
  }
  
  // Report predictions and objective
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(nll);
  
  // Add penalties for invalid parameter values
  if(r_cots < Type(0.0)) nll += Type(1e10);
  if(K_cots < Type(0.0)) nll += Type(1e10);
  if(a_fast < Type(0.0)) nll += Type(1e10);
  if(a_slow < Type(0.0)) nll += Type(1e10);
  if(r_fast < Type(0.0)) nll += Type(1e10);
  if(r_slow < Type(0.0)) nll += Type(1e10);
  if(temp_width < Type(0.0)) nll += Type(1e10);
  
  return nll;
}
