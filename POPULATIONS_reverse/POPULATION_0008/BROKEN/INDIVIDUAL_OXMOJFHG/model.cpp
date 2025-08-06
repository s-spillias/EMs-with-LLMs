#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
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
  PARAMETER(alpha_slow);       // COTS predation rate on slow-growing corals (m^2/individual/year)
  PARAMETER(alpha_fast);       // COTS predation rate on fast-growing corals (m^2/individual/year)

  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Constants for numerical stability
  const Type eps = Type(1e-8);
  const Type min_sd = Type(0.1);    // Minimum standard deviation for observations
  
  // Vectors to store predictions
  int n = sst_dat.size();
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  
  // Initialize predictions for first timestep with positive values
  cots_pred(0) = cots_dat(0) + Type(0.1);  // Add small constant to prevent zeros
  slow_pred(0) = slow_dat(0) + Type(1.0);   // Add small constant to prevent zeros
  fast_pred(0) = fast_dat(0) + Type(1.0);   // Add small constant to prevent zeros
  
  // Model equations - predict future states using only previous predictions
  for(int t = 1; t < n; t++) {
    // 1. Temperature effect on COTS reproduction (Gaussian response)
    Type temp_effect = exp(-pow(sst_dat(t) - temp_opt, 2) / (2 * pow(temp_width, 2)));
    
    // 2. COTS population dynamics with temperature-dependent growth and immigration
    Type cots_growth = r_cots * temp_effect * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1)/K_cots);
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t) + Type(0.1);  // Add small constant
    
    // 3. Coral dynamics with logistic growth and COTS predation
    Type total_cover = (slow_pred(t-1) + fast_pred(t-1))/Type(100.0);
    
    // Slow-growing corals
    Type slow_growth = g_slow * slow_pred(t-1) * (Type(1.0) - total_cover);
    Type slow_mortality = alpha_slow * cots_pred(t-1) * slow_pred(t-1)/Type(100.0);
    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_mortality;
    slow_pred(t) = std::max(Type(1.0), std::min(Type(100.0), slow_pred(t)));  // Bound between 1-100
    
    // Fast-growing corals  
    Type fast_growth = g_fast * fast_pred(t-1) * (Type(1.0) - total_cover);
    Type fast_mortality = alpha_fast * cots_pred(t-1) * fast_pred(t-1)/Type(100.0);
    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_mortality;
    fast_pred(t) = std::max(Type(1.0), std::min(Type(100.0), fast_pred(t)));  // Bound between 1-100
  }
  
  // Observation model using log-normal distribution for all variables
  for(int t = 0; t < n; t++) {
    // COTS observations
    nll -= dnorm(log(cots_dat(t) + eps), 
                 log(cots_pred(t) + eps), 
                 sqrt(pow(min_sd, 2) + pow(0.2, 2)), // CV of 20%
                 true);
    
    // Coral cover observations
    nll -= dnorm(log(slow_dat(t) + eps),
                 log(slow_pred(t) + eps),
                 sqrt(pow(min_sd, 2) + pow(0.15, 2)), // CV of 15%
                 true);
    
    nll -= dnorm(log(fast_dat(t) + eps),
                 log(fast_pred(t) + eps),
                 sqrt(pow(min_sd, 2) + pow(0.15, 2)), // CV of 15%
                 true);
  }
  
  // Report predictions and objective function value
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(nll);               // Report the objective function value
  ADREPORT(cots_pred);
  ADREPORT(slow_pred);
  ADREPORT(fast_pred);
  
  return nll;
}
