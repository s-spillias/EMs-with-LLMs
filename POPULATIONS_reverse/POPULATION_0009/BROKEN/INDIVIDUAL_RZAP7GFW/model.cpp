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
  PARAMETER(r_cots);      // COTS intrinsic growth rate
  PARAMETER(K_cots);      // COTS carrying capacity
  PARAMETER(temp_opt);    // Optimal temperature for COTS reproduction
  PARAMETER(temp_width);  // Temperature tolerance width
  PARAMETER(g_slow);      // Growth rate of slow-growing corals
  PARAMETER(g_fast);      // Growth rate of fast-growing corals
  PARAMETER(K_slow);      // Carrying capacity of slow-growing corals
  PARAMETER(K_fast);      // Carrying capacity of fast-growing corals
  PARAMETER(a_slow);      // COTS feeding rate on slow-growing corals
  PARAMETER(a_fast);      // COTS feeding rate on fast-growing corals

  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Constants for numerical stability
  const Type eps = Type(1e-8);
  
  // Vectors to store predictions
  int n = cots_dat.size();
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  
  // Initialize first time step predictions with first observations
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Time series calculations
  for(int t = 1; t < n; t++) {
    // 1. Temperature effect on COTS reproduction (Gaussian response)
    Type temp_effect = exp(-pow(sst_dat(t-1) - temp_opt, 2.0) / (2.0 * pow(temp_width, 2.0)));
    
    // 2. COTS population dynamics with temperature effect and immigration
    Type cots_growth = temp_effect * r_cots * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1)/(K_cots + eps));
    Type cots_immigration = cotsimm_dat(t-1);
    cots_pred(t) = cots_pred(t-1) + cots_growth + cots_immigration;
    
    // 3. Coral dynamics with COTS predation
    Type slow_growth = g_slow * slow_pred(t-1) * (Type(1.0) - slow_pred(t-1)/(K_slow + eps));
    Type slow_predation = a_slow * cots_pred(t-1) * slow_pred(t-1);
    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_predation;
    
    Type fast_growth = g_fast * fast_pred(t-1) * (Type(1.0) - fast_pred(t-1)/(K_fast + eps));
    Type fast_predation = a_fast * cots_pred(t-1) * fast_pred(t-1);
    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_predation;
    
    // Bound predictions within reasonable ranges
    cots_pred(t) = K_cots / (Type(1.0) + exp(-cots_pred(t)));
    slow_pred(t) = K_slow / (Type(1.0) + exp(-slow_pred(t)/Type(10.0)));
    fast_pred(t) = K_fast / (Type(1.0) + exp(-fast_pred(t)/Type(10.0)));
  }
  
  // Observation model using log-normal distribution
  // Fixed minimum SD to prevent numerical issues
  Type sd_min = Type(0.1);
  
  for(int t = 0; t < cots_dat.size(); t++) {
    // COTS observations
    nll -= dnorm(log(cots_dat(t) + eps), 
                 log(cots_pred(t) + eps),
                 sd_min + cots_pred(t)/K_cots,
                 true);
    
    // Coral observations
    nll -= dnorm(log(slow_dat(t) + eps),
                 log(slow_pred(t) + eps),
                 sd_min + slow_pred(t)/K_slow,
                 true);
    
    nll -= dnorm(log(fast_dat(t) + eps),
                 log(fast_pred(t) + eps),
                 sd_min + fast_pred(t)/K_fast,
                 true);
  }
  
  // Soft bounds on parameters using penalty terms
  nll += 0.01 * pow(exp(r_cots) / (1.0 + exp(r_cots)) - 0.5, 2);    // Keep r_cots moderate
  nll += 0.01 * pow(K_cots - 2.0, 2);                                // Keep K_cots near 2
  nll += 0.01 * pow(temp_opt - 28.0, 2);                            // Keep temp_opt near 28°C
  
  // Calculate objective value
  Type objective = nll;
  
  // Report variables
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(objective);
  
  // AD Report
  ADREPORT(cots_pred);
  ADREPORT(slow_pred);
  ADREPORT(fast_pred);
  ADREPORT(objective);
  
  return objective;
}
