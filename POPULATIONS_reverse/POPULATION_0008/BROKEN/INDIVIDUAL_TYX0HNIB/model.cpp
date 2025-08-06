#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(cots_dat);      // COTS abundance (individuals/m2)
  DATA_VECTOR(slow_dat);      // Slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);      // Fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);       // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);   // COTS immigration rate (individuals/m2/year)
  
  // Parameters
  PARAMETER(log_r_cots);      // Log COTS population growth rate (year^-1)
  PARAMETER(log_K_cots);      // Log COTS carrying capacity (individuals/m2)
  PARAMETER(log_temp_opt);    // Log optimal temperature for COTS (Celsius)
  PARAMETER(log_temp_width);  // Log temperature tolerance width (Celsius)
  PARAMETER(logit_pref_fast); // Logit feeding preference for fast corals (proportion)
  PARAMETER(log_g_slow);      // Log growth rate of slow corals (year^-1)
  PARAMETER(log_g_fast);      // Log growth rate of fast corals (year^-1)
  
  // Convert parameters
  Type r_cots = exp(log_r_cots);
  Type K_cots = exp(log_K_cots);
  Type temp_opt = exp(log_temp_opt);
  Type temp_width = exp(log_temp_width);
  Type pref_fast = 1/(1 + exp(-logit_pref_fast));
  Type g_slow = exp(log_g_slow);
  Type g_fast = exp(log_g_fast);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Constants
  Type eps = Type(1e-8);    // Small constant to prevent division by zero
  
  // Vectors for predictions
  int n = cots_dat.size();
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  
  // Initial conditions
  cots_pred[0] = cots_dat[0];
  slow_pred[0] = slow_dat[0];
  fast_pred[0] = fast_dat[0];
  
  // Process model
  for(int t = 1; t < cots_dat.size(); t++) {
    // 1. Temperature effect on COTS (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat[t] - temp_opt)/temp_width, 2));
    
    // 2. COTS population dynamics
    Type cots_growth = r_cots * cots_pred[t-1] * (1 - cots_pred[t-1]/K_cots) * temp_effect;
    cots_pred[t] = cots_pred[t-1] + cots_growth + cotsimm_dat[t];
    
    // 3. Coral dynamics with preferential feeding
    Type total_coral = slow_pred[t-1] + fast_pred[t-1] + eps;
    Type pref_slow = 1 - pref_fast;
    
    // Feeding pressure on each coral type
    Type feed_slow = cots_pred[t] * pref_slow * (slow_pred[t-1]/total_coral);
    Type feed_fast = cots_pred[t] * pref_fast * (fast_pred[t-1]/total_coral);
    
    // Coral growth and mortality
    slow_pred[t] = slow_pred[t-1] + g_slow * slow_pred[t-1] * (100 - total_coral)/100 - feed_slow;
    fast_pred[t] = fast_pred[t-1] + g_fast * fast_pred[t-1] * (100 - total_coral)/100 - feed_fast;
    
    // Ensure predictions stay positive
    slow_pred[t] = slow_pred[t] < eps ? eps : slow_pred[t];
    fast_pred[t] = fast_pred[t] < eps ? eps : fast_pred[t];
    cots_pred[t] = cots_pred[t] < eps ? eps : cots_pred[t];
  }
  
  // Observation model
  Type cv_cots = Type(0.2);    // Minimum CV for COTS observations
  Type cv_coral = Type(0.1);   // Minimum CV for coral observations
  
  for(int t = 0; t < cots_dat.size(); t++) {
    // Log-normal likelihood for COTS
    nll -= dnorm(log(cots_dat[t]), log(cots_pred[t]), 
                 sqrt(log(1 + pow(cv_cots, 2))), true);
    
    // Log-normal likelihood for corals
    nll -= dnorm(log(slow_dat[t]), log(slow_pred[t]), 
                 sqrt(log(1 + pow(cv_coral, 2))), true);
    nll -= dnorm(log(fast_dat[t]), log(fast_pred[t]), 
                 sqrt(log(1 + pow(cv_coral, 2))), true);
  }
  
  // Report predictions and parameters
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(r_cots);
  REPORT(K_cots);
  REPORT(temp_opt);
  REPORT(temp_width);
  REPORT(pref_fast);
  REPORT(g_slow);
  REPORT(g_fast);
  
  // Report objective function value
  Type objective_function = nll;
  REPORT(objective_function);
  ADREPORT(objective_function);
  
  return objective_function;
}
