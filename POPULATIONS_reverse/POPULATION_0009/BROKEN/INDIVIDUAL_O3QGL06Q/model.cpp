#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(cots_dat);      // Observed COTS density (individuals/m²)
  DATA_VECTOR(slow_dat);      // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);      // Observed fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);       // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);   // COTS immigration rate (individuals/m²/year)
  
  // Parameters
  PARAMETER(log_r_cots);      // Log COTS population growth rate
  PARAMETER(log_K_cots);      // Log COTS carrying capacity
  PARAMETER(log_alpha_fast);  // Log feeding rate on fast coral
  PARAMETER(log_alpha_slow);  // Log feeding rate on slow coral
  PARAMETER(log_r_fast);      // Log fast coral growth rate
  PARAMETER(log_r_slow);      // Log slow coral growth rate
  PARAMETER(log_temp_opt);    // Log optimal temperature for COTS
  PARAMETER(log_temp_width);  // Log temperature tolerance width
  PARAMETER(log_obs_sd);      // Log observation error SD
  
  // Transform parameters
  Type r_cots = exp(log_r_cots);
  Type K_cots = exp(log_K_cots);
  Type alpha_fast = exp(log_alpha_fast);
  Type alpha_slow = exp(log_alpha_slow);
  Type r_fast = exp(log_r_fast);
  Type r_slow = exp(log_r_slow);
  Type temp_opt = exp(log_temp_opt);
  Type temp_width = exp(log_temp_width);
  Type obs_sd = exp(log_obs_sd);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Constants for numerical stability
  Type eps = Type(1e-8);
  Type max_val = Type(1e3);  // Maximum allowed value
  Type min_sd = Type(0.01);  // Minimum standard deviation
  
  // Vectors for predictions
  vector<Type> cots_pred(cots_dat.size());
  vector<Type> slow_pred(slow_dat.size());
  vector<Type> fast_pred(fast_dat.size());
  
  // Initial conditions
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Process model
  for(int t = 1; t < cots_dat.size(); t++) {
    // 1. Temperature effect on COTS (linear ramp)
    Type temp_effect = Type(1.0);
    if(sst_dat(t) < temp_opt - temp_width) {
      temp_effect = Type(0.0);
    } else if(sst_dat(t) < temp_opt) {
      temp_effect = (sst_dat(t) - (temp_opt - temp_width)) / temp_width;
    }
    
    // 2. COTS population dynamics (basic)
    Type cots_growth = r_cots * cots_pred(t-1) * temp_effect;
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t);
    if(cots_pred(t) > K_cots) cots_pred(t) = K_cots;
    
    // 3. Coral dynamics (basic)
    Type total_coral = slow_pred(t-1) + fast_pred(t-1);
    if(total_coral > Type(100.0)) total_coral = Type(100.0);
    
    // Simple growth and predation
    Type slow_growth = r_slow * slow_pred(t-1);
    Type fast_growth = r_fast * fast_pred(t-1);
    
    Type pred_slow = alpha_slow * cots_pred(t);
    Type pred_fast = alpha_fast * cots_pred(t);
    
    slow_pred(t) = slow_pred(t-1) + slow_growth - pred_slow;
    fast_pred(t) = fast_pred(t-1) + fast_growth - pred_fast;
    
    // Ensure predictions stay positive
    slow_pred(t) = slow_pred(t) > eps ? slow_pred(t) : eps;
    fast_pred(t) = fast_pred(t) > eps ? fast_pred(t) : eps;
    cots_pred(t) = cots_pred(t) > eps ? cots_pred(t) : eps;
  }
  
  // Basic observation model
  Type sd = exp(log_obs_sd);
  for(int t = 0; t < cots_dat.size(); t++) {
    if(cots_dat(t) > Type(0.0)) {
      nll -= dnorm(log(cots_dat(t)), log(cots_pred(t) + Type(0.1)), sd, true);
    }
    if(slow_dat(t) > Type(0.0)) {
      nll -= dnorm(log(slow_dat(t)), log(slow_pred(t) + Type(0.1)), sd, true);
    }
    if(fast_dat(t) > Type(0.0)) {
      nll -= dnorm(log(fast_dat(t)), log(fast_pred(t) + Type(0.1)), sd, true);
    }
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  ADREPORT(r_cots);
  ADREPORT(K_cots);
  ADREPORT(alpha_fast);
  ADREPORT(alpha_slow);
  ADREPORT(r_fast);
  ADREPORT(r_slow);
  ADREPORT(temp_opt);
  ADREPORT(temp_width);
  
  return nll;
}
