#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(cots_dat);    // COTS abundance (individuals/m2)
  DATA_VECTOR(slow_dat);    // Slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);    // Fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);     // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat); // COTS immigration rate (individuals/m2/year)
  
  // Parameters
  PARAMETER(log_r_cots);      // COTS intrinsic growth rate (year^-1)
  PARAMETER(log_K_cots);      // COTS carrying capacity (individuals/m2)
  PARAMETER(log_r_slow);      // Slow coral growth rate (year^-1)
  PARAMETER(log_r_fast);      // Fast coral growth rate (year^-1)
  PARAMETER(log_K_slow);      // Slow coral carrying capacity (%)
  PARAMETER(log_K_fast);      // Fast coral carrying capacity (%)
  PARAMETER(log_alpha_slow);  // COTS feeding rate on slow coral (m2/individual/year)
  PARAMETER(log_alpha_fast);  // COTS feeding rate on fast coral (m2/individual/year)
  PARAMETER(log_temp_opt);    // Optimal temperature for COTS (Celsius)
  PARAMETER(log_temp_width);  // Temperature tolerance width (Celsius)
  
  // Transform parameters
  Type r_cots = exp(log_r_cots);
  Type K_cots = exp(log_K_cots);
  Type r_slow = exp(log_r_slow);
  Type r_fast = exp(log_r_fast);
  Type K_slow = exp(log_K_slow);
  Type K_fast = exp(log_K_fast);
  Type alpha_slow = exp(log_alpha_slow);
  Type alpha_fast = exp(log_alpha_fast);
  Type temp_opt = exp(log_temp_opt);
  Type temp_width = exp(log_temp_width);
  
  // Constants for numerical stability
  Type eps = Type(1e-8);    // Small constant to prevent division by zero
  Type min_sd = Type(0.01); // Minimum standard deviation
  Type penalty = Type(1000.0); // Penalty for invalid predictions
  
  // Initialize negative log-likelihood
  Type nll = Type(0.0);
  
  // Process model predictions
  vector<Type> cots_pred(cots_dat.size());
  vector<Type> slow_pred(slow_dat.size());
  vector<Type> fast_pred(fast_dat.size());
  
  // Initial conditions
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Time series predictions
  for(int t = 1; t < cots_dat.size(); t++) {
    // 1. Temperature effect on COTS (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t) - temp_opt) / temp_width, 2));
    
    // 2. COTS population dynamics
    Type cots_growth = r_cots * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1) / K_cots) * temp_effect;
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t);
    
    // 3. Coral dynamics with predation
    // Slow-growing coral
    Type slow_growth = r_slow * slow_pred(t-1) * (Type(1.0) - slow_pred(t-1) / K_slow);
    Type slow_pred_rate = alpha_slow * cots_pred(t) * slow_pred(t-1) / (slow_pred(t-1) + fast_pred(t-1) + eps);
    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_pred_rate;
    
    // Fast-growing coral
    Type fast_growth = r_fast * fast_pred(t-1) * (Type(1.0) - fast_pred(t-1) / K_fast);
    Type fast_pred_rate = alpha_fast * cots_pred(t) * fast_pred(t-1) / (slow_pred(t-1) + fast_pred(t-1) + eps);
    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_pred_rate;
    
    // Ensure predictions stay within reasonable bounds
    cots_pred(t) = cots_pred(t) < eps ? eps : cots_pred(t);
    cots_pred(t) = cots_pred(t) > K_cots ? K_cots : cots_pred(t);
    
    slow_pred(t) = slow_pred(t) < eps ? eps : slow_pred(t);
    slow_pred(t) = slow_pred(t) > K_slow ? K_slow : slow_pred(t);
    
    fast_pred(t) = fast_pred(t) < eps ? eps : fast_pred(t);
    fast_pred(t) = fast_pred(t) > K_fast ? K_fast : fast_pred(t);
  }
  
  // Observation model (log-normal)
  for(int t = 0; t < cots_dat.size(); t++) {
    Type sd_cots = Type(0.2);
    Type sd_slow = Type(0.2);
    Type sd_fast = Type(0.2);
    
    // Add to negative log-likelihood
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t)), sd_cots, true);
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t)), sd_slow, true);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t)), sd_fast, true);
    
    // Add penalties for invalid predictions
    if(cots_pred(t) <= eps || cots_pred(t) >= K_cots) nll += penalty;
    if(slow_pred(t) <= eps || slow_pred(t) >= K_slow) nll += penalty;
    if(fast_pred(t) <= eps || fast_pred(t) >= K_fast) nll += penalty;
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(nll);
  ADREPORT(r_cots);
  ADREPORT(K_cots);
  ADREPORT(r_slow);
  ADREPORT(r_fast);
  ADREPORT(K_slow);
  ADREPORT(K_fast);
  ADREPORT(alpha_slow);
  ADREPORT(alpha_fast);
  ADREPORT(temp_opt);
  ADREPORT(temp_width);
  
  return nll;
}
