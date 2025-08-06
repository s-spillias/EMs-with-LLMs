#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Year);                  // Time series years
  DATA_VECTOR(sst_dat);              // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);          // COTS larval immigration (individuals/m2/year)
  DATA_VECTOR(cots_dat);             // Observed COTS density (individuals/m2)
  DATA_VECTOR(fast_dat);             // Observed fast-growing coral cover (%)
  DATA_VECTOR(slow_dat);             // Observed slow-growing coral cover (%)

  // Parameters
  PARAMETER(log_r_cots);             // COTS intrinsic growth rate
  PARAMETER(log_K_cots);             // COTS carrying capacity
  PARAMETER(log_temp_opt);           // Optimal temperature for COTS
  PARAMETER(log_temp_range);         // Temperature tolerance range
  PARAMETER(logit_fast_pref);        // Preference for fast-growing coral
  PARAMETER(log_half_sat);           // Half-saturation constant for feeding
  PARAMETER(log_r_fast);             // Fast coral growth rate
  PARAMETER(log_r_slow);             // Slow coral growth rate
  PARAMETER(log_process_error);      // Process error SD
  PARAMETER(log_obs_error_cots);     // Observation error SD for COTS
  PARAMETER(log_obs_error_coral);    // Observation error SD for coral

  // Transform parameters
  Type r_cots = exp(log_r_cots);
  Type K_cots = exp(log_K_cots);
  Type temp_opt = exp(log_temp_opt);
  Type temp_range = exp(log_temp_range);
  Type fast_pref = invlogit(logit_fast_pref);
  Type half_sat = exp(log_half_sat);
  Type r_fast = exp(log_r_fast);
  Type r_slow = exp(log_r_slow);
  Type process_error = exp(log_process_error);
  Type obs_error_cots = exp(log_obs_error_cots);
  Type obs_error_coral = exp(log_obs_error_coral);

  // Initialize
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> slow_pred(n);
  
  // Initialize first values
  cots_pred(0) = cots_dat(0);
  fast_pred(0) = fast_dat(0);
  slow_pred(0) = slow_dat(0);

  Type nll = 0.0;  // Negative log-likelihood
  Type eps = Type(1e-8);  // Small constant to prevent division by zero

  // Process model
  for(int i = 1; i < n; i++) {
    // 1. Temperature effect on COTS (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(i) - temp_opt) / temp_range, 2));
    
    // 2. Total coral resource
    Type total_coral = fast_pred(i-1) + slow_pred(i-1) + eps;
    
    // 3. COTS predation allocation
    Type fast_proportion = fast_pred(i-1) / total_coral;
    Type slow_proportion = slow_pred(i-1) / total_coral;
    
    // 4. Functional response (Holling Type II)
    Type feeding_rate = total_coral / (half_sat + total_coral);
    
    // 5. COTS population dynamics
    Type growth = r_cots * cots_pred(i-1) * (Type(1.0) - cots_pred(i-1)/K_cots);
    cots_pred(i) = cots_pred(i-1) + growth * temp_effect * feeding_rate + cotsimm_dat(i);
    
    // 6. Coral dynamics
    Type fast_mortality = fast_pref * cots_pred(i-1) * feeding_rate;
    Type slow_mortality = (Type(1.0) - fast_pref) * cots_pred(i-1) * feeding_rate;
    
    fast_pred(i) = fast_pred(i-1) + r_fast * fast_pred(i-1) * (Type(100.0) - fast_pred(i-1))/Type(100.0) - fast_mortality;
    slow_pred(i) = slow_pred(i-1) + r_slow * slow_pred(i-1) * (Type(100.0) - slow_pred(i-1))/Type(100.0) - slow_mortality;
    
    // Ensure predictions stay positive
    cots_pred(i) = exp(log(cots_pred(i) + eps));
    fast_pred(i) = exp(log(fast_pred(i) + eps));
    slow_pred(i) = exp(log(slow_pred(i) + eps));
  }

  // Observation model (log-normal)
  for(int i = 0; i < n; i++) {
    // Add observation likelihood
    nll -= dnorm(log(cots_dat(i) + eps), log(cots_pred(i) + eps), obs_error_cots, true);
    nll -= dnorm(log(fast_dat(i) + eps), log(fast_pred(i) + eps), obs_error_coral, true);
    nll -= dnorm(log(slow_dat(i) + eps), log(slow_pred(i) + eps), obs_error_coral, true);
  }

  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  REPORT(temp_opt);
  REPORT(temp_range);
  REPORT(fast_pref);
  
  return nll;
}
