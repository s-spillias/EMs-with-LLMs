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
  PARAMETER(log_allee);              // Allee effect threshold
  PARAMETER(log_temp_opt);           // Optimal temperature for COTS
  PARAMETER(log_temp_range);         // Temperature tolerance range
  PARAMETER(logit_fast_pref);        // Preference for fast-growing coral
  PARAMETER(log_half_sat);           // Half-saturation constant for feeding
  PARAMETER(log_r_fast);             // Fast coral growth rate
  PARAMETER(log_r_slow);             // Slow coral growth rate
  PARAMETER(log_obs_error_cots);     // Observation error SD for COTS
  PARAMETER(log_obs_error_coral);    // Observation error SD for coral

  // Transform parameters
  Type r_cots = exp(log_r_cots);
  Type K_cots = exp(log_K_cots);
  Type allee = exp(log_allee);
  Type temp_opt = exp(log_temp_opt);
  Type temp_range = exp(log_temp_range);
  Type fast_pref = invlogit(logit_fast_pref);
  Type half_sat = exp(log_half_sat);
  Type r_fast = exp(log_r_fast);
  Type r_slow = exp(log_r_slow);
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
    
    // 5. COTS population dynamics with Allee effect
    Type N = CppAD::CondExpGt(cots_pred(i-1), Type(0.0), cots_pred(i-1), Type(0.0));
    Type allee_effect = pow(N, Type(2.0)) / (pow(allee, Type(2.0)) + pow(N, Type(2.0)));
    Type density_effect = CppAD::CondExpLt(N, K_cots, Type(1.0) - N/K_cots, Type(0.0));
    Type growth = r_cots * N * density_effect * allee_effect;
    cots_pred(i) = N + growth * temp_effect * feeding_rate + cotsimm_dat(i);
    
    // 6. Coral dynamics
    Type fast_mortality = fast_pref * cots_pred(i-1) * feeding_rate;
    Type slow_mortality = (Type(1.0) - fast_pref) * cots_pred(i-1) * feeding_rate;
    
    fast_pred(i) = fast_pred(i-1) + r_fast * fast_pred(i-1) * (Type(100.0) - fast_pred(i-1))/Type(100.0) - fast_mortality;
    slow_pred(i) = slow_pred(i-1) + r_slow * slow_pred(i-1) * (Type(100.0) - slow_pred(i-1))/Type(100.0) - slow_mortality;
    
    // Ensure predictions stay within bounds
    cots_pred(i) = CppAD::CondExpGt(cots_pred(i), Type(0.0), cots_pred(i), Type(0.0));
    fast_pred(i) = CppAD::CondExpGt(fast_pred(i), Type(0.0), 
                    CppAD::CondExpLt(fast_pred(i), Type(100.0), fast_pred(i), Type(100.0)), 
                    Type(0.0));
    slow_pred(i) = CppAD::CondExpGt(slow_pred(i), Type(0.0),
                    CppAD::CondExpLt(slow_pred(i), Type(100.0), slow_pred(i), Type(100.0)),
                    Type(0.0));
  }

  // Observation model (log-normal)
  for(int i = 0; i < n; i++) {
    // Add observation likelihood using safe log transform
    Type log_cots_obs = CppAD::CondExpGt(cots_dat(i), eps, log(cots_dat(i)), log(eps));
    Type log_cots_pred = CppAD::CondExpGt(cots_pred(i), eps, log(cots_pred(i)), log(eps));
    nll -= dnorm(log_cots_obs, log_cots_pred, obs_error_cots, true);
    
    Type log_fast_obs = CppAD::CondExpGt(fast_dat(i), eps, log(fast_dat(i)), log(eps));
    Type log_fast_pred = CppAD::CondExpGt(fast_pred(i), eps, log(fast_pred(i)), log(eps));
    nll -= dnorm(log_fast_obs, log_fast_pred, obs_error_coral, true);
    
    Type log_slow_obs = CppAD::CondExpGt(slow_dat(i), eps, log(slow_dat(i)), log(eps));
    Type log_slow_pred = CppAD::CondExpGt(slow_pred(i), eps, log(slow_pred(i)), log(eps));
    nll -= dnorm(log_slow_obs, log_slow_pred, obs_error_coral, true);
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
