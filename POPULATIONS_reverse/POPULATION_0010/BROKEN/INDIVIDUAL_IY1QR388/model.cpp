#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Year);                  // Time vector
  DATA_VECTOR(cots_dat);             // COTS abundance data (individuals/m2)
  DATA_VECTOR(slow_dat);             // Slow-growing coral cover data (%)
  DATA_VECTOR(fast_dat);             // Fast-growing coral cover data (%)
  DATA_VECTOR(sst_dat);              // Sea surface temperature data (Celsius)
  DATA_VECTOR(cotsimm_dat);          // COTS immigration rate data (individuals/m2/year)

  // Parameters
  PARAMETER(r_cots);                 // COTS intrinsic growth rate
  PARAMETER(K_cots);                 // COTS carrying capacity
  PARAMETER(r_slow);                 // Slow coral growth rate
  PARAMETER(r_fast);                 // Fast coral growth rate
  PARAMETER(K_coral);                // Total coral carrying capacity
  PARAMETER(alpha_sf);               // Competition effect of fast on slow
  PARAMETER(alpha_fs);               // Competition effect of slow on fast
  PARAMETER(a_slow);                 // COTS attack rate on slow coral
  PARAMETER(a_fast);                 // COTS attack rate on fast coral
  PARAMETER(h_cots);                 // COTS handling time
  PARAMETER(T_opt);                  // Optimal temperature
  PARAMETER(T_range);                // Temperature range tolerance
  PARAMETER(sigma_cots);             // COTS observation error
  PARAMETER(sigma_slow);             // Slow coral observation error
  PARAMETER(sigma_fast);             // Fast coral observation error
  PARAMETER(base_immigration);       // Baseline immigration rate
  PARAMETER(sigma_imm);             // Immigration observation error

  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Small constant to prevent division by zero
  const Type eps = Type(1e-8);

  // Vectors for predictions
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  vector<Type> cotsimm_pred(n);

  // Initial conditions
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  cotsimm_pred(0) = cotsimm_dat(0);

  // Process model
  for(int t = 1; t < n; t++) {
    // Temperature scaling (0-1)
    Type temp_diff = (sst_dat(t) - T_opt) / T_range;
    Type temp_effect = exp(-temp_diff * temp_diff);
    
    // Type II functional responses
    Type total_prey = slow_pred(t-1) + fast_pred(t-1);
    Type denom = Type(1) + h_cots * total_prey;
    Type f_slow = a_slow * slow_pred(t-1) / (denom + eps);
    Type f_fast = a_fast * fast_pred(t-1) / (denom + eps);
    
    // Immigration prediction
    cotsimm_pred(t) = base_immigration * temp_effect;
    
    // COTS dynamics
    Type cots_growth = r_cots * cots_pred(t-1) * (Type(1) - cots_pred(t-1)/K_cots);
    cots_pred(t) = cots_pred(t-1) + temp_effect * cots_growth + cotsimm_pred(t);
    
    // Coral dynamics
    Type slow_comp = Type(1) - (slow_pred(t-1) + alpha_sf * fast_pred(t-1))/K_coral;
    Type fast_comp = Type(1) - (fast_pred(t-1) + alpha_fs * slow_pred(t-1))/K_coral;
    
    Type slow_growth = r_slow * slow_pred(t-1) * slow_comp;
    Type fast_growth = r_fast * fast_pred(t-1) * fast_comp;
    
    slow_pred(t) = slow_pred(t-1) + temp_effect * slow_growth - f_slow * cots_pred(t-1);
    fast_pred(t) = fast_pred(t-1) + temp_effect * fast_growth - f_fast * cots_pred(t-1);
  }

  // Observation model using lognormal distribution
  for(int t = 0; t < n; t++) {
    // Add small constant to prevent log(0)
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots, true);
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_slow, true);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_fast, true);
    nll -= dnorm(log(cotsimm_dat(t) + eps), log(cotsimm_pred(t) + eps), sigma_imm, true);
  }

  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(cotsimm_pred);
  
  return nll;
}
