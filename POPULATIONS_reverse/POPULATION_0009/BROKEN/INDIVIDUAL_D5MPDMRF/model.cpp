#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(cots_dat);     // Observed COTS density (individuals/m2)
  DATA_VECTOR(slow_dat);     // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);     // Observed fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);      // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);  // COTS immigration rate (individuals/m2/year)
  
  // Parameters
  PARAMETER(log_r_cots);     // Log COTS population growth rate (year^-1)
  PARAMETER(log_K_cots);     // Log COTS carrying capacity (individuals/m2)
  PARAMETER(log_r_slow);     // Log slow coral growth rate (year^-1)
  PARAMETER(log_r_fast);     // Log fast coral growth rate (year^-1)
  PARAMETER(log_alpha_slow); // Log COTS predation rate on slow coral (m2/individual/year)
  PARAMETER(log_alpha_fast); // Log COTS predation rate on fast coral (m2/individual/year)
  PARAMETER(log_temp_opt);   // Log optimal temperature for coral growth (Celsius)
  PARAMETER(log_temp_width); // Log temperature tolerance width (Celsius)
  
  // Transform parameters
  Type r_cots = exp(log_r_cots);
  Type K_cots = exp(log_K_cots);
  Type r_slow = exp(log_r_slow);
  Type r_fast = exp(log_r_fast);
  Type alpha_slow = exp(log_alpha_slow);
  Type alpha_fast = exp(log_alpha_fast);
  Type temp_opt = exp(log_temp_opt);
  Type temp_width = exp(log_temp_width);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Minimum standard deviations to prevent numerical issues
  Type min_sd = Type(0.01);
  
  // Storage for predictions
  vector<Type> cots_pred(cots_dat.size());
  vector<Type> slow_pred(slow_dat.size());
  vector<Type> fast_pred(fast_dat.size());
  
  // Initial conditions
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Process model
  for(int t = 1; t < cots_dat.size(); t++) {
    // 1. Temperature effect on coral growth (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t) - temp_opt) / temp_width, 2));
    
    // 2. COTS population dynamics with immigration
    Type cots_growth = r_cots * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1)/K_cots);
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t);
    
    // 3. Slow-growing coral dynamics
    Type slow_growth = r_slow * temp_effect * slow_pred(t-1) * (Type(1.0) - slow_pred(t-1)/Type(100.0));
    Type slow_pred_loss = alpha_slow * cots_pred(t-1) * slow_pred(t-1);
    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_pred_loss;
    
    // 4. Fast-growing coral dynamics  
    Type fast_growth = r_fast * temp_effect * fast_pred(t-1) * (Type(1.0) - fast_pred(t-1)/Type(100.0));
    Type fast_pred_loss = alpha_fast * cots_pred(t-1) * fast_pred(t-1);
    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_pred_loss;
    
    // Ensure predictions stay positive and within bounds
    cots_pred(t) = posfun(cots_pred(t), Type(1e-8), nll);
    slow_pred(t) = posfun(slow_pred(t), Type(1e-8), nll);
    fast_pred(t) = posfun(fast_pred(t), Type(1e-8), nll);
    
    // Add penalty for coral cover exceeding 100%
    if (slow_pred(t) > Type(100.0)) nll += pow(slow_pred(t) - Type(100.0), 2);
    if (fast_pred(t) > Type(100.0)) nll += pow(fast_pred(t) - Type(100.0), 2);
  }
  
  // Observation model (log-normal)
  for(int t = 0; t < cots_dat.size(); t++) {
    // COTS observations
    Type sd_cots = min_sd > Type(0.2) ? min_sd : Type(0.2);
    nll -= dnorm(log(cots_dat(t)), log(cots_pred(t)), sd_cots, true);
    
    // Coral cover observations  
    Type sd_coral = min_sd > Type(0.2) ? min_sd : Type(0.2);
    nll -= dnorm(log(slow_dat(t)), log(slow_pred(t)), sd_coral, true);
    nll -= dnorm(log(fast_dat(t)), log(fast_pred(t)), sd_coral, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  ADREPORT(r_cots);
  ADREPORT(K_cots);
  ADREPORT(r_slow);
  ADREPORT(r_fast);
  ADREPORT(alpha_slow);
  ADREPORT(alpha_fast);
  ADREPORT(temp_opt);
  ADREPORT(temp_width);
  
  REPORT(nll);
  return nll;
}
