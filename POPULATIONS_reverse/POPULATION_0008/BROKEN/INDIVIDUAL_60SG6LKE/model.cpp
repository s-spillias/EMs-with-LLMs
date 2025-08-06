#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(cots_dat);    // COTS abundance (individuals/m^2)
  DATA_VECTOR(slow_dat);    // Slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);    // Fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);     // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat); // COTS immigration rate (individuals/m^2/year)
  
  // Parameters
  PARAMETER(r_cots);        // COTS intrinsic growth rate
  PARAMETER(K_cots);        // COTS carrying capacity
  PARAMETER(a_fast);        // COTS predation rate on fast coral
  PARAMETER(a_slow);        // COTS predation rate on slow coral
  PARAMETER(r_fast);        // Fast coral growth rate
  PARAMETER(r_slow);        // Slow coral growth rate
  PARAMETER(temp_opt);      // Optimal temperature for COTS
  PARAMETER(temp_width);    // Temperature tolerance width
  PARAMETER(log_sigma_cots);    // Log SD for COTS observations
  PARAMETER(log_sigma_coral);   // Log SD for coral observations
  PARAMETER(c_fast_slow);   // Competition effect of fast on slow coral
  PARAMETER(c_slow_fast);   // Competition effect of slow on fast coral
  
  // Derived values
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_coral = exp(log_sigma_coral);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Transform parameters to ensure positivity
  Type r_cots_bound = exp(r_cots);
  Type K_cots_bound = exp(K_cots);
  Type a_fast_bound = exp(a_fast);
  Type a_slow_bound = exp(a_slow);
  Type r_fast_bound = exp(r_fast);
  Type r_slow_bound = exp(r_slow);
  
  // Vectors for predictions
  vector<Type> cots_pred(cots_dat.size());
  vector<Type> slow_pred(slow_dat.size());
  vector<Type> fast_pred(fast_dat.size());
  
  // Initial conditions
  cots_pred(0) = std::max(eps, cots_dat(0));
  slow_pred(0) = std::max(eps, slow_dat(0));
  fast_pred(0) = std::max(eps, fast_dat(0));
  
  // Time series predictions
  for(int t = 1; t < cots_dat.size(); t++) {
    // 1. Temperature effect on COTS growth (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t) - temp_opt)/temp_width, 2));
    
    // 2. COTS population dynamics
    Type cots_growth = r_cots_bound * temp_effect * cots_pred(t-1) * (1 - cots_pred(t-1)/K_cots_bound);
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t);
    
    // 3. Coral dynamics with competition
    Type slow_growth = r_slow_bound * slow_pred(t-1) * (100 - slow_pred(t-1) - c_fast_slow * fast_pred(t-1))/100;
    Type fast_growth = r_fast_bound * fast_pred(t-1) * (100 - fast_pred(t-1) - c_slow_fast * slow_pred(t-1))/100;
    
    // 4. COTS predation impact
    Type slow_pred_impact = a_slow_bound * cots_pred(t) * slow_pred(t-1)/(slow_pred(t-1) + eps);
    Type fast_pred_impact = a_fast_bound * cots_pred(t) * fast_pred(t-1)/(fast_pred(t-1) + eps);
    
    // 5. Update coral cover
    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_pred_impact;
    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_pred_impact;
    
    // 6. Bound predictions to biological limits
    if(slow_pred(t) < 0) slow_pred(t) = 0;
    if(slow_pred(t) > 100) slow_pred(t) = 100;
    if(fast_pred(t) < 0) fast_pred(t) = 0;
    if(fast_pred(t) > 100) fast_pred(t) = 100;
    if(cots_pred(t) < 0) cots_pred(t) = 0;
  }
  
  // Observation model using normal distribution
  for(int t = 0; t < cots_dat.size(); t++) {
    // Add small constant to prevent taking log of zero
    Type cots_obs = cots_dat(t) + eps;
    Type cots_model = cots_pred(t) + eps;
    Type slow_obs = slow_dat(t) + eps;
    Type slow_model = slow_pred(t) + eps;
    Type fast_obs = fast_dat(t) + eps;
    Type fast_model = fast_pred(t) + eps;
    
    nll -= dnorm(log(cots_obs), log(cots_model), sigma_cots, true);
    nll -= dnorm(log(slow_obs), log(slow_model), sigma_coral, true);
    nll -= dnorm(log(fast_obs), log(fast_model), sigma_coral, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  ADREPORT(sigma_cots);
  ADREPORT(sigma_coral);
  
  return nll;
}
