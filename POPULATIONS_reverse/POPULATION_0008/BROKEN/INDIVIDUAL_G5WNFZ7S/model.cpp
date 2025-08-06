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
  PARAMETER(recruit_rate);      // Coral recruitment rate multiplier
  
  // Derived values
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_coral = exp(log_sigma_coral);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Vectors for predictions
  vector<Type> cots_pred(cots_dat.size());
  vector<Type> slow_pred(slow_dat.size());
  vector<Type> fast_pred(fast_dat.size());
  
  // Initial conditions
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Time series predictions
  for(int t = 1; t < cots_dat.size(); t++) {
    // 1. Temperature effect on COTS growth (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t) - temp_opt)/temp_width, 2));
    
    // 2. COTS population dynamics
    Type cots_growth = r_cots * temp_effect * cots_pred(t-1) * (1 - cots_pred(t-1)/K_cots);
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t);
    
    // 3. Coral dynamics with temperature-dependent growth
    Type coral_temp_effect = exp(-0.5 * pow((sst_dat(t) - temp_opt)/temp_width, 2));
    Type slow_growth = r_slow * (1 + recruit_rate * coral_temp_effect) * slow_pred(t-1) * (1 - slow_pred(t-1)/100);
    Type fast_growth = r_fast * (1 + recruit_rate * coral_temp_effect) * fast_pred(t-1) * (1 - fast_pred(t-1)/100);
    
    // 4. COTS predation impact
    Type slow_pred_impact = a_slow * cots_pred(t) * slow_pred(t-1)/(slow_pred(t-1) + eps);
    Type fast_pred_impact = a_fast * cots_pred(t) * fast_pred(t-1)/(fast_pred(t-1) + eps);
    
    // 5. Update coral cover
    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_pred_impact;
    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_pred_impact;
    
    // 6. Bound predictions to biological limits
    slow_pred(t) = std::max(Type(0), std::min(Type(100), slow_pred(t)));
    fast_pred(t) = std::max(Type(0), std::min(Type(100), fast_pred(t)));
    cots_pred(t) = std::max(Type(0), cots_pred(t));
  }
  
  // Observation model using lognormal distribution
  for(int t = 0; t < cots_dat.size(); t++) {
    // COTS observations
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots, true);
    
    // Coral cover observations
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_coral, true);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_coral, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  ADREPORT(sigma_cots);
  ADREPORT(sigma_coral);
  
  return nll;
}
