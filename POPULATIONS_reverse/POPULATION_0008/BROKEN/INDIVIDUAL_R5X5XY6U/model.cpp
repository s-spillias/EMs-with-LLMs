#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(cots_dat);    // Observed COTS abundance (individuals/m^2)
  DATA_VECTOR(slow_dat);    // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);    // Observed fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);     // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat); // COTS immigration rate (individuals/m^2/year)
  
  // Parameters
  PARAMETER(r_cots);        // COTS intrinsic growth rate
  PARAMETER(K_cots);        // COTS carrying capacity
  PARAMETER(r_slow);        // Slow coral growth rate
  PARAMETER(r_fast);        // Fast coral growth rate
  PARAMETER(K_slow);        // Slow coral carrying capacity
  PARAMETER(K_fast);        // Fast coral carrying capacity
  PARAMETER(alpha_slow);    // COTS feeding rate on slow corals
  PARAMETER(alpha_fast);    // COTS feeding rate on fast corals
  PARAMETER(temp_opt);      // Optimal temperature for COTS
  PARAMETER(temp_width);    // Temperature tolerance width
  PARAMETER(log_sigma_cots);    // Log of COTS observation error SD
  PARAMETER(log_sigma_coral);   // Log of coral observation error SD
  
  // Derived values
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_coral = exp(log_sigma_coral);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Small constant to prevent division by zero
  const Type eps = Type(1e-8);
  
  // Vectors for predictions
  vector<Type> cots_pred(cots_dat.size());
  vector<Type> slow_pred(slow_dat.size());
  vector<Type> fast_pred(fast_dat.size());
  
  // Initial values
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Time series predictions
  for(int t = 1; t < cots_dat.size(); t++) {
    // 1. Temperature effect on COTS reproduction (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t) - temp_opt)/temp_width, 2));
    
    // 2. COTS population dynamics with temperature effect and immigration
    Type cots_growth = r_cots * temp_effect * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1)/(K_cots + eps));
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t);
    cots_pred(t) = cots_pred(t) < eps ? eps : cots_pred(t); // Ensure positive values
    
    // 3. Coral dynamics with COTS predation
    Type slow_growth = r_slow * slow_pred(t-1) * (Type(1.0) - slow_pred(t-1)/K_slow) * 
                      (Type(1.0) - Type(0.5) * fast_pred(t-1)/K_fast);
    Type fast_growth = r_fast * fast_pred(t-1) * (Type(1.0) - fast_pred(t-1)/K_fast) * 
                      (Type(1.0) - Type(0.5) * slow_pred(t-1)/K_slow);
    
    // 4. Predation rates with smooth transition near zero
    Type pred_slow = alpha_slow * cots_pred(t) * slow_pred(t-1)/(slow_pred(t-1) + eps);
    Type pred_fast = alpha_fast * cots_pred(t) * fast_pred(t-1)/(fast_pred(t-1) + eps);
    
    // 5. Update coral cover
    slow_pred(t) = slow_pred(t-1) + slow_growth - pred_slow;
    fast_pred(t) = fast_pred(t-1) + fast_growth - pred_fast;
    
    // Ensure coral cover stays positive
    slow_pred(t) = slow_pred(t) < eps ? eps : slow_pred(t);
    fast_pred(t) = fast_pred(t) < eps ? eps : fast_pred(t);
  }
  
  // Observation model using lognormal distribution
  for(int t = 0; t < cots_dat.size(); t++) {
    // 6. COTS abundance likelihood
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots, true);
    
    // 7. Coral cover likelihood
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
