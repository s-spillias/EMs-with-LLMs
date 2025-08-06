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
  PARAMETER(r_slow);        // Slow-growing coral growth rate
  PARAMETER(r_fast);        // Fast-growing coral growth rate
  PARAMETER(K_coral);       // Total coral carrying capacity
  PARAMETER(alpha_slow);    // COTS feeding rate on slow coral
  PARAMETER(alpha_fast);    // COTS feeding rate on fast coral
  PARAMETER(temp_opt);      // Optimal temperature for COTS
  PARAMETER(temp_width);    // Temperature tolerance width
  PARAMETER(log_sigma_cots);    // Log of COTS observation error SD
  PARAMETER(log_sigma_coral);   // Log of coral observation error SD
  PARAMETER(pred_sat);         // Predation saturation coefficient
  
  // Derived values
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_coral = exp(log_sigma_coral);
  Type small_number = Type(1e-8);
  int n = cots_dat.size();
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Vectors for predictions
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  
  // Initial conditions
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Time series predictions
  for(int t = 1; t < n; t++) {
    // 1. Temperature effect on COTS (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t) - temp_opt)/temp_width, 2));
    
    // 2. Total coral cover and space limitation
    Type total_coral = slow_pred(t-1) + fast_pred(t-1);
    Type space_available = K_coral/(total_coral + small_number);
    
    // 3. COTS population dynamics
    Type cots_growth = r_cots * temp_effect * cots_pred(t-1) * (1 - cots_pred(t-1)/K_cots);
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t);
    
    // 4. Coral dynamics with COTS predation
    Type slow_growth = r_slow * slow_pred(t-1) * (1 - total_coral/K_coral);
    Type fast_growth = r_fast * fast_pred(t-1) * (1 - total_coral/K_coral);
    
    slow_pred(t) = slow_pred(t-1) + slow_growth - 
                   alpha_slow * cots_pred(t-1) * slow_pred(t-1) / (1 + pred_sat * slow_pred(t-1));
    
    fast_pred(t) = fast_pred(t-1) + fast_growth - 
                   alpha_fast * cots_pred(t-1) * fast_pred(t-1) / (1 + pred_sat * fast_pred(t-1));
    
    // Ensure predictions stay positive
    slow_pred(t) = slow_pred(t) > 0 ? slow_pred(t) : small_number;
    fast_pred(t) = fast_pred(t) > 0 ? fast_pred(t) : small_number;
    cots_pred(t) = cots_pred(t) > 0 ? cots_pred(t) : small_number;
  }
  
  // Observation model using lognormal distribution
  for(int t = 0; t < n; t++) {
    // COTS observations
    nll -= dnorm(log(cots_dat(t)), log(cots_pred(t)), sigma_cots, true);
    
    // Coral observations
    nll -= dnorm(log(slow_dat(t)), log(slow_pred(t)), sigma_coral, true);
    nll -= dnorm(log(fast_dat(t)), log(fast_pred(t)), sigma_coral, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  ADREPORT(sigma_cots);
  ADREPORT(sigma_coral);
  
  return nll;
}
