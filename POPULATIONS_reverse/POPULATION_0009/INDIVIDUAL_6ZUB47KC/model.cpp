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
  PARAMETER(r_cots);      // COTS intrinsic growth rate
  PARAMETER(K_cots);      // COTS carrying capacity
  PARAMETER(r_slow);      // Slow coral growth rate
  PARAMETER(r_fast);      // Fast coral growth rate
  PARAMETER(K_slow);      // Slow coral carrying capacity
  PARAMETER(K_fast);      // Fast coral carrying capacity
  PARAMETER(a_slow);      // Attack rate on slow corals
  PARAMETER(a_fast);      // Attack rate on fast corals
  PARAMETER(temp_opt);    // Optimal temperature
  PARAMETER(temp_width);  // Temperature tolerance width
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Constants for numerical stability
  const Type eps = Type(1e-8);
  
  // Vectors for predictions
  vector<Type> cots_pred(cots_dat.size());
  vector<Type> slow_pred(slow_dat.size());
  vector<Type> fast_pred(fast_dat.size());
  
  // Initial conditions
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Model equations
  for(int t = 1; t < cots_dat.size(); t++) {
    // 1. Temperature effect on COTS reproduction (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t) - temp_opt)/temp_width, 2));
    
    // 2. COTS population dynamics with immigration
    Type cots_growth = r_cots * temp_effect * cots_pred(t-1) * (1 - cots_pred(t-1)/K_cots);
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t);
    
    // 3. Coral dynamics with COTS predation
    Type slow_growth = r_slow * slow_pred(t-1) * (1 - slow_pred(t-1)/K_slow);
    Type fast_growth = r_fast * fast_pred(t-1) * (1 - fast_pred(t-1)/K_fast);
    
    // 4. Predation rates with smooth saturation
    Type slow_pred_rate = (a_slow * slow_pred(t-1) * cots_pred(t-1)) / (1 + eps + slow_pred(t-1));
    Type fast_pred_rate = (a_fast * fast_pred(t-1) * cots_pred(t-1)) / (1 + eps + fast_pred(t-1));
    
    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_pred_rate;
    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_pred_rate;
    
    // Ensure predictions stay positive
    slow_pred(t) = slow_pred(t) < eps ? eps : slow_pred(t);
    fast_pred(t) = fast_pred(t) < eps ? eps : fast_pred(t);
    cots_pred(t) = cots_pred(t) < eps ? eps : cots_pred(t);
  }
  
  // Likelihood calculations using log-normal distribution
  // Fixed minimum SD to prevent numerical issues
  Type min_sd = Type(0.1);
  
  for(int t = 0; t < cots_dat.size(); t++) {
    // COTS abundance likelihood
    nll -= dnorm(log(cots_dat(t)), log(cots_pred(t)), min_sd + Type(0.2), true);
    
    // Coral cover likelihoods
    nll -= dnorm(log(slow_dat(t)), log(slow_pred(t)), min_sd + Type(0.3), true);
    nll -= dnorm(log(fast_dat(t)), log(fast_pred(t)), min_sd + Type(0.3), true);
  }
  
  // Add smooth penalties for parameter bounds
  nll += Type(0.01) * pow(r_cots < Type(0) ? -r_cots : Type(0), 2);
  nll += Type(0.01) * pow(K_cots < Type(0) ? -K_cots : Type(0), 2);
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  ADREPORT(cots_pred);
  ADREPORT(slow_pred);
  ADREPORT(fast_pred);
  
  return nll;
}
