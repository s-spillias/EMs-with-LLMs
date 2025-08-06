#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(cots_dat);     // COTS abundance observations (individuals/m^2)
  DATA_VECTOR(slow_dat);     // Slow-growing coral cover observations (%)
  DATA_VECTOR(fast_dat);     // Fast-growing coral cover observations (%)
  DATA_VECTOR(sst_dat);      // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);  // COTS immigration rate (individuals/m^2/year)
  
  // Parameters
  PARAMETER(r_cots);         // COTS intrinsic growth rate
  PARAMETER(K_cots);         // COTS carrying capacity
  PARAMETER(r_slow);         // Slow coral growth rate
  PARAMETER(r_fast);         // Fast coral growth rate
  PARAMETER(K_slow);         // Slow coral carrying capacity
  PARAMETER(K_fast);         // Fast coral carrying capacity
  PARAMETER(alpha_slow);     // COTS feeding rate on slow corals
  PARAMETER(alpha_fast);     // COTS feeding rate on fast corals
  PARAMETER(temp_opt);       // Optimal temperature
  PARAMETER(temp_width);     // Temperature tolerance width
  PARAMETER(log_sigma_cots); // Log SD for COTS observations
  PARAMETER(log_sigma_coral);// Log SD for coral observations
  
  // Derived values
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_coral = exp(log_sigma_coral);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Initialize vectors for predictions
  vector<Type> cots_pred(cots_dat.size());
  vector<Type> slow_pred(slow_dat.size());
  vector<Type> fast_pred(fast_dat.size());
  
  // Small constant to prevent division by zero
  const Type eps = Type(1e-8);
  
  // Initial values
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Time series predictions
  for(int t = 1; t < cots_dat.size(); t++) {
    // 1. Temperature effect on coral growth (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t) - temp_opt)/temp_width, 2));
    
    // 2. COTS population dynamics
    Type total_food = (slow_pred(t-1) + fast_pred(t-1)) / (100.0 + eps); // Convert % to proportion
    Type cots_growth = r_cots * cots_pred(t-1) * (1.0 - cots_pred(t-1)/K_cots) * total_food;
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t);
    
    // 3. Coral dynamics
    // Temperature effect on COTS predation (increases with temperature)
    Type pred_effect = Type(1.0) + Type(0.1) * (sst_dat(t) - temp_opt);
    
    // Slow-growing corals
    Type slow_growth = r_slow * slow_pred(t-1) * (1.0 - slow_pred(t-1)/K_slow) * temp_effect;
    Type slow_pred_temp = slow_pred(t-1) + slow_growth - 
                         alpha_slow * pred_effect * cots_pred(t-1) * slow_pred(t-1)/(100.0 + eps);
    slow_pred(t) = Type(0.5) * (slow_pred_temp + sqrt(pow(slow_pred_temp, 2) + eps));
    
    // Fast-growing corals
    Type fast_growth = r_fast * fast_pred(t-1) * (1.0 - fast_pred(t-1)/K_fast) * temp_effect;
    Type fast_pred_temp = fast_pred(t-1) + fast_growth - 
                         alpha_fast * pred_effect * cots_pred(t-1) * fast_pred(t-1)/(100.0 + eps);
    fast_pred(t) = Type(0.5) * (fast_pred_temp + sqrt(pow(fast_pred_temp, 2) + eps));
  }
  
  // Observation model
  // Use lognormal distribution for COTS (strictly positive)
  for(int t = 0; t < cots_dat.size(); t++) {
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots, true);
  }
  
  // Use normal distribution for coral cover (can be near zero)
  for(int t = 0; t < slow_dat.size(); t++) {
    nll -= dnorm(slow_dat(t), slow_pred(t), sigma_coral + eps, true);
    nll -= dnorm(fast_dat(t), fast_pred(t), sigma_coral + eps, true);
  }
  
  // Reporting
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  ADREPORT(sigma_cots);
  ADREPORT(sigma_coral);
  
  return nll;
}
