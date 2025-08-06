#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(cots_dat);      // COTS abundance observations (individuals/m²)
  DATA_VECTOR(slow_dat);      // Slow-growing coral cover observations (%)
  DATA_VECTOR(fast_dat);      // Fast-growing coral cover observations (%)
  DATA_VECTOR(sst_dat);       // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);   // COTS immigration rate (individuals/m²/year)
  
  // Parameters
  PARAMETER(log_r_cots);      // Log COTS population growth rate (year⁻¹)
  PARAMETER(log_K_cots);      // Log COTS carrying capacity (individuals/m²)
  PARAMETER(log_r_slow);      // Log slow coral growth rate (year⁻¹)
  PARAMETER(log_r_fast);      // Log fast coral growth rate (year⁻¹)
  PARAMETER(log_alpha_slow);  // Log COTS predation rate on slow coral (m²/individual/year)
  PARAMETER(log_alpha_fast);  // Log COTS predation rate on fast coral (m²/individual/year)
  PARAMETER(log_temp_opt);    // Log optimal temperature for coral growth (°C)
  PARAMETER(log_temp_width);  // Log temperature tolerance width (°C)
  
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
  Type nll = Type(0.0);
  
  // Constants for numerical stability
  Type eps = Type(1e-8);
  Type max_coral = Type(100.0);
  
  // Vectors for predictions
  int n = cots_dat.size();
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  
  // Initial conditions
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Process model
  for(int t = 1; t < cots_dat.size(); t++) {
    // Temperature effect (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t) - temp_opt) / temp_width, 2));
    
    // COTS dynamics
    Type cots_growth = r_cots * cots_pred(t-1) * (1.0 - cots_pred(t-1) / K_cots);
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t);
    cots_pred(t) = cots_pred(t) < eps ? eps : cots_pred(t);
    
    // Coral dynamics
    Type slow_growth = r_slow * slow_pred(t-1) * (max_coral - slow_pred(t-1)) / max_coral * temp_effect;
    Type fast_growth = r_fast * fast_pred(t-1) * (max_coral - fast_pred(t-1)) / max_coral * temp_effect;
    
    // Predation
    Type pred_slow = alpha_slow * cots_pred(t-1) * slow_pred(t-1);
    Type pred_fast = alpha_fast * cots_pred(t-1) * fast_pred(t-1);
    
    // Update corals
    slow_pred(t) = slow_pred(t-1) + slow_growth - pred_slow;
    fast_pred(t) = fast_pred(t-1) + fast_growth - pred_fast;
    
    // Bound predictions
    slow_pred(t) = slow_pred(t) < eps ? eps : (slow_pred(t) > max_coral ? max_coral : slow_pred(t));
    fast_pred(t) = fast_pred(t) < eps ? eps : (fast_pred(t) > max_coral ? max_coral : fast_pred(t));
  }
  
  // Observation model using lognormal distribution
  Type cv_cots = Type(0.2);   // Minimum coefficient of variation
  Type cv_coral = Type(0.15); // Minimum coefficient of variation
  
  for(int t = 0; t < cots_dat.size(); t++) {
    // COTS likelihood
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), 
                sqrt(log(1.0 + pow(cv_cots, 2))), true);
    
    // Coral likelihoods
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps),
                sqrt(log(1.0 + pow(cv_coral, 2))), true);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps),
                sqrt(log(1.0 + pow(cv_coral, 2))), true);
  }
  
  // Report variables
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(r_cots);
  REPORT(K_cots);
  REPORT(r_slow);
  REPORT(r_fast);
  REPORT(alpha_slow);
  REPORT(alpha_fast);
  REPORT(temp_opt);
  REPORT(temp_width);
  REPORT(nll);
  
  // Report objective function value
  Type objective = nll;
  ADREPORT(objective);
  
  return objective;
}
