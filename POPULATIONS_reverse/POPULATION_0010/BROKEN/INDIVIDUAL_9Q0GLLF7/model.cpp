#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Year);           // Time vector (years)
  DATA_VECTOR(cots_dat);       // COTS abundance data (individuals/m2)
  DATA_VECTOR(slow_dat);       // Slow-growing coral cover data (%)
  DATA_VECTOR(fast_dat);       // Fast-growing coral cover data (%)
  DATA_VECTOR(sst_dat);        // Sea surface temperature data (Celsius)
  DATA_VECTOR(cotsimm_dat);    // COTS immigration rate data (individuals/m2/year)
  
  // Parameters
  PARAMETER(log_r_cots);       // COTS intrinsic growth rate (year^-1)
  PARAMETER(log_K_cots);       // COTS carrying capacity (individuals/m2)
  PARAMETER(log_r_slow);       // Slow coral intrinsic growth rate (year^-1)
  PARAMETER(log_r_fast);       // Fast coral intrinsic growth rate (year^-1)
  PARAMETER(log_K_coral);      // Combined coral carrying capacity (%)
  PARAMETER(log_alpha_slow);   // COTS attack rate on slow coral (m2/ind/year)
  PARAMETER(log_alpha_fast);   // COTS attack rate on fast coral (m2/ind/year)
  PARAMETER(log_comp_coef);    // Competition coefficient between coral types
  
  // Standard deviations for observation model
  PARAMETER(log_sigma_cots);   // SD for COTS observations
  PARAMETER(log_sigma_slow);   // SD for slow coral observations
  PARAMETER(log_sigma_fast);   // SD for fast coral observations
  
  // Transform parameters
  Type r_cots = exp(log_r_cots);
  Type K_cots = exp(log_K_cots);
  Type r_slow = exp(log_r_slow);
  Type r_fast = exp(log_r_fast);
  Type K_coral = exp(log_K_coral);
  Type alpha_slow = exp(log_alpha_slow);
  Type alpha_fast = exp(log_alpha_fast);
  Type comp_coef = exp(log_comp_coef);
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_slow = exp(log_sigma_slow);
  Type sigma_fast = exp(log_sigma_fast);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Parameter bounds checking
  if(!R_FINITE(asDouble(r_cots)) || !R_FINITE(asDouble(K_cots)) || 
     !R_FINITE(asDouble(r_slow)) || !R_FINITE(asDouble(r_fast)) || 
     !R_FINITE(asDouble(K_coral)) || !R_FINITE(asDouble(alpha_slow)) || 
     !R_FINITE(asDouble(alpha_fast)) || !R_FINITE(asDouble(h_cots)) || 
     !R_FINITE(asDouble(temp_opt)) || !R_FINITE(asDouble(temp_tol)) || 
     !R_FINITE(asDouble(comp_coef)) || !R_FINITE(asDouble(sigma_cots)) || 
     !R_FINITE(asDouble(sigma_slow)) || !R_FINITE(asDouble(sigma_fast))) {
    return Type(INFINITY);
  }
  
  // Vectors for predictions
  vector<Type> cots_pred(Year.size());
  vector<Type> slow_pred(Year.size());
  vector<Type> fast_pred(Year.size());
  vector<Type> cotsimm_pred(Year.size());
  
  // Initial conditions - ensure positive values
  cots_pred(0) = cots_dat(0) > eps ? cots_dat(0) : eps;
  slow_pred(0) = slow_dat(0) > eps ? slow_dat(0) : eps;
  fast_pred(0) = fast_dat(0) > eps ? fast_dat(0) : eps;
  cotsimm_pred(0) = cotsimm_dat(0);
  
  // Process model
  for(int t = 1; t < Year.size(); t++) {
    // Basic space limitation
    Type space_limit = Type(1.0) - (slow_pred(t-1) + fast_pred(t-1)) / K_coral;
    space_limit = space_limit > Type(0.0) ? space_limit : Type(0.0);
    
    // Simple predation rates
    Type f_slow = alpha_slow * slow_pred(t-1);
    Type f_fast = alpha_fast * fast_pred(t-1);
    
    // Basic COTS logistic growth plus immigration
    cots_pred(t) = cots_pred(t-1) + 
                   r_cots * cots_pred(t-1) * (1 - cots_pred(t-1)/K_cots) +
                   cotsimm_dat(t);
    
    // Basic coral dynamics
    slow_pred(t) = slow_pred(t-1) + 
                   r_slow * slow_pred(t-1) * space_limit * (1 - comp_coef * fast_pred(t-1)/K_coral) -
                   f_slow * cots_pred(t-1);
    
    fast_pred(t) = fast_pred(t-1) + 
                   r_fast * fast_pred(t-1) * space_limit * (1 - comp_coef * slow_pred(t-1)/K_coral) -
                   f_fast * cots_pred(t-1);
    
    // Simple immigration prediction
    cotsimm_pred(t) = cotsimm_dat(t);
    
    // Ensure predictions stay positive
    cots_pred(t) = cots_pred(t) < eps ? eps : cots_pred(t);
    slow_pred(t) = slow_pred(t) < eps ? eps : slow_pred(t);
    fast_pred(t) = fast_pred(t) < eps ? eps : fast_pred(t);
  }
  
  // Observation model (lognormal)
  for(int t = 0; t < Year.size(); t++) {
    nll -= dnorm(log(cots_dat(t)), log(cots_pred(t)), sigma_cots, true);
    nll -= dnorm(log(slow_dat(t)), log(slow_pred(t)), sigma_slow, true);
    nll -= dnorm(log(fast_dat(t)), log(fast_pred(t)), sigma_fast, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(cotsimm_pred);
  
  return nll;
}
