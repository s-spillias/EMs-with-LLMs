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
  PARAMETER(log_h_cots);       // COTS handling time (year)
  PARAMETER(log_temp_opt);     // Optimal temperature for COTS (Celsius)
  PARAMETER(log_temp_tol);     // Temperature tolerance range (Celsius)
  PARAMETER(log_comp_coef);    // Competition coefficient between coral types
  PARAMETER(log_stress_coef);  // Coefficient for stress-induced growth reduction
  
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
  Type h_cots = exp(log_h_cots);
  Type temp_opt = exp(log_temp_opt);
  Type temp_tol = exp(log_temp_tol);
  Type comp_coef = exp(log_comp_coef);
  Type stress_coef = exp(log_stress_coef);
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_slow = exp(log_sigma_slow);
  Type sigma_fast = exp(log_sigma_fast);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Initialize parameters with reasonable bounds
  Type r_cots_bounded = exp(log_r_cots);
  Type K_cots_bounded = exp(log_K_cots); 
  Type r_slow_bounded = exp(log_r_slow);
  Type r_fast_bounded = exp(log_r_fast);
  Type K_coral_bounded = exp(log_K_coral);
  Type alpha_slow_bounded = exp(log_alpha_slow);
  Type alpha_fast_bounded = exp(log_alpha_fast);
  Type h_cots_bounded = exp(log_h_cots);
  Type temp_opt_bounded = exp(log_temp_opt);
  Type temp_tol_bounded = exp(log_temp_tol);
  Type comp_coef_bounded = exp(log_comp_coef);
  Type stress_coef_bounded = exp(log_stress_coef);
  
  // Vectors for predictions
  vector<Type> cots_pred(Year.size());
  vector<Type> slow_pred(Year.size());
  vector<Type> fast_pred(Year.size());
  vector<Type> cotsimm_pred(Year.size());
  
  // Initial conditions with bounds
  cots_pred(0) = cots_dat(0) < eps ? eps : cots_dat(0);
  slow_pred(0) = slow_dat(0) < eps ? eps : slow_dat(0);
  fast_pred(0) = fast_dat(0) < eps ? eps : fast_dat(0);
  cotsimm_pred(0) = cotsimm_dat(0) < eps ? eps : cotsimm_dat(0);
  
  // Process model
  for(int t = 1; t < Year.size(); t++) {
    // Ensure positive values for state variables
    cots_pred(t-1) = cots_pred(t-1) < eps ? eps : cots_pred(t-1);
    slow_pred(t-1) = slow_pred(t-1) < eps ? eps : slow_pred(t-1);
    fast_pred(t-1) = fast_pred(t-1) < eps ? eps : fast_pred(t-1);
    
    // Temperature scaling function (Gaussian)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t) - temp_opt) / temp_tol, 2));
    
    // Total coral cover (with minimum bound)
    Type total_coral = slow_pred(t-1) + fast_pred(t-1);
    total_coral = total_coral < eps ? eps : total_coral;
    
    // Space limitation factor (bounded between 0 and 1)
    Type space_limit = (K_coral - total_coral) / K_coral;
    space_limit = space_limit < 0 ? 0 : (space_limit > 1 ? 1 : space_limit);
    
    // Functional responses (with protection against division by zero)
    Type denominator = 1 + h_cots * (alpha_slow * slow_pred(t-1) + alpha_fast * fast_pred(t-1));
    denominator = denominator < eps ? eps : denominator;
    Type f_slow = alpha_slow * slow_pred(t-1) / denominator;
    Type f_fast = alpha_fast * fast_pred(t-1) / denominator;
    
    // COTS dynamics with bounded parameters
    cots_pred(t) = cots_pred(t-1) + 
                   temp_effect * r_cots_bounded * cots_pred(t-1) * (1 - cots_pred(t-1)/K_cots_bounded) +
                   cotsimm_dat(t);
    
    // Calculate stress factor from COTS predation pressure (bounded between 0 and 1)
    Type rel_cots = cots_pred(t-1)/K_cots_bounded;
    rel_cots = rel_cots < eps ? eps : rel_cots;
    Type stress_factor = 1 / (1 + stress_coef_bounded * rel_cots);
    stress_factor = stress_factor < eps ? eps : (stress_factor > 1 ? 1 : stress_factor);
    
    // Competition terms (bounded)
    Type comp_slow = comp_coef_bounded * fast_pred(t-1)/K_coral_bounded;
    Type comp_fast = comp_coef_bounded * slow_pred(t-1)/K_coral_bounded;
    comp_slow = comp_slow < 0 ? 0 : (comp_slow > 1 ? 1 : comp_slow);
    comp_fast = comp_fast < 0 ? 0 : (comp_fast > 1 ? 1 : comp_fast);
    
    // Coral dynamics with competition and stress using bounded parameters
    slow_pred(t) = slow_pred(t-1) + 
                   r_slow_bounded * slow_pred(t-1) * space_limit * stress_factor * (1 - comp_slow) -
                   alpha_slow_bounded * slow_pred(t-1) / (1 + h_cots_bounded * (alpha_slow_bounded * slow_pred(t-1) + alpha_fast_bounded * fast_pred(t-1))) * cots_pred(t-1);
    
    fast_pred(t) = fast_pred(t-1) + 
                   r_fast_bounded * fast_pred(t-1) * space_limit * stress_factor * (1 - comp_fast) -
                   alpha_fast_bounded * fast_pred(t-1) / (1 + h_cots_bounded * (alpha_slow_bounded * slow_pred(t-1) + alpha_fast_bounded * fast_pred(t-1))) * cots_pred(t-1);
    
    // Model COTS immigration as a function of temperature and previous COTS density
    cotsimm_pred(t) = temp_effect * (cots_pred(t-1)/K_cots) * Type(2.0);
    
    // Ensure predictions stay positive
    cots_pred(t) = cots_pred(t) < eps ? eps : cots_pred(t);
    slow_pred(t) = slow_pred(t) < eps ? eps : slow_pred(t);
    fast_pred(t) = fast_pred(t) < eps ? eps : fast_pred(t);
  }
  
  // Observation model (lognormal with protection against log(0))
  for(int t = 0; t < Year.size(); t++) {
    // Ensure positive values for data and predictions
    Type cots_obs = cots_dat(t) < eps ? eps : cots_dat(t);
    Type slow_obs = slow_dat(t) < eps ? eps : slow_dat(t);
    Type fast_obs = fast_dat(t) < eps ? eps : fast_dat(t);
    
    Type cots_model = cots_pred(t) < eps ? eps : cots_pred(t);
    Type slow_model = slow_pred(t) < eps ? eps : slow_pred(t);
    Type fast_model = fast_pred(t) < eps ? eps : fast_pred(t);
    
    // Calculate negative log-likelihood
    nll -= dnorm(log(cots_obs), log(cots_model), sigma_cots, true);
    nll -= dnorm(log(slow_obs), log(slow_model), sigma_slow, true);
    nll -= dnorm(log(fast_obs), log(fast_model), sigma_fast, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(cotsimm_pred);
  
  return nll;
}
