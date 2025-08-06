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
  
  // Parameters - Phase 1
  PARAMETER(log_r_cots);       // COTS intrinsic growth rate (year^-1)
  PARAMETER(log_r_slow);       // Slow coral intrinsic growth rate (year^-1)
  PARAMETER(log_r_fast);       // Fast coral intrinsic growth rate (year^-1)
  PARAMETER(log_alpha_slow);   // COTS attack rate on slow coral (m2/ind/year)
  PARAMETER(log_alpha_fast);   // COTS attack rate on fast coral (m2/ind/year)
  PARAMETER(log_feed_temp_opt);// Optimal temperature for COTS feeding (Celsius)
  PARAMETER(log_feed_temp_tol);// Temperature tolerance for feeding (Celsius)
  
  // Parameters - Phase 2
  PARAMETER(log_K_cots);       // COTS carrying capacity (individuals/m2)
  PARAMETER(log_K_coral);      // Combined coral carrying capacity (%)
  PARAMETER(log_h_cots);       // COTS handling time (year)
  PARAMETER(log_temp_opt);     // Optimal temperature for COTS (Celsius)
  PARAMETER(log_temp_tol);     // Temperature tolerance range (Celsius)
  PARAMETER(log_comp_coef);    // Competition coefficient between coral types
  
  // Parameters - Phase 3 (observation error)
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
  Type feed_temp_opt = exp(log_feed_temp_opt);
  Type feed_temp_tol = exp(log_feed_temp_tol);
  Type comp_coef = exp(log_comp_coef);
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_slow = exp(log_sigma_slow);
  Type sigma_fast = exp(log_sigma_fast);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Vectors for predictions
  vector<Type> cots_pred(Year.size());
  vector<Type> slow_pred(Year.size());
  vector<Type> fast_pred(Year.size());
  vector<Type> cotsimm_pred(Year.size());
  
  // Initial conditions
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  cotsimm_pred(0) = cotsimm_dat(0);
  
  // Process model
  for(int t = 1; t < Year.size(); t++) {
    // Temperature scaling function (Gaussian) with bounds
    Type temp_effect = exp(-0.5 * pow((sst_dat(t) - temp_opt) / (temp_tol + eps), 2));
    temp_effect = CppAD::CondExpGe(temp_effect, eps, temp_effect, eps);
    
    // Total coral cover with bounds
    Type total_coral = slow_pred(t-1) + fast_pred(t-1);
    total_coral = CppAD::CondExpLe(total_coral, K_coral, total_coral, K_coral);
    
    // Space limitation factor with bounds
    Type space_limit = (K_coral - total_coral) / K_coral;
    space_limit = CppAD::CondExpGe(space_limit, Type(0), space_limit, Type(0));
    
    // Temperature-dependent feeding efficiency (simplified)
    Type temp_diff = sst_dat(t) - feed_temp_opt;
    Type feed_efficiency = Type(1.0) / (Type(1.0) + pow(temp_diff/feed_temp_tol, 2));
    
    // Functional responses
    Type total_prey = slow_pred(t-1) + fast_pred(t-1);
    Type f_slow = feed_efficiency * alpha_slow * slow_pred(t-1) / (Type(1.0) + h_cots * total_prey);
    Type f_fast = feed_efficiency * alpha_fast * fast_pred(t-1) / (Type(1.0) + h_cots * total_prey);
    
    // COTS dynamics with bounded growth term
    Type growth_term = temp_effect * r_cots * cots_pred(t-1) * (1 - cots_pred(t-1)/K_cots);
    growth_term = CppAD::CondExpGe(growth_term, -cots_pred(t-1), growth_term, -cots_pred(t-1));
    cots_pred(t) = cots_pred(t-1) + growth_term + cotsimm_dat(t);
    
    // Coral dynamics with competition and bounded growth
    Type slow_growth = r_slow * slow_pred(t-1) * space_limit * (1 - comp_coef * fast_pred(t-1)/K_coral);
    Type fast_growth = r_fast * fast_pred(t-1) * space_limit * (1 - comp_coef * slow_pred(t-1)/K_coral);
    
    slow_growth = CppAD::CondExpGe(slow_growth, -slow_pred(t-1), slow_growth, -slow_pred(t-1));
    fast_growth = CppAD::CondExpGe(fast_growth, -fast_pred(t-1), fast_growth, -fast_pred(t-1));
    
    slow_pred(t) = slow_pred(t-1) + slow_growth - f_slow * cots_pred(t-1);
    fast_pred(t) = fast_pred(t-1) + fast_growth - f_fast * cots_pred(t-1);
    
    // Model COTS immigration as a function of temperature and previous COTS density
    cotsimm_pred(t) = temp_effect * (cots_pred(t-1)/K_cots) * Type(2.0);
    
    // Ensure predictions stay positive using smooth transitions
    cots_pred(t) = CppAD::CondExpGe(cots_pred(t), eps, cots_pred(t), eps);
    slow_pred(t) = CppAD::CondExpGe(slow_pred(t), eps, slow_pred(t), eps);
    fast_pred(t) = CppAD::CondExpGe(fast_pred(t), eps, fast_pred(t), eps);
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
