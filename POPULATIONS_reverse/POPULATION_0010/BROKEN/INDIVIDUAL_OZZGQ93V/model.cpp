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
  PARAMETER(log_comp_coef);    // Base competition coefficient between coral types
  PARAMETER(log_comp_density); // Density-dependent competition scaling
  
  // Standard deviations for observation model
  PARAMETER(log_sigma_cots);   // SD for COTS observations
  PARAMETER(log_sigma_slow);   // SD for slow coral observations
  PARAMETER(log_sigma_fast);   // SD for fast coral observations
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Transform parameters with bounds
  Type r_cots = exp(log_r_cots) + eps;
  Type K_cots = exp(log_K_cots) + eps;
  Type r_slow = exp(log_r_slow) + eps;
  Type r_fast = exp(log_r_fast) + eps;
  Type K_coral = exp(log_K_coral) + eps;
  Type alpha_slow = exp(log_alpha_slow) + eps;
  Type alpha_fast = exp(log_alpha_fast) + eps;
  Type h_cots = exp(log_h_cots) + eps;
  Type temp_opt = exp(log_temp_opt);
  Type temp_tol = exp(log_temp_tol) + eps;
  Type comp_coef = exp(log_comp_coef) + eps;
  Type comp_density = exp(log_comp_density) + eps;
  Type sigma_cots = exp(log_sigma_cots) + eps;
  Type sigma_slow = exp(log_sigma_slow) + eps;
  Type sigma_fast = exp(log_sigma_fast) + eps;
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
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
    Type temp_diff = (sst_dat(t) - temp_opt) / temp_tol;
    Type temp_effect = exp(-0.5 * pow(temp_diff, 2));
    temp_effect = temp_effect < eps ? eps : (temp_effect > Type(1.0) ? Type(1.0) : temp_effect);
    
    // Total coral cover with bounds
    Type total_coral = slow_pred(t-1) + fast_pred(t-1);
    total_coral = total_coral < eps ? eps : (total_coral > K_coral ? K_coral : total_coral);
    
    // Space limitation factor with bounds
    Type space_limit = (K_coral - total_coral) / K_coral;
    space_limit = space_limit < eps ? eps : (space_limit > Type(1.0) ? Type(1.0) : space_limit);
    
    // Functional responses with protection against division by zero
    Type denom = Type(1.0) + h_cots * (alpha_slow * slow_pred(t-1) + alpha_fast * fast_pred(t-1));
    denom = denom < eps ? eps : denom;
    Type f_slow = alpha_slow * slow_pred(t-1) / denom;
    Type f_fast = alpha_fast * fast_pred(t-1) / denom;
    
    // COTS dynamics with bounded growth term
    Type cots_growth = temp_effect * r_cots * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1)/K_cots);
    cots_growth = cots_growth < -cots_pred(t-1) ? -cots_pred(t-1) : cots_growth;
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t);
    
    // Competition term with dampened effect
    Type coral_ratio = total_coral/K_coral;
    coral_ratio = coral_ratio < eps ? eps : (coral_ratio > Type(1.0) ? Type(1.0) : coral_ratio);
    Type comp_base = Type(0.05) * pow(coral_ratio, comp_density); // Reduced baseline competition
    Type comp_strength = comp_coef * (Type(1.0) + comp_base);
    
    // Coral dynamics with bounded growth terms
    Type slow_growth = r_slow * slow_pred(t-1) * space_limit * (Type(1.0) - comp_strength * fast_pred(t-1)/K_coral);
    slow_growth = slow_growth < -slow_pred(t-1) ? -slow_pred(t-1) : slow_growth;
    slow_pred(t) = slow_pred(t-1) + slow_growth - f_slow * cots_pred(t-1);
    
    Type fast_growth = r_fast * fast_pred(t-1) * space_limit * (Type(1.0) - comp_strength * slow_pred(t-1)/K_coral);
    fast_growth = fast_growth < -fast_pred(t-1) ? -fast_pred(t-1) : fast_growth;
    fast_pred(t) = fast_pred(t-1) + fast_growth - f_fast * cots_pred(t-1);
    
    // Model COTS immigration with bounds
    Type cots_ratio = cots_pred(t-1)/K_cots;
    cots_ratio = cots_ratio < eps ? eps : (cots_ratio > Type(1.0) ? Type(1.0) : cots_ratio);
    cotsimm_pred(t) = temp_effect * cots_ratio;
    
    // Ensure predictions stay positive
    cots_pred(t) = cots_pred(t) < eps ? eps : cots_pred(t);
    slow_pred(t) = slow_pred(t) < eps ? eps : slow_pred(t);
    fast_pred(t) = fast_pred(t) < eps ? eps : fast_pred(t);
  }
  
  // Observation model (lognormal) with protection against log(0)
  for(int t = 0; t < Year.size(); t++) {
    Type obs_cots = cots_dat(t) < eps ? eps : cots_dat(t);
    Type obs_slow = slow_dat(t) < eps ? eps : slow_dat(t);
    Type obs_fast = fast_dat(t) < eps ? eps : fast_dat(t);
    
    Type pred_cots = cots_pred(t) < eps ? eps : cots_pred(t);
    Type pred_slow = slow_pred(t) < eps ? eps : slow_pred(t);
    Type pred_fast = fast_pred(t) < eps ? eps : fast_pred(t);
    
    nll -= dnorm(log(obs_cots), log(pred_cots), sigma_cots, true);
    nll -= dnorm(log(obs_slow), log(pred_slow), sigma_slow, true);
    nll -= dnorm(log(obs_fast), log(pred_fast), sigma_fast, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(cotsimm_pred);
  
  return nll;
}
