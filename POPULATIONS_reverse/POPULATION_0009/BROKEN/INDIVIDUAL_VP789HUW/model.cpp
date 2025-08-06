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
  PARAMETER(log_alpha_slow);   // Log feeding rate on slow-growing coral (m²/individual/year)
  PARAMETER(log_alpha_fast);   // Log feeding rate on fast-growing coral (m²/individual/year)
  PARAMETER(log_r_slow);      // Log slow-growing coral recovery rate (year⁻¹)
  PARAMETER(log_r_fast);      // Log fast-growing coral recovery rate (year⁻¹)
  PARAMETER(temp_opt);        // Optimal temperature for COTS (°C)
  PARAMETER(temp_width);      // Temperature tolerance width (°C)
  
  // Transform parameters
  Type r_cots = exp(log_r_cots);
  Type K_cots = exp(log_K_cots);
  Type alpha_slow = exp(log_alpha_slow);
  Type alpha_fast = exp(log_alpha_fast);
  Type r_slow = exp(log_r_slow);
  Type r_fast = exp(log_r_fast);
  
  // Initialize negative log-likelihood
  Type nll = Type(0.0);
  const Type eps = Type(1e-8);
  const Type inf = Type(1e10);
  
  vector<Type> cots_pred(cots_dat.size());
  vector<Type> slow_pred(slow_dat.size());
  vector<Type> fast_pred(fast_dat.size());
  
  // Initial conditions with bounds checking
  cots_pred(0) = CppAD::CondExpGe(cots_dat(0), eps, cots_dat(0), eps);
  slow_pred(0) = CppAD::CondExpGe(slow_dat(0), eps, slow_dat(0), eps);
  fast_pred(0) = CppAD::CondExpGe(fast_dat(0), eps, fast_dat(0), eps);
  
  // Process model
  for(int t = 1; t < cots_dat.size(); t++) {
    // 1. Temperature effect on COTS (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t) - temp_opt) / temp_width, 2));
    
    // 2. COTS population dynamics
    Type cots_growth = r_cots * cots_pred(t-1) * (1 - cots_pred(t-1) / K_cots) * temp_effect;
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t);
    
    // 3. Coral dynamics with smooth transitions
    Type slow_consumed = alpha_slow * cots_pred(t-1) * slow_pred(t-1) / (slow_pred(t-1) + eps);
    Type fast_consumed = alpha_fast * cots_pred(t-1) * fast_pred(t-1) / (fast_pred(t-1) + eps);
    
    slow_pred(t) = slow_pred(t-1) + r_slow * slow_pred(t-1) * (1 - slow_pred(t-1)/100) - slow_consumed;
    fast_pred(t) = fast_pred(t-1) + r_fast * fast_pred(t-1) * (1 - fast_pred(t-1)/100) - fast_consumed;
    
    // Ensure predictions stay within reasonable bounds using CppAD conditional expressions
    slow_pred(t) = CppAD::CondExpGe(slow_pred(t), Type(0), 
                    CppAD::CondExpLe(slow_pred(t), Type(100), slow_pred(t), Type(100)),
                    Type(0));
    fast_pred(t) = CppAD::CondExpGe(fast_pred(t), Type(0),
                    CppAD::CondExpLe(fast_pred(t), Type(100), fast_pred(t), Type(100)),
                    Type(0));
    cots_pred(t) = CppAD::CondExpGe(cots_pred(t), Type(0), cots_pred(t), Type(0));
  }
  
  // Observation model using lognormal distribution with robust error handling
  Type cv_cots = Type(0.2);   // Minimum CV for COTS observations
  Type cv_coral = Type(0.1);  // Minimum CV for coral observations
  
  for(int t = 0; t < cots_dat.size(); t++) {
    // Only include non-zero observations in likelihood
    if(cots_dat(t) > eps && cots_pred(t) > eps) {
      Type sd_cots = sqrt(log(1.0 + pow(cv_cots, 2)));
      nll -= dnorm(log(cots_dat(t)), log(cots_pred(t)), sd_cots, true);
    }
    
    if(slow_dat(t) > eps && slow_pred(t) > eps) {
      Type sd_coral = sqrt(log(1.0 + pow(cv_coral, 2)));
      nll -= dnorm(log(slow_dat(t)), log(slow_pred(t)), sd_coral, true);
    }
    
    if(fast_dat(t) > eps && fast_pred(t) > eps) {
      Type sd_coral = sqrt(log(1.0 + pow(cv_coral, 2)));
      nll -= dnorm(log(fast_dat(t)), log(fast_pred(t)), sd_coral, true);
    }
  }
  
  // Add penalty for invalid predictions
  Type penalty = Type(0.0);
  for(int t = 0; t < cots_dat.size(); t++) {
    if(cots_pred(t) < eps) penalty += Type(1e10);
    if(slow_pred(t) < eps) penalty += Type(1e10);
    if(fast_pred(t) < eps) penalty += Type(1e10);
  }
  
  // Combine likelihood and penalty
  Type objective = nll + penalty;
  
  // Report predictions and parameters
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(temp_opt);
  REPORT(temp_width);
  REPORT(r_cots);
  REPORT(K_cots);
  REPORT(alpha_slow);
  REPORT(alpha_fast);
  REPORT(r_slow);
  REPORT(r_fast);
  REPORT(objective);
  
  return objective;
}
