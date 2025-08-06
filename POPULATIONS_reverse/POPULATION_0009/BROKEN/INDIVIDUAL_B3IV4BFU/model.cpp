#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(cots_dat);      // Observed COTS density (individuals/m²)
  DATA_VECTOR(slow_dat);      // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);      // Observed fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);       // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);   // COTS immigration rate (individuals/m²/year)
  
  // Parameters
  PARAMETER(log_r);           // Log of COTS population growth rate
  PARAMETER(log_K);           // Log of COTS carrying capacity
  PARAMETER(log_alpha_slow);  // Log of COTS feeding rate on slow corals
  PARAMETER(log_alpha_fast);  // Log of COTS feeding rate on fast corals
  PARAMETER(log_g_slow);      // Log of slow coral growth rate
  PARAMETER(log_g_fast);      // Log of fast coral growth rate
  PARAMETER(temp_opt);        // Optimal temperature for COTS
  PARAMETER(temp_range);      // Temperature tolerance range
  PARAMETER(log_sigma_cots);  // Log of observation error SD for COTS
  PARAMETER(log_sigma_coral); // Log of observation error SD for corals
  
  // Transform parameters
  Type r = exp(log_r);
  Type K = exp(log_K);
  Type alpha_slow = exp(log_alpha_slow);
  Type alpha_fast = exp(log_alpha_fast);
  Type g_slow = exp(log_g_slow);
  Type g_fast = exp(log_g_fast);
  Type sigma_cots = exp(log_sigma_cots);
  Type sigma_coral = exp(log_sigma_coral);
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Initialize vectors for predictions
  int n = cots_dat.size();
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  
  // Constants to prevent numerical issues
  Type eps = Type(1e-8);
  Type max_pred = Type(1000.0);
  
  // Initialize predictions
  for(int i = 0; i < n; i++) {
    cots_pred(i) = Type(0.0);
    slow_pred(i) = Type(0.0);
    fast_pred(i) = Type(0.0);
  }
  
  // Set initial conditions
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Model equations
  for(int i = 0; i < (n-1); i++) {
    // 1. Temperature effect on COTS (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(i) - temp_opt)/temp_range, 2));
    
    // 2. COTS population dynamics using previous predictions
    Type cots_growth = r * cots_pred(i) * (1 - cots_pred(i)/K) * temp_effect;
    cots_pred(i+1) = cots_pred(i) + cots_growth + cotsimm_dat(i);
    
    // 3. Coral dynamics with smooth bounded growth using predictions
    Type slow_consumed = alpha_slow * cots_pred(i) * slow_pred(i)/(slow_pred(i) + eps);
    Type fast_consumed = alpha_fast * cots_pred(i) * fast_pred(i)/(fast_pred(i) + eps);
    
    slow_pred(i+1) = slow_pred(i) + g_slow * slow_pred(i) * (1 - slow_pred(i)/100) - slow_consumed;
    fast_pred(i+1) = fast_pred(i) + g_fast * fast_pred(i) * (1 - fast_pred(i)/100) - fast_consumed;
    
    // Ensure predictions stay within biological bounds
    cots_pred(i+1) = CppAD::CondExpLt(cots_pred(i+1), Type(0.0), 
                                     Type(0.0), 
                                     CppAD::CondExpGt(cots_pred(i+1), max_pred,
                                                    max_pred, 
                                                    cots_pred(i+1)));
                                                    
    slow_pred(i+1) = CppAD::CondExpLt(slow_pred(i+1), Type(0.0),
                                     Type(0.0),
                                     CppAD::CondExpGt(slow_pred(i+1), Type(100.0),
                                                    Type(100.0),
                                                    slow_pred(i+1)));
                                                    
    fast_pred(i+1) = CppAD::CondExpLt(fast_pred(i+1), Type(0.0),
                                     Type(0.0),
                                     CppAD::CondExpGt(fast_pred(i+1), Type(100.0),
                                                    Type(100.0),
                                                    fast_pred(i+1)));
  }
  
  // Calculate likelihood using lognormal distribution
  Type sigma_min = Type(0.01);  // Minimum standard deviation
  
  // Calculate effective standard deviations with minimum bounds
  Type sigma_cots_eff = CppAD::CondExpLt(sigma_cots, sigma_min, sigma_min, sigma_cots);
  Type sigma_coral_eff = CppAD::CondExpLt(sigma_coral, sigma_min, sigma_min, sigma_coral);
  
  for(int i = 0; i < n; i++) {
    // Ensure positive values for log transform
    Type cots_obs = CppAD::CondExpLt(cots_dat(i), eps, eps, cots_dat(i));
    Type cots_mod = CppAD::CondExpLt(cots_pred(i), eps, eps, cots_pred(i));
    Type slow_obs = CppAD::CondExpLt(slow_dat(i), eps, eps, slow_dat(i));
    Type slow_mod = CppAD::CondExpLt(slow_pred(i), eps, eps, slow_pred(i));
    Type fast_obs = CppAD::CondExpLt(fast_dat(i), eps, eps, fast_dat(i));
    Type fast_mod = CppAD::CondExpLt(fast_pred(i), eps, eps, fast_pred(i));
    
    // Add to negative log-likelihood
    nll -= dnorm(log(cots_obs), log(cots_mod), sigma_cots_eff, true);
    nll -= dnorm(log(slow_obs), log(slow_mod), sigma_coral_eff, true);
    nll -= dnorm(log(fast_obs), log(fast_mod), sigma_coral_eff, true);
  }
  
  // Report predictions and objective value
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  REPORT(r);
  REPORT(K);
  REPORT(alpha_slow);
  REPORT(alpha_fast);
  REPORT(g_slow);
  REPORT(g_fast);
  REPORT(temp_opt);
  REPORT(temp_range);
  REPORT(nll);
  ADREPORT(nll);
  
  return nll;
}
