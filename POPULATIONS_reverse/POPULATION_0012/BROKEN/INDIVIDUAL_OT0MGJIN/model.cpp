#include <TMB.hpp>

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Year);                  // Time vector
  DATA_VECTOR(cots_dat);             // COTS abundance (individuals/m2)
  DATA_VECTOR(slow_dat);             // Slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);             // Fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);              // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);          // COTS immigration rate (individuals/m2/year)

  // Parameters
  PARAMETER(r_cots);                 // COTS intrinsic growth rate
  PARAMETER(K_cots);                 // COTS carrying capacity
  PARAMETER(temp_opt);               // Optimal temperature for COTS
  PARAMETER(temp_tol);               // Temperature tolerance for COTS
  PARAMETER(temp_opt_fast);          // Optimal temperature for fast-growing coral
  PARAMETER(temp_tol_fast);          // Temperature tolerance for fast-growing coral
  PARAMETER(temp_opt_slow);          // Optimal temperature for slow-growing coral
  PARAMETER(temp_tol_slow);          // Temperature tolerance for slow-growing coral
  PARAMETER(a_fast);                 // Attack rate on fast coral
  PARAMETER(a_slow);                 // Attack rate on slow coral
  PARAMETER(h_fast);                 // Handling time for fast coral
  PARAMETER(h_slow);                 // Handling time for slow coral
  PARAMETER(r_fast);                 // Fast coral growth rate
  PARAMETER(r_slow);                 // Slow coral growth rate
  PARAMETER(K_coral);                // Total coral carrying capacity
  
  // Small constant to prevent numerical issues
  Type eps = Type(1e-8);
  
  // Transform parameters to ensure positivity
  Type r_cots_pos = exp(r_cots);
  Type K_cots_pos = exp(K_cots);
  Type temp_tol_pos = exp(temp_tol);
  Type temp_tol_fast_pos = exp(temp_tol_fast);
  Type temp_tol_slow_pos = exp(temp_tol_slow);
  Type a_fast_pos = exp(a_fast);
  Type a_slow_pos = exp(a_slow);
  Type h_fast_pos = exp(h_fast);
  Type h_slow_pos = exp(h_slow);
  Type r_fast_pos = exp(r_fast);
  Type r_slow_pos = exp(r_slow);
  Type K_coral_pos = exp(K_coral);

  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Initialize vectors for predictions
  vector<Type> cots_pred(Year.size());
  vector<Type> slow_pred(Year.size());
  vector<Type> fast_pred(Year.size());
  
  // Set initial conditions
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Model equations
  for(int t = 1; t < Year.size(); t++) {
    // 1. Temperature effects (simplified Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t-1) - temp_opt) / temp_tol_pos, 2.0));
    Type temp_effect_fast = exp(-0.5 * pow((sst_dat(t-1) - temp_opt_fast) / temp_tol_fast_pos, 2.0));
    Type temp_effect_slow = exp(-0.5 * pow((sst_dat(t-1) - temp_opt_slow) / temp_tol_slow_pos, 2.0));
    
    // 2. Type II functional responses for COTS predation
    Type total_handling = 1.0 + a_fast_pos * h_fast_pos * fast_pred(t-1) + a_slow_pos * h_slow_pos * slow_pred(t-1);
    Type pred_fast = (a_fast_pos * fast_pred(t-1)) / total_handling;
    Type pred_slow = (a_slow_pos * slow_pred(t-1)) / total_handling;
    
    // 4. COTS population dynamics with temperature effect and immigration
    cots_pred(t) = cots_pred(t-1) + 
                   temp_effect * r_cots_pos * cots_pred(t-1) * (1.0 - cots_pred(t-1)/K_cots_pos) +
                   cotsimm_dat(t-1);
    
    // 5. Coral dynamics with competition, temperature effects, and COTS predation
    Type total_cover = fast_pred(t-1) + slow_pred(t-1);
    Type competition = CppAD::CondExpGe(total_cover/K_coral_pos, Type(1), Type(1), total_cover/K_coral_pos);
    
    fast_pred(t) = fast_pred(t-1) + 
                   temp_effect_fast * r_fast_pos * fast_pred(t-1) * (1.0 - competition) -
                   pred_fast * cots_pred(t-1);
    
    slow_pred(t) = slow_pred(t-1) + 
                   temp_effect_slow * r_slow_pos * slow_pred(t-1) * (1.0 - competition) -
                   pred_slow * cots_pred(t-1);
    
    // Ensure predictions stay positive and within bounds
    cots_pred(t) = CppAD::CondExpGe(cots_pred(t), Type(0), cots_pred(t), Type(0));
    fast_pred(t) = CppAD::CondExpGe(fast_pred(t), Type(0), fast_pred(t), Type(0));
    slow_pred(t) = CppAD::CondExpGe(slow_pred(t), Type(0), slow_pred(t), Type(0));
  }
  
  // Calculate negative log-likelihood
  Type sigma = Type(0.3);
  
  for(int t = 0; t < Year.size(); t++) {
    // Add small constant to prevent log(0)
    Type obs_cots = cots_dat(t) + eps;
    Type obs_fast = fast_dat(t) + eps;
    Type obs_slow = slow_dat(t) + eps;
    
    Type pred_cots = cots_pred(t) + eps;
    Type pred_fast = fast_pred(t) + eps;
    Type pred_slow = slow_pred(t) + eps;
    
    // Simple log-normal likelihood
    nll -= dnorm(log(obs_cots), log(pred_cots), sigma, true);
    nll -= dnorm(log(obs_fast), log(pred_fast), sigma, true);
    nll -= dnorm(log(obs_slow), log(pred_slow), sigma, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
