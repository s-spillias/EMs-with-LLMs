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
  PARAMETER(temp_tol);               // Temperature tolerance
  PARAMETER(temp_stress_effect);     // Effect of temperature stress on coral vulnerability
  PARAMETER(a_fast);                 // Attack rate on fast coral  
  PARAMETER(a_slow);                 // Attack rate on slow coral
  PARAMETER(h_fast);                 // Handling time for fast coral
  PARAMETER(h_slow);                 // Handling time for slow coral
  PARAMETER(r_fast);                 // Fast coral growth rate
  PARAMETER(r_slow);                 // Slow coral growth rate
  PARAMETER(K_coral);                // Total coral carrying capacity
  
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
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Bound parameters to prevent numerical issues
  if(r_cots < eps || K_cots < eps || temp_tol < eps || 
     a_fast < eps || a_slow < eps || h_fast < eps || h_slow < eps ||
     r_fast < eps || r_slow < eps || K_coral < eps) {
    return Type(INFINITY);
  }
  
  // Model equations
  for(int t = 1; t < Year.size(); t++) {
    // 1. Temperature effects on COTS growth
    Type temp_effect = exp(-pow(sst_dat(t-1) - temp_opt, 2.0) / (2.0 * pow(temp_tol, 2.0)));
    
    // 2. Type II functional responses for COTS predation
    Type pred_fast = (a_fast * fast_pred(t-1)) / 
                    (1.0 + a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1));
    Type pred_slow = (a_slow * slow_pred(t-1)) / 
                    (1.0 + a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1));
    
    // Apply temperature stress effect to predation rates when above optimal temperature
    if(sst_dat(t-1) > temp_opt) {
      pred_fast *= (Type(1.0) + temp_stress_effect);
      pred_slow *= (Type(1.0) + temp_stress_effect);
    }
    
    // 3. COTS population dynamics with temperature effect and immigration
    cots_pred(t) = cots_pred(t-1) + 
                   temp_effect * r_cots * cots_pred(t-1) * (1.0 - cots_pred(t-1)/K_cots) +
                   cotsimm_dat(t-1);
    
    // 4. Coral dynamics with competition and COTS predation
    Type total_cover = fast_pred(t-1) + slow_pred(t-1);
    
    fast_pred(t) = fast_pred(t-1) + 
                   r_fast * fast_pred(t-1) * (1.0 - total_cover/K_coral) -
                   pred_fast * cots_pred(t-1);
    
    slow_pred(t) = slow_pred(t-1) + 
                   r_slow * slow_pred(t-1) * (1.0 - total_cover/K_coral) -
                   pred_slow * cots_pred(t-1);
    
    // Ensure predictions stay positive and bounded
    cots_pred(t) = CppAD::CondExpLt(cots_pred(t), eps, eps, cots_pred(t));
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), K_cots, K_cots, cots_pred(t));
    
    fast_pred(t) = CppAD::CondExpLt(fast_pred(t), eps, eps, fast_pred(t));
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), K_coral, K_coral, fast_pred(t));
    
    slow_pred(t) = CppAD::CondExpLt(slow_pred(t), eps, eps, slow_pred(t));
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), K_coral, K_coral, slow_pred(t));
  }
  
  // Calculate negative log-likelihood using lognormal distribution
  Type sigma_cots = Type(0.5);  // Increased SD to handle larger variations
  Type sigma_coral = Type(0.5);
  
  for(int t = 0; t < Year.size(); t++) {
    // Add small constant inside log to prevent numerical issues
    Type obs_cots = cots_dat(t) + eps;
    Type pred_cots = cots_pred(t) + eps;
    Type obs_fast = fast_dat(t) + eps;
    Type pred_fast = fast_pred(t) + eps;
    Type obs_slow = slow_dat(t) + eps;
    Type pred_slow = slow_pred(t) + eps;
    
    // COTS likelihood
    nll -= dnorm(log(obs_cots), log(pred_cots), sigma_cots, true);
    
    // Coral likelihoods
    nll -= dnorm(log(obs_fast), log(pred_fast), sigma_coral, true);
    nll -= dnorm(log(obs_slow), log(pred_slow), sigma_coral, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
