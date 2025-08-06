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
  PARAMETER(stress_rate);            // Rate of vulnerability increase with temperature
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
  
  // Small constant to prevent division by zero and bound predictions
  Type eps = Type(1e-4);
  Type max_val = Type(1e3);
  
  // Model equations
  for(int t = 1; t < Year.size(); t++) {
    // 1. Temperature effects
    Type temp_effect = exp(-pow(sst_dat(t-1) - temp_opt, 2.0) / (2.0 * pow(temp_tol, 2.0)));
    
    // Temperature stress increases coral vulnerability (simple linear response with bounds)
    Type temp_stress = Type(1.0) + stress_rate * CppAD::CondExpGt(sst_dat(t-1), temp_opt,
                      sst_dat(t-1) - temp_opt, Type(0.0));
    
    // 2. Type II functional responses for COTS predation with temperature-modified attack rates
    Type pred_fast = (temp_stress * a_fast * fast_pred(t-1)) / 
                    (1.0 + temp_stress * (a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1)));
    Type pred_slow = (temp_stress * a_slow * slow_pred(t-1)) / 
                    (1.0 + temp_stress * (a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1)));
    
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
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), eps,
                    CppAD::CondExpLt(cots_pred(t), max_val, cots_pred(t), max_val),
                    eps);
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), eps,
                    CppAD::CondExpLt(fast_pred(t), max_val, fast_pred(t), max_val),
                    eps);
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), eps,
                    CppAD::CondExpLt(slow_pred(t), max_val, slow_pred(t), max_val),
                    eps);
  }
  
  // Parameter bounds checking
  if(r_cots < Type(0.0) || K_cots < Type(0.0) || a_fast < Type(0.0) || 
     a_slow < Type(0.0) || r_fast < Type(0.0) || r_slow < Type(0.0) || 
     K_coral < Type(0.0) || stress_rate < Type(0.0)) {
    return Type(1e10);  // Return large value if parameters invalid
  }

  // Calculate negative log-likelihood
  Type sigma_cots = Type(0.5);
  Type sigma_coral = Type(0.5);
  
  for(int t = 0; t < Year.size(); t++) {
    // Add small constant inside log to prevent undefined gradients
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), sigma_cots, true);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), sigma_coral, true);
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), sigma_coral, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
