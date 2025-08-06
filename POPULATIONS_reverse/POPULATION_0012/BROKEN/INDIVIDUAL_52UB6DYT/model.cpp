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
  PARAMETER(a_fast);                 // Attack rate on fast coral
  PARAMETER(a_slow);                 // Attack rate on slow coral
  PARAMETER(h_fast);                 // Handling time for fast coral
  PARAMETER(h_slow);                 // Handling time for slow coral
  PARAMETER(r_fast);                 // Fast coral growth rate
  PARAMETER(r_slow);                 // Slow coral growth rate
  PARAMETER(K_coral);                // Total coral carrying capacity
  PARAMETER(d_coral);                // Density-dependent predation coefficient
  
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
  
  // Model equations
  for(int t = 1; t < Year.size(); t++) {
    // 1. Temperature effect on COTS growth (Gaussian response)
    Type temp_effect = exp(-pow(sst_dat(t-1) - temp_opt, 2.0) / (2.0 * pow(temp_tol, 2.0)));
    
    // 2. Type II functional responses with linear density dependence
    Type total_cover = fast_pred(t-1) + slow_pred(t-1);
    Type density_effect = Type(1.0) + d_coral * (total_cover/K_coral);
    
    // Calculate predation terms with safeguards
    Type pred_fast = (a_fast * density_effect * fast_pred(t-1)) / 
                    (Type(1.0) + a_fast * h_fast * fast_pred(t-1) + eps);
    Type pred_slow = (a_slow * density_effect * slow_pred(t-1)) / 
                    (Type(1.0) + a_slow * h_slow * slow_pred(t-1) + eps);
    
    // 3. COTS population dynamics with temperature effect and immigration
    cots_pred(t) = cots_pred(t-1) + 
                   temp_effect * r_cots * cots_pred(t-1) * (1.0 - cots_pred(t-1)/K_cots) +
                   cotsimm_dat(t-1);
    
    // 4. Coral dynamics with competition and COTS predation
    fast_pred(t) = fast_pred(t-1) + 
                   r_fast * fast_pred(t-1) * (1.0 - total_cover/K_coral) -
                   pred_fast * cots_pred(t-1);
    
    slow_pred(t) = slow_pred(t-1) + 
                   r_slow * slow_pred(t-1) * (1.0 - total_cover/K_coral) -
                   pred_slow * cots_pred(t-1);
    
    // Bound predictions to be positive and below carrying capacity
    cots_pred(t) = CppAD::CondExpGt(cots_pred(t), Type(0.0),
                    CppAD::CondExpLt(cots_pred(t), K_cots, cots_pred(t), K_cots),
                    Type(0.0));
    fast_pred(t) = CppAD::CondExpGt(fast_pred(t), Type(0.0),
                    CppAD::CondExpLt(fast_pred(t), K_coral, fast_pred(t), K_coral),
                    Type(0.0));
    slow_pred(t) = CppAD::CondExpGt(slow_pred(t), Type(0.0),
                    CppAD::CondExpLt(slow_pred(t), K_coral, slow_pred(t), K_coral),
                    Type(0.0));
  }
  
  // Calculate negative log-likelihood with robust error handling
  Type sigma_cots = Type(0.2);
  Type sigma_coral = Type(0.2);
  
  for(int t = 0; t < Year.size(); t++) {
    if(cots_dat(t) > eps && cots_pred(t) > eps) {
      nll -= dnorm(log(cots_dat(t)), log(cots_pred(t)), sigma_cots, true);
    }
    if(fast_dat(t) > eps && fast_pred(t) > eps) {
      nll -= dnorm(log(fast_dat(t)), log(fast_pred(t)), sigma_coral, true);
    }
    if(slow_dat(t) > eps && slow_pred(t) > eps) {
      nll -= dnorm(log(slow_dat(t)), log(slow_pred(t)), sigma_coral, true);
    }
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
