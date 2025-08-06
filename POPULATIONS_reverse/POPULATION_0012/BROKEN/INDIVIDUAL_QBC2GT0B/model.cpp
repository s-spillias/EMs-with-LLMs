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
  PARAMETER(f_facilitation);         // Facilitation effect of fast coral on slow coral
  
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
  Type eps = Type(1e-4);
  
  // Model equations
  for(int t = 1; t < Year.size(); t++) {
    // 1. Temperature effect on COTS growth (Gaussian response)
    Type temp_diff = sst_dat(t-1) - temp_opt;
    Type temp_effect = exp(-(temp_diff * temp_diff) / (2.0 * temp_tol * temp_tol));
    
    // 2. Type II functional responses for COTS predation
    Type pred_fast = (a_fast * fast_pred(t-1)) / (1.0 + a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1));
    Type pred_slow = (a_slow * slow_pred(t-1)) / (1.0 + a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1));
    
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
                   r_slow * slow_pred(t-1) * (1.0 - total_cover/K_coral) * (1.0 + f_facilitation * fast_pred(t-1)/(K_coral + total_cover)) -
                   pred_slow * cots_pred(t-1);
    
    // Ensure predictions stay positive
    cots_pred(t) = cots_pred(t) > eps ? cots_pred(t) : eps;
    fast_pred(t) = fast_pred(t) > eps ? fast_pred(t) : eps;
    slow_pred(t) = slow_pred(t) > eps ? slow_pred(t) : eps;
  }
  
  // Calculate negative log-likelihood using lognormal distribution
  Type sigma_cots = Type(0.5);  // Increased SD to handle more variation
  Type sigma_coral = Type(0.5);
  
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
