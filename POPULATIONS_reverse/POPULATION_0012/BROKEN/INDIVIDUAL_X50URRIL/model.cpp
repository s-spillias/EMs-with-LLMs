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
  PARAMETER(beta_prey);              // Prey availability effect on COTS growth
  
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
    
    // 2. Type II functional responses for COTS predation
    Type pred_fast = (a_fast * fast_pred(t-1)) / (1.0 + a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1));
    Type pred_slow = (a_slow * slow_pred(t-1)) / (1.0 + a_fast * h_fast * fast_pred(t-1) + a_slow * h_slow * slow_pred(t-1));
    
    // 3. COTS population dynamics with temperature effect, prey availability, and immigration
    Type total_prey = fast_pred(t-1) + slow_pred(t-1) + Type(1e-4);
    Type prey_ratio = total_prey / (K_coral + Type(1e-4));
    Type prey_effect = Type(0.2) + Type(0.8) * prey_ratio / (prey_ratio + beta_prey + Type(1e-4));
    
    Type dd_term = Type(1.0) - cots_pred(t-1)/(K_cots + Type(1e-4));
    Type growth = temp_effect * prey_effect * r_cots * cots_pred(t-1) * dd_term;
    cots_pred(t) = cots_pred(t-1) + growth + cotsimm_dat(t-1);
    
    // Bound predictions to prevent extreme values
    if(cots_pred(t) < Type(0.0)) cots_pred(t) = Type(0.0);
    if(cots_pred(t) > Type(2.0) * K_cots) cots_pred(t) = Type(2.0) * K_cots;
    
    // 4. Coral dynamics with competition and COTS predation
    Type total_cover = fast_pred(t-1) + slow_pred(t-1);
    
    fast_pred(t) = fast_pred(t-1) + 
                   r_fast * fast_pred(t-1) * (1.0 - total_cover/K_coral) -
                   pred_fast * cots_pred(t-1);
    
    slow_pred(t) = slow_pred(t-1) + 
                   r_slow * slow_pred(t-1) * (1.0 - total_cover/K_coral) -
                   pred_slow * cots_pred(t-1);
    
    // Bound coral predictions
    if(fast_pred(t) < Type(0.0)) fast_pred(t) = Type(0.0);
    if(fast_pred(t) > K_coral) fast_pred(t) = K_coral;
    if(slow_pred(t) < Type(0.0)) slow_pred(t) = Type(0.0);
    if(slow_pred(t) > K_coral) slow_pred(t) = K_coral;
  }
  
  // Calculate negative log-likelihood using lognormal distribution
  Type sigma_cots = Type(0.5);  // Increased SD to handle larger variations
  Type sigma_coral = Type(0.5);
  
  for(int t = 0; t < Year.size(); t++) {
    // Add small constant to both data and predictions to prevent log(0)
    Type safe_cots_dat = cots_dat(t) + Type(1e-4);
    Type safe_cots_pred = cots_pred(t) + Type(1e-4);
    Type safe_fast_dat = fast_dat(t) + Type(1e-4);
    Type safe_fast_pred = fast_pred(t) + Type(1e-4);
    Type safe_slow_dat = slow_dat(t) + Type(1e-4);
    Type safe_slow_pred = slow_pred(t) + Type(1e-4);
    
    // COTS likelihood
    nll -= dnorm(log(safe_cots_dat), log(safe_cots_pred), sigma_cots, true);
    
    // Coral likelihoods
    nll -= dnorm(log(safe_fast_dat), log(safe_fast_pred), sigma_coral, true);
    nll -= dnorm(log(safe_slow_dat), log(safe_slow_pred), sigma_coral, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
