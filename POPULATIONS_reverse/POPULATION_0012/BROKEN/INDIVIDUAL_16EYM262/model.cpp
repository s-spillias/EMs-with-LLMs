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
  
  // Small constant to prevent division by zero and log(0)
  Type eps = Type(1e-4);
  
  // Model equations
  for(int t = 1; t < Year.size(); t++) {
    // 1. Temperature effects (Gaussian response)
    Type temp_diff = sst_dat(t-1) - temp_opt;
    Type temp_effect = exp(-(temp_diff * temp_diff) / (Type(2.0) * temp_tol * temp_tol));
    
    // 2. Temperature-dependent attack rates 
    Type a_fast_temp = a_fast * temp_effect;
    Type a_slow_temp = a_slow * temp_effect;
    
    // 3. Type II functional responses for COTS predation
    Type total_handle = Type(1.0) + a_fast_temp * h_fast * fast_pred(t-1) + a_slow_temp * h_slow * slow_pred(t-1);
    Type pred_fast = (a_fast_temp * fast_pred(t-1)) / total_handle;
    Type pred_slow = (a_slow_temp * slow_pred(t-1)) / total_handle;
    
    // 3. COTS population dynamics with temperature effect and immigration
    Type cots_growth = temp_effect * r_cots * cots_pred(t-1) * (Type(1.0) - cots_pred(t-1)/K_cots);
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t-1);
    
    // 4. Coral dynamics with competition and COTS predation
    Type total_cover = fast_pred(t-1) + slow_pred(t-1);
    Type competition = Type(1.0) - total_cover/K_coral;
    
    Type fast_growth = r_fast * fast_pred(t-1) * competition;
    Type fast_loss = pred_fast * cots_pred(t-1);
    fast_pred(t) = fast_pred(t-1) + fast_growth - fast_loss;
    
    Type slow_growth = r_slow * slow_pred(t-1) * competition;
    Type slow_loss = pred_slow * cots_pred(t-1);
    slow_pred(t) = slow_pred(t-1) + slow_growth - slow_loss;
    
    // Ensure predictions stay positive
    cots_pred(t) = cots_pred(t) + eps;
    fast_pred(t) = fast_pred(t) + eps;
    slow_pred(t) = slow_pred(t) + eps;
  }
  
  // Calculate negative log-likelihood
  Type sigma_cots = Type(1.0);  // Larger SD for overdispersed count data
  Type sigma_coral = Type(1.0); 
  
  for(int t = 0; t < Year.size(); t++) {
    // COTS likelihood (gamma distribution for positive continuous data)
    nll -= dgamma(cots_dat(t) + eps, Type(2.0), cots_pred(t)/(Type(2.0)), true);
    
    // Coral likelihoods (beta distribution for proportions)
    nll -= dbeta(fast_dat(t)/Type(100.0) + eps, Type(2.0) * fast_pred(t)/Type(100.0), Type(2.0), true);
    nll -= dbeta(slow_dat(t)/Type(100.0) + eps, Type(2.0) * slow_pred(t)/Type(100.0), Type(2.0), true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(fast_pred);
  REPORT(slow_pred);
  
  return nll;
}
