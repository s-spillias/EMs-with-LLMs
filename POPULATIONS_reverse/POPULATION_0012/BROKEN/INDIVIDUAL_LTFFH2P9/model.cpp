#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Year);           // Time vector (years)
  DATA_VECTOR(sst_dat);        // Sea surface temperature (°C)
  DATA_VECTOR(cotsimm_dat);    // COTS immigration rate (individuals/m2/year)
  DATA_VECTOR(cots_dat);       // Observed COTS density (individuals/m2)
  DATA_VECTOR(slow_dat);       // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);       // Observed fast-growing coral cover (%)
  
  // Parameters
  PARAMETER(r_cots);           // COTS population growth rate (year^-1)
  PARAMETER(K_cots);           // COTS carrying capacity (individuals/m2)
  PARAMETER(temp_opt);         // Optimal temperature for COTS survival (°C)
  PARAMETER(temp_tol);         // Temperature tolerance range (°C)
  PARAMETER(alpha_slow);       // Attack rate on slow corals (m2/individual/year)
  PARAMETER(alpha_fast);       // Attack rate on fast corals (m2/individual/year)
  PARAMETER(r_slow);           // Growth rate of slow corals (year^-1)
  PARAMETER(r_fast);           // Growth rate of fast corals (year^-1)
  PARAMETER(K_coral);          // Combined coral carrying capacity (%)
  PARAMETER(obs_sd_cots);      // Observation SD for COTS
  PARAMETER(obs_sd_coral);     // Observation SD for coral cover
  PARAMETER(beta_switch);      // Prey switching coefficient
  
  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Initialize vectors for predictions
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  
  // Set initial conditions
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Small constant to prevent division by zero
  Type eps = Type(1e-8);
  
  // Model predictions
  for(int t = 1; t < n; t++) {
    // 1. Temperature effect on COTS (Gaussian response)
    Type temp_effect = exp(-0.5 * pow((sst_dat(t-1) - temp_opt) / temp_tol, 2));
    
    // 2. Total coral cover (food availability)
    Type total_coral = slow_pred(t-1) + fast_pred(t-1) + eps;
    
    // 3. COTS dynamics with temperature effect and immigration
    Type cots_growth = r_cots * cots_pred(t-1) * (1 - cots_pred(t-1)/K_cots) * temp_effect;
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t-1);
    cots_pred(t) = exp(log(cots_pred(t) + eps)); // Ensure positive values
    
    // 4. Coral predation rates with prey switching
    Type prey_ratio = fast_pred(t-1) / (slow_pred(t-1) + fast_pred(t-1) + eps);
    Type switch_factor = Type(1.0) + beta_switch * (prey_ratio - Type(0.5));
    
    // Bounded attack rates
    Type alpha_slow_adj = alpha_slow * CppAD::CondExpGe(switch_factor, Type(0.1),
                         switch_factor, Type(0.1));
    Type alpha_fast_adj = alpha_fast * CppAD::CondExpLe(switch_factor, Type(2.0),
                         switch_factor, Type(2.0));
    
    // Standard functional response with bounded attack rates
    Type slow_consumed = (alpha_slow_adj * cots_pred(t-1) * slow_pred(t-1)) / 
                        (Type(1.0) + alpha_slow * slow_pred(t-1) + alpha_fast * fast_pred(t-1));
    
    Type fast_consumed = (alpha_fast_adj * cots_pred(t-1) * fast_pred(t-1)) / 
                        (Type(1.0) + alpha_slow * slow_pred(t-1) + alpha_fast * fast_pred(t-1));
    
    // 5. Coral dynamics with logistic growth and predation
    Type available_space = (K_coral - total_coral) / K_coral;
    slow_pred(t) = slow_pred(t-1) + r_slow * slow_pred(t-1) * available_space - slow_consumed;
    fast_pred(t) = fast_pred(t-1) + r_fast * fast_pred(t-1) * available_space - fast_consumed;
    
    // Ensure coral cover stays positive
    slow_pred(t) = exp(log(slow_pred(t) + eps));
    fast_pred(t) = exp(log(fast_pred(t) + eps));
  }
  
  // Observation model using log-normal distribution
  for(int t = 0; t < n; t++) {
    // COTS observations
    nll -= dnorm(log(cots_dat(t) + eps), log(cots_pred(t) + eps), obs_sd_cots, true);
    
    // Coral cover observations
    nll -= dnorm(log(slow_dat(t) + eps), log(slow_pred(t) + eps), obs_sd_coral, true);
    nll -= dnorm(log(fast_dat(t) + eps), log(fast_pred(t) + eps), obs_sd_coral, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
