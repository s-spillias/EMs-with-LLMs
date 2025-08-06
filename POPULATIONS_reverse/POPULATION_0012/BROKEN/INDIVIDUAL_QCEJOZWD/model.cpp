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
  PARAMETER(pred_threshold);    // Coral cover threshold affecting COTS predation efficiency (%)
  
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
  Type eps = Type(1e-4);
  
  // Model predictions
  for(int t = 1; t < n; t++) {
    // 1. Temperature effect on COTS (simplified response)
    Type temp_diff = (sst_dat(t-1) - temp_opt) / temp_tol;
    Type temp_effect = Type(1.0) / (Type(1.0) + temp_diff * temp_diff);
    
    // 2. Total coral cover (food availability)
    Type total_coral = slow_pred(t-1) + fast_pred(t-1);
    total_coral = total_coral + eps;
    
    // 3. COTS dynamics with bounded growth
    Type rel_density = cots_pred(t-1) / K_cots;
    Type cots_growth = r_cots * cots_pred(t-1) * (Type(1.0) - rel_density) * temp_effect;
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t-1);
    cots_pred(t) = cots_pred(t) > eps ? cots_pred(t) : eps;
    
    // 4. Coral predation rates
    Type pred_ratio = total_coral / pred_threshold;
    Type pred_efficiency = pred_ratio / (Type(1.0) + pred_ratio);
    
    // Calculate bounded consumption rates
    Type max_slow = slow_pred(t-1) * Type(0.9); // Max 90% consumption
    Type max_fast = fast_pred(t-1) * Type(0.9);
    
    Type slow_consumed = pred_efficiency * alpha_slow * cots_pred(t-1) * slow_pred(t-1);
    Type fast_consumed = pred_efficiency * alpha_fast * cots_pred(t-1) * fast_pred(t-1);
    
    slow_consumed = slow_consumed < max_slow ? slow_consumed : max_slow;
    fast_consumed = fast_consumed < max_fast ? fast_consumed : max_fast;
    
    // 5. Coral dynamics with logistic growth and predation
    Type available_space = (K_coral - total_coral) / (K_coral + eps);
    available_space = available_space > Type(0) ? available_space : Type(0); // Ensure non-negative
    
    // Update predictions with bounded values
    slow_pred(t) = slow_pred(t-1) + r_slow * slow_pred(t-1) * available_space - slow_consumed;
    slow_pred(t) = slow_pred(t) > eps ? slow_pred(t) : eps;
    
    fast_pred(t) = fast_pred(t-1) + r_fast * fast_pred(t-1) * available_space - fast_consumed;
    fast_pred(t) = fast_pred(t) > eps ? fast_pred(t) : eps;
  }
  
  // Simple observation model with constant SD
  for(int t = 0; t < n; t++) {
    // COTS observations
    nll -= dnorm(cots_dat(t), cots_pred(t), obs_sd_cots, true);
    
    // Coral cover observations  
    nll -= dnorm(slow_dat(t), slow_pred(t), obs_sd_coral, true);
    nll -= dnorm(fast_dat(t), fast_pred(t), obs_sd_coral, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
