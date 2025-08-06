#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_VECTOR(Year);                  // Time vector (years)
  DATA_VECTOR(cots_dat);             // Observed COTS density (individuals/m2)
  DATA_VECTOR(slow_dat);             // Observed slow-growing coral cover (%)
  DATA_VECTOR(fast_dat);             // Observed fast-growing coral cover (%)
  DATA_VECTOR(sst_dat);              // Sea surface temperature (Celsius)
  DATA_VECTOR(cotsimm_dat);          // COTS immigration rate (individuals/m2/year)
  
  // Parameters
  PARAMETER(r_cots);                 // COTS intrinsic growth rate
  PARAMETER(K_cots);                 // COTS carrying capacity
  PARAMETER(temp_opt_cots);          // Optimal temperature for COTS
  PARAMETER(temp_tol_cots);          // Temperature tolerance for COTS
  PARAMETER(temp_opt_coral);         // Optimal temperature for coral growth
  PARAMETER(temp_tol_coral);         // Temperature tolerance for coral
  PARAMETER(g_slow);                 // Slow coral growth rate
  PARAMETER(g_fast);                 // Fast coral growth rate
  PARAMETER(K_coral);                // Total coral carrying capacity
  PARAMETER(alpha_slow);             // Predation rate on slow corals
  PARAMETER(alpha_fast);             // Predation rate on fast corals
  PARAMETER(h_coral);                // Half-saturation for predation
  PARAMETER(sigma_cots);             // SD for COTS observations
  PARAMETER(sigma_coral);            // SD for coral observations

  // Initialize negative log-likelihood
  Type nll = 0.0;
  
  // Initialize vectors for predictions
  int n = Year.size();
  vector<Type> cots_pred(n);
  vector<Type> slow_pred(n);
  vector<Type> fast_pred(n);
  
  // Constants to prevent numerical issues
  Type eps = Type(1e-8);
  
  // Initialize with first observation
  cots_pred(0) = cots_dat(0);
  slow_pred(0) = slow_dat(0);
  fast_pred(0) = fast_dat(0);
  
  // Time series predictions
  for(int t = 1; t < n; t++) {
    // 1. Temperature effects (bounded between 0.2 and 1)
    Type temp_sq_cots = pow(sst_dat(t-1) - temp_opt_cots, 2);
    Type temp_sq_coral = pow(sst_dat(t-1) - temp_opt_coral, 2);
    
    Type temp_effect_cots = Type(0.2) + Type(0.8) * exp(-temp_sq_cots / (2 * pow(temp_tol_cots, 2) + Type(1)));
    Type temp_effect_coral = Type(0.2) + Type(0.8) * exp(-temp_sq_coral / (2 * pow(temp_tol_coral, 2) + Type(1)));
    
    // 2. Total coral cover for density dependence
    Type total_coral = slow_pred(t-1) + fast_pred(t-1);
    
    // 3. Coral-dependent carrying capacity
    Type effective_K = K_cots * total_coral / (total_coral + h_coral);
    
    // 4. COTS population dynamics with bounded growth
    Type density_effect = Type(1) - cots_pred(t-1) / (effective_K + eps);
    density_effect = CppAD::CondExpGe(density_effect, Type(0), density_effect, Type(0));
    
    Type cots_growth = r_cots * temp_effect_cots * cots_pred(t-1) * density_effect;
    
    cots_pred(t) = cots_pred(t-1) + cots_growth + cotsimm_dat(t-1);
    cots_pred(t) = CppAD::CondExpGe(cots_pred(t), Type(eps), cots_pred(t), Type(eps));
    
    // 5. Coral predation rates with functional response
    Type pred_slow = (alpha_slow * cots_pred(t-1) * slow_pred(t-1)) / 
                    (h_coral + slow_pred(t-1) + fast_pred(t-1));
    Type pred_fast = (alpha_fast * cots_pred(t-1) * fast_pred(t-1)) / 
                    (h_coral + slow_pred(t-1) + fast_pred(t-1));
    
    // 6. Coral dynamics with logistic growth and predation
    Type available_space = CppAD::CondExpGe(K_coral - (slow_pred(t-1) + fast_pred(t-1)), Type(0),
      (K_coral - (slow_pred(t-1) + fast_pred(t-1))) / K_coral,
      Type(0));
    
    Type slow_change = g_slow * temp_effect_coral * slow_pred(t-1) * available_space - pred_slow;
    slow_change = CppAD::CondExpGe(slow_change, Type(-slow_pred(t-1)), slow_change, Type(-slow_pred(t-1)));
    slow_pred(t) = slow_pred(t-1) + slow_change;
    
    Type fast_change = g_fast * temp_effect_coral * fast_pred(t-1) * available_space - pred_fast;
    fast_change = CppAD::CondExpGe(fast_change, Type(-fast_pred(t-1)), fast_change, Type(-fast_pred(t-1)));
    fast_pred(t) = fast_pred(t-1) + fast_change;
    
    // Ensure predictions stay positive
    slow_pred(t) = CppAD::CondExpGe(slow_pred(t), Type(0), slow_pred(t), Type(eps));
    fast_pred(t) = CppAD::CondExpGe(fast_pred(t), Type(0), fast_pred(t), Type(eps));
  }
  
  // Observation model using log-normal distribution with bounds
  for(int t = 0; t < n; t++) {
    Type obs_cots = CppAD::CondExpGe(cots_dat(t), Type(0.1), cots_dat(t), Type(0.1));
    Type pred_cots = CppAD::CondExpGe(cots_pred(t), Type(0.1), cots_pred(t), Type(0.1));
    Type obs_slow = CppAD::CondExpGe(slow_dat(t), Type(0.1), slow_dat(t), Type(0.1));
    Type pred_slow = CppAD::CondExpGe(slow_pred(t), Type(0.1), slow_pred(t), Type(0.1));
    Type obs_fast = CppAD::CondExpGe(fast_dat(t), Type(0.1), fast_dat(t), Type(0.1));
    Type pred_fast = CppAD::CondExpGe(fast_pred(t), Type(0.1), fast_pred(t), Type(0.1));
    
    nll -= dnorm(log(obs_cots), log(pred_cots), sigma_cots, true);
    nll -= dnorm(log(obs_slow), log(pred_slow), sigma_coral, true);
    nll -= dnorm(log(obs_fast), log(pred_fast), sigma_coral, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
