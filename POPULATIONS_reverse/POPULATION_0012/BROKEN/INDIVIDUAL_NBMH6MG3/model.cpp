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
  PARAMETER(temp_opt);               // Optimal temperature for COTS
  PARAMETER(temp_tol);               // Temperature tolerance for COTS
  PARAMETER(temp_opt_slow);          // Optimal temperature for slow corals
  PARAMETER(temp_tol_slow);          // Temperature tolerance for slow corals
  PARAMETER(temp_opt_fast);          // Optimal temperature for fast corals  
  PARAMETER(temp_tol_fast);          // Temperature tolerance for fast corals
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
    // 1. Temperature effect on COTS growth (Gaussian response)
    Type temp_effect = exp(-pow(sst_dat(t-1) - temp_opt, 2) / (2 * pow(temp_tol, 2)));
    
    // 2. Total coral cover for density dependence
    Type total_coral = slow_pred(t-1) + fast_pred(t-1);
    
    // 3. Coral-dependent carrying capacity
    Type effective_K = K_cots * total_coral / (total_coral + h_coral);
    
    // 4. COTS population dynamics with temperature effect and immigration
    Type density_effect = CppAD::CondExpGt(effective_K, eps,
                         (Type(1) - cots_pred(t-1) / effective_K),
                         Type(0));
    Type cots_growth = r_cots * temp_effect * cots_pred(t-1) * density_effect;
    
    // Ensure predictions stay positive but bounded
    cots_pred(t) = CppAD::CondExpLt(
                     cots_pred(t-1) + cots_growth + cotsimm_dat(t-1),
                     eps,
                     eps,
                     cots_pred(t-1) + cots_growth + cotsimm_dat(t-1));
    
    // 5. Coral predation rates with functional response
    Type total_coral_available = slow_pred(t-1) + fast_pred(t-1) + h_coral;
    Type pred_slow = CppAD::CondExpGt(total_coral_available, eps,
                    (alpha_slow * cots_pred(t-1) * slow_pred(t-1)) / total_coral_available,
                    Type(0));
    Type pred_fast = CppAD::CondExpGt(total_coral_available, eps,
                    (alpha_fast * cots_pred(t-1) * fast_pred(t-1)) / total_coral_available,
                    Type(0));
    
    // 6. Temperature effect on coral growth (bounded between 0 and 1)
    Type temp_effect_slow = exp(-pow(sst_dat(t-1) - temp_opt_slow, 2) / (2 * pow(temp_tol_slow + eps, 2)));
    Type temp_effect_fast = exp(-pow(sst_dat(t-1) - temp_opt_fast, 2) / (2 * pow(temp_tol_fast + eps, 2)));
    temp_effect_slow = CppAD::CondExpGt(temp_effect_slow, Type(1), Type(1), temp_effect_slow);
    temp_effect_fast = CppAD::CondExpGt(temp_effect_fast, Type(1), Type(1), temp_effect_fast);
    
    // 7. Coral dynamics with bounded growth and predation
    Type total_cover = slow_pred(t-1) + fast_pred(t-1);
    Type available_space = CppAD::CondExpGt(total_cover, K_coral,
                                          Type(0),
                                          (K_coral - total_cover) / K_coral);
    
    Type slow_growth = g_slow * temp_effect_slow * slow_pred(t-1) * available_space;
    Type fast_growth = g_fast * temp_effect_fast * fast_pred(t-1) * available_space;
    
    slow_pred(t) = CppAD::CondExpLt(slow_pred(t-1) + slow_growth - pred_slow, Type(0),
                                   Type(eps),
                                   slow_pred(t-1) + slow_growth - pred_slow);
    fast_pred(t) = CppAD::CondExpLt(fast_pred(t-1) + fast_growth - pred_fast, Type(0),
                                   Type(eps),
                                   fast_pred(t-1) + fast_growth - pred_fast);
  }
  
  // Observation model using log-normal distribution
  for(int t = 0; t < n; t++) {
    // Use CppAD::CondExpGt to prevent taking log of negative values
    Type cots_obs = CppAD::CondExpLt(cots_dat(t), eps, eps, cots_dat(t));
    Type cots_mod = CppAD::CondExpLt(cots_pred(t), eps, eps, cots_pred(t));
    Type slow_obs = CppAD::CondExpLt(slow_dat(t), eps, eps, slow_dat(t));
    Type slow_mod = CppAD::CondExpLt(slow_pred(t), eps, eps, slow_pred(t));
    Type fast_obs = CppAD::CondExpLt(fast_dat(t), eps, eps, fast_dat(t));
    Type fast_mod = CppAD::CondExpLt(fast_pred(t), eps, eps, fast_pred(t));
    
    nll -= dnorm(log(cots_obs), log(cots_mod), sigma_cots, true);
    nll -= dnorm(log(slow_obs), log(slow_mod), sigma_coral, true);
    nll -= dnorm(log(fast_obs), log(fast_mod), sigma_coral, true);
  }
  
  // Report predictions
  REPORT(cots_pred);
  REPORT(slow_pred);
  REPORT(fast_pred);
  
  return nll;
}
